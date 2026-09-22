"""Style-aware conditional mask generator (paper Sec. 3.2.1, Fig. 3).

Topology is taken directly from Figure 3 and Equation (4):

* an **input pyramid** ``{Z_0, Z_1, Z_2}`` built by downsampling, with an
  optional head downsample;
* **Level 2** (coarsest): stride-2 downsampling, a trunk of **Style ResBlocks**
  modulated by AdaIN with style code ``w``, then UpConv to a "low mask";
* **Levels 1 and 0**: coarse-to-fine. The upsampled output of the previous level
  is fused with locally-downsampled features at the current resolution by
  **element-wise addition**, refined by **Local ResBlocks**, then UpConv;
* an optional final **bicubic** interpolation back to the input resolution.

Style enters **only at Level 2** (the paper's explicit design choice), so content
comes from ``Z`` and style from ``w``.

Channel widths and the number of blocks are NOT given by the paper (G-010) and
are therefore configuration, not constants.

CPU-only; no CUDA paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GeneratorConfig:
    """Sizing knobs. Topology is fixed by Fig. 3; these sizes are ours (G-010)."""

    latent_dim: int = 256          # paper: z-dim 256 (Sec. 4.1)
    style_dim: int = 256
    base_channels: int = 32
    n_style_blocks: int = 4        # "Style ResBlocks x n"; n not given by the paper
    n_local_blocks: int = 2
    mapping_layers: int = 4
    head_downsample: int = 1       # "optional head downsample"; 1 = off
    #: Scale applied to the style code w. Implements the paper's Sec. 3.2.2
    #: statement that z is treated as a reparameterised Gaussian whose scale (or
    #: truncation in w) is controlled "to anneal diversity across training
    #: stages". With an under-trained generator the logit spread across latents
    #: can fall far below the 0.5 binarisation threshold, so every sampled mask
    #: binarises identically, every advantage is 0, and the policy gradient
    #: vanishes. This knob is the paper's own remedy.
    style_scale: float = 1.0


class StyleMapping(nn.Module):
    """Lightweight MLP ``f_phi: z -> w`` (paper Sec. 3.2.1)."""

    def __init__(self, latent_dim: int, style_dim: int, layers: int):
        super().__init__()
        seq: list[nn.Module] = []
        d = latent_dim
        for _ in range(layers - 1):
            seq += [nn.Linear(d, style_dim), nn.LeakyReLU(0.2)]
            d = style_dim
        seq += [nn.Linear(d, style_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: ``(B, latent_dim)`` -> ``w``: ``(B, style_dim)``."""
        return self.net(z)


class AdaIN(nn.Module):
    """Adaptive instance normalization, paper Equation (4).

        AdaIN(x; w) = gamma(w) * (x - mu(x)) / (sigma(x) + eps) + beta(w)

    ``mu`` and ``sigma`` are per-sample, per-channel spatial statistics.
    """

    def __init__(self, channels: int, style_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = nn.Linear(style_dim, 2 * channels)
        # Start NEAR identity (gamma ~ 1, beta ~ 0) but keep a small w-dependent
        # component. A hard zero-init would make AdaIN independent of w, so the
        # generator would emit identical masks for every latent -- mode collapse
        # by construction, which is precisely what this architecture exists to
        # avoid. Pinned by test_generator.py::test_distinct_latents_give_distinct_masks.
        nn.init.normal_(self.affine.weight, std=0.02)
        with torch.no_grad():
            self.affine.bias[:channels] = 1.0
            self.affine.bias[channels:] = 0.0

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B,C,H,W)``, ``w``: ``(B,style_dim)`` -> ``(B,C,H,W)``."""
        mu = x.mean(dim=(2, 3), keepdim=True)
        sigma = x.std(dim=(2, 3), keepdim=True, unbiased=False)
        gamma, beta = self.affine(w).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * (x - mu) / (sigma + self.eps) + beta


class StyleResBlock(nn.Module):
    """Fig. 3 inset: ``x -> AdaIN(w) -> Conv -> AdaIN(w) -> Conv -> (+x) -> y``."""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm1 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = AdaIN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x, w)))
        h = self.conv2(self.act(self.norm2(h, w)))
        return x + h


class LocalResBlock(nn.Module):
    """Plain (style-free) residual block used at Levels 1 and 0."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class StyleAwareGenerator(nn.Module):
    """Conditional mask sampler ``G(Z, q) -> Y`` (logits).

    Inputs
    ------
    design : ``(B, 1, H, W)`` float32, values in ``{0,1}``
    z      : ``(B, latent_dim)`` float32, ``z ~ N(0, I)``

    Output
    ------
    ``(B, 1, H, W)`` float32 **logits**. The mask is ``sigmoid(Y)``; the binary
    action is ``M = 1[Y > 0.5]`` exactly as the paper writes it (Sec. 3.3.2).
    """

    def __init__(self, cfg: GeneratorConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or GeneratorConfig()
        c = cfg.base_channels

        self.mapping = StyleMapping(cfg.latent_dim, cfg.style_dim, cfg.mapping_layers)

        # ---- Level 2 (coarsest): stride-2 downsample -> style trunk -> UpConv
        self.l2_in = nn.Conv2d(1, c, 3, padding=1)
        self.l2_down = nn.Sequential(
            nn.Conv2d(c, 2 * c, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(2 * c, 4 * c, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.l2_style = nn.ModuleList(
            [StyleResBlock(4 * c, cfg.style_dim) for _ in range(cfg.n_style_blocks)]
        )
        self.l2_up = nn.Sequential(
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.l2_out = nn.Conv2d(c, 1, 3, padding=1)

        # ---- Levels 1 and 0: local downsample + additive fusion + local ResBlocks
        self.l1_local = nn.Sequential(nn.Conv2d(1, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.l1_fuse = nn.Conv2d(1, c, 3, padding=1)
        self.l1_blocks = nn.Sequential(*[LocalResBlock(c) for _ in range(cfg.n_local_blocks)])
        self.l1_up = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.l1_out = nn.Conv2d(c, 1, 3, padding=1)

        self.l0_local = nn.Sequential(nn.Conv2d(1, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.l0_fuse = nn.Conv2d(1, c, 3, padding=1)
        self.l0_blocks = nn.Sequential(*[LocalResBlock(c) for _ in range(cfg.n_local_blocks)])
        self.l0_up = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
        self.l0_out = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, design: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        w = self.mapping(z) * cfg.style_scale

        z0 = design
        if cfg.head_downsample > 1:
            z0 = F.avg_pool2d(z0, cfg.head_downsample)
        z1 = F.avg_pool2d(z0, 2)
        z2 = F.avg_pool2d(z1, 2)

        # Level 2 -- the only place style is injected.
        h = self.l2_down(self.l2_in(z2))
        for blk in self.l2_style:
            h = blk(h, w)
        low2 = self.l2_out(self.l2_up(h))                       # (B,1,H/4,W/4)

        # Level 1: fuse upsampled level-2 output with local features (addition).
        up2 = F.interpolate(low2, size=z1.shape[-2:], mode="bilinear", align_corners=False)
        h1 = self.l1_local(z1) + self.l1_fuse(F.avg_pool2d(up2, 2))
        h1 = self.l1_up(self.l1_blocks(h1))
        low1 = self.l1_out(h1) + up2                            # (B,1,H/2,W/2)

        # Level 0: same pattern at full resolution.
        up1 = F.interpolate(low1, size=z0.shape[-2:], mode="bilinear", align_corners=False)
        h0 = self.l0_local(z0) + self.l0_fuse(F.avg_pool2d(up1, 2))
        h0 = self.l0_up(self.l0_blocks(h0))
        out = self.l0_out(h0) + up1

        if cfg.head_downsample > 1:
            # "An optional final bicubic interpolation restores the original resolution."
            out = F.interpolate(
                out, size=design.shape[-2:], mode="bicubic", align_corners=False
            )
        return out

    def sample_latent(self, batch: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``z ~ N(0, I)``, shape ``(batch, latent_dim)``."""
        return torch.randn(batch, self.cfg.latent_dim, generator=generator)


class Discriminator(nn.Module):
    """Conditional WGAN-GP critic ``D(M, Z) -> R`` (paper Eq. 5).

    No final activation and no batch norm (required for a valid gradient penalty).
    """

    def __init__(self, base_channels: int = 32, levels: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_c, c = 2, base_channels          # 2 channels: mask and design
        for _ in range(levels):
            layers += [nn.Conv2d(in_c, c, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
            in_c, c = c, min(c * 2, 256)
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_c, 1, 3, padding=1)

    def forward(self, mask: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
        """``mask``/``design``: ``(B,1,H,W)`` -> critic score ``(B,)``."""
        h = self.head(self.body(torch.cat([mask, design], dim=1)))
        return h.mean(dim=(1, 2, 3))
