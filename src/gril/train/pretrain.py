"""Stage-1 generative pretraining: WGAN-GP + reconstruction (paper Eq. 5).

    min_G max_D  E_M[D(M,Z)] - E_{Z,q}[D(G(Z,q),Z)]
                 + lambda_1 * E[ l_rec(G(Z,q), M) ]
                 - lambda_2 * E[ (||grad_{M_hat} D(M_hat,Z)||_2 - 1)^2 ]

with ``M_hat = eps*M + (1-eps)*G(Z,q)``, ``eps ~ U(0,1)``.

The paper leaves ``l_rec`` ("e.g. l1 or l2"), ``lambda_1`` and ``lambda_2``
unspecified (G-020); they are configuration here.

CPU-only; no CUDA paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass
class PretrainConfig:
    """Stage-1 hyperparameters. (paper) marks values given in Sec. 4.1."""

    epochs: int = 50                       # (paper)
    batch_size: int = 16                   # (paper)
    lr_discriminator: float = 2e-4         # (paper)
    betas_discriminator: tuple = (0.5, 0.999)   # (paper)
    lr_generator: float = 1e-4             # paper uses Prodigy; Adam here (G-020)
    n_critic: int = 5                      # not given by the paper
    lambda_rec: float = 100.0              # lambda_1, not given by the paper
    lambda_gp: float = 10.0                # lambda_2, not given by the paper
    rec_loss: Literal["l1", "l2"] = "l1"   # paper says "e.g. l1 or l2"
    latent_dim: int = 256                  # (paper)
    seed: int = 0


def gradient_penalty(
    critic: torch.nn.Module, real: torch.Tensor, fake: torch.Tensor, design: torch.Tensor
) -> torch.Tensor:
    """WGAN-GP term ``E[(||grad D(M_hat)||_2 - 1)^2]`` (paper Eq. 5)."""
    eps = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=real.dtype)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(interp, design)
    grads = torch.autograd.grad(
        outputs=scores.sum(), inputs=interp, create_graph=True, retain_graph=True
    )[0]
    norms = grads.flatten(1).norm(dim=1)
    return ((norms - 1.0) ** 2).mean()


def reconstruction_loss(fake_logits: torch.Tensor, real: torch.Tensor, kind: str) -> torch.Tensor:
    """``l_rec`` between ``sigmoid(Y)`` and the ground-truth mask."""
    pred = torch.sigmoid(fake_logits)
    return F.l1_loss(pred, real) if kind == "l1" else F.mse_loss(pred, real)


def critic_step(
    critic, generator, design, real_mask, cfg: PretrainConfig, opt_d
) -> dict[str, float]:
    """One discriminator update. Returns scalar diagnostics."""
    opt_d.zero_grad(set_to_none=True)
    with torch.no_grad():
        fake = torch.sigmoid(generator(design, torch.randn(design.shape[0], cfg.latent_dim)))
    # Critic maximises E[D(real)] - E[D(fake)]; we minimise the negation.
    wass = critic(fake, design).mean() - critic(real_mask, design).mean()
    gp = gradient_penalty(critic, real_mask, fake, design)
    loss = wass + cfg.lambda_gp * gp
    loss.backward()
    opt_d.step()
    return {
        "d_loss": loss.detach().item(),
        "wasserstein": (-wass).detach().item(),
        "gradient_penalty": gp.detach().item(),
    }


def generator_step(
    critic, generator, design, real_mask, cfg: PretrainConfig, opt_g
) -> dict[str, float]:
    """One generator update (adversarial + reconstruction)."""
    opt_g.zero_grad(set_to_none=True)
    logits = generator(design, torch.randn(design.shape[0], cfg.latent_dim))
    adv = -critic(torch.sigmoid(logits), design).mean()
    rec = reconstruction_loss(logits, real_mask, cfg.rec_loss)
    loss = adv + cfg.lambda_rec * rec
    loss.backward()
    opt_g.step()
    return {
        "g_loss": loss.detach().item(),
        "g_adv": adv.detach().item(),
        "g_rec": rec.detach().item(),
    }
