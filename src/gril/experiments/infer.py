"""Posterior sampling + batched ILT refinement + best-candidate selection.

This is the paper's inference loop (abstract, Fig. 1c, Sec. 3.3.2): sample a
small batch of masks from the generator, run fast batched ILT refinement on all
of them, evaluate lithography metrics, and select the best candidate.

Also provides the **low-resolution refinement loop** used for the RL reward
(paper Sec. 4.1: downsample factor 8, 100 iterations, upsample via low-res
pooling followed by bicubic interpolation, binarize at 0.5).

CPU-only; no CUDA paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel
from gril.metrics.core import epe_violations, l2_loss, pv_band


@dataclass
class SamplingConfig:
    """Inference-time sampling. (paper) marks values from Sec. 4.1."""

    group_size: int = 16          # K (paper)
    latent_dim: int = 256         # (paper)
    selection_metric: str = "epe"  # "epe" | "l2" | "pvb"
    epe_tolerance: float = 3      # (paper) training/stress threshold, NANOMETRES
    pixel_nm: float = 1.0         # canvas pixel pitch; 1.0 on the 2048 ICCAD13 canvas
    seed: int = 0


def low_res_refine(
    masks: torch.Tensor,
    target: torch.Tensor,
    litho: LithoModel,
    downsample: int = 8,
    iterations: int = 100,
    step_size: float = 0.2,
    mask_steepness: float = 8.0,
) -> torch.Tensor:
    """The paper's short, low-resolution ILT loop used for the RL reward.

    Downsamples by ``downsample``, refines for ``iterations`` steps, then
    upsamples back with average pooling + **bicubic** interpolation and
    binarizes at 0.5, exactly as Sec. 4.1 describes.

    This is physically sound because the SOCS kernels span a fixed *physical*
    extent of the canvas, so coarsening the pixel grid preserves the optics --
    verified to 0.17% of peak at 8x in ``tests_gril/physics``.

    Parameters
    ----------
    masks : ``(K,1,H,W)`` binary initial masks
    target : ``(1,1,H,W)`` or ``(H,W)`` binary design

    Returns
    -------
    ``(K,1,H,W)`` binary refined masks.
    """
    full_hw = masks.shape[-2:]
    tgt = target.view(1, 1, *full_hw)

    m_low = F.avg_pool2d(masks, downsample)
    t_low = (F.avg_pool2d(tgt, downsample) > 0.5).float()

    # Warm-start the parameter field from the sampled mask.
    init = mask_steepness * (2.0 * m_low - 1.0)
    res = solve(
        t_low.expand(m_low.shape[0], -1, -1, -1).squeeze(1),
        litho,
        ILTConfig(
            iterations=iterations,
            step_size=step_size,
            mask_steepness=mask_steepness,
            init_scale=1.0,
        ),
        init_params=init.squeeze(1),
    )
    refined = res.mask.unsqueeze(1)
    up = F.interpolate(refined, size=full_hw, mode="bicubic", align_corners=False)
    return (up > 0.5).float()


def score_candidates(
    masks: torch.Tensor,
    target: torch.Tensor,
    litho: LithoModel,
    tolerance: float,
    pixel_nm: float = 1.0,
) -> list[dict]:
    """Score each candidate mask. ``masks``: ``(K,1,H,W)``; returns K dicts."""
    out = []
    tgt2d = target.view(*target.shape[-2:])
    with torch.no_grad():
        for k in range(masks.shape[0]):
            b_nom, b_max, b_min = litho.binary(masks[k, 0])
            ein, eout = epe_violations(b_nom, tgt2d, tolerance, pixel_nm)
            out.append(
                {
                    "index": k,
                    "l2": l2_loss(b_nom, tgt2d),
                    "pvb": pv_band(b_max, b_min),
                    "epe": ein + eout,
                }
            )
    return out


def select_best(scores: list[dict], metric: str = "epe") -> dict:
    """Pick the best candidate. Deterministic tie-break: metric, then L2, then index."""
    return min(scores, key=lambda s: (s[metric], s["l2"], s["index"]))


def sample_and_refine(
    generator: torch.nn.Module,
    design: torch.Tensor,
    litho: LithoModel,
    cfg: SamplingConfig,
    ilt_cfg: ILTConfig | None = None,
) -> dict:
    """Full inference: sample K, batched-refine, score, select.

    Parameters
    ----------
    design : ``(1,1,H,W)`` binary design
    ilt_cfg : refinement budget; defaults to 150 iterations at full resolution.

    Returns
    -------
    dict with ``best`` (the winning score dict), ``scores`` (all K), ``masks``
    (``(K,1,H,W)`` refined), and ``diversity`` (mean pairwise L1 between the
    sampled masks before refinement -- 0 means the sampler collapsed).
    """
    ilt_cfg = ilt_cfg or ILTConfig(iterations=150, step_size=0.2, mask_steepness=8.0)
    gen_rng = torch.Generator().manual_seed(cfg.seed)

    with torch.no_grad():
        z = torch.randn(cfg.group_size, cfg.latent_dim, generator=gen_rng)
        logits = generator(design.expand(cfg.group_size, -1, -1, -1), z)
        initial = (logits > 0.5).float()

    k = cfg.group_size
    if k > 1:
        flat = initial.flatten(1)
        pairs = [
            (flat[i] - flat[j]).abs().mean().item()
            for i in range(k) for j in range(i + 1, k)
        ]
        diversity = sum(pairs) / len(pairs)
    else:
        diversity = 0.0

    target2d = design.view(*design.shape[-2:])
    init_params = ilt_cfg.mask_steepness * (2.0 * initial.squeeze(1) - 1.0)
    res = solve(
        target2d.unsqueeze(0).expand(k, -1, -1),
        litho,
        ilt_cfg,
        init_params=init_params,
    )
    refined = res.mask.unsqueeze(1)
    scores = score_candidates(refined, design, litho, cfg.epe_tolerance, cfg.pixel_nm)
    return {
        "best": select_best(scores, cfg.selection_metric),
        "scores": scores,
        "masks": refined,
        "diversity": diversity,
        "seconds": res.seconds,
    }
