"""Reinforcement finetuning: GRPO + ILT-guided imitation (paper Sec. 3.3.2).

Implements Equations (6)-(10) literally.

**Documented paper defect (G-012):** Sec. 3.3.2 defines and argues for a
*teacher-relative* baseline (Eq. 6), while Sec. 4.1 states the runs used "a
self-critical baseline given by the group mean". These are different algorithms
and the paper does not reconcile them. Both are implemented and selectable; the
default follows Eqs. 6-7, and experiment X-07 measures the difference.

CPU-only; no CUDA paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """Finetuning hyperparameters. Values marked (paper) are from Sec. 4.1."""

    group_size: int = 16                 # K (paper)
    latent_dim: int = 256                # (paper)
    lambda_pg: float = 500.0             # (paper)
    lambda_imit: float = 1.0             # (paper)
    smoothing_kernel: int = 25           # 25x25 stride-1 average pool (paper)
    advantage: Literal["teacher_relative", "group_mean"] = "teacher_relative"
    logprob_reduction: Literal["mean", "sum"] = "mean"
    epe_tolerance: int = 3               # stricter threshold used in training (paper)
    seed: int = 0


def smooth_target(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    """``S(.)``: mild low-pass smoothing of the ILT-refined mask (paper Eq. 9).

    ``25x25`` stride-1 average pooling, as the paper specifies. Input/output
    ``(B,1,H,W)``.
    """
    if kernel <= 1:
        return mask
    # count_include_pad=False is load-bearing. With the default (True) the zero
    # padding enters the denominator, so a kernel//2 border is attenuated toward
    # zero -- at the RL loop's 256x256 resolution a 25x25 kernel would corrupt
    # ~5% of each edge and pull the generator toward 0 there. The paper does not
    # state the padding convention; preserving constants is the only choice that
    # keeps S(.) a genuine low-pass operator.
    return F.avg_pool2d(
        mask,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
        count_include_pad=False,
    )


def policy_log_prob(
    logits: torch.Tensor, actions: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """``log pi(M | Z, q)`` under a pixel-wise Bernoulli policy (paper Eq. 8).

    The paper treats pixels as independent Bernoulli with probability
    ``sigmoid(Y)`` and uses BCE as a surrogate for ``-log P``. ``actions`` must
    already be detached, so gradients flow only through ``logits``.

    Parameters
    ----------
    logits : ``(B,1,H,W)``
    actions : ``(B,1,H,W)`` binary ``{0,1}``, detached

    Returns
    -------
    ``(B,)`` log-probabilities.
    """
    bce = F.binary_cross_entropy_with_logits(logits, actions, reduction="none")
    per_sample = bce.flatten(1)
    return -(per_sample.mean(1) if reduction == "mean" else per_sample.sum(1))


def advantages(
    rewards: torch.Tensor, teacher_rewards: torch.Tensor | None, mode: str
) -> torch.Tensor:
    """Group advantages ``A_k``.

    ``teacher_relative`` (paper Eq. 6): ``A_k = R_k - R_k^T``.
    ``group_mean`` (original GRPO, and what paper Sec. 4.1 says was used):
    ``A_k = R_k - mean_j R_j``.
    """
    if mode == "teacher_relative":
        if teacher_rewards is None:
            raise ValueError("teacher_relative advantage requires teacher rewards")
        return rewards - teacher_rewards
    if mode == "group_mean":
        return rewards - rewards.mean()
    raise ValueError(f"unknown advantage mode: {mode}")


def finetune_step(
    generator: torch.nn.Module,
    teacher: torch.nn.Module | None,
    design: torch.Tensor,
    refine: Callable[[torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: GRPOConfig,
    generator_rng: torch.Generator | None = None,
) -> dict[str, float | torch.Tensor]:
    """One GRPO finetuning step on a single design (paper: batch 8, one design per step).

    Parameters
    ----------
    generator:
        Policy ``G``; receives gradients.
    teacher:
        Frozen pretrained ``G_T`` used for the teacher-relative baseline. Must be
        in eval mode with gradients disabled. May be ``None`` for ``group_mean``.
    design:
        ``(1,1,H,W)`` binary design ``Z``.
    refine:
        ``M -> M_ILT``: the short low-resolution ILT loop, batched over K.
    reward_fn:
        ``M_ILT -> (K,)`` rewards, i.e. ``-EPE(M_ILT, Z)``.
    cfg:
        Hyperparameters.

    Returns
    -------
    dict with the total loss tensor (``loss``, differentiable) plus scalar
    diagnostics: ``loss_pg``, ``loss_imit``, ``reward_mean``, ``reward_std``,
    ``advantage_mean``, ``advantage_std``.
    """
    k = cfg.group_size
    z = torch.randn(k, cfg.latent_dim, generator=generator_rng)
    designs = design.expand(k, -1, -1, -1)

    logits = generator(designs, z)                        # (K,1,H,W)
    # Actions are binarised and DETACHED (paper Eq. 8).
    actions = (logits > 0.5).to(logits.dtype).detach()

    with torch.no_grad():
        refined = refine(actions)                         # (K,1,H,W)
        rewards = reward_fn(refined)                      # (K,)

        teacher_rewards = None
        if cfg.advantage == "teacher_relative":
            if teacher is None:
                raise ValueError("teacher_relative advantage requires a teacher model")
            t_logits = teacher(designs, z)
            t_actions = (t_logits > 0.5).to(t_logits.dtype)
            teacher_rewards = reward_fn(refine(t_actions))

        adv = advantages(rewards, teacher_rewards, cfg.advantage)

    log_p = policy_log_prob(logits, actions, cfg.logprob_reduction)
    loss_pg = -(adv * log_p).mean()                       # paper Eq. 8

    target = smooth_target(refined, cfg.smoothing_kernel)
    loss_imit = ((logits - target) ** 2).flatten(1).mean(1).mean()   # paper Eq. 9

    loss = cfg.lambda_pg * loss_pg + cfg.lambda_imit * loss_imit     # paper Eq. 10

    return {
        "loss": loss,
        "loss_pg": loss_pg.detach().item(),
        "loss_imit": loss_imit.detach().item(),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()) if k > 1 else 0.0,
        "advantage_mean": float(adv.mean()),
        "advantage_std": float(adv.std()) if k > 1 else 0.0,
    }
