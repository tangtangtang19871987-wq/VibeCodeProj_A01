"""Gradient-based numerical ILT solver (CPU).

Provenance note (G-015): arXiv 2602.19027 does NOT specify its ILT solver. It
refers to CurvyILT [ref. 4, Yang & Ren, ISPD'25] for both the solver and the
morphological MRC handling, and gives no step size, optimizer, mask
parameterization, loss weights, or convergence rule. Everything in this module
is therefore a **reconstruction** in the documented MOSAIC / GAN-OPC / CurvyILT
lineage, not a transcription. Every constant is config-exposed and logged.

What IS taken from the paper:
* the forward model and constant-threshold resist (Eqs. 1-2);
* the curvilinear-mask assumption with morphological opening/cleanup for MRC
  compliance before printability evaluation (Sec. 2 and Sec. 4.1);
* the iteration budgets used for comparison (150 ours / 300 baseline, Table 1).

No CUDA is used anywhere; this runs on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from gril.litho.resist import LithoModel, resist


@dataclass
class ILTConfig:
    """Solver settings. Defaults are ours, not the paper's (see G-015)."""

    iterations: int = 150
    step_size: float = 1.0
    mask_steepness: float = 4.0        # beta_m in M = sigmoid(beta_m * P)
    weight_nominal: float = 1.0        # L2 against the target at the nominal corner
    weight_pvb: float = 0.0            # process-window term ||Z_max - Z_min||^2
    init_scale: float = 2.0            # P_0 = init_scale * (2*target - 1)
    optimizer: str = "adam"            # "adam" | "sgd"
    early_stop_rtol: float = 0.0       # 0 disables; fixed budgets keep comparisons exact
    mrc_open_size: int = 0             # morphological opening kernel, 0 = off
    checkpoints: tuple[int, ...] = ()  # iteration counts at which to snapshot the mask
    seed: int = 0


@dataclass
class ILTResult:
    """Outcome of one ILT run."""

    mask: torch.Tensor                 # binary {0,1}, (B,H,W) or (H,W)
    params: torch.Tensor               # raw parameter field P
    loss_history: list[float] = field(default_factory=list)
    iterations_run: int = 0
    seconds: float = 0.0
    #: iteration count -> binarised mask, for iteration-budget curves (X-10).
    checkpoint_masks: dict[int, torch.Tensor] = field(default_factory=dict)


def morphological_open(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Binary opening (erosion then dilation) with a square structuring element.

    Used for curvilinear MRC cleanup, as the paper specifies (Sec. 4.1). Operates
    on a binary ``{0,1}`` tensor of shape ``(B,H,W)`` or ``(H,W)``.
    """
    if size <= 1:
        return mask
    squeeze = mask.dim() == 2
    x = mask.unsqueeze(0) if squeeze else mask
    x = x.unsqueeze(1)
    pad = size // 2
    eroded = -F.max_pool2d(-x, kernel_size=size, stride=1, padding=pad)
    opened = F.max_pool2d(eroded, kernel_size=size, stride=1, padding=pad)
    opened = opened[:, :, : x.shape[-2], : x.shape[-1]].squeeze(1)
    return opened.squeeze(0) if squeeze else opened


def ilt_loss(
    params: torch.Tensor, target: torch.Tensor, litho: LithoModel, cfg: ILTConfig
) -> torch.Tensor:
    """Differentiable ILT objective.

    ``M = sigmoid(beta_m * P)``; the loss is the squared error of the nominal
    resist image against the target, plus an optional process-window term.
    Returns a scalar (summed over the batch).
    """
    mask = torch.sigmoid(cfg.mask_steepness * params)
    z_nom = resist(litho.aerial_nominal(mask), litho.cfg)
    loss = cfg.weight_nominal * ((z_nom - target) ** 2).sum()
    if cfg.weight_pvb > 0:
        # Only pay for the outer corners when the process-window term is active.
        i_max, i_min = litho.aerial_outer(mask)
        z_max, z_min = resist(i_max, litho.cfg), resist(i_min, litho.cfg)
        loss = loss + cfg.weight_pvb * ((z_max - z_min) ** 2).sum()
    return loss


def solve(
    target: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig | None = None,
    init_params: torch.Tensor | None = None,
    progress: bool = False,
) -> ILTResult:
    """Run gradient-based ILT.

    Parameters
    ----------
    target:
        Binary ``{0,1}`` design, ``(H,W)`` or ``(B,H,W)``, float32, CPU.
    litho:
        Three-corner forward model.
    cfg:
        Solver configuration.
    init_params:
        Optional warm start for ``P``. When ``None``, ``P_0 = init_scale*(2*target-1)``,
        i.e. the design itself — the standard cold start. A generative warm start
        supplies this tensor instead.
    progress:
        Print the loss every 25 iterations.

    Returns
    -------
    ILTResult
        ``mask`` is the binarised (and optionally MRC-opened) result.
    """
    cfg = cfg or ILTConfig()
    torch.manual_seed(cfg.seed)

    params = (
        (cfg.init_scale * (2.0 * target - 1.0)).clone()
        if init_params is None
        else init_params.clone()
    ).requires_grad_(True)

    opt = (
        torch.optim.Adam([params], lr=cfg.step_size)
        if cfg.optimizer == "adam"
        else torch.optim.SGD([params], lr=cfg.step_size)
    )

    def _binarize(p: torch.Tensor) -> torch.Tensor:
        m = (torch.sigmoid(cfg.mask_steepness * p) >= 0.5).to(target.dtype)
        return morphological_open(m, cfg.mrc_open_size) if cfg.mrc_open_size > 1 else m

    history: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    start = time.time()
    ran = 0
    for step in range(cfg.iterations):
        opt.zero_grad(set_to_none=True)
        loss = ilt_loss(params, target, litho, cfg)
        loss.backward()
        opt.step()
        history.append(loss.detach().item())
        ran = step + 1
        if ran in cfg.checkpoints:
            with torch.no_grad():
                snapshots[ran] = _binarize(params)
        if progress and step % 25 == 0:
            print(f"    iter {step:4d}  loss {float(loss):.1f}", flush=True)
        if cfg.early_stop_rtol > 0 and len(history) > 10:
            rel = abs(history[-11] - history[-1]) / max(abs(history[-11]), 1e-12)
            if rel < cfg.early_stop_rtol:
                break

    with torch.no_grad():
        mask = _binarize(params)

    return ILTResult(
        mask=mask,
        params=params.detach(),
        loss_history=history,
        iterations_run=ran,
        seconds=time.time() - start,
        checkpoint_masks=snapshots,
    )
