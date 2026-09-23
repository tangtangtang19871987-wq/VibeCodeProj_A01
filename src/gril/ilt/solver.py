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

Optimizers
----------
Four are available, none dictated by the paper (G-015):

* ``"adam"`` (default, unchanged from earlier versions of this module --
  existing experiment results are reproduced bit-for-bit).
* ``"sgd"``  plain or classical-momentum gradient descent.
* ``"nesterov"``  Nesterov-accelerated gradient descent.
* ``"lbfgs"``  quasi-Newton L-BFGS with a strong-Wolfe line search
  (``torch.optim.LBFGS``). See the ``ILTConfig`` docstring for why an
  "iteration" of L-BFGS is not comparable to one of Adam/SGD, and why batched
  L-BFGS runs independently per example rather than jointly.

Beta annealing (continuation)
------------------------------
``ILTConfig.beta_init`` (default ``None`` = off, exactly the original constant-
beta behaviour) ramps the mask steepness from a low starting value up to
``mask_steepness`` over the course of the run instead of holding it fixed.
This addresses F-SAT-01 (docs/findings.md): the main ICCAD13 baseline's own
constants (``init_scale=2.0``, ``mask_steepness=8.0``) saturate the mask
sigmoid so hard that SGD/Nesterov/L-BFGS make no measurable progress from a
cold start, while Adam's per-parameter normalisation happens to be insensitive
to it. Annealing fixes the underlying conditioning problem directly rather
than routing around it with a different starting point (F-ANNEAL-01): verified
to let L-BFGS train from those exact saturated constants.

No CUDA is used anywhere; this runs on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from gril.litho.resist import LithoModel, resist

_KNOWN_OPTIMIZERS = ("adam", "sgd", "nesterov", "lbfgs")
_KNOWN_BETA_SCHEDULES = ("linear", "exponential")


@dataclass
class ILTConfig:
    """Solver settings. Defaults are ours, not the paper's (see G-015).

    A note on comparing optimizers fairly
    --------------------------------------
    ``iterations`` means one parameter *update* for every optimizer, so an
    "iteration" is NOT the same amount of compute across optimizers:

    * Adam/SGD/Nesterov: exactly one forward + one backward pass per iteration.
    * L-BFGS: one quasi-Newton step (``lbfgs_max_iter_per_step`` inner L-BFGS
      iterations, default 1), each of which runs a strong-Wolfe line search
      that may evaluate the closure (forward + backward) several times.

    ``ILTResult.n_func_evals`` records the actual number of forward/backward
    evaluations performed, which is the number to use for any *fair*
    cross-optimizer comparison -- comparing raw ``iterations`` between L-BFGS
    and Adam is comparing different units and is never done in this project's
    reports without also stating ``n_func_evals``.
    """

    iterations: int = 150
    step_size: float = 1.0
    mask_steepness: float = 4.0        # beta_m in M = sigmoid(beta_m * P)
    weight_nominal: float = 1.0        # L2 against the target at the nominal corner
    weight_pvb: float = 0.0            # process-window term ||Z_max - Z_min||^2
    weight_tv: float = 0.0             # total-variation mask-complexity regulariser, 0 = off
    init_scale: float = 2.0            # P_0 = init_scale * (2*target - 1)
    optimizer: str = "adam"            # "adam" | "sgd" | "nesterov" | "lbfgs"
    momentum: float = 0.0              # for "sgd" / "nesterov"; nesterov requires > 0
    grad_clip: float = 0.0             # max gradient norm; 0 disables clipping
    early_stop_rtol: float = 0.0       # 0 disables; fixed budgets keep comparisons exact
    convergence_grad_tol: float = 0.0  # 0 disables; stop when ||grad||_inf falls below this
    mrc_open_size: int = 0             # morphological opening kernel, 0 = off
    checkpoints: tuple[int, ...] = ()  # iteration counts at which to snapshot the mask
    # --- beta annealing / continuation (see F-SAT-01) ---
    #: If set, the mask steepness used DURING optimization starts here and
    #: anneals to `mask_steepness` by the final iteration, instead of using
    #: `mask_steepness` for the whole run. None (default) preserves the exact
    #: prior behaviour -- constant beta = mask_steepness throughout, bit-for-
    #: bit compatible with every committed result. Binarization of the output
    #: mask is UNAFFECTED by this: sigmoid(beta*P) >= 0.5 iff P >= 0 for any
    #: beta > 0, so the final {0,1} mask depends on beta only through the
    #: OPTIMIZATION TRAJECTORY it shapes, never through where the threshold
    #: falls. See docs/findings.md F-SAT-01 for why a fixed high beta stalls
    #: raw-gradient optimizers, and F-ANNEAL-01 for what this fixes.
    beta_init: float | None = None
    beta_schedule: str = "linear"      # "linear" | "exponential"; ignored if beta_init is None
    # --- L-BFGS-specific (ignored by the other optimizers) ---
    lbfgs_history_size: int = 10
    lbfgs_max_iter_per_step: int = 1   # inner L-BFGS iterations per solve() "iteration"
    lbfgs_line_search: str | None = "strong_wolfe"
    #: Reserved. The solver is deterministic, so this is recorded in experiment
    #: manifests for provenance but is intentionally NOT applied to the global RNG.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.optimizer not in _KNOWN_OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {self.optimizer!r}; expected one of {_KNOWN_OPTIMIZERS}. "
                "(Earlier versions of this module silently fell back to SGD for any "
                "unrecognised name -- that silent fallback has been removed.)"
            )
        if self.optimizer == "nesterov" and self.momentum <= 0:
            raise ValueError(
                "optimizer='nesterov' requires momentum > 0 (e.g. 0.9); "
                "Nesterov acceleration is undefined without it."
            )
        if self.beta_init is not None:
            if self.beta_schedule not in _KNOWN_BETA_SCHEDULES:
                raise ValueError(
                    f"Unknown beta_schedule {self.beta_schedule!r}; "
                    f"expected one of {_KNOWN_BETA_SCHEDULES}."
                )
            if self.beta_init <= 0:
                raise ValueError(f"beta_init must be > 0, got {self.beta_init}")
            if self.mask_steepness <= 0:
                raise ValueError(f"mask_steepness must be > 0, got {self.mask_steepness}")


@dataclass
class ILTResult:
    """Outcome of one ILT run."""

    mask: torch.Tensor                 # binary {0,1}, (B,H,W) or (H,W)
    params: torch.Tensor               # raw parameter field P
    loss_history: list[float] = field(default_factory=list)
    iterations_run: int = 0
    #: Total forward+backward (closure) evaluations. Equals iterations_run for
    #: Adam/SGD/Nesterov (one per iteration); typically LARGER than
    #: iterations_run for L-BFGS because of its internal line search. This is
    #: the number to use for a compute-fair comparison across optimizers.
    n_func_evals: int = 0
    optimizer: str = "adam"
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


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    """Anisotropic L1 total variation of a (continuous-relaxed) mask.

    ``sum |dM/dx| + |dM/dy|`` over adjacent pixels. Standard ILT-literature
    regulariser (e.g. MOSAIC): penalising boundary roughness in mask space is a
    cheap differentiable proxy for lower e-beam shot count / a simpler
    curvilinear boundary, both of which the paper cares about for MRC
    compliance (Sec. 2, Sec. 4.1) without prescribing this specific mechanism
    (G-015). Computed on the *continuous* mask so it is differentiable; applied
    with weight 0 (off) by default.
    """
    dh = (mask[..., 1:, :] - mask[..., :-1, :]).abs().sum()
    dw = (mask[..., :, 1:] - mask[..., :, :-1]).abs().sum()
    return dh + dw


def _beta_for_step(cfg: ILTConfig, step: int) -> float:
    """Mask steepness at optimizer step ``step`` (0-indexed).

    Returns ``cfg.mask_steepness`` unchanged when ``cfg.beta_init`` is None --
    this is what keeps every existing config's behaviour bit-for-bit identical.
    Otherwise interpolates from ``beta_init`` (step 0) to ``mask_steepness``
    (the final step), linearly or geometrically per ``cfg.beta_schedule``.
    """
    if cfg.beta_init is None:
        return cfg.mask_steepness
    total = max(cfg.iterations - 1, 1)
    frac = min(step / total, 1.0)
    if cfg.beta_schedule == "linear":
        return cfg.beta_init + (cfg.mask_steepness - cfg.beta_init) * frac
    # "exponential": geometric interpolation, i.e. linear in log(beta). Smoother
    # early ramp than "linear" when beta_init << mask_steepness, since beta
    # roughly doubles every fixed fraction of the run rather than growing by a
    # fixed additive amount each step.
    return cfg.beta_init * (cfg.mask_steepness / cfg.beta_init) ** frac


def ilt_loss(
    params: torch.Tensor,
    target: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig,
    beta: float | None = None,
) -> torch.Tensor:
    """Differentiable ILT objective.

    ``M = sigmoid(beta * P)``; the loss is the squared error of the nominal
    resist image against the target, plus optional process-window and
    total-variation terms. Returns a scalar (summed over the batch).

    Parameters
    ----------
    beta:
        Overrides ``cfg.mask_steepness`` for this call. ``None`` (default)
        uses ``cfg.mask_steepness`` directly -- existing call sites that don't
        pass this are completely unaffected. ``solve()`` passes the current
        annealed value here when ``cfg.beta_init`` is set.
    """
    mask = torch.sigmoid((cfg.mask_steepness if beta is None else beta) * params)
    z_nom = resist(litho.aerial_nominal(mask), litho.cfg)
    loss = cfg.weight_nominal * ((z_nom - target) ** 2).sum()
    if cfg.weight_pvb > 0:
        # Only pay for the outer corners when the process-window term is active.
        i_max, i_min = litho.aerial_outer(mask)
        z_max, z_min = resist(i_max, litho.cfg), resist(i_min, litho.cfg)
        loss = loss + cfg.weight_pvb * ((z_max - z_min) ** 2).sum()
    if cfg.weight_tv > 0:
        loss = loss + cfg.weight_tv * total_variation(mask)
    return loss


def _binarize(params: torch.Tensor, target_dtype: torch.dtype, cfg: ILTConfig) -> torch.Tensor:
    m = (torch.sigmoid(cfg.mask_steepness * params) >= 0.5).to(target_dtype)
    return morphological_open(m, cfg.mrc_open_size) if cfg.mrc_open_size > 1 else m


def _make_first_order_optimizer(params: torch.Tensor, cfg: ILTConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam([params], lr=cfg.step_size)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD([params], lr=cfg.step_size, momentum=cfg.momentum)
    if cfg.optimizer == "nesterov":
        return torch.optim.SGD(
            [params], lr=cfg.step_size, momentum=cfg.momentum, nesterov=True
        )
    raise AssertionError(f"_make_first_order_optimizer called with {cfg.optimizer!r}")


def _solve_first_order(
    target: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig,
    params: torch.Tensor,
    progress: bool,
) -> tuple[torch.Tensor, list[float], int, int, dict[int, torch.Tensor]]:
    """Adam / SGD / Nesterov: fully vectorised across the batch dimension."""
    opt = _make_first_order_optimizer(params, cfg)
    history: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    ran = 0
    for step in range(cfg.iterations):
        opt.zero_grad(set_to_none=True)
        loss = ilt_loss(params, target, litho, cfg, beta=_beta_for_step(cfg, step))
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([params], cfg.grad_clip)
        grad_norm = params.grad.detach().abs().max().item() if params.grad is not None else 0.0
        opt.step()
        history.append(loss.detach().item())
        ran = step + 1
        if ran in cfg.checkpoints:
            with torch.no_grad():
                snapshots[ran] = _binarize(params, target.dtype, cfg)
        if progress and step % 25 == 0:
            print(f"    iter {step:4d}  loss {float(loss):.1f}", flush=True)
        if cfg.early_stop_rtol > 0 and len(history) > 10:
            rel = abs(history[-11] - history[-1]) / max(abs(history[-11]), 1e-12)
            if rel < cfg.early_stop_rtol:
                break
        if cfg.convergence_grad_tol > 0 and grad_norm < cfg.convergence_grad_tol:
            break
    return params, history, ran, ran, snapshots  # one func-eval per iteration


def _solve_lbfgs_one(
    target_2d: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig,
    params_2d: torch.Tensor,
    progress: bool,
) -> tuple[torch.Tensor, list[float], int, int, dict[int, torch.Tensor]]:
    """L-BFGS on a single (unbatched) example.

    Run per-example rather than jointly across a batch: ``torch.optim.LBFGS``
    builds one shared low-rank Hessian approximation over the *whole* flattened
    parameter tensor it is given. For a batch of otherwise-independent ILT
    problems that would silently couple their curvature estimates together --
    mathematically still convergent (the objective is separable, so the joint
    minimum coincides with the per-example minima), but it would break the
    project-wide ``batched == looped single-case`` equivalence that every other
    optimizer satisfies (see ``test_lbfgs_batched_equals_single``). Running one
    independent ``torch.optim.LBFGS`` instance per example costs a Python-level
    loop but keeps that guarantee exactly, not approximately.
    """
    opt = torch.optim.LBFGS(
        [params_2d],
        lr=cfg.step_size,
        max_iter=cfg.lbfgs_max_iter_per_step,
        history_size=cfg.lbfgs_history_size,
        line_search_fn=cfg.lbfgs_line_search,
    )
    history: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    n_evals = 0
    last_grad_norm = [0.0]
    # Beta must stay FIXED across every closure call within one outer step's
    # line search (the function being searched must not move under it) and
    # only advance once per outer step. A mutable cell lets the closure read
    # the value the outer loop set just before calling opt.step().
    current_beta = [cfg.mask_steepness]

    def closure() -> torch.Tensor:
        nonlocal n_evals
        opt.zero_grad(set_to_none=True)
        loss = ilt_loss(params_2d, target_2d, litho, cfg, beta=current_beta[0])
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([params_2d], cfg.grad_clip)
        if params_2d.grad is not None:
            last_grad_norm[0] = params_2d.grad.detach().abs().max().item()
        n_evals += 1
        return loss

    ran = 0
    for step in range(cfg.iterations):
        current_beta[0] = _beta_for_step(cfg, step)
        loss = opt.step(closure)
        history.append(loss.detach().item())
        ran = step + 1
        if ran in cfg.checkpoints:
            with torch.no_grad():
                snapshots[ran] = _binarize(params_2d, target_2d.dtype, cfg)
        if progress and step % 25 == 0:
            print(f"    L-BFGS iter {step:4d}  loss {float(loss):.1f}  evals {n_evals}", flush=True)
        if cfg.early_stop_rtol > 0 and len(history) > 10:
            rel = abs(history[-11] - history[-1]) / max(abs(history[-11]), 1e-12)
            if rel < cfg.early_stop_rtol:
                break
        if cfg.convergence_grad_tol > 0 and last_grad_norm[0] < cfg.convergence_grad_tol:
            break
    return params_2d, history, ran, n_evals, snapshots


def _solve_lbfgs(
    target: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig,
    params: torch.Tensor,
    progress: bool,
) -> tuple[torch.Tensor, list[float], int, int, dict[int, torch.Tensor]]:
    """Dispatch L-BFGS over an optional batch dimension, one example at a time."""
    if target.dim() == 2:
        p, hist, ran, evals, snaps = _solve_lbfgs_one(target, litho, cfg, params, progress)
        return p, hist, ran, evals, snaps

    batch = target.shape[0]
    per_example_params: list[torch.Tensor] = []
    per_example_history: list[list[float]] = []
    per_example_snapshots: list[dict[int, torch.Tensor]] = []
    total_evals = 0
    max_ran = 0
    for b in range(batch):
        p_b = params[b].detach().clone().requires_grad_(True)
        p_b, hist_b, ran_b, evals_b, snaps_b = _solve_lbfgs_one(
            target[b], litho, cfg, p_b, progress and b == 0
        )
        per_example_params.append(p_b.detach())
        per_example_history.append(hist_b)
        per_example_snapshots.append(snaps_b)
        total_evals += evals_b
        max_ran = max(max_ran, ran_b)

    stacked_params = torch.stack(per_example_params, dim=0)

    # Joint loss curve: sum of per-example losses at each outer-step index.
    # Examples that early-stopped are padded with their own final loss so the
    # joint curve stays defined at every index up to max_ran, rather than being
    # truncated to the shortest-running example.
    joint_history = [
        sum(
            hist[i] if i < len(hist) else hist[-1]
            for hist in per_example_history
        )
        for i in range(max_ran)
    ]

    joint_snapshots: dict[int, torch.Tensor] = {}
    for cp in cfg.checkpoints:
        if all(cp in s for s in per_example_snapshots):
            joint_snapshots[cp] = torch.stack(
                [s[cp] for s in per_example_snapshots], dim=0
            )

    return stacked_params, joint_history, max_ran, total_evals, joint_snapshots


def solve(
    target: torch.Tensor,
    litho: LithoModel,
    cfg: ILTConfig | None = None,
    init_params: torch.Tensor | None = None,
    progress: bool = False,
) -> ILTResult:
    """Run gradient-based (or quasi-Newton) numerical ILT.

    Parameters
    ----------
    target:
        Binary ``{0,1}`` design, ``(H,W)`` or ``(B,H,W)``, float32, CPU.
    litho:
        Three-corner forward model.
    cfg:
        Solver configuration. ``cfg.optimizer`` selects Adam (default), SGD,
        Nesterov-accelerated SGD, or L-BFGS -- see ``ILTConfig`` for the
        fairness caveat on comparing "iterations" across optimizers.
    init_params:
        Optional warm start for ``P``. When ``None``, ``P_0 = init_scale*(2*target-1)``,
        i.e. the design itself — the standard cold start. A generative warm start
        supplies this tensor instead.
    progress:
        Print the loss periodically.

    Returns
    -------
    ILTResult
        ``mask`` is the binarised (and optionally MRC-opened) result.
        ``n_func_evals`` is the number to use for a compute-fair comparison
        across optimizers (see ``ILTConfig``).
    """
    cfg = cfg or ILTConfig()
    # NOTE: deliberately no torch.manual_seed here. This solver is deterministic
    # by construction (fixed initialisation, no stochastic operators), so seeding
    # buys nothing -- and seeding globally would silently reset the CALLER's RNG
    # stream on every call. solve() runs inside the RL reward loop, so that would
    # have reset the training RNG on every single refinement and destroyed
    # reproducibility. Pinned by test_solve_does_not_touch_global_rng.
    # solve() runs its own optimisation, so it must hold a gradient scope even
    # when called from inside torch.no_grad() (e.g. during evaluation).
    grad_ctx = torch.enable_grad()
    grad_ctx.__enter__()

    params = (
        (cfg.init_scale * (2.0 * target - 1.0)).clone()
        if init_params is None
        else init_params.clone()
    ).requires_grad_(True)

    start = time.time()
    dispatch: Callable = _solve_lbfgs if cfg.optimizer == "lbfgs" else _solve_first_order
    params, history, ran, n_evals, snapshots = dispatch(target, litho, cfg, params, progress)

    with torch.no_grad():
        mask = _binarize(params, target.dtype, cfg)
    grad_ctx.__exit__(None, None, None)

    return ILTResult(
        mask=mask,
        params=params.detach(),
        loss_history=history,
        iterations_run=ran,
        n_func_evals=n_evals,
        optimizer=cfg.optimizer,
        seconds=time.time() - start,
        checkpoint_masks=snapshots,
    )
