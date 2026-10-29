"""Fast re-optimisation of an ILT mask after a recipe change (X-20 .. X-26).

Setting: a reference mask parameter ``P_ref`` has been optimised for one
recipe (resist threshold, dose, target bias). For a *changed* recipe, find a
mask that prints nearly as well as a full re-optimisation, in seconds. Every
method here searches a small space but always evaluates the **exact** model
(``ILTLeastSquares``); nothing is fitted to data.

What the experiments established (``docs/findings.md`` F-ROM-01 .. 05), and
therefore what this module keeps:

* :func:`solve_in_subspace` -- Levenberg-Marquardt over ``P_ref + U a`` for a
  fixed orthonormal basis ``U``. With ``U`` from :func:`pod_basis` over 3-4
  solved recipes of the same knob it recovers ~95% of the re-optimisation gain
  in ~3 s at 512^2 (X-23), and the basis transfers across layouts (X-24).
* :class:`NormalDisplacementField` + :func:`solve_in_normal_field` -- the
  snapshot-free route: perturb ``P`` by a smoothly varying normal edge
  displacement ``dP = b(x) |grad P_ref|`` with ``b`` bilinear on a coarse
  grid. At a grid spacing near the optical resolution (16 or 8 px = 64 or
  32 nm) it recovers 95-96% in ~4 s on every layout and recipe tested (X-26).

Not kept (refuted): the once-linearised operator ROM caps at ~25-50%
(``reduced_order.ReducedILT``, still available for the record), and one-step
implicit-function / Newton predictors cap at ~30% (X-26, script only).

Recovery is the fraction of the frozen -> re-optimised printed-L2 gap that a
candidate closes; see :func:`recovery`.
"""

from __future__ import annotations

import copy
import dataclasses
import time

import torch
import torch.nn.functional as F

from gril.ilt.gauss_newton import ILTLeastSquares, _cg
from gril.litho.resist import LithoModel, ProcessConfig


# ------------------------------------------------------------------- recipes

def biased_target(target: torch.Tensor, bias_px: int) -> torch.Tensor:
    """Grow (``bias_px > 0``) or shrink the target by ``|bias_px|`` pixels (3x3 morphology)."""
    t = target[None, None]
    for _ in range(abs(bias_px)):
        t = F.max_pool2d(t, 3, 1, 1) if bias_px > 0 else -F.max_pool2d(-t, 3, 1, 1)
    return t[0, 0]


def apply_recipe(litho: LithoModel, target: torch.Tensor, base: ProcessConfig, *,
                 threshold: float | None = None, dose: float = 1.0, bias: int = 0
                 ) -> tuple[LithoModel, torch.Tensor]:
    """A shallow copy of ``litho`` with the recipe's resist threshold and dose, and the biased target.

    ``dose`` scales all three process corners (nominal, max, min) together;
    the optics (kernels) are shared, not copied.
    """
    cfg = dataclasses.replace(base, target_density=base.target_density if threshold is None else threshold,
                              dose_nom=dose * base.dose_nom, dose_max=dose * base.dose_max,
                              dose_min=dose * base.dose_min)
    lm = copy.copy(litho)
    lm.cfg = cfg
    return lm, biased_target(target, bias)


def recovery(frozen_l2: float, candidate_l2: float, truth_l2: float) -> float:
    """Fraction of the frozen -> truth printed-L2 gap closed (1 = as good as the truth; can exceed 1)."""
    return (frozen_l2 - candidate_l2) / (frozen_l2 - truth_l2)


# ------------------------------------------------------------ snapshot route

def pod_basis(deltas: list[torch.Tensor], rel_tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis ``(H*W, k)`` of the snapshot span; drops singular values below ``rel_tol * s_max``."""
    s = torch.stack([d.reshape(-1) for d in deltas], dim=1)
    u, sv, _ = torch.linalg.svd(s, full_matrices=False)
    return u[:, sv > rel_tol * sv[0]]


def solve_in_subspace(target: torch.Tensor, litho: LithoModel, p0: torch.Tensor, u: torch.Tensor, *,
                      beta: float, iterations: int = 15, damping: float = 1.0) -> torch.Tensor:
    """Levenberg-Marquardt over ``P = p0 + U a`` with the full model; ``J U`` is recomputed every step.

    ``U`` has ``k`` columns (k JVPs per step), so this is for small ``k``
    (POD bases); for large structured bases use :func:`solve_in_normal_field`.
    """
    model = ILTLeastSquares(target, litho, beta)
    k = u.shape[1]
    a = torch.zeros(k)
    st = model.state(p0)
    lam, nu = damping, 2.0
    eye = torch.eye(k)
    for _ in range(iterations):
        ju = torch.stack([model.jvp(st, u[:, j].view(p0.shape)).reshape(-1) for j in range(k)], dim=1)
        g = ju.T @ st.r.reshape(-1)
        step = torch.linalg.solve(ju.T @ ju + lam * eye, -g)
        pred = -(2 * float(g @ step) + float((ju @ step).pow(2).sum()))
        trial = model.state((p0.reshape(-1) + u @ (a + step)).reshape(p0.shape))
        rho = (st.f - trial.f) / pred if pred > 0 else -1.0
        if rho > 1e-4:
            a, st = a + step, trial
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:
            lam *= nu
            nu *= 2.0
    return st.p


# -------------------------------------------------------- snapshot-free route

def grad_mag(p: torch.Tensor) -> torch.Tensor:
    """``|grad P|`` by central differences (the level-set normal-speed weight)."""
    gy, gx = torch.gradient(p)
    return (gx * gx + gy * gy).sqrt()


def _cg_psd(apply_a, b: torch.Tensor, iters: int) -> torch.Tensor:
    """Plain CG from 0 for a positive semi-definite operator; stops on non-positive curvature."""
    x, r = torch.zeros_like(b), b.clone()
    d, rs = r.clone(), float((b * b).sum())
    for _ in range(iters):
        ad = apply_a(d)
        dad = float((d * ad).sum())
        if dad <= 0:
            break
        a = rs / dad
        x += a * d
        r -= a * ad
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 < 1e-3 * float(b.norm()):
            break
        d, rs = r + (rs_new / rs) * d, rs_new
    return x


class NormalDisplacementField:
    """Linear map ``b -> dP = b(x) |grad P_ref|``, with ``b`` bilinear on a coarse grid.

    In level-set terms the zero contour of ``P_ref + dP`` moves by ``-b`` pixels
    along its normal, so ``b`` is a local edge displacement. ``spacing`` is the
    grid pitch in pixels (``"global"`` = one constant ``b``, the uniform-bias
    mode). Parameters live on an ``(H//spacing + 1, W//spacing + 1)`` grid,
    interpolated with ``align_corners=True``.
    """

    def __init__(self, p_ref: torch.Tensor, spacing: int | str):
        self.g = grad_mag(p_ref)
        self.hw = tuple(p_ref.shape)
        self.n = (1, 1) if spacing == "global" else tuple(s // spacing + 1 for s in self.hw)

    @property
    def dims(self) -> int:
        return self.n[0] * self.n[1]

    def up(self, c: torch.Tensor) -> torch.Tensor:
        """Parameters ``(n0, n1)`` -> mask-parameter perturbation ``(H, W)``."""
        if self.n == (1, 1):
            return c.reshape(()) * self.g
        return F.interpolate(c[None, None], size=self.hw, mode="bilinear", align_corners=True)[0, 0] * self.g

    def up_t(self, v: torch.Tensor) -> torch.Tensor:
        """Exact adjoint of :meth:`up` (via autograd of the linear map)."""
        c = torch.zeros(self.n, requires_grad=True)
        with torch.enable_grad():
            return torch.autograd.grad((self.up(c) * v).sum(), c)[0]

    def project(self, delta: torch.Tensor, iters: int = 300) -> torch.Tensor:
        """Least-squares fit ``argmin_b ||up(b) - delta||``, returned as ``up(b)``."""
        x = _cg_psd(lambda d: self.up_t(self.up(d)), self.up_t(delta), iters)
        return self.up(x)


def solve_in_normal_field(target: torch.Tensor, litho: LithoModel, p_ref: torch.Tensor,
                          field: NormalDisplacementField, *, beta: float, iterations: int = 20,
                          checkpoints: tuple[int, ...] = (), max_cg: int = 10, cg_rtol: float = 0.1,
                          damping_init: float = 1.0
                          ) -> tuple[torch.Tensor, dict[int, tuple[torch.Tensor, float]]]:
    """Levenberg-Marquardt over the field parameters, full model, matrix-free CG.

    Each CG step costs one JVP + one VJP whatever the field's dimension.
    Returns ``(P, snapshots)`` where ``snapshots[k] = (P_k, seconds)`` for every
    ``k`` in ``checkpoints``.
    """
    model = ILTLeastSquares(target, litho, beta)
    c = torch.zeros(field.n)
    st = model.state(p_ref)
    lam, nu, snaps, t0 = damping_init, 2.0, {}, time.time()
    with torch.no_grad():
        for it in range(1, iterations + 1):
            jtr = field.up_t(model.vjp(st, st.r))

            def apply_a(d, st=st, lam=lam):
                jd = model.jvp(st, field.up(d))
                return field.up_t(model.vjp(st, jd)) + lam * d, jd

            delta, j_delta = _cg(apply_a, -jtr, max_cg, cg_rtol, lambda x: model.jvp(st, field.up(x)))
            pred = -2.0 * float((jtr * delta).sum()) - float((j_delta * j_delta).sum())
            trial = model.state(p_ref + field.up(c + delta))
            rho = (st.f - trial.f) / pred if pred > 0 else -1.0
            if rho > 1e-4:
                c, st = c + delta, trial
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                nu = 2.0
            else:
                lam *= nu
                nu *= 2.0
            if it in checkpoints:
                snaps[it] = (st.p.clone(), time.time() - t0)
    return st.p, snaps
