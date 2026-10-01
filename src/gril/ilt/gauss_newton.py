"""Matrix-free Levenberg-Marquardt (damped Gauss-Newton) for the ILT objective.

ILT's nominal objective is nonlinear least squares::

    f(P) = w_n * || r(P) ||^2,   r = sigmoid(g*(I(M) - th)) - target,
    M = sigmoid(beta*P),          I(M) = sum_k w_k |h_k * (dose*M)|^2

so the Gauss-Newton matrix ``J^T J`` is the natural curvature model -- always
positive semi-definite and exact up to the residual-curvature term. Each outer
step solves ``(w_n J^T J + lam I) d = -w_n J^T r`` with conjugate gradients
using only Jacobian-vector (JVP) and vector-Jacobian (VJP) products, never
forming ``J`` (4M x 4M at 2048^2). ``lam`` is adapted with Nielsen's rule from
the ratio of actual to predicted reduction.

Cost per CG step is one JVP + one VJP against a field ``E_k = h_k * (dose*M)``
cached once per outer step: the JVP is 1 FFT + K inverse FFTs and the VJP is
K FFTs + 1 inverse FFT (the per-kernel adjoint convolutions are summed in the
frequency domain before a single inverse transform). That is ~50 FFTs versus
~98 for one forward+backward of the autograd path, so a CG step costs about
half a gradient evaluation.

Only the plain nominal-L2 objective is supported; any other loss term raises
rather than being silently dropped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from gril.litho.resist import LithoModel
from gril.litho.socs import _apply_kernel_corners, _complex_of, _match, convolve


@dataclass
class _GNState:
    p: torch.Tensor
    m: torch.Tensor
    s_m: torch.Tensor      # dM/dP
    e: torch.Tensor        # (K,H,W) complex field per kernel
    s_z: torch.Tensor      # dZ/dI
    r: torch.Tensor        # Z - target
    f: float


class ILTLeastSquares:
    """Residual, JVP and VJP of the nominal ILT objective for one (H, W) problem."""

    def __init__(self, target: torch.Tensor, litho: LithoModel, beta: float, weight_nominal: float = 1.0):
        if target.dim() != 2:
            raise ValueError(f"ILTLeastSquares works on one (H,W) problem, got {tuple(target.shape)}")
        self.t = target
        self.beta = beta
        self.wn = weight_nominal
        ks = litho.focus
        self.dose = litho.cfg.dose_nom
        self.n = litho.cfg.num_kernels
        self.gamma = litho.cfg.print_steepness
        self.theta = litho.cfg.target_density
        self.cdtype = _complex_of(target.dtype)
        self.kern = _match(ks.kernels, self.cdtype)
        self.kern_conj = self.kern.conj()
        self.w = ks.scales[: self.n].to(target.dtype).view(-1, 1, 1)
        self.n_fwd = self.n_jvp = self.n_vjp = 0

    def state(self, p: torch.Tensor) -> _GNState:
        self.n_fwd += 1
        m = torch.sigmoid(self.beta * p)
        s_m = self.beta * m * (1 - m)
        e = convolve((self.dose * m).to(self.cdtype).unsqueeze(0), self.kern, self.n)[0]
        intensity = (self.w * (e.real**2 + e.imag**2)).sum(0)
        z = torch.sigmoid(self.gamma * (intensity - self.theta))
        r = z - self.t
        return _GNState(p=p, m=m, s_m=s_m, e=e, s_z=self.gamma * z * (1 - z), r=r,
                        f=float(self.wn * (r * r).sum()))

    def jvp(self, st: _GNState, v: torch.Tensor) -> torch.Tensor:
        """``J v``: how the residual moves for a parameter perturbation ``v``."""
        self.n_jvp += 1
        hv = convolve((self.dose * st.s_m * v).to(self.cdtype).unsqueeze(0), self.kern, self.n)[0]
        d_int = 2.0 * (self.w * (st.e.real * hv.real + st.e.imag * hv.imag)).sum(0)
        return st.s_z * d_int

    def vjp(self, st: _GNState, u: torch.Tensor) -> torch.Tensor:
        """``J^T u``: the adjoint, summed over kernels before one inverse FFT."""
        self.n_vjp += 1
        g = (st.s_z * u).to(self.cdtype)
        spec = torch.fft.fft2((st.e * g).unsqueeze(0), norm="forward")
        weighted = _apply_kernel_corners(spec, self.kern_conj, self.n) * self.w.unsqueeze(0)
        back = torch.fft.ifft2(weighted.sum(1), norm="forward")[0]
        return st.s_m * (2.0 * self.dose) * back.real

    def gradient(self, st: _GNState) -> torch.Tensor:
        """``df/dP = 2 w_n J^T r`` -- equal to autograd's gradient of ``ilt_loss``."""
        return 2.0 * self.wn * self.vjp(st, st.r)


def _cg(apply_a, b: torch.Tensor, max_iter: int, rtol: float, jvp_of):
    """CG for ``A x = b`` from ``x0 = 0``; also returns ``J x`` accumulated for free."""
    x = torch.zeros_like(b)
    jx = None
    res = b.clone()
    d = res.clone()
    rs = float((res * res).sum())
    b_norm = rs ** 0.5
    for _ in range(max_iter):
        ad, jd = apply_a(d)
        dad = float((d * ad).sum())
        if dad <= 0:
            break
        alpha = rs / dad
        x.add_(d, alpha=alpha)
        jx = jd * alpha if jx is None else jx.add_(jd, alpha=alpha)
        res.sub_(ad, alpha=alpha)
        rs_new = float((res * res).sum())
        if rs_new ** 0.5 <= rtol * b_norm:
            break
        d = res + (rs_new / rs) * d
        rs = rs_new
    if jx is None:
        jx = jvp_of(x)
    return x, jx


def solve_levenberg_marquardt(
    target: torch.Tensor,
    litho: LithoModel,
    params: torch.Tensor,
    *,
    iterations: int,
    beta: float,
    weight_nominal: float = 1.0,
    max_cg: int = 10,
    cg_rtol: float = 0.1,
    damping_init: float = 1.0,
    time_budget_s: float = 0.0,
    progress: bool = False,
) -> tuple[torch.Tensor, list[float], int, int, list[float]]:
    """Run LM on one (H,W) problem.

    Returns ``(params, loss_history, outer_iterations, work_units, time_history)``
    where ``work_units = n_fwd + (n_jvp + n_vjp) / 2``, i.e. measured in
    forward+backward-equivalents (see module docstring); wall-clock in
    ``time_history`` is the fair cross-optimizer unit.
    """
    model = ILTLeastSquares(target, litho, beta, weight_nominal)
    wn = model.wn
    start = time.time()
    with torch.no_grad():
        st = model.state(params.detach().clone())
        lam, nu = damping_init, 2.0
        history, times = [st.f], [0.0]
        ran = 0
        jtr = None
        for it in range(iterations):
            if time_budget_s > 0 and time.time() - start >= time_budget_s:
                break
            if jtr is None:  # only recomputed after an accepted step moves the state
                jtr = model.vjp(st, st.r)
            b = -wn * jtr

            def apply_a(d, st=st, lam=lam):
                jd = model.jvp(st, d)
                return wn * model.vjp(st, jd) + lam * d, jd

            delta, j_delta = _cg(apply_a, b, max_cg, cg_rtol, lambda x: model.jvp(st, x))
            pred = -2.0 * wn * float((jtr * delta).sum()) - wn * float((j_delta * j_delta).sum())
            trial = model.state(st.p + delta)
            rho = (st.f - trial.f) / pred if pred > 0 else -1.0
            if rho > 1e-4:
                st = trial
                jtr = None
                lam *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = 2.0
            else:
                lam *= nu
                nu *= 2.0
            ran = it + 1
            history.append(st.f)
            times.append(time.time() - start)
            if progress and it % 5 == 0:
                print(f"    LM iter {it:3d}  loss {st.f:.1f}  lam {lam:.3g}  rho {rho:.3f}", flush=True)
    work = model.n_fwd + (model.n_jvp + model.n_vjp) // 2
    return st.p, history, ran, work, times
