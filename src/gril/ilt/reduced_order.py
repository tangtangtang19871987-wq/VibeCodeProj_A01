"""Physics-based reduced-order model of ILT around a converged reference solution.

Why a tiny subspace is enough -- an exact property, not an approximation
------------------------------------------------------------------------
The nominal residual is ``r = sigmoid(g*(I - th)) - target`` with
``I = sum_k w_k |h_k * (dose*M)|^2`` and ``M = sigmoid(beta*P)``. Its Jacobian
factorises as ``J = S_z . A . B . S_m`` where ``S_z``/``S_m`` are pointwise
(diagonal) sigmoid slopes and ``B`` projects a mask perturbation onto the
pupil band: every SOCS kernel lives on the same (2h+1)x(2h+1) block of
low frequencies (35x35 for the ICCAD13 kernels, h = 17), so a mask
perturbation can only change the image through its spectrum inside that
block. A real image's spectrum on a symmetric block has exactly
``(2h+1)^2`` real degrees of freedom, hence

    rank(J) <= (2h+1)^2 = 1225     (for 262,144 pixels at 512^2, 4.2M at 2048^2)

and ``range(J^T) = S_m . BL`` exactly, where ``BL`` is the space of real
band-limited images. Everything a Gauss-Newton step can do lives in that
subspace; everything orthogonal to it is invisible to the image to first
order. :func:`gn_spectrum` therefore computes the *exact* nonzero spectrum
of ``J^T J`` by building the 1225 basis vectors ``W_j = s_m * f_j`` (``f_j``
the real Fourier modes of the band) and solving a 1225x1225 eigenproblem --
no Lanczos, no randomisation, no samples.

The reduced model
-----------------
Keeping the top-``r`` eigenvectors ``V_r`` of ``J^T J`` at the reference
``P0`` gives the parameterisation ``P = P0 + V_r a``. The expensive part --
the optics -- is linearised once: ``I(P0 + V_r a) ~ I0 + K_r a`` with
``K_r = (dI/dP) V_r`` (``r`` JVPs, computed offline). Everything that
depends on the *recipe* -- resist threshold, dose, target -- is pointwise
and is kept exact: dose scales intensity exactly as ``dose^2``; threshold
and target enter only the pointwise resist/residual. So a new recipe is
solved with matrix-vector products only (no FFTs): see :class:`ReducedILT`.

Nothing here is fitted to data. Validation is always done by binarising the
reduced solution and printing it with the full forward model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gril.ilt.gauss_newton import ILTLeastSquares, _GNState
from gril.litho.socs import convolve


def band_frequencies(half: int = 17) -> list[tuple[int, int]]:
    """One representative of each +/- frequency pair in the pupil block (DC excluded)."""
    reps = []
    for ky in range(0, half + 1):
        for kx in range(-half, half + 1):
            if ky > 0 or kx > 0:
                reps.append((ky, kx))
    return reps


def band_basis(h: int, w: int, half: int = 17, dtype=torch.float32) -> torch.Tensor:
    """All ``(2*half+1)^2`` real Fourier modes of the pupil block, shape ``(m, h, w)``.

    Ordering: DC, then ``cos, sin`` for each representative frequency. Each
    mode has unit RMS, so the basis is orthonormal up to the factor ``h*w``.
    """
    y = torch.arange(h, dtype=torch.float64).view(-1, 1) / h
    x = torch.arange(w, dtype=torch.float64).view(1, -1) / w
    modes = [torch.ones(h, w, dtype=torch.float64)]
    for ky, kx in band_frequencies(half):
        phase = 2 * torch.pi * (ky * y + kx * x)
        modes.append(2**0.5 * torch.cos(phase))
        modes.append(2**0.5 * torch.sin(phase))
    return torch.stack(modes).to(dtype)


def intensity_jvp_batch(model: ILTLeastSquares, st: _GNState, v: torch.Tensor) -> torch.Tensor:
    """``(dI/dP) v`` for a batch ``v`` of shape ``(B, H, W)``; returns ``(B, H, W)``.

    Intensity-level (no resist slope), so it is independent of resist
    threshold and target -- the property :class:`ReducedILT` relies on.
    """
    hv = convolve((model.dose * st.s_m * v).to(model.cdtype), model.kern, model.n)
    e = st.e.unsqueeze(0)
    w = model.w.unsqueeze(0)
    return 2.0 * (w * (e.real * hv.real + e.imag * hv.imag)).sum(1)


@dataclass
class GNSpectrum:
    """Exact nonzero spectrum of ``J^T J`` at a reference point."""

    eigenvalues: torch.Tensor      # (m',) descending
    basis_coeffs: torch.Tensor     # (m, m'): eigenvector j = W @ basis_coeffs[:, j]
    w_mat: torch.Tensor            # (N, m) the range basis W = s_m * f_j, flattened
    k_mat: torch.Tensor            # (N, m) (dI/dP) W, flattened
    s_z: torch.Tensor              # (N,) resist slope at the reference
    dropped: int                   # basis directions dropped as numerically degenerate

    def vectors(self, r: int) -> torch.Tensor:
        """Top-``r`` eigenvectors as an ``(N, r)`` matrix (orthonormal)."""
        return self.w_mat @ self.basis_coeffs[:, :r]

    def k_reduced(self, r: int) -> torch.Tensor:
        """``(dI/dP) V_r`` as an ``(N, r)`` matrix."""
        return self.k_mat @ self.basis_coeffs[:, :r]


def gn_spectrum(model: ILTLeastSquares, st: _GNState, half: int = 17, batch: int = 16,
                rel_tol: float = 1e-10) -> GNSpectrum:
    """Exact nonzero eigenpairs of ``J^T J`` at state ``st``.

    Builds ``W_j = s_m * f_j`` for the ``(2*half+1)^2`` band modes, computes
    ``K W`` with batched JVPs, whitens ``W`` (``W^T W = U L U^T``; directions
    with ``L < rel_tol * max(L)`` are dropped and counted), and solves the
    small symmetric eigenproblem. Since ``range(J^T J)`` is contained in
    ``span(W)``, these are exactly the nonzero eigenvalues of ``J^T J``.
    """
    h, w = st.p.shape
    f = band_basis(h, w, half, dtype=st.p.dtype)
    m = f.shape[0]
    w_mat = (st.s_m.unsqueeze(0) * f).reshape(m, -1)            # (m, N)
    k_rows = []
    for i in range(0, m, batch):
        k_rows.append(intensity_jvp_batch(model, st, w_mat[i:i + batch].view(-1, h, w)).reshape(-1, h * w))
    k_mat = torch.cat(k_rows).T.contiguous()                      # (N, m)
    w_mat = w_mat.T.contiguous()                                  # (N, m)
    s_z = st.s_z.reshape(-1)
    j_mat = s_z.unsqueeze(1) * k_mat                              # (N, m)

    gram_w = (w_mat.T.double() @ w_mat.double())
    lw, uw = torch.linalg.eigh(gram_w)
    keep = lw > rel_tol * lw.max()
    t = uw[:, keep] / lw[keep].sqrt()                             # whitening: (W t)^T (W t) = I
    jt = j_mat.double() @ t
    lam, y = torch.linalg.eigh(jt.T @ jt)
    order = torch.argsort(lam, descending=True)
    coeffs = (t @ y[:, order]).to(st.p.dtype)
    return GNSpectrum(eigenvalues=lam[order].clamp_min(0), basis_coeffs=coeffs, w_mat=w_mat,
                      k_mat=k_mat, s_z=s_z, dropped=int((~keep).sum()))


class ReducedILT:
    """``P = P0 + V_r a`` with linearised optics and exact pointwise recipe physics.

    Online cost per evaluation is two ``(N x r)`` matrix-vector products --
    no FFTs. Supported recipe changes (all exact at fixed optics): resist
    threshold, nominal dose (``I ~ dose^2``), resist steepness, and target.
    """

    def __init__(self, spec: GNSpectrum, r: int, p0: torch.Tensor, i0: torch.Tensor, dose0: float,
                 beta: float):
        self.shape = p0.shape
        self.v = spec.vectors(r)
        self.k = spec.k_reduced(r)
        self.p0 = p0.reshape(-1)
        self.i0 = i0.reshape(-1)
        self.dose0 = dose0
        self.beta = beta

    def residual_and_jac(self, a, threshold, steepness, dose, target):
        scale = (dose / self.dose0) ** 2
        intensity = scale * (self.i0 + self.k @ a)
        z = torch.sigmoid(steepness * (intensity - threshold))
        r = z - target
        jac = (steepness * z * (1 - z) * scale).unsqueeze(1) * self.k
        return r, jac

    def solve(self, threshold: float, steepness: float, dose: float, target: torch.Tensor,
              iterations: int = 30, damping: float = 1.0) -> tuple[torch.Tensor, list[float]]:
        """Levenberg-Marquardt on the ``r``-dimensional reduced problem."""
        t = target.reshape(-1).to(self.k.dtype)
        a = torch.zeros(self.k.shape[1], dtype=self.k.dtype)
        r, jac = self.residual_and_jac(a, threshold, steepness, dose, t)
        f = float((r * r).sum())
        hist, lam, nu = [f], damping, 2.0
        eye = torch.eye(a.numel(), dtype=a.dtype)
        for _ in range(iterations):
            g = jac.T @ r
            jtj = jac.T @ jac
            step = torch.linalg.solve(jtj + lam * eye, -g)
            pred = -(2 * float(g @ step) + float((jac @ step).pow(2).sum()))
            r_new, jac_new = self.residual_and_jac(a + step, threshold, steepness, dose, t)
            f_new = float((r_new * r_new).sum())
            rho = (f - f_new) / pred if pred > 0 else -1.0
            if rho > 1e-4:
                a, r, jac, f = a + step, r_new, jac_new, f_new
                lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
                nu = 2.0
            else:
                lam *= nu
                nu *= 2.0
            hist.append(f)
        return a, hist

    def params(self, a: torch.Tensor) -> torch.Tensor:
        return (self.p0 + self.v @ a).reshape(self.shape)
