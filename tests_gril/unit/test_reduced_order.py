"""Tests for the physics-based reduced-order ILT model (gril.ilt.reduced_order)."""

from __future__ import annotations

import os

import pytest
import torch

from gril.ilt.gauss_newton import ILTLeastSquares
from gril.ilt.reduced_order import ReducedILT, band_basis, gn_spectrum
from gril.litho.resist import LithoModel, ProcessConfig

KERNEL_DIR = "/home/user/openopc/openilt/kernel"
pytestmark = pytest.mark.skipif(not os.path.isdir(KERNEL_DIR), reason="ICCAD13 kernels not present")
H = 96


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())
    t = torch.zeros(H, H, dtype=torch.float64)
    t[30:66, 40:54] = 1
    t[22:34, 14:80] = 1
    p = 0.5 * (2 * t - 1) + 0.3 * torch.randn_like(t)
    model = ILTLeastSquares(t, litho, beta=8.0)
    st = model.state(p)
    return litho, t, p, model, st, gn_spectrum(model, st, batch=64)


def test_band_basis_has_exactly_the_pupil_dimension_and_is_orthonormal():
    f = band_basis(H, H, dtype=torch.float64)
    assert f.shape[0] == 35 * 35
    g = f.reshape(f.shape[0], -1) @ f.reshape(f.shape[0], -1).T / (H * H)
    assert torch.allclose(g, torch.eye(g.shape[0], dtype=g.dtype), atol=1e-9)


def test_gauss_newton_range_is_inside_the_band_subspace(setup):
    """rank(J) <= 1225: J^T J of ANY vector lies in span(s_m * band modes)."""
    _, _, _, model, st, spec = setup
    x = torch.randn(H, H, dtype=torch.float64)
    y = model.vjp(st, model.jvp(st, x)).reshape(-1)
    q = spec.vectors(spec.eigenvalues.numel()).double()
    assert float((y - q @ (q.T @ y)).norm() / y.norm()) < 1e-10


def test_complement_of_band_subspace_is_invisible_to_the_image(setup):
    _, _, _, model, st, spec = setup
    x = torch.randn(H, H, dtype=torch.float64)
    q = spec.vectors(spec.eigenvalues.numel()).double()
    n = x.reshape(-1) - q @ (q.T @ x.reshape(-1))
    assert float(n.norm() / x.norm()) > 0.5          # most of the space is null
    assert float(model.jvp(st, n.view(H, H)).norm() / model.jvp(st, x).norm()) < 1e-10


def test_spectrum_eigenpairs_are_exact(setup):
    _, _, _, model, st, spec = setup
    v = spec.vectors(3).double()
    for j in range(3):
        av = model.vjp(st, model.jvp(st, v[:, j].view(H, H))).reshape(-1)
        assert float((av - spec.eigenvalues[j] * v[:, j]).norm() / (spec.eigenvalues[j] * v[:, j]).norm()) < 1e-9


def test_reduced_model_matches_full_model_to_first_order(setup):
    """At a = 0 the reduced residual is exact; its Jacobian equals the full JVP on V_r."""
    litho, t, p, model, st, spec = setup
    with torch.no_grad():
        i0 = litho.aerial_nominal(torch.sigmoid(8.0 * p))
    rom = ReducedILT(spec, 20, p, i0, litho.cfg.dose_nom, 8.0)
    c = litho.cfg
    r, jac = rom.residual_and_jac(torch.zeros(20, dtype=torch.float64), c.target_density,
                                  c.print_steepness, c.dose_nom, t.reshape(-1))
    assert float((r - st.r.reshape(-1)).abs().max()) < 1e-10
    v = spec.vectors(20).double()
    full = torch.stack([model.jvp(st, v[:, j].view(H, H)).reshape(-1) for j in range(20)], dim=1)
    assert float((jac - full).norm() / full.norm()) < 1e-10


def test_reduced_solve_decreases_the_reduced_objective(setup):
    litho, t, p, model, st, spec = setup
    with torch.no_grad():
        i0 = litho.aerial_nominal(torch.sigmoid(8.0 * p))
    rom = ReducedILT(spec, 30, p, i0, litho.cfg.dose_nom, 8.0)
    c = litho.cfg
    a, hist = rom.solve(c.target_density * 1.05, c.print_steepness, c.dose_nom, t, iterations=15)
    assert hist[-1] < hist[0]
    assert all(b <= a_ + 1e-9 for a_, b in zip(hist, hist[1:]))
