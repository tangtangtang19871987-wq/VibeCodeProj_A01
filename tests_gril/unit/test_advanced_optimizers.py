"""Tests for the SOTA first-order optimizers and the Levenberg-Marquardt solver."""

from __future__ import annotations

import os

import pytest
import torch

from gril.ilt.gauss_newton import ILTLeastSquares
from gril.ilt.optimizers import SOAP, Muon, ScheduleFreeAdamW, newton_schulz5
from gril.ilt.solver import ILTConfig, ilt_loss, solve
from gril.litho.resist import LithoModel, ProcessConfig

KERNEL_DIR = "/home/user/openopc/openilt/kernel"
needs_kernels = pytest.mark.skipif(not os.path.isdir(KERNEL_DIR), reason="ICCAD13 kernels not present")


@pytest.fixture(scope="module")
def litho():
    return LithoModel(KERNEL_DIR, ProcessConfig())


def _target(h=64, dtype=torch.float32):
    t = torch.zeros(h, h, dtype=dtype)
    t[20:44, 26:36] = 1
    t[14:22, 10:54] = 1
    return t


# ----------------------------------------------------------- Gauss-Newton core

@needs_kernels
def test_gn_gradient_matches_autograd(litho):
    torch.manual_seed(0)
    t = _target(dtype=torch.float64)
    p = 0.5 * (2 * t - 1) + 0.3 * torch.randn_like(t)
    model = ILTLeastSquares(t, litho, beta=8.0)
    pp = p.clone().requires_grad_(True)
    ilt_loss(pp, t, litho, ILTConfig(mask_steepness=8.0)).backward()
    g = model.gradient(model.state(p))
    assert float((g - pp.grad).norm() / pp.grad.norm()) < 1e-10


@needs_kernels
def test_gn_jvp_vjp_are_adjoint(litho):
    torch.manual_seed(1)
    t = _target(dtype=torch.float64)
    model = ILTLeastSquares(t, litho, beta=8.0)
    st = model.state(0.5 * (2 * t - 1) + 0.3 * torch.randn_like(t))
    v, u = torch.randn_like(t), torch.randn_like(t)
    lhs = float((model.jvp(st, v) * u).sum())
    rhs = float((v * model.vjp(st, u)).sum())
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


@needs_kernels
def test_gn_jvp_matches_finite_difference(litho):
    torch.manual_seed(2)
    t = _target(dtype=torch.float64)
    model = ILTLeastSquares(t, litho, beta=8.0)
    p = 0.5 * (2 * t - 1) + 0.3 * torch.randn_like(t)
    v = torch.randn_like(t)
    eps = 1e-6
    fd = (model.state(p + eps * v).r - model.state(p - eps * v).r) / (2 * eps)
    jv = model.jvp(model.state(p), v)
    assert float((jv - fd).norm() / fd.norm()) < 1e-6


@needs_kernels
def test_gauss_newton_loss_is_monotone_and_decreases(litho):
    t = _target()
    res = solve(t, litho, ILTConfig(optimizer="gauss_newton", iterations=6, mask_steepness=8.0,
                                    init_scale=0.5, gn_max_cg=5))
    h = res.loss_history
    assert all(b <= a + 1e-6 for a, b in zip(h, h[1:])), h  # LM only accepts decreases
    assert h[-1] < 0.8 * h[0]
    assert len(res.time_history) == len(h)


def test_gauss_newton_rejects_unsupported_terms():
    with pytest.raises(ValueError, match="nominal-L2"):
        ILTConfig(optimizer="gauss_newton", weight_pvb=0.1)
    with pytest.raises(ValueError, match="nominal-L2"):
        ILTConfig(optimizer="gauss_newton", weight_epe=1.0)


@needs_kernels
def test_gauss_newton_rejects_beta_annealing(litho):
    with pytest.raises(ValueError, match="annealing"):
        solve(_target(), litho, ILTConfig(optimizer="gauss_newton", iterations=1, beta_init=1.0))


# ------------------------------------------------------- first-order optimizers

def test_newton_schulz_orthogonalises_square_rectangular_and_batched():
    """The reference quintic coefficients trade exact convergence for speed:
    singular values land in ~[0.68, 1.2], except that ones starting near zero
    (here < 1e-3 of the Frobenius norm) are only partially lifted in 5 steps.
    That is Muon's documented behaviour, so the check is: nothing blows up
    past ~1.2, and every singular value that started non-negligible is in band.
    """
    torch.manual_seed(3)
    for shape in [(40, 40), (30, 50), (50, 30), (3, 32, 32)]:
        g = torch.randn(*shape)
        o = newton_schulz5(g)
        assert o.shape == g.shape
        for m, gi in zip(o.reshape(-1, *shape[-2:]), g.reshape(-1, *shape[-2:])):
            s = torch.linalg.svdvals(m)
            s_in = torch.linalg.svdvals(gi / gi.norm())
            assert float(s.max()) < 1.25
            assert float(s[s_in > 1e-2].min()) > 0.6
    batched = torch.randn(3, 24, 24)
    stacked = torch.stack([newton_schulz5(b) for b in batched])
    assert torch.allclose(newton_schulz5(batched), stacked, atol=1e-5)


def _quadratic_descends(opt_cls, steps=60, **kw):
    torch.manual_seed(4)
    a = torch.randn(16, 16)
    p = torch.zeros(16, 16, requires_grad=True)
    opt = opt_cls([p], **kw)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        loss = ((p - a) ** 2).sum()
        first = first if first is not None else float(loss.detach())
        loss.backward()
        opt.step()
    if isinstance(opt, ScheduleFreeAdamW):
        opt.eval()
    return first, float(((p.detach() - a) ** 2).sum())


@pytest.mark.parametrize("cls,kw", [
    (ScheduleFreeAdamW, dict(lr=0.1)),
    # Muon takes fixed-length spectral-norm steps, so it needs more of them
    # on a quadratic (measured: 31% of the initial loss left at 60, 1.3% at 150).
    (Muon, dict(lr=0.05, steps=150)),
    (SOAP, dict(lr=0.1)),
])
def test_first_order_optimizers_descend_a_quadratic(cls, kw):
    first, last = _quadratic_descends(cls, **kw)
    assert last < 0.2 * first


def test_schedule_free_eval_train_round_trip_is_exact():
    torch.manual_seed(5)
    p = torch.randn(8, 8, requires_grad=True)
    opt = ScheduleFreeAdamW([p], lr=0.1)
    for _ in range(5):
        opt.zero_grad()
        (p ** 2).sum().backward()
        opt.step()
    y = p.detach().clone()
    opt.eval()
    assert not torch.allclose(p.detach(), y)
    opt.train()
    assert torch.allclose(p.detach(), y, atol=1e-6)


@pytest.mark.parametrize("cls", [Muon, SOAP])
def test_batched_matrix_optimizers_treat_slices_independently(cls):
    """A (B,H,W) parameter must behave exactly like B separate (H,W) ones."""
    torch.manual_seed(6)
    a = torch.randn(2, 12, 12)
    pb = torch.zeros(2, 12, 12, requires_grad=True)
    ps = [torch.zeros(12, 12, requires_grad=True) for _ in range(2)]
    ob = cls([pb], lr=0.05)
    os_ = [cls([q], lr=0.05) for q in ps]
    for _ in range(12):
        ob.zero_grad()
        ((pb - a) ** 2).sum().backward()
        ob.step()
        for i, (q, o) in enumerate(zip(ps, os_)):
            o.zero_grad()
            ((q - a[i]) ** 2).sum().backward()
            o.step()
    for i in range(2):
        assert torch.allclose(pb.detach()[i], ps[i].detach(), atol=1e-5)


@needs_kernels
@pytest.mark.parametrize("name,lr", [("sf_adamw", 0.1), ("muon", 0.1), ("soap", 0.1)])
def test_solve_with_new_first_order_optimizers_reduces_ilt_loss(litho, name, lr):
    t = _target()
    res = solve(t, litho, ILTConfig(optimizer=name, iterations=25, step_size=lr,
                                    mask_steepness=8.0, init_scale=0.5))
    assert res.loss_history[-1] < res.loss_history[0]
    assert res.mask.shape == t.shape and set(res.mask.unique().tolist()) <= {0.0, 1.0}


@needs_kernels
def test_time_budget_stops_early(litho):
    res = solve(_target(), litho, ILTConfig(optimizer="adam", iterations=10_000, step_size=0.1,
                                            mask_steepness=8.0, init_scale=0.5, time_budget_s=0.5))
    assert res.iterations_run < 10_000
    assert res.time_history and res.time_history[-1] < 5.0


def test_schema_accepts_new_optimizer_names():
    from gril.config.schema import ILTConfigSchema
    for name in ("sf_adamw", "muon", "soap", "gauss_newton"):
        ILTConfigSchema(optimizer=name)
