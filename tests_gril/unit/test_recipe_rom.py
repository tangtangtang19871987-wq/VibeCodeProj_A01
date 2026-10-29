"""Tests for recipe re-optimisation helpers (gril.ilt.recipe_rom) and asset paths (gril.paths)."""

from __future__ import annotations

import pytest
import torch

from gril import paths
from gril.ilt.gauss_newton import ILTLeastSquares
from gril.ilt.recipe_rom import (NormalDisplacementField, apply_recipe, biased_target, grad_mag, pod_basis,
                                 recovery, solve_in_normal_field, solve_in_subspace)
from gril.litho.resist import ProcessConfig

H = 96
BETA = 8.0


def _target():
    t = torch.zeros(H, H)
    t[30:66, 40:54] = 1
    t[22:34, 14:80] = 1
    return t


def _p_ref():
    torch.manual_seed(0)
    return 0.5 * (2 * _target() - 1) + 0.05 * torch.randn(H, H)


# ------------------------------------------------------------ pure geometry

def test_biased_target_grows_and_shrinks_by_one_pixel_per_step():
    t = torch.zeros(20, 20)
    t[5:15, 5:15] = 1
    assert int(biased_target(t, 1).sum()) == 12 * 12
    assert int(biased_target(t, -2).sum()) == 6 * 6
    assert torch.equal(biased_target(t, 0), t)


def test_apply_recipe_changes_only_the_recipe_and_leaves_the_original_untouched():
    class _Litho:
        cfg = ProcessConfig()

    base, litho = ProcessConfig(), _Litho()
    lm, tgt = apply_recipe(litho, _target(), base, threshold=0.2, dose=1.03, bias=1)
    assert lm.cfg.target_density == 0.2
    assert lm.cfg.dose_nom == pytest.approx(1.03 * base.dose_nom)
    assert lm.cfg.dose_max == pytest.approx(1.03 * base.dose_max)
    assert lm.cfg.dose_min == pytest.approx(1.03 * base.dose_min)
    assert litho.cfg == base
    assert tgt.sum() > _target().sum()


def test_recovery_is_the_closed_fraction_of_the_gap():
    assert recovery(100.0, 100.0, 60.0) == 0.0
    assert recovery(100.0, 60.0, 60.0) == 1.0
    assert recovery(100.0, 50.0, 60.0) == pytest.approx(1.25)


def test_pod_basis_is_orthonormal_and_drops_dependent_snapshots():
    torch.manual_seed(1)
    a, b = torch.randn(H, H), torch.randn(H, H)
    u = pod_basis([a, b, a + 2 * b])
    assert u.shape == (H * H, 2)
    assert torch.allclose(u.T @ u, torch.eye(2), atol=1e-5)
    residual = a.reshape(-1) - u @ (u.T @ a.reshape(-1))
    assert float(residual.norm() / a.norm()) < 1e-5


@pytest.mark.parametrize("spacing", ["global", 32, 8])
def test_normal_field_adjoint_is_exact(spacing):
    f = NormalDisplacementField(_p_ref(), spacing)
    torch.manual_seed(2)
    c, v = torch.randn(f.n), torch.randn(H, H)
    assert float((f.up(c) * v).sum()) == pytest.approx(float((c * f.up_t(v)).sum()), rel=1e-5)


def test_normal_field_dims_and_global_mode():
    p = _p_ref()
    assert NormalDisplacementField(p, "global").dims == 1
    assert NormalDisplacementField(p, 16).dims == (H // 16 + 1) ** 2
    f = NormalDisplacementField(p, "global")
    assert torch.allclose(f.up(torch.full((1, 1), 0.3)), 0.3 * grad_mag(p))


def test_normal_field_projection_recovers_a_field_in_its_range():
    """``project`` runs CG on the normal equations to a 1e-3 relative residual, so a
    field already in the range comes back to within the conditioning of U^T U
    (measured 3e-3 here), not to machine precision."""
    f = NormalDisplacementField(_p_ref(), 16)
    torch.manual_seed(3)
    delta = f.up(torch.randn(f.n))
    proj = f.project(delta)
    normal_residual = f.up_t(proj - delta)
    assert float(normal_residual.norm() / f.up_t(delta).norm()) < 1e-3
    assert float((proj - delta).norm() / delta.norm()) < 1e-2


# ---------------------------------------------------------- with the optics

@pytest.fixture(scope="module")
def litho():
    from gril.litho.resist import LithoModel

    kd = paths.find_kernel_dir()
    if kd is None:
        pytest.skip("ICCAD13 kernels not present; run scripts/fetch_data.sh")
    return LithoModel(kd, ProcessConfig())


def test_solve_in_normal_field_reduces_the_new_recipe_objective(litho):
    p = _p_ref()
    lm, tgt = apply_recipe(litho, _target(), litho.cfg, threshold=0.95 * litho.cfg.target_density, bias=1)
    f0 = ILTLeastSquares(tgt, lm, BETA).state(p).f
    out, snaps = solve_in_normal_field(tgt, lm, p, NormalDisplacementField(p, 16), beta=BETA, iterations=6,
                                       checkpoints=(3, 6))
    f = [ILTLeastSquares(tgt, lm, BETA).state(snaps[k][0]).f for k in (3, 6)]
    assert f[0] < f0 and f[1] <= f[0] + 1e-6
    assert torch.equal(out, snaps[6][0])


def test_solve_in_subspace_is_monotone_and_stays_in_the_span(litho):
    p = _p_ref()
    lm, tgt = apply_recipe(litho, _target(), litho.cfg, dose=1.03)
    field = NormalDisplacementField(p, "global")
    torch.manual_seed(4)
    u = pod_basis([field.up(torch.ones(1, 1)), torch.randn(H, H)])
    out = solve_in_subspace(tgt, lm, p, u, beta=BETA, iterations=5)
    model = ILTLeastSquares(tgt, lm, BETA)
    assert model.state(out).f < model.state(p).f
    d = (out - p).reshape(-1)
    assert float((d - u @ (u.T @ d)).norm() / d.norm()) < 1e-4


# ------------------------------------------------------------------ paths

def test_paths_prefer_an_existing_explicit_dir_and_ignore_auto(tmp_path, monkeypatch):
    (tmp_path / "kernels").mkdir()
    (tmp_path / "scales").mkdir()
    assert paths.find_kernel_dir(str(tmp_path)) == str(tmp_path)
    monkeypatch.setenv("GRIL_KERNEL_DIR", str(tmp_path))
    assert paths.find_kernel_dir("auto") == str(tmp_path)


def test_paths_raise_with_the_fix_when_nothing_is_found(monkeypatch):
    monkeypatch.setattr(paths, "find_bench_dir", lambda configured=None: None)
    with pytest.raises(FileNotFoundError, match="fetch_data.sh"):
        paths.bench_dir()
