"""Tests for the numerical ILT solver."""
import pytest
import torch

from gril.ilt.solver import ILTConfig, ILTResult, morphological_open, solve
from gril.litho.resist import LithoModel, ProcessConfig


@pytest.fixture(scope="module")
def litho(kernel_dir):
    return LithoModel(kernel_dir, ProcessConfig())


@pytest.fixture(scope="module")
def target():
    t = torch.zeros(128, 128)
    t[40:90, 40:90] = 1.0
    return t


def test_solve_returns_binary_mask(litho, target):
    res = solve(target, litho, ILTConfig(iterations=5))
    assert set(res.mask.unique().tolist()) <= {0.0, 1.0}
    assert res.mask.shape == target.shape


def test_solve_reduces_loss(litho, target):
    res = solve(target, litho, ILTConfig(iterations=30, step_size=0.2, mask_steepness=8.0))
    assert res.loss_history[-1] < res.loss_history[0]


def test_solve_improves_on_no_opc(litho, target):
    """ILT must beat using the target itself as the mask."""
    res = solve(target, litho, ILTConfig(iterations=60, step_size=0.2, mask_steepness=8.0))
    with torch.no_grad():
        opt_l2 = float(((litho.binary(res.mask)[0] - target) ** 2).sum())
        raw_l2 = float(((litho.binary(target)[0] - target) ** 2).sum())
    assert opt_l2 < raw_l2


def test_solve_is_deterministic(litho, target):
    cfg = ILTConfig(iterations=8, seed=3)
    assert torch.equal(solve(target, litho, cfg).mask, solve(target, litho, cfg).mask)


def test_solve_works_inside_no_grad(litho, target):
    """Regression: solve() runs its own optimisation and must hold a grad scope.

    Calling it from evaluation code wrapped in torch.no_grad() used to raise
    "element 0 of tensors does not require grad".
    """
    with torch.no_grad():
        res = solve(target, litho, ILTConfig(iterations=3))
    assert len(res.loss_history) == 3


def test_batched_matches_single(litho, target):
    other = torch.zeros(128, 128)
    other[30:70, 50:100] = 1.0
    batch = torch.stack([target, other])
    cfg = ILTConfig(iterations=6, step_size=0.2)
    got = solve(batch, litho, cfg).mask
    for i, t in enumerate([target, other]):
        assert torch.equal(got[i], solve(t, litho, cfg).mask)


def test_checkpoints_are_recorded(litho, target):
    res = solve(target, litho, ILTConfig(iterations=10, checkpoints=(3, 7, 10)))
    assert sorted(res.checkpoint_masks) == [3, 7, 10]
    assert torch.equal(res.checkpoint_masks[10], res.mask)


def test_morphological_open_removes_isolated_speck():
    x = torch.zeros(32, 32)
    x[10:22, 10:22] = 1.0     # survives
    x[2, 2] = 1.0             # single pixel, removed by a 3x3 opening
    out = morphological_open(x, 3)
    assert out[2, 2] == 0
    assert out[15, 15] == 1


def test_morphological_open_is_identity_for_size_one():
    x = torch.rand(16, 16).round()
    assert torch.equal(morphological_open(x, 1), x)
