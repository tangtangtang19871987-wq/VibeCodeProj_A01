"""Tests for the ILT solver's optimizer/regularisation upgrades.

Covers: SGD (plain + momentum), Nesterov, L-BFGS, total-variation
regularisation, gradient clipping, gradient-norm convergence, and the
correctness invariants each new path must preserve (determinism; batched ==
looped single-case).
"""
import pytest
import torch

from gril.ilt.solver import ILTConfig, solve, total_variation
from gril.litho.resist import LithoModel, ProcessConfig


@pytest.fixture(scope="module")
def litho(kernel_dir):
    return LithoModel(kernel_dir, ProcessConfig())


@pytest.fixture(scope="module")
def target():
    t = torch.zeros(96, 96)
    t[25:70, 25:70] = 1.0
    return t


#: init_scale=0.5 keeps sigmoid(mask_steepness * P) well away from saturation at
#: init (see test_saturation_makes_raw_gradient_methods_stall below and
#: docs/findings.md F-SAT-01). The project's DEFAULT init_scale=2.0 combined
#: with mask_steepness=8.0 saturates the sigmoid so hard that raw-gradient
#: methods (SGD, Nesterov, L-BFGS) cannot move in a small iteration budget --
#: only Adam's per-parameter normalisation can. Tests that exercise SGD/
#: Nesterov/L-BFGS use this well-conditioned starting point instead, which is
#: also the practical guidance this project gives anyone using those
#: optimizers: keep init_scale * mask_steepness moderate, or warm-start from
#: something already close to a reasonable mask.
WELL_CONDITIONED_INIT = 0.5


# ---------------------------------------------------------------- validation

def test_unknown_optimizer_raises():
    """Earlier versions silently fell back to SGD for any unrecognised name."""
    with pytest.raises(ValueError, match="Unknown optimizer"):
        ILTConfig(optimizer="rmsprop")


def test_nesterov_without_momentum_raises():
    with pytest.raises(ValueError, match="momentum"):
        ILTConfig(optimizer="nesterov", momentum=0.0)


def test_nesterov_with_momentum_is_valid():
    ILTConfig(optimizer="nesterov", momentum=0.9)  # must not raise


# ------------------------------------------------- saturation characterisation

def test_saturation_makes_raw_gradient_methods_stall(litho, target):
    """Characterisation test, not a bug report.

    At the project's DEFAULT init_scale=2.0 with mask_steepness=8.0, the
    argument to the mask sigmoid is +-16 at every pixel, which saturates it so
    hard that the raw gradient magnitude is ~1e-6 (measured). Adam's
    per-parameter step normalisation (dividing by an estimate of the gradient's
    own RMS) is insensitive to this: it still takes an order-1 step. SGD does
    not normalise, so at any step size that is sane for a WELL-conditioned
    problem it makes negligible progress in a modest iteration budget. This is
    exactly why every existing ICCAD13 experiment in this repository uses Adam
    (G-015 -- not a paper requirement, but a consequence of this
    parameterisation's own conditioning at the values chosen for it).
    """
    cfg_adam = ILTConfig(optimizer="adam", iterations=20, step_size=0.2, mask_steepness=8.0)
    cfg_sgd = ILTConfig(optimizer="sgd", iterations=20, step_size=0.2, mask_steepness=8.0)

    adam_drop = solve(target, litho, cfg_adam).loss_history
    sgd_drop = solve(target, litho, cfg_sgd).loss_history

    adam_rel_change = abs(adam_drop[-1] - adam_drop[0]) / adam_drop[0]
    sgd_rel_change = abs(sgd_drop[-1] - sgd_drop[0]) / sgd_drop[0]

    assert adam_rel_change > 0.01, "Adam should visibly move even from a saturated start"
    assert sgd_rel_change < 1e-6, "SGD should stall at a saturated start without a huge step size"


# ---------------------------------------------------------------------- SGD

def test_sgd_plain_reduces_loss(litho, target):
    cfg = ILTConfig(
        optimizer="sgd", iterations=40, step_size=0.05, mask_steepness=8.0,
        init_scale=WELL_CONDITIONED_INIT,
    )
    res = solve(target, litho, cfg)
    assert res.loss_history[-1] < res.loss_history[0]
    assert res.optimizer == "sgd"
    assert res.n_func_evals == res.iterations_run  # one eval per SGD step


def test_sgd_momentum_reduces_loss(litho, target):
    cfg = ILTConfig(
        optimizer="sgd", iterations=40, step_size=0.05, mask_steepness=8.0, momentum=0.9,
        init_scale=WELL_CONDITIONED_INIT,
    )
    res = solve(target, litho, cfg)
    assert res.loss_history[-1] < res.loss_history[0]


def test_nesterov_reduces_loss(litho, target):
    cfg = ILTConfig(
        optimizer="nesterov", iterations=40, step_size=0.05, mask_steepness=8.0, momentum=0.9,
        init_scale=WELL_CONDITIONED_INIT,
    )
    res = solve(target, litho, cfg)
    assert res.loss_history[-1] < res.loss_history[0]


# -------------------------------------------------------------------- LBFGS

def test_lbfgs_reduces_loss(litho, target):
    cfg = ILTConfig(
        optimizer="lbfgs", iterations=15, step_size=1.0, mask_steepness=8.0,
        init_scale=WELL_CONDITIONED_INIT,
    )
    res = solve(target, litho, cfg)
    assert res.loss_history[-1] < res.loss_history[0]
    assert res.optimizer == "lbfgs"


def test_lbfgs_records_more_func_evals_than_iterations(litho, target):
    """Strong-Wolfe line search evaluates the closure more than once per step."""
    cfg = ILTConfig(optimizer="lbfgs", iterations=10, mask_steepness=8.0)
    res = solve(target, litho, cfg)
    assert res.n_func_evals >= res.iterations_run


def test_lbfgs_is_deterministic(litho, target):
    cfg = ILTConfig(optimizer="lbfgs", iterations=8, mask_steepness=8.0)
    a = solve(target, litho, cfg).mask
    b = solve(target, litho, cfg).mask
    assert torch.equal(a, b)


def test_lbfgs_does_not_touch_global_rng(litho, target):
    torch.manual_seed(42)
    expected = torch.randn(4)
    torch.manual_seed(42)
    solve(target, litho, ILTConfig(optimizer="lbfgs", iterations=3, mask_steepness=8.0))
    assert torch.equal(torch.randn(4), expected)


def test_lbfgs_batched_equals_single(litho, target):
    """Per-example independent L-BFGS instances must reproduce single-case runs
    exactly -- this is the whole reason batched L-BFGS does NOT share one joint
    optimizer instance across the batch (see solver.py's _solve_lbfgs docstring).
    """
    other = torch.zeros(96, 96)
    other[10:50, 40:90] = 1.0
    batch = torch.stack([target, other])
    cfg = ILTConfig(optimizer="lbfgs", iterations=6, mask_steepness=8.0)

    batched = solve(batch, litho, cfg).mask
    for i, t in enumerate([target, other]):
        single = solve(t, litho, cfg).mask
        assert torch.equal(batched[i], single)


def test_lbfgs_checkpoints_recorded_for_batch(litho, target):
    other = torch.zeros(96, 96)
    other[5:40, 5:40] = 1.0
    batch = torch.stack([target, other])
    cfg = ILTConfig(optimizer="lbfgs", iterations=6, mask_steepness=8.0, checkpoints=(3, 6))
    res = solve(batch, litho, cfg)
    assert sorted(res.checkpoint_masks) == [3, 6]
    assert res.checkpoint_masks[6].shape == batch.shape


def test_lbfgs_max_iter_per_step_is_config_exposed(litho, target):
    """Two L-BFGS steps of max_iter=2 must not be identical to one of max_iter=1
    followed by nothing -- i.e. the knob actually changes the trajectory.
    """
    cfg_a = ILTConfig(optimizer="lbfgs", iterations=3, lbfgs_max_iter_per_step=1, mask_steepness=8.0)
    cfg_b = ILTConfig(optimizer="lbfgs", iterations=3, lbfgs_max_iter_per_step=3, mask_steepness=8.0)
    res_a = solve(target, litho, cfg_a)
    res_b = solve(target, litho, cfg_b)
    assert res_b.n_func_evals >= res_a.n_func_evals


# ---------------------------------------------------------- regularisation

def test_total_variation_zero_for_constant_field():
    x = torch.full((32, 32), 0.5)
    assert float(total_variation(x)) == 0.0


def test_total_variation_positive_for_a_boundary():
    x = torch.zeros(32, 32)
    x[10:20, 10:20] = 1.0
    assert float(total_variation(x)) > 0.0


def test_tv_regularisation_reduces_final_mask_roughness(litho, target):
    """Turning on weight_tv must reduce the TV of the (continuous) mask produced,
    relative to an identical run without it, all else equal.
    """
    base = ILTConfig(iterations=60, step_size=0.2, mask_steepness=8.0, weight_tv=0.0)
    reg = ILTConfig(iterations=60, step_size=0.2, mask_steepness=8.0, weight_tv=50.0)

    res_base = solve(target, litho, base)
    res_reg = solve(target, litho, reg)

    tv_base = float(total_variation(torch.sigmoid(base.mask_steepness * res_base.params)))
    tv_reg = float(total_variation(torch.sigmoid(reg.mask_steepness * res_reg.params)))
    assert tv_reg < tv_base


# -------------------------------------------------------------- grad clip

def test_grad_clip_bounds_gradient_norm(litho, target):
    """With clipping on, the FIRST step's gradient norm must not exceed the cap."""
    cfg = ILTConfig(iterations=1, mask_steepness=8.0, grad_clip=1.0)
    params = (cfg.init_scale * (2.0 * target - 1.0)).clone().requires_grad_(True)
    from gril.ilt.solver import ilt_loss

    loss = ilt_loss(params, target, litho, cfg)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([params], cfg.grad_clip)
    assert params.grad.norm().item() <= cfg.grad_clip + 1e-4


def test_grad_clip_changes_trajectory(litho, target):
    """Sanity: clipping must actually alter the run relative to unclipped."""
    unclipped = ILTConfig(iterations=10, step_size=0.5, mask_steepness=8.0, grad_clip=0.0)
    clipped = ILTConfig(iterations=10, step_size=0.5, mask_steepness=8.0, grad_clip=10.0)
    a = solve(target, litho, unclipped).loss_history
    b = solve(target, litho, clipped).loss_history
    assert a != b


# --------------------------------------------------------- grad-norm stop

def test_convergence_grad_tol_can_stop_early(litho, target):
    """A very loose gradient-norm tolerance must trigger before the iteration cap."""
    cfg = ILTConfig(iterations=300, step_size=0.2, mask_steepness=8.0, convergence_grad_tol=1e6)
    res = solve(target, litho, cfg)
    assert res.iterations_run < 300
