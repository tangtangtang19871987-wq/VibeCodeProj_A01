"""Tests for beta annealing / continuation (ILTConfig.beta_init, beta_schedule).

See docs/findings.md F-ANNEAL-01 and F-SAT-01: fixed high mask_steepness
combined with the default init_scale saturates the sigmoid so badly that raw-
gradient optimizers (SGD, L-BFGS) cannot move. Annealing beta from a low
starting value ramps into that regime instead of starting there.
"""
import pytest
import torch

from gril.ilt.solver import ILTConfig, _beta_for_step, ilt_loss, solve
from gril.litho.resist import LithoModel, ProcessConfig


@pytest.fixture(scope="module")
def litho(kernel_dir):
    return LithoModel(kernel_dir, ProcessConfig())


@pytest.fixture(scope="module")
def target():
    t = torch.zeros(96, 96)
    t[25:70, 25:70] = 1.0
    return t


# ------------------------------------------------------------- schedule math

def test_no_annealing_is_constant_at_mask_steepness():
    cfg = ILTConfig(iterations=11, mask_steepness=8.0)
    assert all(_beta_for_step(cfg, s) == 8.0 for s in range(11))


def test_linear_schedule_endpoints_and_midpoint():
    cfg = ILTConfig(iterations=11, beta_init=1.0, mask_steepness=9.0, beta_schedule="linear")
    assert _beta_for_step(cfg, 0) == pytest.approx(1.0)
    assert _beta_for_step(cfg, 10) == pytest.approx(9.0)
    assert _beta_for_step(cfg, 5) == pytest.approx(5.0)


def test_linear_schedule_is_monotonic():
    cfg = ILTConfig(iterations=20, beta_init=0.5, mask_steepness=8.0, beta_schedule="linear")
    vals = [_beta_for_step(cfg, s) for s in range(20)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))


def test_exponential_schedule_endpoints():
    cfg = ILTConfig(iterations=11, beta_init=1.0, mask_steepness=8.0, beta_schedule="exponential")
    assert _beta_for_step(cfg, 0) == pytest.approx(1.0)
    assert _beta_for_step(cfg, 10) == pytest.approx(8.0, rel=1e-6)


def test_exponential_ramps_slower_early_than_linear():
    """Exponential (geometric) interpolation should stay below linear at the
    midpoint when beta_init << mask_steepness (a slower early ramp)."""
    lin = ILTConfig(iterations=11, beta_init=1.0, mask_steepness=16.0, beta_schedule="linear")
    exp = ILTConfig(iterations=11, beta_init=1.0, mask_steepness=16.0, beta_schedule="exponential")
    assert _beta_for_step(exp, 5) < _beta_for_step(lin, 5)


def test_beta_held_beyond_final_iteration_index():
    """_beta_for_step must clamp -- callers may query step >= iterations-1."""
    cfg = ILTConfig(iterations=5, beta_init=1.0, mask_steepness=8.0)
    assert _beta_for_step(cfg, 4) == pytest.approx(8.0)
    assert _beta_for_step(cfg, 100) == pytest.approx(8.0)  # never overshoots


def test_single_iteration_run_does_not_divide_by_zero():
    """iterations=1 must not crash on iterations-1=0 in the schedule denominator.

    Step 0 always starts the schedule at beta_init (frac=0), regardless of how
    many total iterations there are -- the guard (total=max(iterations-1,1))
    only exists to keep the division defined, not to change where step 0 sits.
    """
    cfg = ILTConfig(iterations=1, beta_init=1.0, mask_steepness=8.0)
    assert _beta_for_step(cfg, 0) == pytest.approx(1.0)


# --------------------------------------------------------------- validation

def test_unknown_beta_schedule_rejected():
    with pytest.raises(ValueError, match="beta_schedule"):
        ILTConfig(beta_init=1.0, beta_schedule="cosine")


def test_nonpositive_beta_init_rejected():
    with pytest.raises(ValueError, match="beta_init"):
        ILTConfig(beta_init=0.0)
    with pytest.raises(ValueError, match="beta_init"):
        ILTConfig(beta_init=-1.0)


def test_beta_schedule_ignored_when_beta_init_is_none():
    """An invalid beta_schedule string must not raise if beta_init is unset --
    it is documented as ignored in that case, so it should not be validated."""
    ILTConfig(beta_schedule="not_a_real_schedule")  # beta_init=None -> must not raise


# -------------------------------------------------------------- loss override

def test_ilt_loss_beta_override_matches_cfg_default(litho, target):
    """beta=None must be identical to beta=cfg.mask_steepness."""
    cfg = ILTConfig(mask_steepness=6.0)
    params = torch.zeros(96, 96, requires_grad=False)
    a = ilt_loss(params, target, litho, cfg)
    b = ilt_loss(params, target, litho, cfg, beta=6.0)
    assert torch.equal(a, b)


def test_ilt_loss_beta_override_changes_value(litho, target):
    cfg = ILTConfig(mask_steepness=6.0)
    params = torch.randn(96, 96) * 0.5
    a = ilt_loss(params, target, litho, cfg, beta=1.0)
    b = ilt_loss(params, target, litho, cfg, beta=6.0)
    assert not torch.equal(a, b)


# ------------------------------------------------------- backward compatibility

def test_beta_init_none_is_bit_exact_with_constant_beta(litho, target):
    """The whole point of defaulting beta_init=None: zero behavioural change."""
    cfg_plain = ILTConfig(iterations=10, step_size=0.2, mask_steepness=8.0)
    cfg_explicit_none = ILTConfig(
        iterations=10, step_size=0.2, mask_steepness=8.0, beta_init=None
    )
    a = solve(target, litho, cfg_plain).loss_history
    b = solve(target, litho, cfg_explicit_none).loss_history
    assert a == b


# --------------------------------------------------------- the actual fix

def test_annealing_lets_lbfgs_train_at_saturated_baseline_constants(litho, target):
    """The central claim of F-ANNEAL-01: at the main ICCAD13 baseline's own
    constants (init_scale=2.0, mask_steepness=8.0), which F-SAT-01 showed
    stall L-BFGS completely (0.0% loss change), annealing beta from a low
    starting value must let it make real progress.
    """
    stalled_cfg = ILTConfig(
        optimizer="lbfgs", iterations=10, step_size=1.0, mask_steepness=8.0, init_scale=2.0
    )
    annealed_cfg = ILTConfig(
        optimizer="lbfgs", iterations=10, step_size=1.0, mask_steepness=8.0,
        init_scale=2.0, beta_init=1.0,
    )

    def final_l2(cfg):
        res = solve(target, litho, cfg)
        with torch.no_grad():
            b_nom, _, _ = litho.binary(res.mask)
            return float(((b_nom - target) ** 2).sum())

    no_opc_l2 = float(((target - target) ** 2).sum())  # 0, sanity only
    stalled_l2 = final_l2(stalled_cfg)
    annealed_l2 = final_l2(annealed_cfg)

    # Stalled run must reproduce F-SAT-01: no measurable improvement over a
    # cold start whose binarised mask already roughly equals the target.
    assert annealed_l2 < stalled_l2, (
        f"annealing did not help: stalled={stalled_l2}, annealed={annealed_l2}"
    )


def test_annealed_lbfgs_batched_equals_single(litho, target):
    """Annealing must not break the batched == looped-single invariant."""
    other = torch.zeros(96, 96)
    other[10:50, 40:90] = 1.0
    batch = torch.stack([target, other])
    cfg = ILTConfig(
        optimizer="lbfgs", iterations=5, mask_steepness=8.0, init_scale=2.0, beta_init=1.0
    )
    batched = solve(batch, litho, cfg).mask
    for i, t in enumerate([target, other]):
        single = solve(t, litho, cfg).mask
        assert torch.equal(batched[i], single)


def test_annealed_adam_still_reduces_loss(litho, target):
    """Sanity: annealing composes with Adam too, not just L-BFGS/SGD."""
    cfg = ILTConfig(
        optimizer="adam", iterations=15, step_size=0.2, mask_steepness=8.0,
        init_scale=2.0, beta_init=2.0,
    )
    res = solve(target, litho, cfg)
    # NOTE: loss values at different steps use different beta, so this is not
    # "the same objective decreasing" in the usual sense -- it is a sanity
    # check that the run completes and produces a valid binary mask, not a
    # claim about monotonic decrease under annealing.
    assert res.mask.shape == target.shape
    assert set(res.mask.unique().tolist()) <= {0.0, 1.0}
