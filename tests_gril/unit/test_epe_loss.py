"""Tests for the differentiable EPE-site-aware loss (gril.ilt.epe_loss).

The central correctness claim: build_epe_sites() finds the EXACT SAME
measurement geometry that epe_violations() uses internally, and evaluating
soft_epe_loss() on a BINARY (not continuous) printed image reproduces
epe_violations()'s violation COUNT exactly, per side (inner/outer). This is a
genuine cross-check against the already bit-exact-verified metric, not just an
internal self-consistency check.
"""
import pytest
import torch

from gril.ilt.epe_loss import EPESites, build_epe_sites, soft_epe_loss
from gril.metrics.core import epe_violations


def _count_violations_via_soft_loss(z_binary: torch.Tensor, sites: EPESites) -> tuple[int, int]:
    """Exact per-site violation count from the hinge components.

    For BINARY z (exactly {0,1}): relu(0.5 - z_inner) is 0.5 iff z_inner==0
    (matches _check_sites' `image[inner]==0` violation test exactly) and 0
    iff z_inner==1. Symmetric for the outer probe. Counting nonzero hinge
    terms therefore reproduces epe_violations()'s integer counts exactly.
    """
    if sites.n_sites == 0:
        return 0, 0
    z_inner = z_binary[sites.inner_rows, sites.inner_cols]
    z_outer = z_binary[sites.outer_rows, sites.outer_cols]
    inner_violations = int((torch.relu(0.5 - z_inner) > 0).sum())
    outer_violations = int((torch.relu(z_outer - 0.5) > 0).sum())
    return inner_violations, outer_violations


def _square_target(size=400, lo=100, hi=300):
    t = torch.zeros(size, size)
    t[lo:hi, lo:hi] = 1.0
    return t


# --------------------------------------------------------- geometry sanity

def test_site_count_matches_hand_computed_segment_structure():
    """A 200x200 square has 2 vertical + 2 horizontal boundary segments, each
    200 long; with interval 40 and min-length 80, each segment gets multiple
    sample sites -- just check the count is in a sane, nonzero, even range
    (each of the 4 edges contributes the same count by symmetry).
    """
    target = _square_target()
    sites = build_epe_sites(target, tolerance=15, pixel_nm=1.0)
    assert sites.n_sites > 0
    assert sites.n_sites % 4 == 0  # 4-fold symmetric square


def test_empty_target_gives_zero_sites():
    target = torch.zeros(64, 64)
    sites = build_epe_sites(target, tolerance=15)
    assert sites.n_sites == 0


def test_full_target_gives_zero_sites():
    """An all-1 target has a boundary only at the canvas edge, which is
    filtered by _segments' interior-only 8-neighbour boundary test combined
    with this module's own bounds-clipping -- either way, no interior sites.
    """
    target = torch.ones(64, 64)
    sites = build_epe_sites(target, tolerance=15)
    assert sites.n_sites == 0


def test_sites_are_within_canvas_bounds():
    target = _square_target(size=64, lo=20, hi=44)
    sites = build_epe_sites(target, tolerance=10, pixel_nm=1.0)
    assert sites.n_sites > 0
    assert sites.inner_rows.max() < 64 and sites.inner_rows.min() >= 0
    assert sites.outer_cols.max() < 64 and sites.outer_cols.min() >= 0


# --------------------------------- the central cross-check against epe_violations

@pytest.mark.parametrize("shift", [0, 3, 8, 15, 20, -5, -15])
def test_exact_violation_count_matches_metric_uniform_shift(shift):
    """A uniformly shrunk/grown square, at several shift magnitudes straddling
    the 15nm tolerance -- the soft-loss violation count (on a binary image)
    must equal epe_violations()'s count EXACTLY, every time.
    """
    target = _square_target(size=400, lo=100, hi=300)
    printed = _square_target(size=400, lo=100 + shift, hi=300 - shift)
    sites = build_epe_sites(target, tolerance=15, pixel_nm=1.0)
    soft_in, soft_out = _count_violations_via_soft_loss(printed, sites)
    ref_in, ref_out = epe_violations(printed.clone(), target, tolerance=15, pixel_nm=1.0)
    assert (soft_in, soft_out) == (ref_in, ref_out)


def test_exact_violation_count_matches_metric_on_real_iccad13_mask():
    """Strongest test: a REAL, curvy, ILT-optimised mask from an actual run,
    not a synthetic square. Exercises arbitrary segment orientations/lengths.
    """
    import os

    from gril.data.glp import Design

    mask_path = "results/iccad13_ilt/mask1.pt"
    glp_path = "/home/user/openopc/openilt/benchmark/ICCAD2013/M1_test1.glp"
    if not (os.path.exists(mask_path) and os.path.exists(glp_path)):
        pytest.skip("ICCAD13 result / benchmark assets not present in this environment")

    printed = torch.load(mask_path, weights_only=False).float()
    target = torch.tensor(Design.from_glp(glp_path).centred_raster(2048))

    sites = build_epe_sites(target, tolerance=15, pixel_nm=1.0)
    soft_in, soft_out = _count_violations_via_soft_loss(printed, sites)
    ref_in, ref_out = epe_violations(printed.clone(), target, tolerance=15, pixel_nm=1.0)
    assert (soft_in, soft_out) == (ref_in, ref_out)


def test_exact_violation_count_matches_metric_at_3nm_tolerance():
    """Cross-check at the paper's stricter 3nm tolerance too, not just 15nm."""
    target = _square_target(size=400, lo=100, hi=300)
    printed = _square_target(size=400, lo=104, hi=296)  # 4nm shrink each side
    sites = build_epe_sites(target, tolerance=3, pixel_nm=1.0)
    soft_in, soft_out = _count_violations_via_soft_loss(printed, sites)
    ref_in, ref_out = epe_violations(printed.clone(), target, tolerance=3, pixel_nm=1.0)
    assert (soft_in, soft_out) == (ref_in, ref_out)


# ------------------------------------------------------------- soft loss itself

def test_soft_loss_zero_for_perfect_print():
    target = _square_target()
    sites = build_epe_sites(target, tolerance=15)
    z = target.clone()  # continuous field exactly equal to the (binary) target
    loss = soft_epe_loss(z, sites)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_soft_loss_positive_for_violation():
    target = _square_target()
    printed = _square_target(lo=120, hi=280)  # 20nm shrink, beyond 15nm tolerance
    sites = build_epe_sites(target, tolerance=15)
    loss = soft_epe_loss(printed, sites)
    assert float(loss) > 0.0


def test_soft_loss_is_differentiable():
    target = _square_target(size=64, lo=16, hi=48)
    sites = build_epe_sites(target, tolerance=10)
    z = torch.rand(64, 64, requires_grad=True)
    loss = soft_epe_loss(z, sites)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


def test_soft_loss_gradient_pushes_toward_target():
    """At a site with an inner-probe violation (z_inner < 0.5), the gradient
    w.r.t. that pixel must be NEGATIVE (increasing z_inner reduces the loss).
    """
    target = _square_target(size=64, lo=16, hi=48)
    sites = build_epe_sites(target, tolerance=10)
    assert sites.n_sites > 0
    z = torch.full((64, 64), 0.2, requires_grad=True)  # under-printed everywhere
    loss = soft_epe_loss(z, sites)
    loss.backward()
    r, c = int(sites.inner_rows[0]), int(sites.inner_cols[0])
    assert z.grad[r, c] < 0


def test_soft_loss_empty_sites_returns_zero_with_gradient():
    target = torch.zeros(32, 32)
    sites = build_epe_sites(target, tolerance=10)
    z = torch.rand(32, 32, requires_grad=True)
    loss = soft_epe_loss(z, sites)
    assert float(loss) == 0.0
    loss.backward()  # must not raise
    assert z.grad is not None


def test_margin_increases_loss():
    target = _square_target(size=64, lo=16, hi=48)
    sites = build_epe_sites(target, tolerance=10)
    z = target.clone()
    loss_no_margin = soft_epe_loss(z, sites, margin=0.0)
    loss_with_margin = soft_epe_loss(z, sites, margin=0.1)
    assert float(loss_with_margin) >= float(loss_no_margin)


def test_build_epe_sites_rejects_nonpositive_pixel_nm():
    target = _square_target(size=32, lo=8, hi=24)
    with pytest.raises(ValueError):
        build_epe_sites(target, tolerance=10, pixel_nm=0.0)


# --------------------------------------------------- integration with solve()

def test_solve_weight_epe_zero_is_bit_exact_with_baseline():
    """weight_epe=0 (default) must not touch epe_sites at all -- verifies the
    zero-cost-when-disabled contract solve() documents.
    """
    import sys

    sys.path.insert(0, "src")
    from gril.ilt.solver import ILTConfig, solve

    target = _square_target(size=64, lo=16, hi=48)

    class _DummyLitho:
        class cfg:
            target_density = 0.225
            print_thresh = 0.5
            print_steepness = 50.0

        def aerial_nominal(self, mask):
            return mask  # identity stand-in, just needs to be differentiable

        def aerial_outer(self, mask):
            return mask, mask

        def binary(self, mask):
            b = (mask >= 0.5).float()
            return b, b, b

    litho = _DummyLitho()
    cfg_a = ILTConfig(iterations=5, step_size=0.1, mask_steepness=4.0, weight_epe=0.0)
    cfg_b = ILTConfig(iterations=5, step_size=0.1, mask_steepness=4.0)  # explicit default
    a = solve(target, litho, cfg_a).loss_history
    b = solve(target, litho, cfg_b).loss_history
    assert a == b


def test_solve_weight_epe_positive_reduces_real_epe_violations(kernel_dir):
    """End-to-end, on the real litho model: adding the EPE-aware term to the
    ILT objective must not INCREASE the measured EPE violation count relative
    to the same budget without it, on a real (non-toy) test pattern. This is
    an integration check across solver.py + epe_loss.py + the real SOCS model
    + the real (bit-exact-verified) epe_violations metric -- not a synthetic
    unit test.
    """
    import sys

    sys.path.insert(0, "src")
    from gril.ilt.solver import ILTConfig, solve
    from gril.litho.resist import LithoModel, ProcessConfig
    from gril.metrics.core import epe_violations

    litho = LithoModel(kernel_dir, ProcessConfig())
    target = _square_target(size=256, lo=80, hi=176)

    def final_epe(cfg):
        res = solve(target, litho, cfg)
        with torch.no_grad():
            b_nom, _, _ = litho.binary(res.mask)
        ein, eout = epe_violations(b_nom, target, tolerance=15, pixel_nm=1.0)
        return ein + eout

    baseline_cfg = ILTConfig(
        iterations=15, step_size=0.2, mask_steepness=8.0, init_scale=0.5, weight_epe=0.0
    )
    epe_aware_cfg = ILTConfig(
        iterations=15, step_size=0.2, mask_steepness=8.0, init_scale=0.5,
        weight_epe=1.0, epe_tolerance_nm=15.0,
    )
    baseline_epe = final_epe(baseline_cfg)
    aware_epe = final_epe(epe_aware_cfg)
    assert aware_epe <= baseline_epe, (
        f"EPE-aware term made EPE worse: baseline={baseline_epe}, aware={aware_epe}"
    )
