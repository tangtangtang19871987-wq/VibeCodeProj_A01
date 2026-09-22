"""Hand-computed unit tests for the scoring metrics."""
import torch

from gril.metrics.core import epe_violations, l2_loss, pv_band


def test_l2_hand_computed():
    target = torch.zeros(8, 8)
    target[2:6, 2:6] = 1.0          # 16 target pixels
    printed = torch.zeros(8, 8)
    printed[2:6, 2:5] = 1.0         # 12 printed pixels, 4 missing
    assert l2_loss(printed, target) == 4.0


def test_l2_zero_for_perfect_print():
    target = torch.zeros(8, 8)
    target[1:4, 1:4] = 1.0
    assert l2_loss(target.clone(), target) == 0.0


def test_l2_counts_both_polarities():
    target = torch.zeros(4, 4)
    target[0, 0] = 1.0              # missing -> 1
    printed = torch.zeros(4, 4)
    printed[3, 3] = 1.0             # extra   -> 1
    assert l2_loss(printed, target) == 2.0


def test_pvband_hand_computed():
    outer = torch.zeros(8, 8)
    outer[2:6, 2:6] = 1.0           # 16 px
    inner = torch.zeros(8, 8)
    inner[3:5, 3:5] = 1.0           # 4 px, nested
    assert pv_band(outer, inner) == 12.0


def test_pvband_zero_when_corners_identical():
    img = torch.zeros(8, 8)
    img[2:6, 2:6] = 1.0
    assert pv_band(img, img.clone()) == 0.0


def test_epe_zero_for_exact_print():
    """A perfect print has no edge displacement anywhere."""
    target = torch.zeros(400, 400)
    target[100:300, 100:300] = 1.0
    inner, outer = epe_violations(target.clone(), target, tolerance=15)
    assert (inner, outer) == (0, 0)


def test_epe_detects_uniform_shrink():
    """Shrinking every edge by more than the tolerance must flag inner violations."""
    target = torch.zeros(400, 400)
    target[100:300, 100:300] = 1.0
    printed = torch.zeros(400, 400)
    printed[120:280, 120:280] = 1.0     # 20 nm pull-back > 15 nm tolerance
    inner, outer = epe_violations(printed, target, tolerance=15)
    assert inner > 0
    assert outer == 0


def test_epe_detects_uniform_bloat():
    target = torch.zeros(400, 400)
    target[100:300, 100:300] = 1.0
    printed = torch.zeros(400, 400)
    printed[80:320, 80:320] = 1.0       # 20 nm push-out
    inner, outer = epe_violations(printed, target, tolerance=15)
    assert outer > 0
    assert inner == 0


def test_epe_violations_monotone_in_tolerance():
    """Loosening the tolerance can never increase the violation count."""
    target = torch.zeros(400, 400)
    target[100:300, 100:300] = 1.0
    printed = torch.zeros(400, 400)
    printed[112:288, 112:288] = 1.0
    counts = [sum(epe_violations(printed, target, tolerance=t)) for t in (3, 6, 9, 12, 15)]
    assert all(a >= b for a, b in zip(counts, counts[1:])), counts


def test_epe_shrink_within_tolerance_is_clean():
    target = torch.zeros(400, 400)
    target[100:300, 100:300] = 1.0
    printed = torch.zeros(400, 400)
    printed[105:295, 105:295] = 1.0     # 5 nm < 15 nm tolerance
    assert epe_violations(printed, target, tolerance=15) == (0, 0)
