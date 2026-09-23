"""Tests for gril.experiments.run_multistart_ablation's building blocks."""
import torch

from gril.experiments.run_multistart_ablation import _diversity, _score_candidates


def test_diversity_zero_for_identical_masks():
    masks = torch.ones(4, 8, 8)
    assert _diversity(masks) == 0.0


def test_diversity_positive_for_distinct_masks():
    masks = torch.zeros(2, 8, 8)
    masks[1, :4, :4] = 1.0
    assert _diversity(masks) > 0.0


def test_diversity_single_mask_is_zero():
    masks = torch.rand(1, 8, 8)
    assert _diversity(masks) == 0.0


def test_diversity_matches_hand_computation():
    """Two 4x4 masks differing in exactly 4 of 16 pixels -> L1 mean = 4/16."""
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    b[0, :] = 1.0  # 4 pixels differ
    masks = torch.stack([a, b])
    assert abs(_diversity(masks) - 4 / 16) < 1e-6


def test_score_candidates_counts_epe_violations(kernel_dir):
    from gril.litho.resist import LithoModel, ProcessConfig

    litho = LithoModel(kernel_dir, ProcessConfig())
    design = torch.zeros(1, 1, 128, 128)
    design[0, 0, 40:88, 40:88] = 1.0
    # Two candidates: one identical to target (K=1 shown as a 2-batch for shape),
    # one badly shrunk.
    good = design.clone()
    bad = torch.zeros(1, 1, 128, 128)
    bad[0, 0, 55:73, 55:73] = 1.0
    candidates = torch.cat([good, bad], dim=0)
    scores = _score_candidates(candidates, design, litho, tolerance=15, pixel_nm=1.0)
    assert len(scores) == 2
    assert scores[0] <= scores[1]  # the good candidate must score no worse
