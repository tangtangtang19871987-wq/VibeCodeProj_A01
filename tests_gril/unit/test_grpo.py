"""Tests for GRPO finetuning (paper Eqs. 6-10)."""
import copy

import pytest
import torch

from gril.models.generator import GeneratorConfig, StyleAwareGenerator
from gril.train.grpo import (
    GRPOConfig,
    advantages,
    finetune_step,
    policy_log_prob,
    smooth_target,
)


def test_policy_log_prob_matches_bernoulli():
    """Eq. 8's BCE surrogate must equal the exact Bernoulli log-probability."""
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 8, 8)
    actions = (logits > 0.5).float()
    mine = policy_log_prob(logits, actions, reduction="sum")
    ref = torch.distributions.Bernoulli(logits=logits).log_prob(actions).flatten(1).sum(1)
    assert torch.allclose(mine, ref, atol=1e-5)


def test_log_prob_is_negative():
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 8, 8)
    assert (policy_log_prob(logits, (logits > 0.5).float()) <= 0).all()


def test_group_mean_advantage_is_zero_mean():
    adv = advantages(torch.tensor([1.0, 5.0, 2.0, 8.0]), None, "group_mean")
    assert abs(float(adv.mean())) < 1e-6


def test_teacher_relative_advantage():
    r = torch.tensor([1.0, 2.0, 3.0])
    t = torch.tensor([0.5, 2.5, 1.0])
    assert torch.allclose(advantages(r, t, "teacher_relative"), r - t)


def test_teacher_relative_requires_teacher():
    with pytest.raises(ValueError):
        advantages(torch.tensor([1.0]), None, "teacher_relative")


def test_unknown_advantage_mode_rejected():
    with pytest.raises(ValueError):
        advantages(torch.tensor([1.0]), None, "nonsense")


def test_smoothing_preserves_constant_regions():
    """S(.) is an average pool, so a constant field is unchanged."""
    x = torch.full((1, 1, 32, 32), 0.7)
    assert torch.allclose(smooth_target(x, 5), x, atol=1e-6)


def test_smoothing_is_identity_for_kernel_one():
    x = torch.rand(1, 1, 16, 16)
    assert torch.equal(smooth_target(x, 1), x)


def _setup():
    torch.manual_seed(0)
    cfg_g = GeneratorConfig(base_channels=8, n_style_blocks=1, n_local_blocks=1, latent_dim=16, style_dim=16)
    gen = StyleAwareGenerator(cfg_g)
    teacher = copy.deepcopy(gen).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    design = torch.zeros(1, 1, 32, 32)
    design[..., 8:24, 8:24] = 1.0
    return gen, teacher, design


def test_finetune_step_leaves_teacher_frozen():
    gen, teacher, design = _setup()
    before = [p.clone() for p in teacher.parameters()]
    cfg = GRPOConfig(group_size=3, latent_dim=16, smoothing_kernel=3)
    out = finetune_step(
        gen, teacher, design,
        refine=lambda m: m,
        reward_fn=lambda m: -(m - design).abs().flatten(1).sum(1),
        cfg=cfg,
    )
    out["loss"].backward()
    assert all(torch.equal(a, b) for a, b in zip(before, teacher.parameters()))
    assert all(p.grad is None for p in teacher.parameters())


def test_finetune_step_produces_generator_gradient():
    gen, teacher, design = _setup()
    cfg = GRPOConfig(group_size=3, latent_dim=16, smoothing_kernel=3)
    out = finetune_step(
        gen, teacher, design,
        refine=lambda m: m,
        reward_fn=lambda m: -(m - design).abs().flatten(1).sum(1),
        cfg=cfg,
    )
    out["loss"].backward()
    total = sum(p.grad.abs().sum() for p in gen.parameters() if p.grad is not None)
    assert total > 0


def test_reward_improves_when_mask_improves():
    """Sanity check on the reward direction: R = -EPE, so better mask -> higher R."""
    design = torch.zeros(1, 1, 32, 32)
    design[..., 8:24, 8:24] = 1.0
    reward = lambda m: -(m - design).abs().flatten(1).sum(1)
    good = design.clone()
    bad = torch.zeros_like(design)
    assert float(reward(good)) > float(reward(bad))
