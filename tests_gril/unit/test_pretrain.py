"""Tests for WGAN-GP pretraining (paper Eq. 5)."""
import torch

from gril.models.generator import Discriminator, GeneratorConfig, StyleAwareGenerator
from gril.train.pretrain import (
    PretrainConfig,
    critic_step,
    gradient_penalty,
    reconstruction_loss,
)


class _ZeroCritic(torch.nn.Module):
    """Critic whose gradient w.r.t. the input is exactly zero."""

    def forward(self, mask, design):
        return (mask * 0).flatten(1).sum(1)


def test_gradient_penalty_finite_at_zero_gradient():
    """Regression: ``.norm()`` has an infinite derivative at 0 and gave NaN.

    An untrained critic really can produce an exactly-zero input gradient, which
    poisoned the whole loss on the first step.
    """
    real = torch.rand(2, 1, 32, 32)
    gp = gradient_penalty(_ZeroCritic(), real, real.clone(), torch.rand(2, 1, 32, 32))
    assert torch.isfinite(gp)
    assert abs(float(gp) - 1.0) < 1e-4      # ||0|| = 0 -> (0-1)^2 = 1


def test_gradient_penalty_zero_for_unit_norm_map():
    """A map with unit input-gradient norm must give a zero penalty."""

    class Unit(torch.nn.Module):
        def forward(self, mask, design):
            return mask.flatten(1)[:, 0]

    real = torch.rand(3, 1, 8, 8)
    gp = gradient_penalty(Unit(), real, torch.rand(3, 1, 8, 8), torch.rand(3, 1, 8, 8))
    assert float(gp) < 1e-6


def test_reconstruction_loss_zero_on_match():
    target = torch.rand(2, 1, 16, 16).round()
    logits = torch.where(target > 0.5, 20.0, -20.0)      # sigmoid -> ~{0,1}
    for kind in ("l1", "l2"):
        assert float(reconstruction_loss(logits, target, kind)) < 1e-6


def test_critic_step_stays_finite():
    torch.manual_seed(0)
    gen = StyleAwareGenerator(
        GeneratorConfig(base_channels=8, n_style_blocks=1, n_local_blocks=1, latent_dim=16)
    )
    critic = Discriminator(base_channels=8, levels=2)
    cfg = PretrainConfig(latent_dim=16)
    opt = torch.optim.Adam(critic.parameters(), lr=2e-4)
    design = torch.rand(2, 1, 64, 64).round()
    real = torch.rand(2, 1, 64, 64).round()
    for _ in range(3):
        stats = critic_step(critic, gen, design, real, cfg, opt)
        assert all(torch.isfinite(torch.tensor(v)) for v in stats.values()), stats
