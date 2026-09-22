"""Tests for the style-aware generator (paper Sec. 3.2.1, Fig. 3, Eq. 4)."""
import torch

from gril.models.generator import (
    AdaIN,
    Discriminator,
    GeneratorConfig,
    StyleAwareGenerator,
    StyleResBlock,
)

CFG = GeneratorConfig(base_channels=8, n_style_blocks=2, n_local_blocks=1, latent_dim=32, style_dim=32)


def _gen(seed: int = 0) -> StyleAwareGenerator:
    torch.manual_seed(seed)
    return StyleAwareGenerator(CFG)


def test_output_shape_and_dtype():
    g = _gen()
    design = torch.rand(2, 1, 64, 64).round()
    out = g(design, g.sample_latent(2))
    assert out.shape == (2, 1, 64, 64)
    assert out.dtype == torch.float32


def test_same_latent_is_deterministic():
    g = _gen()
    design = torch.rand(1, 1, 64, 64).round()
    z = g.sample_latent(1)
    assert torch.equal(g(design, z), g(design, z))


def test_distinct_latents_give_distinct_masks():
    """The generator is a SAMPLER: different z must give different output.

    Regression guard for a real bug -- zero-initialising the AdaIN affine makes
    AdaIN independent of w, collapsing every latent to the same mask.
    """
    g = _gen()
    design = torch.rand(1, 1, 64, 64).round()
    a = g(design, g.sample_latent(1))
    b = g(design, g.sample_latent(1))
    assert (a - b).abs().max() > 1e-6


def test_style_mapping_receives_gradient():
    g = _gen()
    design = torch.rand(1, 1, 64, 64).round()
    g(design, g.sample_latent(1)).sum().backward()
    total = sum(p.grad.abs().sum() for p in g.mapping.parameters() if p.grad is not None)
    assert total > 0


def test_design_conditioning_dominates_style():
    """Content comes from Z, style only perturbs -- the paper's stated intent."""
    g = _gen()
    z = g.sample_latent(1)
    d1 = torch.zeros(1, 1, 64, 64)
    d1[..., 16:48, 16:48] = 1.0
    d2 = torch.zeros(1, 1, 64, 64)
    d2[..., 8:24, 8:24] = 1.0
    design_effect = (g(d1, z) - g(d2, z)).abs().max()
    style_effect = (g(d1, z) - g(d1, g.sample_latent(1))).abs().max()
    assert design_effect > style_effect


def test_adain_applies_commanded_statistics():
    """With gamma/beta forced, AdaIN output must hit those per-channel stats."""
    torch.manual_seed(0)
    norm = AdaIN(channels=4, style_dim=6)
    with torch.no_grad():
        norm.affine.weight.zero_()
        norm.affine.bias[:4] = torch.tensor([2.0, 3.0, 0.5, 1.0])
        norm.affine.bias[4:] = torch.tensor([1.0, -1.0, 0.0, 5.0])
    x = torch.randn(2, 4, 16, 16) * 7 + 3
    out = norm(x, torch.randn(2, 6))
    assert torch.allclose(out.mean(dim=(2, 3)), torch.tensor([[1.0, -1.0, 0.0, 5.0]] * 2), atol=1e-4)
    assert torch.allclose(
        out.std(dim=(2, 3), unbiased=False),
        torch.tensor([[2.0, 3.0, 0.5, 1.0]] * 2),
        atol=1e-3,
    )


def test_style_resblock_is_residual():
    """A residual block with zeroed output convs must be the identity."""
    torch.manual_seed(0)
    blk = StyleResBlock(channels=4, style_dim=6)
    with torch.no_grad():
        blk.conv2.weight.zero_()
        blk.conv2.bias.zero_()
    x = torch.randn(1, 4, 8, 8)
    assert torch.allclose(blk(x, torch.randn(1, 6)), x, atol=1e-6)


def test_discriminator_is_scalar_per_sample():
    d = Discriminator(base_channels=8, levels=2)
    out = d(torch.rand(3, 1, 64, 64), torch.rand(3, 1, 64, 64))
    assert out.shape == (3,)


def test_generator_runs_on_cpu_only():
    """This project must never require CUDA."""
    g = _gen()
    assert all(p.device.type == "cpu" for p in g.parameters())
