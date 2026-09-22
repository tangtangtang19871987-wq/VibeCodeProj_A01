"""Adjoint / gradient validation (REPRODUCTION_SPEC.md Sec. 5).

Context: the public reference implementation this project validates against
(OpenILT ``pylitho/exact.py``) computes its backward pass with the shipped
``ct_*`` kernel tensors. Those are NOT the adjoint of its own forward operator,
and the resulting gradient is ~8.7% off. See ``docs/findings.md`` (F-ADJ-01).
This repository instead uses ``conj(H)``, which is the exact adjoint of
``IFFT . diag(H) . FFT``; these tests pin that down.
"""
import pytest
import torch

from gril.litho.kernels import NUM_KERNELS
from gril.litho.socs import aerial_image


def _pure_autograd_forward(mask, kernel_set, dose, num_kernels=NUM_KERNELS):
    """Forward built from plain differentiable ops; autograd differentiates it exactly."""
    n, hh = num_kernels, kernel_set.kernels.shape[-1] // 2
    height, width = mask.shape[-2], mask.shape[-1]
    field = (dose * mask).to(torch.complex128)[None, None]
    spec = torch.fft.fft2(field, norm="forward")
    k = kernel_set.kernels[:n].to(torch.complex128)[None]

    def place(block, r0, r1, c0, c1):
        t = torch.zeros((1, n, height, width), dtype=torch.complex128)
        t[:, :, r0:r1, c0:c1] = block
        return t

    acc = (
        place(spec[:, :, : hh + 1, : hh + 1] * k[:, :, -(hh + 1):, -(hh + 1):], 0, hh + 1, 0, hh + 1)
        + place(spec[:, :, : hh + 1, -hh:] * k[:, :, -(hh + 1):, :hh], 0, hh + 1, width - hh, width)
        + place(spec[:, :, -hh:, : hh + 1] * k[:, :, :hh, -(hh + 1):], height - hh, height, 0, hh + 1)
        + place(spec[:, :, -hh:, -hh:] * k[:, :, :hh, :hh], height - hh, height, width - hh, width)
    )
    img = torch.fft.ifft2(acc, norm="forward")
    w = kernel_set.scales[:n].double().view(1, -1, 1, 1)
    return (w * img.abs() ** 2).sum(1)[0]


@pytest.mark.parametrize("dose", [0.98, 1.00, 1.02])
def test_forward_matches_pure_autograd(focus_kernels, dose):
    torch.manual_seed(0)
    mask = torch.rand(48, 48, dtype=torch.float64)
    a = aerial_image(mask, focus_kernels, dose, NUM_KERNELS)
    b = _pure_autograd_forward(mask, focus_kernels, dose)
    assert (a - b).abs().max() < 1e-14


@pytest.mark.parametrize("dose", [0.98, 1.00, 1.02])
def test_analytic_adjoint_is_exact(focus_kernels, dose):
    """Analytic adjoint must equal exact autograd to machine precision."""
    torch.manual_seed(0)
    mask = torch.rand(48, 48, dtype=torch.float64)
    weight = torch.randn(48, 48, dtype=torch.float64)

    a = mask.clone().requires_grad_(True)
    (aerial_image(a, focus_kernels, dose, NUM_KERNELS) * weight).sum().backward()
    b = mask.clone().requires_grad_(True)
    (_pure_autograd_forward(b, focus_kernels, dose) * weight).sum().backward()

    assert (a.grad - b.grad).norm() / b.grad.norm() < 1e-12


def test_finite_difference_directional_derivative(focus_kernels):
    """Central finite differences in float64 along random directions."""
    torch.manual_seed(0)
    mask = torch.rand(48, 48, dtype=torch.float64)
    weight = torch.randn(48, 48, dtype=torch.float64)
    a = mask.clone().requires_grad_(True)
    (aerial_image(a, focus_kernels, 1.0, NUM_KERNELS) * weight).sum().backward()

    eps = 1e-6
    for seed in range(4):
        torch.manual_seed(100 + seed)
        v = torch.randn(48, 48, dtype=torch.float64)
        v /= v.norm()
        with torch.no_grad():
            lp = (aerial_image(mask + eps * v, focus_kernels, 1.0, NUM_KERNELS) * weight).sum()
            lm = (aerial_image(mask - eps * v, focus_kernels, 1.0, NUM_KERNELS) * weight).sum()
        fd = ((lp - lm) / (2 * eps)).item()
        analytic = (a.grad * v).sum().item()
        # Tolerance is set by finite-difference cancellation, not by the adjoint.
        assert abs(fd - analytic) <= 1e-4 * max(abs(fd), 1e-12) + 1e-9


def test_gradient_batched_equals_single(focus_kernels):
    torch.manual_seed(0)
    batch = torch.rand(2, 48, 48, dtype=torch.float64)
    weight = torch.randn(2, 48, 48, dtype=torch.float64)
    b = batch.clone().requires_grad_(True)
    (aerial_image(b, focus_kernels, 1.0, NUM_KERNELS) * weight).sum().backward()
    for i in range(2):
        s = batch[i].clone().requires_grad_(True)
        (aerial_image(s, focus_kernels, 1.0, NUM_KERNELS) * weight[i]).sum().backward()
        assert (b.grad[i] - s.grad).abs().max() < 1e-12
