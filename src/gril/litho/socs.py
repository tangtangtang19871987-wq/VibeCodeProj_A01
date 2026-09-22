"""Hopkins/SOCS forward imaging with an exact analytic adjoint.

Conventions (see REPRODUCTION_SPEC.md Sec. 3 -- these are load-bearing)
----------------------------------------------------------------------
* ``torch.fft.fft2(..., norm="forward")``: the 1/N^2 factor sits on the forward
  transform. The inverse carries none.
* **No fftshift.** DC stays at index ``[0, 0]`` and the band-limited 35x35 kernel
  is applied to the four corner blocks of the spectrum, which are the low
  frequencies. An equivalent centred implementation is provided as
  :func:`convolve_centred` purely so the two can be cross-checked in tests.
* **No padding**: the FFT is circular, so the canvas boundary is periodic.
"""

from __future__ import annotations

import torch

from gril.litho.kernels import KernelSet


def _apply_kernel_corners(spectrum: torch.Tensor, kernel: torch.Tensor, num_kernels: int) -> torch.Tensor:
    """Multiply a mask spectrum by band-limited kernels in the FFT corner layout.

    Parameters
    ----------
    spectrum:
        Mask spectrum, shape ``(B, 1, H, W)``, complex.
    kernel:
        Kernels, shape ``(K, kh, kw)``, complex.
    num_kernels:
        How many leading kernels to use (SOCS truncation).

    Returns
    -------
    torch.Tensor
        Shape ``(B, num_kernels, H, W)``, complex, same device/dtype as ``spectrum``.
    """
    kh, kw = kernel.shape[-2:]
    hh, hw = kh // 2, kw // 2
    kernel = kernel.to(spectrum.device)
    b, _, height, width = spectrum.shape
    # The four corner blocks span rows [0, hh] and [height-hh, height-1]. If the
    # canvas is smaller than the kernel these overlap, and the later writes
    # silently clobber the earlier ones -- producing a plausible-looking but
    # physically wrong image. Fail loudly instead.
    if height < kh or width < kw:
        raise ValueError(
            f"Canvas {height}x{width} is smaller than the {kh}x{kw} optical kernel; "
            "the FFT corner blocks would overlap and the aerial image would be "
            "silently wrong. Use a canvas of at least the kernel size (the public "
            "ICCAD13 kernels are 35x35, so >=64 px is recommended)."
        )

    out = torch.zeros(
        (b, num_kernels, height, width), dtype=spectrum.dtype, device=spectrum.device
    )
    k = kernel[None, :num_kernels]
    # Four corners of the spectrum == the low frequencies.
    out[:, :, : hh + 1, : hw + 1] = spectrum[:, :, : hh + 1, : hw + 1] * k[:, :, -(hh + 1) :, -(hw + 1) :]
    out[:, :, : hh + 1, -hw:] = spectrum[:, :, : hh + 1, -hw:] * k[:, :, -(hh + 1) :, :hw]
    out[:, :, -hh:, : hw + 1] = spectrum[:, :, -hh:, : hw + 1] * k[:, :, :hh, -(hw + 1) :]
    out[:, :, -hh:, -hw:] = spectrum[:, :, -hh:, -hw:] * k[:, :, :hh, :hw]
    return out


def convolve(field: torch.Tensor, kernel: torch.Tensor, num_kernels: int) -> torch.Tensor:
    """Band-limited convolution of a complex field with each SOCS kernel.

    Parameters
    ----------
    field:
        Complex mask field, shape ``(B, H, W)``.
    kernel:
        Complex kernels, shape ``(K, kh, kw)``.
    num_kernels:
        SOCS truncation order.

    Returns
    -------
    torch.Tensor
        Complex, shape ``(B, num_kernels, H, W)``.
    """
    spectrum = torch.fft.fft2(field.unsqueeze(1), norm="forward")
    return torch.fft.ifft2(_apply_kernel_corners(spectrum, kernel, num_kernels), norm="forward")


def convolve_centred(field: torch.Tensor, kernel: torch.Tensor, num_kernels: int) -> torch.Tensor:
    """Reference implementation using explicit fftshift and a centred kernel.

    Mathematically identical to :func:`convolve`; kept only as an independent
    cross-check (see ``tests/physics/test_fft_convention.py``). Slower.
    """
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(field.unsqueeze(1), norm="forward"), dim=(-2, -1)
    )
    b, _, height, width = spectrum.shape
    kh, kw = kernel.shape[-2:]
    y0, x0 = height // 2 - kh // 2, width // 2 - kw // 2
    out = torch.zeros((b, num_kernels, height, width), dtype=spectrum.dtype, device=spectrum.device)
    out[:, :, y0 : y0 + kh, x0 : x0 + kw] = (
        spectrum[:, :, y0 : y0 + kh, x0 : x0 + kw] * kernel[None, :num_kernels].to(spectrum.device)
    )
    return torch.fft.ifft2(torch.fft.ifftshift(out, dim=(-2, -1)), norm="forward")


def _complex_of(real_dtype: torch.dtype) -> torch.dtype:
    """Complex dtype matching a real dtype (float32 -> complex64, float64 -> complex128)."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _match(kernel: torch.Tensor, cdtype: torch.dtype) -> torch.Tensor:
    """Cast kernels to the working complex precision (needed for float64 grad checks)."""
    return kernel if kernel.dtype == cdtype else kernel.to(cdtype)


class _SocsIntensity(torch.autograd.Function):
    """``I = sum_k w_k |h_k conv (dose*M)|^2`` with the analytic adjoint.

    The backward pass uses the conjugate-transpose kernel set so the gradient is
    exact rather than a finite-difference or autograd-through-FFT approximation.
    Verified against float64 central differences in ``tests/physics``.
    """

    @staticmethod
    def forward(ctx, mask, dose, kernels, scales, ct_kernels, num_kernels):  # type: ignore[override]
        ctx.save_for_backward(mask, kernels, scales, ct_kernels)
        ctx.dose = dose
        ctx.num_kernels = num_kernels
        cdtype = _complex_of(mask.dtype)
        field = (dose * mask).to(cdtype)
        images = convolve(field, _match(kernels, cdtype), num_kernels)
        w = scales[:num_kernels].to(mask.dtype).view(1, -1, 1, 1)
        return (w * images.abs() ** 2).sum(dim=1)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        mask, kernels, scales, ct_kernels = ctx.saved_tensors
        dose, num_kernels = ctx.dose, ctx.num_kernels
        cdtype = _complex_of(mask.dtype)
        field = (dose * mask).to(cdtype)
        w = scales[:num_kernels].to(mask.dtype).view(1, -1, 1, 1)
        g = grad_out.unsqueeze(1).to(cdtype)

        # d/dM sum_k w_k |h_k * f|^2 = 2 * dose * Re{ sum_k w_k h_k^dagger * ( (h_k * f) . g ) }
        e_fwd = convolve(field, _match(kernels, cdtype), num_kernels)
        term = _conv_stack(e_fwd * g, _match(kernels, cdtype).conj(), num_kernels)
        grad_mask = 2.0 * dose * (w * term).sum(dim=1).real
        return grad_mask, None, None, None, None, None


def _conv_stack(stack: torch.Tensor, kernel: torch.Tensor, num_kernels: int) -> torch.Tensor:
    """Per-kernel convolution of an already-stacked field ``(B, K, H, W)``."""
    spectrum = torch.fft.fft2(stack, norm="forward")
    kh, kw = kernel.shape[-2:]
    hh, hw = kh // 2, kw // 2
    k = kernel[None, :num_kernels].to(spectrum.device)
    out = torch.zeros_like(spectrum)
    out[:, :, : hh + 1, : hw + 1] = spectrum[:, :, : hh + 1, : hw + 1] * k[:, :, -(hh + 1) :, -(hw + 1) :]
    out[:, :, : hh + 1, -hw:] = spectrum[:, :, : hh + 1, -hw:] * k[:, :, -(hh + 1) :, :hw]
    out[:, :, -hh:, : hw + 1] = spectrum[:, :, -hh:, : hw + 1] * k[:, :, :hh, -(hw + 1) :]
    out[:, :, -hh:, -hw:] = spectrum[:, :, -hh:, -hw:] * k[:, :, :hh, :hw]
    return torch.fft.ifft2(out, norm="forward")


def aerial_image(
    mask: torch.Tensor, kernel_set: KernelSet, dose: float = 1.0, num_kernels: int = 24
) -> torch.Tensor:
    """Aerial intensity for a real mask.

    Parameters
    ----------
    mask:
        Real mask in ``[0, 1]``, shape ``(H, W)`` or ``(B, H, W)``, float32/float64.
    kernel_set:
        Loaded :class:`~gril.litho.kernels.KernelSet` for the desired focus condition.
    dose:
        Dimensionless dose multiplier (1.00 nominal, 1.02 max, 0.98 min).
    num_kernels:
        SOCS truncation order.

    Returns
    -------
    torch.Tensor
        Non-negative intensity, same batch shape as ``mask``, normalized so that an
        infinite clear field gives ``open_field_intensity(kernel_set)`` (~0.95154
        for the public 24-kernel set).
    """
    squeeze = mask.dim() == 2
    if squeeze:
        mask = mask.unsqueeze(0)
    out = _SocsIntensity.apply(
        mask, dose, kernel_set.kernels, kernel_set.scales, kernel_set.ct_kernels, num_kernels
    )
    return out.squeeze(0) if squeeze else out
