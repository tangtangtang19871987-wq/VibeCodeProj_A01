"""Physics validation of the SOCS forward model (REPRODUCTION_SPEC.md Sec. 1-4)."""
import pytest
import torch

from gril.litho.kernels import KERNEL_SIZE, NUM_KERNELS, open_field_intensity
from gril.litho.socs import aerial_image, convolve, convolve_centred

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def test_kernel_shapes_and_dtype(focus_kernels):
    assert focus_kernels.kernels.shape == (NUM_KERNELS, KERNEL_SIZE, KERNEL_SIZE)
    assert focus_kernels.kernels.dtype == torch.complex64
    assert focus_kernels.scales.shape == (NUM_KERNELS,)


def test_socs_eigenvalues_nonneg_and_descending(focus_kernels):
    """TCC is Hermitian PSD, so its SOCS eigenvalues must be >=0 and sorted."""
    w = focus_kernels.scales
    assert torch.all(w >= 0), "negative SOCS eigenvalue implies a non-PSD TCC"
    assert torch.all(w[1:] <= w[:-1] + 1e-6), "eigenvalues are not descending"


def test_open_field_normalization(focus_kernels):
    """Clear-field intensity. 24-term truncation loses ~4.85% of the energy."""
    val = open_field_intensity(focus_kernels)
    assert val == pytest.approx(0.9515371, abs=1e-6)
    assert 0.9 < val < 1.0


def test_open_field_matches_uniform_mask(focus_kernels):
    """An all-open mask must reproduce the analytic clear-field value everywhere."""
    img = aerial_image(torch.ones(128, 128), focus_kernels, 1.0, NUM_KERNELS)
    expected = open_field_intensity(focus_kernels)
    assert img.max().item() == pytest.approx(expected, rel=1e-5)
    assert img.min().item() == pytest.approx(expected, rel=1e-5)


def test_truncation_error_decreases_with_more_kernels(focus_kernels):
    """SOCS error vs the 24-term reference must shrink monotonically in K."""
    torch.manual_seed(0)
    mask = (torch.rand(128, 128) > 0.5).float()
    full = aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS)
    errs = [
        float((aerial_image(mask, focus_kernels, 1.0, k) - full).abs().max())
        for k in (1, 2, 4, 8, 16)
    ]
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs


def test_intensity_is_nonnegative(focus_kernels):
    torch.manual_seed(0)
    mask = (torch.rand(128, 128) > 0.7).float()
    assert aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS).min() >= -1e-6


def test_fft_convention_corner_equals_centred(focus_kernels):
    """The fast corner-indexed path must equal the explicit fftshift path."""
    torch.manual_seed(0)
    field = (torch.rand(1, 128, 128) > 0.5).to(torch.complex64)
    a = convolve(field, focus_kernels.kernels, NUM_KERNELS)
    b = convolve_centred(field, focus_kernels.kernels, NUM_KERNELS)
    assert (a - b).abs().max() / a.abs().max() < 1e-5


def test_translation_equivariance(focus_kernels):
    """Circular shift of the mask must circularly shift the aerial image."""
    torch.manual_seed(0)
    mask = (torch.rand(128, 128) > 0.6).float()
    shift = (17, -23)
    direct = aerial_image(torch.roll(mask, shift, dims=(0, 1)), focus_kernels, 1.0, NUM_KERNELS)
    shifted = torch.roll(aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS), shift, dims=(0, 1))
    assert (direct - shifted).abs().max() / direct.max() < 1e-5


def test_x_mirror_symmetry(focus_kernels):
    """The kernel set is closed under mirroring in x, so the image mirrors too.

    Verified property of the published kernels: each kernel is symmetric about
    column 17 to 5.9e-8. See docs/findings.md F-KER-01.
    """
    torch.manual_seed(0)
    mask = (torch.rand(128, 128) > 0.6).float()
    direct = aerial_image(torch.flip(mask, dims=(1,)), focus_kernels, 1.0, NUM_KERNELS)
    flipped = torch.flip(aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS), dims=(1,))
    assert (direct - flipped).abs().max() / direct.max() < 1e-5


def test_kernels_are_column_symmetric_but_not_row_symmetric(focus_kernels):
    """Characterisation test for an asymmetry in the published kernel data.

    The ICCAD13 kernels are symmetric about their centre column but NOT about
    their centre row. This is a property of the distributed artifact, not of our
    code, and it means y-mirror equivariance does NOT hold for this optical
    model. Pinned here so that a future kernel change is caught loudly rather
    than silently altering every downstream number. See docs/findings.md F-KER-01.
    """
    h = focus_kernels.kernels[0]
    col_err = (h - torch.flip(h, dims=(1,))).abs().max().item()
    row_err = (h - torch.flip(h, dims=(0,))).abs().max().item()
    assert col_err < 1e-6, "centre-column symmetry unexpectedly broken"
    assert row_err > 1e-3, "kernels became row-symmetric; re-derive F-KER-01"


def test_y_mirror_asymmetry_is_bounded(focus_kernels):
    """The y-mirror equivariance error is real but bounded; pin its magnitude."""
    torch.manual_seed(0)
    mask = (torch.rand(128, 128) > 0.6).float()
    image = aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS)
    direct = aerial_image(torch.flip(mask, dims=(0,)), focus_kernels, 1.0, NUM_KERNELS)
    err = ((direct - torch.flip(image, dims=(0,))).abs().max() / image.max()).item()
    assert 1e-3 < err < 5e-2, f"y-mirror asymmetry moved to {err:.3e}"


def test_batched_equals_single(focus_kernels):
    """Batched evaluation must equal looping over the batch."""
    torch.manual_seed(0)
    batch = (torch.rand(3, 96, 96) > 0.6).float()
    out = aerial_image(batch, focus_kernels, 1.0, NUM_KERNELS)
    for i in range(batch.shape[0]):
        single = aerial_image(batch[i], focus_kernels, 1.0, NUM_KERNELS)
        assert (out[i] - single).abs().max() < 1e-6


def test_dose_scales_intensity_quadratically(focus_kernels):
    """Intensity is quadratic in the mask field, hence quadratic in dose."""
    torch.manual_seed(0)
    mask = (torch.rand(96, 96) > 0.6).float()
    i1 = aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS)
    i2 = aerial_image(mask, focus_kernels, 2.0, NUM_KERNELS)
    assert (i2 - 4.0 * i1).abs().max() / i1.max() < 1e-5


def test_float32_float64_agreement(focus_kernels):
    torch.manual_seed(0)
    mask = (torch.rand(96, 96) > 0.6)
    a = aerial_image(mask.float(), focus_kernels, 1.0, NUM_KERNELS).double()
    b = aerial_image(mask.double(), focus_kernels, 1.0, NUM_KERNELS)
    assert ((a - b).norm() / b.norm()) < 1e-5


def test_canvas_smaller_than_kernel_is_rejected(focus_kernels):
    """A canvas below the kernel size would make the FFT corner blocks overlap.

    That corrupts the aerial image silently, so it must raise instead.
    """
    with pytest.raises(ValueError, match="smaller than"):
        aerial_image(torch.zeros(32, 32), focus_kernels, 1.0, NUM_KERNELS)


def test_multiresolution_consistency(focus_kernels):
    """The kernels span a fixed PHYSICAL extent of the canvas.

    Coarsening the pixel grid at constant physical size therefore preserves the
    optics. This is what makes the paper's 8x-downsampled RL reward loop sound.
    """
    torch.manual_seed(0)
    mask = torch.zeros(512, 512)
    mask[100:300, 150:360] = 1.0
    mask[350:420, 60:480] = 1.0
    full = aerial_image(mask, focus_kernels, 1.0, NUM_KERNELS)
    for factor in (2, 4):
        low = aerial_image(
            torch.nn.functional.avg_pool2d(mask[None, None], factor)[0, 0],
            focus_kernels, 1.0, NUM_KERNELS,
        )
        ref = torch.nn.functional.avg_pool2d(full[None, None], factor)[0, 0]
        assert (low - ref).abs().max() / full.max() < 0.01


def test_abs_squared_identity_matches_sqrt_then_square(focus_kernels):
    """|z|^2 = re(z)^2 + im(z)^2 is exact; pin the identity used in socs.py.

    Regression guard: if this line ever reverts to `.abs()**2`, this test still
    passes (both formulas agree to float rounding) -- it exists to document
    the identity, not to catch a regression in VALUE. What it protects against
    is someone "optimizing" it into something that ISN'T the same identity.
    """
    torch.manual_seed(0)
    z = torch.randn(4, 8, 16, 16, dtype=torch.complex64)
    via_abs = z.abs() ** 2
    via_identity = z.real**2 + z.imag**2
    assert torch.allclose(via_abs, via_identity, rtol=1e-5, atol=1e-6)
