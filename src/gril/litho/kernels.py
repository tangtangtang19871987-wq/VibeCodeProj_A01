"""SOCS optical kernel loading for the ICCAD 2013 lithography model.

Provenance: the kernel tensors are a public artifact redistributed by the
Apache-2.0 licensed OpenILT project (see ``docs/source_inventory.md`` S2.2).
They are NOT produced by this repository; ``scripts/fetch_data.sh`` retrieves
them at a pinned commit and verifies ``data/CHECKSUMS.sha256``.

Physics
-------
SOCS decomposes the Hopkins bilinear TCC operator into an incoherent sum of
coherent systems::

    I(x) = sum_k w_k * | (h_k conv M)(x) |^2

``h_k`` are the coherent kernels (eigenvectors of TCC) stored in the *frequency*
domain; ``w_k`` are the eigenvalues, in descending order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

#: Number of SOCS terms in the public ICCAD13 kernel set.
NUM_KERNELS = 24
#: Spatial-frequency support of each stored kernel, in samples (square).
KERNEL_SIZE = 35


@dataclass(frozen=True)
class KernelSet:
    """One focus condition's SOCS kernels plus its adjoint (conjugate) partner.

    Attributes
    ----------
    kernels:
        Complex kernels, shape ``(NUM_KERNELS, 35, 35)``, dtype ``complex64``,
        frequency domain, dimensionless.
    scales:
        Real eigenvalues ``w_k``, shape ``(NUM_KERNELS,)``, dtype ``float32``,
        descending, non-negative, dimensionless.
    ct_kernels:
        Conjugate-transpose kernels used to make the adjoint exact. Same shape
        and dtype as ``kernels``.
    name:
        ``"focus"`` or ``"defocus"``.
    """

    kernels: torch.Tensor
    scales: torch.Tensor
    ct_kernels: torch.Tensor
    name: str

    def to(self, device: torch.device | str) -> "KernelSet":
        """Return a copy of this set on ``device``. Pure; does not mutate."""
        return KernelSet(
            kernels=self.kernels.to(device),
            scales=self.scales.to(device),
            ct_kernels=self.ct_kernels.to(device),
            name=self.name,
        )


def _load_tensor(path: str, device: torch.device | str) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing lithography asset: {path}\n"
            "Run scripts/fetch_data.sh to retrieve the public ICCAD13 kernels."
        )
    return torch.load(path, map_location=device, weights_only=False)


def load_kernel_set(kernel_dir: str, defocus: bool, device: torch.device | str = "cpu") -> KernelSet:
    """Load one focus condition.

    Parameters
    ----------
    kernel_dir:
        Directory holding ``kernels/`` and ``scales/`` (the OpenILT layout).
    defocus:
        ``False`` -> nominal focus, ``True`` -> defocused condition.
    device:
        Torch device for the returned tensors.

    Returns
    -------
    KernelSet

    Notes
    -----
    Stored layout is ``(35, 35, 24)``; it is permuted to ``(24, 35, 35)`` so that
    the kernel index leads, matching the convolution code in :mod:`gril.litho.socs`.
    """
    stem = "defocus" if defocus else "focus"
    kernels = _load_tensor(os.path.join(kernel_dir, "kernels", f"{stem}.pt"), device)
    ct = _load_tensor(os.path.join(kernel_dir, "kernels", f"ct_{stem}.pt"), device)
    scales = _load_tensor(os.path.join(kernel_dir, "scales", f"{stem}.pt"), device)

    kernels = kernels.permute(2, 0, 1).contiguous()
    ct = ct.permute(2, 0, 1).contiguous()

    if kernels.shape != (NUM_KERNELS, KERNEL_SIZE, KERNEL_SIZE):
        raise ValueError(f"Unexpected kernel shape {tuple(kernels.shape)}")
    if scales.shape != (NUM_KERNELS,):
        raise ValueError(f"Unexpected scale shape {tuple(scales.shape)}")

    return KernelSet(kernels=kernels, scales=scales, ct_kernels=ct, name=stem)


def open_field_intensity(ks: KernelSet, num_kernels: int = NUM_KERNELS) -> float:
    """Intensity of an infinite clear field: ``sum_k w_k |h_k(0,0)|^2``.

    This is the SOCS truncation anchor. With the public 24-kernel set it equals
    ~0.95154 rather than 1.0; the 4.85% deficit is the truncation error and is
    reported rather than normalized away.
    """
    centre = KERNEL_SIZE // 2
    dc = ks.kernels[:num_kernels, centre, centre]
    return float((ks.scales[:num_kernels] * dc.abs() ** 2).sum())
