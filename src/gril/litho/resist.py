"""Constant-threshold resist model and process corners (ICCAD13 settings)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gril.litho.kernels import KernelSet, load_kernel_set
from gril.litho.socs import aerial_image


@dataclass(frozen=True)
class ProcessConfig:
    """ICCAD13 process settings. Defaults are the public contest values (S2.5)."""

    target_density: float = 0.225   # I_th, the resist threshold intensity
    print_thresh: float = 0.5       # threshold on the sigmoid resist output
    print_steepness: float = 50.0   # beta_r
    dose_nom: float = 1.00
    dose_max: float = 1.02
    dose_min: float = 0.98
    num_kernels: int = 24


def resist(intensity: torch.Tensor, cfg: ProcessConfig) -> torch.Tensor:
    """Sigmoid resist: ``Z = sigmoid(beta_r * (I - I_th))``.

    ``Z >= print_thresh`` is exactly equivalent to ``I >= target_density`` when
    ``print_thresh == 0.5``, which is the ICCAD13 setting.
    """
    return torch.sigmoid(cfg.print_steepness * (intensity - cfg.target_density))


class LithoModel:
    """Three-corner lithography forward model.

    Corners (S2.5): nominal = dose 1.00 at focus, max = dose 1.02 at focus,
    min = dose 0.98 defocused.
    """

    def __init__(self, kernel_dir: str, cfg: ProcessConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg or ProcessConfig()
        self.device = torch.device(device)
        self.focus: KernelSet = load_kernel_set(kernel_dir, defocus=False, device=self.device)
        self.defocus: KernelSet = load_kernel_set(kernel_dir, defocus=True, device=self.device)

    def aerial(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(I_nom, I_max, I_min)`` aerial intensities."""
        c, n = self.cfg, self.cfg.num_kernels
        return (
            aerial_image(mask, self.focus, c.dose_nom, n),
            aerial_image(mask, self.focus, c.dose_max, n),
            aerial_image(mask, self.defocus, c.dose_min, n),
        )

    def __call__(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(Z_nom, Z_max, Z_min)`` resist images in ``(0, 1)``."""
        return tuple(resist(i, self.cfg) for i in self.aerial(mask))  # type: ignore[return-value]

    def binary(self, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the three hard-thresholded printed images as float ``{0,1}``."""
        return tuple((z >= self.cfg.print_thresh).to(mask.dtype) for z in self(mask))  # type: ignore[return-value]
