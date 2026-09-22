"""ICCAD13 scoring metrics: L2, PV Band, and EPE violations.

The EPE routine reproduces the public contest checker's sampling scheme exactly
(``docs/source_inventory.md`` S2.4); it is validated bit-for-bit against that
reference on all ten ICCAD13 cases in ``tests/regression``.

Units: the canvas is 1 nm/pixel, so L2 and PV Band are in nm^2 and the EPE
tolerance is in nm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

#: Reference contest EPE tolerance, nm (S2.4).
EPE_TOLERANCE_NM = 15
#: Spacing between EPE measurement sites along a segment, nm.
EPE_CHECK_INTERVAL = 40
#: Segments shorter than this are measured only at their midpoint, nm.
MIN_EPE_CHECK_LENGTH = 80
#: Offset of the first measurement site from a segment end, nm.
EPE_CHECK_START_INTERVAL = 40


def l2_loss(printed_nom: torch.Tensor, target: torch.Tensor) -> float:
    """Squared error between the printed nominal image and the target, in nm^2.

    Both inputs must already be binary ``{0,1}`` and the same shape.
    """
    return float(((printed_nom - target) ** 2).sum())


def pv_band(printed_max: torch.Tensor, printed_min: torch.Tensor) -> float:
    """Process-variation band area: pixels that differ between the outer corners."""
    return float((printed_max != printed_min).sum())


def _segments(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract vertical and horizontal boundary segments of a binary target.

    Returns
    -------
    (vposes, hposes)
        Each ``(N, 2, 2)`` float tensors holding, per segment, the start and end
        ``(row, col)`` coordinates.
    """
    padded = F.pad(target[None, None], pad=(1, 1, 1, 1))[0, 0]
    upper, lower = padded[2:, 1:-1] == 1, padded[:-2, 1:-1] == 1
    left, right = padded[1:-1, :-2] == 1, padded[1:-1, 2:] == 1
    ul, ur = padded[2:, :-2] == 1, padded[2:, 2:] == 1
    ll, lr = padded[:-2, :-2] == 1, padded[:-2, 2:] == 1

    boundary = target == 1
    boundary[upper & lower & left & right & ul & ur & ll & lr] = False

    padded = F.pad(boundary[None, None].float(), pad=(1, 1, 1, 1))[0, 0]
    upper, lower = padded[2:, 1:-1] == 1, padded[:-2, 1:-1] == 1
    left, right = padded[1:-1, :-2] == 1, padded[1:-1, 2:] == 1
    center = padded[1:-1, 1:-1] == 1

    def runs(mask2d: torch.Tensor, axis: int) -> torch.Tensor:
        sites = mask2d.nonzero()
        if sites.numel() == 0:
            return torch.zeros((0, 2, 2))
        a, b = (0, 1) if axis == 0 else (1, 0)
        order = np.lexsort(
            (sites[:, a].cpu().numpy(), sites[:, b].cpu().numpy())
        )
        sites = sites[order]
        coord = sites[:, a]
        start = torch.cat((torch.tensor([True]), coord[1:] != coord[:-1] + 1))
        end = torch.cat((coord[1:] != coord[:-1] + 1, torch.tensor([True])))
        return torch.stack(
            (sites[start.nonzero()[:, 0]], sites[end.nonzero()[:, 0]]), dim=2
        ).float()

    vertical = center.clone()
    vertical[left & right] = False
    horizontal = center.clone()
    horizontal[upper & lower] = False
    return runs(vertical, 0), runs(horizontal, 1)


def _check_sites(
    image: torch.Tensor, sample: torch.Tensor, target: torch.Tensor, axis: str, tol: int
) -> tuple[int, int]:
    """Count inner and outer EPE violations at the given measurement sites."""
    r0, c0 = sample[0, 0].long(), sample[0, 1].long()
    if axis == "v":
        pos, neg = target[r0, c0 + 1] == 1, target[r0, c0 - 1] == 1
        step = torch.tensor([0, tol], dtype=sample.dtype)
    else:
        pos, neg = target[r0 + 1, c0] == 1, target[r0 - 1, c0] == 1
        step = torch.tensor([tol, 0], dtype=sample.dtype)

    if pos and not neg:
        inner_off, outer_off = step, -step
    elif neg and not pos:
        inner_off, outer_off = -step, step
    else:
        return 0, 0

    inner_pt = (sample + inner_off).long()
    outer_pt = (sample + outer_off).long()
    inner = int((image[inner_pt[:, 0], inner_pt[:, 1]] == 0).sum())
    outer = int((image[outer_pt[:, 0], outer_pt[:, 1]] == 1).sum())
    return inner, outer


def epe_violations(
    printed_nom: torch.Tensor, target: torch.Tensor, tolerance: int = EPE_TOLERANCE_NM
) -> tuple[int, int]:
    """Count (inner, outer) edge-placement-error violations.

    Parameters
    ----------
    printed_nom:
        Binary ``{0,1}`` printed image at the nominal corner, ``(H, W)``.
    target:
        Binary ``{0,1}`` design target, ``(H, W)``.
    tolerance:
        EPE tolerance in nm. The contest uses 15; the paper reports 3 (G-016).

    Returns
    -------
    (inner, outer)
        Counts of sites pulled in / pushed out beyond ``tolerance``.
    """
    vposes, hposes = _segments(target)
    inner = outer = 0
    for poses, axis, coord in ((vposes, "v", 0), (hposes, "h", 1)):
        for idx in range(poses.shape[0]):
            seg = poses[idx]
            lo, hi = seg[coord, 0], seg[coord, 1]
            centre = ((seg[:, 0] + seg[:, 1]) / 2).int().float().unsqueeze(0)
            if (hi - lo) <= MIN_EPE_CHECK_LENGTH:
                sample = centre
            else:
                mid = centre[0, coord]
                vals = torch.cat(
                    (
                        torch.arange(lo + EPE_CHECK_START_INTERVAL, mid + 1, EPE_CHECK_INTERVAL),
                        torch.arange(hi - EPE_CHECK_START_INTERVAL, mid, -EPE_CHECK_INTERVAL),
                    )
                ).unique()
                sample = seg[:, 0].repeat(vals.shape[0], 1)
                sample[:, coord] = vals
            i, o = _check_sites(printed_nom, sample, target, axis, tolerance)
            inner += i
            outer += o
    return inner, outer


@dataclass
class Scores:
    """Full ICCAD13 score card for one mask."""

    l2: float
    pvb: float
    epe_in: int
    epe_out: int
    tolerance: int

    @property
    def epe(self) -> int:
        """Total EPE violations."""
        return self.epe_in + self.epe_out


def evaluate(
    mask: torch.Tensor, target: torch.Tensor, litho, tolerance: int = EPE_TOLERANCE_NM
) -> Scores:
    """Score a binary mask against a target using the three-corner litho model."""
    with torch.no_grad():
        binary = (mask >= 0.5).to(target.dtype)
        b_nom, b_max, b_min = litho.binary(binary)
        ein, eout = epe_violations(b_nom, target, tolerance)
        return Scores(
            l2=l2_loss(b_nom, target),
            pvb=pv_band(b_max, b_min),
            epe_in=ein,
            epe_out=eout,
            tolerance=tolerance,
        )
