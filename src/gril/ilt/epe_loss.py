"""Differentiable EPE-site-aware loss, for use inside the ILT objective.

The EPE metric itself (``gril.metrics.core.epe_violations``) is fundamentally
non-differentiable: it operates on an already-binarized printed image, with
integer pixel indexing and boolean comparisons. This module does NOT
differentiate through that function. It instead builds a **smooth proxy**
that measures the *same physical quantity, at the same measurement sites*, but
evaluated on the continuous resist field ``Z`` (before thresholding) with a
hinge/margin loss instead of a hard 0/1 violation count.

Why this exists (G-015): the ILT objective currently minimizes global L2
between the resist image and the target, which is not the same thing as
minimizing EPE -- L2 treats every pixel equally, while EPE only cares about a
sparse set of measurement sites along the target boundary, sampled exactly as
the contest checker samples them (interval 40nm, min segment 80nm, start
offset 40nm -- see ``metrics.core``). A large fraction of L2 error can live
far from any EPE measurement site (e.g. in a resist region that is
oversized/undersized in the interior of a wide feature, which no EPE site
would ever probe) and be irrelevant to the paper's headline metric, while a
site-local deviation that most affects EPE might be a vanishingly small
contribution to global L2. Weighting the loss toward the actual measurement
geometry targets what is actually being compared to the paper's numbers.

This module deliberately does NOT modify or import-and-monkeypatch
``epe_violations()`` or ``_check_sites()`` -- both are bit-exact-verified
against the reference contest checker on all 10 ICCAD13 cases, and are left
completely untouched. The site-extraction geometry here is a parallel,
independently-written implementation of the same sampling scheme, cross-
checked against the metric's own site count in tests (not against its
internal exact bit sequence, since one produces a *shape*, the other a
*binarization test*).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gril.metrics.core import (
    EPE_CHECK_INTERVAL,
    EPE_CHECK_START_INTERVAL,
    MIN_EPE_CHECK_LENGTH,
    _segments,
)


@dataclass
class EPESites:
    """Precomputed measurement geometry for one target. Reused across every
    ILT iteration -- the target does not change during optimization, so this
    is computed once by ``solve()`` rather than once per iteration.

    All fields are ``(N,)`` LongTensors of flat pixel indices into a
    ``target.flatten()``-shaped tensor, ready for direct indexing.
    """

    inner_rows: torch.Tensor
    inner_cols: torch.Tensor
    outer_rows: torch.Tensor
    outer_cols: torch.Tensor

    @property
    def n_sites(self) -> int:
        return int(self.inner_rows.shape[0])


def _site_positions(target: torch.Tensor, tol_px: int, interval_px: int, min_len_px: float, start_px: int):
    """Yield (sample_rc, axis, direction_sign) for every measurement site.

    Mirrors the sampling geometry in ``metrics.core.epe_violations`` exactly
    (same interval/min-length/start-offset logic), but does not touch a
    printed mask at all -- it only needs the target, so it can run once,
    outside the optimization loop.

    ``direction_sign`` is ``+1`` if the target is "inside" at larger index
    along the probe axis (i.e. the outward normal points toward smaller
    index), ``-1`` for the opposite orientation, or ``None`` for a site where
    neither side is unambiguously inside (skipped, exactly as
    ``_check_sites`` skips it by returning ``(0, 0)``).
    """
    vposes, hposes = _segments(target)
    for poses, axis, coord in ((vposes, "v", 0), (hposes, "h", 1)):
        for idx in range(poses.shape[0]):
            seg = poses[idx]
            lo, hi = seg[coord, 0], seg[coord, 1]
            centre = ((seg[:, 0] + seg[:, 1]) / 2).int().float().unsqueeze(0)
            if (hi - lo) <= min_len_px:
                samples = centre
            else:
                mid = centre[0, coord]
                vals = torch.cat(
                    (
                        torch.arange(lo + start_px, mid + 1, interval_px),
                        torch.arange(hi - start_px, mid, -interval_px),
                    )
                ).unique()
                samples = seg[:, 0].repeat(vals.shape[0], 1)
                samples[:, coord] = vals

            h, w = target.shape[-2], target.shape[-1]
            for s in range(samples.shape[0]):
                r0, c0 = int(samples[s, 0]), int(samples[s, 1])
                # Bounds-check BEFORE indexing the neighbour used for the
                # direction test. A target whose boundary sits within 1 pixel
                # of the canvas edge (e.g. an all-1 target, where _segments'
                # zero-padding makes every edge pixel look like a boundary)
                # would otherwise index out of range here. NONE of this
                # project's real targets are ever this close to the edge
                # (verified: the largest ICCAD13 bbox is 828x640 nm inside a
                # 2048 nm canvas), so this never fires on real data -- it is a
                # defensive guard for a pathological input, not a fix that
                # changes any real result. See docs/findings.md F-EDGE-01: the
                # SAME out-of-bounds access exists, unguarded, in the already
                # bit-exact-verified epe_violations()/_check_sites(), which
                # this module deliberately does not modify.
                if axis == "v":
                    if not (0 <= c0 - 1 and c0 + 1 < w):
                        continue
                    pos = bool(target[r0, c0 + 1] == 1)
                    neg = bool(target[r0, c0 - 1] == 1)
                else:
                    if not (0 <= r0 - 1 and r0 + 1 < h):
                        continue
                    pos = bool(target[r0 + 1, c0] == 1)
                    neg = bool(target[r0 - 1, c0] == 1)
                if pos == neg:
                    continue  # ambiguous site, skipped -- matches _check_sites
                sign = 1 if (pos and not neg) else -1
                yield (r0, c0), axis, sign


def build_epe_sites(
    target: torch.Tensor, tolerance: float, pixel_nm: float = 1.0
) -> EPESites:
    """Precompute the inner/outer probe coordinates for every EPE site.

    Parameters
    ----------
    target:
        Binary ``{0,1}`` design, ``(H, W)``.
    tolerance:
        EPE tolerance in nm (matches ``epe_violations``'s convention).
    pixel_nm:
        Canvas pixel pitch in nm.

    Returns
    -------
    EPESites
        Empty (``n_sites == 0``) if the target has no interior boundary
        segments (e.g. an all-zero or all-one canvas).
    """
    if pixel_nm <= 0:
        raise ValueError(f"pixel_nm must be positive, got {pixel_nm}")
    tol_px = max(1, int(round(tolerance / pixel_nm)))
    interval_px = max(1, int(round(EPE_CHECK_INTERVAL / pixel_nm)))
    min_len_px = MIN_EPE_CHECK_LENGTH / pixel_nm
    start_px = max(1, int(round(EPE_CHECK_START_INTERVAL / pixel_nm)))

    h, w = target.shape[-2], target.shape[-1]
    inner_rc: list[tuple[int, int]] = []
    outer_rc: list[tuple[int, int]] = []
    for (r0, c0), axis, sign in _site_positions(target, tol_px, interval_px, min_len_px, start_px):
        d = tol_px * sign
        if axis == "v":
            inner = (r0, c0 + d)
            outer = (r0, c0 - d)
        else:
            inner = (r0 + d, c0)
            outer = (r0 - d, c0)
        # Clip to canvas bounds -- a probe point can fall outside the canvas
        # near the border for a large tolerance; skip such sites rather than
        # index out of range (epe_violations has the same implicit
        # requirement: its callers never place targets within `tolerance` of
        # the canvas edge, verified true for all 10 ICCAD13 cases in
        # tests_gril/regression).
        if not (0 <= inner[0] < h and 0 <= inner[1] < w and 0 <= outer[0] < h and 0 <= outer[1] < w):
            continue
        inner_rc.append(inner)
        outer_rc.append(outer)

    if not inner_rc:
        z = torch.zeros(0, dtype=torch.long)
        return EPESites(inner_rows=z, inner_cols=z.clone(), outer_rows=z.clone(), outer_cols=z.clone())

    inner_t = torch.tensor(inner_rc, dtype=torch.long)
    outer_t = torch.tensor(outer_rc, dtype=torch.long)
    return EPESites(
        inner_rows=inner_t[:, 0], inner_cols=inner_t[:, 1],
        outer_rows=outer_t[:, 0], outer_cols=outer_t[:, 1],
    )


def soft_epe_loss(z_nom: torch.Tensor, sites: EPESites, margin: float = 0.0) -> torch.Tensor:
    """Hinge loss on the continuous resist field at the EPE measurement sites.

    ``z_nom`` should be printed (``>= 0.5 + margin``) at every ``inner`` probe
    point and not printed (``<= 0.5 - margin``) at every ``outer`` probe
    point -- the same physical requirement ``epe_violations`` checks with a
    hard threshold, made smooth so it is differentiable w.r.t. the mask.

    Parameters
    ----------
    z_nom:
        Continuous nominal resist field, ``(H, W)``, values in ``(0, 1)``.
        Must NOT be batched -- call once per example (mirrors the rest of the
        per-example EPE machinery).
    sites:
        Precomputed by ``build_epe_sites`` from the (fixed) target -- pass the
        SAME instance on every call within one ``solve()`` run rather than
        rebuilding it, since it is independent of the mask being optimized.
    margin:
        Extra margin pushed past the 0.5 threshold before the hinge goes to
        zero. 0.0 (default) is a bare hinge at the exact decision boundary.

    Returns
    -------
    torch.Tensor
        Scalar. Zero sites (an all-uniform target, or none of the segments
        survived the canvas-bounds clip) returns exactly ``0.0`` with a
        correct (empty-sum) gradient, not a division or NaN.
    """
    if sites.n_sites == 0:
        return z_nom.sum() * 0.0  # zero, but keeps z_nom in the autograd graph
    z_inner = z_nom[sites.inner_rows, sites.inner_cols]
    z_outer = z_nom[sites.outer_rows, sites.outer_cols]
    loss_inner = torch.relu((0.5 + margin) - z_inner)
    loss_outer = torch.relu(z_outer - (0.5 - margin))
    return loss_inner.sum() + loss_outer.sum()
