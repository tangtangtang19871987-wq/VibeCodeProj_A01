"""Keep-out (bridge) and keep-in (pinch) zones: the generic, geometry-only checker.

keep_out: pixels within s_min/2 of the mid-line between two *different*
          target polygons (and not inside any target).  Any print here at any
          corner => the printed gap between those polygons is < s_min (bridge).
          Covers lateral, tip-to-tip and diagonal (line-end to side) bridges.
keep_in : the central strip (width w_min) of every line, kept `end_keep` away
          from the line end, plus a thin centre-line spine that reaches to
          (end - pb_max).  Any unprinted pixel here at any corner => the line
          is narrower than w_min or the end pulled back more than pb_max.
"""
from __future__ import annotations

import numpy as np

from layout import Clip, y_in_seg


def _coords(n, L):
    c = (np.arange(n) + 0.5) * L / n
    return np.meshgrid(c, c, indexing="xy")


def segment_distance(clip: Clip, s, X, Y):
    """Euclidean distance (periodic) from every pixel centre to a segment rectangle."""
    L = clip.L
    xc = clip.xc(s.track)
    dx = np.abs(((X - xc + L / 2) % L) - L / 2) - s.width / 2
    yc = s.y0 + s.length / 2
    dy = np.abs(((Y - yc + L / 2) % L) - L / 2) - s.length / 2
    return np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))


def make_zones(clip: Clip, n: int, s_min=28.0, w_min=28.0, pb_max=15.0,
               end_keep=30.0, spine=6.0, d_max=60.0):
    X, Y = _coords(n, clip.L)
    D = np.stack([segment_distance(clip, s, X, Y) for s in clip.segs])  # [S,n,n]
    # merge segments that touch (same polygon) -- none in our clips, keep simple
    order = np.sort(D, axis=0)
    d1, d2 = order[0], order[1]
    keep_out = (d1 > 0) & (np.abs(d2 - d1) <= s_min) & (d1 <= d_max)
    keep_in = np.zeros((n, n), bool)
    L = clip.L
    for s in clip.segs:
        xc = clip.xc(s.track)
        ax = np.abs(((X - xc + L / 2) % L) - L / 2)
        t = (Y - s.y0) % L                       # position along the segment
        if s.length >= L - 1e-6:                  # full-length line: no ends
            keep_in |= ax <= min(w_min, s.width) / 2
            continue
        core = (ax <= min(w_min, s.width) / 2) & (t >= end_keep) & (t <= s.length - end_keep)
        spn = (ax <= spine / 2) & (t >= pb_max) & (t <= s.length - pb_max)
        keep_in |= core | spn
    return keep_out, keep_in
