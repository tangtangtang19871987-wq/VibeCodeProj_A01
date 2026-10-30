"""Test clips: pitch-74.5 unidirectional metal, 8 tracks, periodic 596 nm square."""
from __future__ import annotations

import numpy as np

from layout import Clip, Seg

PITCH, TRACKS = 74.5, 8
L = PITCH * TRACKS


def random_clip(seed=1, widths=(30.0, 37.0, 44.0), wprob=(0.25, 0.5, 0.25),
                gaps=(50.0, 60.0, 80.0, 140.0), p_empty_run=0.25, name=None,
                min_len=130.0):
    """Random unidirectional metal.  Some tracks get long empty runs (room for
    SRAF); line widths vary so neighbours create narrow spaces / thin lines."""
    rng = np.random.default_rng(seed)
    segs = []
    for k in range(TRACKS):
        nseg = int(rng.integers(1, 4))
        g = rng.choice(gaps, size=nseg)
        if rng.random() < p_empty_run:
            g[rng.integers(nseg)] = rng.uniform(180, 260)
        while nseg > 1 and L - g.sum() < nseg * min_len:
            nseg -= 1
            g = g[:nseg]
        free = L - g.sum()
        # every segment at least min_len long, the rest split at random
        lens = min_len + rng.dirichlet(np.ones(nseg)) * max(free - nseg * min_len, 0)
        y = rng.uniform(0, L)
        for j in range(nseg):
            w = float(rng.choice(widths, p=wprob))
            segs.append(Seg(k, y % L, float(lens[j]), w))
            y += lens[j] + g[j]
    return Clip(PITCH, TRACKS, L, segs, name or f"rand{seed}")


def dense_clip(width=37.0):
    return Clip(PITCH, TRACKS, L, [Seg(k, 0.0, L, width) for k in range(TRACKS)], "dense")


def conflict_clip(t2t_gap=50.0, thin=30.0, wide=44.0):
    """Hand-built clip with known hard spots:
       * a wide/thin neighbour pair (narrow space next to a thin line),
       * a short tip-to-tip gap,
       * an isolated line with an empty track next to it (SRAF room)."""
    s = []
    # tracks 0..2: wide - thin - wide sandwich (thin line is pinch-prone, the
    # two spaces are bridge-prone)
    s += [Seg(0, 0.0, L, wide), Seg(1, 0.0, L, thin), Seg(2, 0.0, L, wide)]
    # track 3: two lines with a tight tip-to-tip gap and a normal one
    a = 250.0
    s += [Seg(3, 30.0, a, 37.0), Seg(3, 30.0 + a + t2t_gap, L - a - t2t_gap - 90.0, 37.0)]
    # track 4: regular line
    s += [Seg(4, 0.0, L, 37.0)]
    # track 5: empty except a short piece (SRAF room next to track 4 and 6)
    s += [Seg(5, 400.0, 120.0, 37.0)]
    # track 6: isolated-ish line (neighbour track 5 mostly empty, 7 empty)
    s += [Seg(6, 0.0, L, 37.0)]
    return Clip(PITCH, TRACKS, L, s, f"conflict_g{int(t2t_gap)}")
