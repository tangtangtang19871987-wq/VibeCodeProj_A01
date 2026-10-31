"""Pattern fingerprints, catalogs and search.

A pattern is the binary target window around an anchor.  Fingerprints are canonical
under the 8 symmetries of the square (D4: rotations by 90 deg and mirrors), so the same
layout context in any orientation gets the same id.  Search uses FFT correlation of the
+-1 encoded window against the whole layer, for all 8 orientations.
"""
from __future__ import annotations

import hashlib

import numpy as np


def window(binary, row, col, radius_px):
    pad = np.pad(np.asarray(binary, bool), radius_px, constant_values=False)
    return pad[row:row + 2 * radius_px + 1, col:col + 2 * radius_px + 1]


def d4(w):
    out = []
    for k in range(4):
        r = np.rot90(w, k)
        out += [r, np.fliplr(r)]
    return out


def canonical(w, grid=2):
    """Down-sample by `grid` (block majority) and pick the lexicographically smallest of the
    8 symmetric variants.  Returns (hash, canonical array)."""
    h = (w.shape[0] // grid) * grid
    v = w[:h, :h].reshape(h // grid, grid, h // grid, grid).mean(axis=(1, 3)) >= 0.5
    best = min(d4(v), key=lambda a: np.packbits(a).tobytes())
    return hashlib.sha1(np.packbits(best).tobytes() + bytes([best.shape[0]])).hexdigest()[:12], best


def catalog(binary, anchors, radius_px, grid=2):
    """anchors: list of (row, col, label). Groups anchors by canonical fingerprint."""
    groups = {}
    for r, c, lab in anchors:
        hsh, _ = canonical(window(binary, r, c, radius_px), grid)
        groups.setdefault(hsh, []).append((r, c, lab))
    return groups


def search(binary, query, max_mismatch_frac=0.03, min_sep_px=None):
    """Locations (row, col of window centre) where `query` (any of its 8 orientations)
    matches the layer with at most max_mismatch_frac differing pixels."""
    B = np.where(np.asarray(binary, bool), 1.0, -1.0)
    H, W = B.shape
    n = query.size
    hits = {}
    FB = np.fft.rfft2(B, s=(H + query.shape[0], W + query.shape[1]))
    for q in {np.packbits(v).tobytes(): v for v in d4(query)}.values():
        Q = np.where(q, 1.0, -1.0)[::-1, ::-1]
        corr = np.fft.irfft2(FB * np.fft.rfft2(Q, s=(H + q.shape[0], W + q.shape[1])),
                             s=(H + q.shape[0], W + q.shape[1]))
        hr, hc = q.shape[0] // 2, q.shape[1] // 2
        corr = corr[hr:hr + H, hc:hc + W]                   # centre-aligned
        mism = (n - corr) / 2.0
        ok = np.argwhere(mism <= max_mismatch_frac * n)
        for r, c in ok:
            # windows must lie fully inside the layer
            if r - hr < 0 or c - hc < 0 or r + hr >= H or c + hc >= W:
                continue
            key = (int(r), int(c))
            hits[key] = min(hits.get(key, np.inf), float(mism[r, c] / n))
    # non-maximum suppression
    sep = min_sep_px or max(2, query.shape[0] // 4)
    out = []
    for (r, c), m in sorted(hits.items(), key=lambda kv: kv[1]):
        if all(abs(r - r2) > sep or abs(c - c2) > sep for r2, c2, _ in out):
            out.append((r, c, m))
    return out
