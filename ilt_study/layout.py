"""Unidirectional metal clips + the cutline 'sites' used by both cost and checker.

A clip is a set of vertical line segments on tracks of pitch 74.5 nm.  The
clip is periodic in x and y (FFT imaging), so segments may wrap in y.

Site kinds (each is a straight cutline sampled every `ds` nm):
  pinch  : across a line, measures printed (bright) width      -> w_min
  bridge : across a space between two lines, measures dark gap -> s_min
  t2t    : along a track across a tip-to-tip gap, dark length  -> s_min_t2t
  lend   : along a line end, printed extent past (end - L0)    -> pull-back
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Seg:
    track: int
    y0: float           # start (may be > y1 when wrapping)
    length: float
    width: float = 37.0


@dataclass
class Clip:
    pitch: float
    tracks: int
    L: float
    segs: list
    name: str = "clip"

    def xc(self, k):
        return (k % self.tracks + 0.5) * self.pitch


def y_in_seg(y, s: Seg, L):
    return ((y - s.y0) % L) < s.length


def rasterize(clip: Clip, n: int, supersample: int = 4) -> np.ndarray:
    """Area-sampled target in [0, 1] (grey at edges)."""
    m = n * supersample
    d = clip.L / m
    c = (np.arange(m) + 0.5) * d
    X, Y = np.meshgrid(c, c, indexing="xy")
    T = np.zeros((m, m), bool)
    for s in clip.segs:
        xc = clip.xc(s.track)
        dxw = np.abs(((X - xc + clip.L / 2) % clip.L) - clip.L / 2) < s.width / 2
        T |= dxw & y_in_seg(Y, s, clip.L)
    return T.reshape(n, supersample, n, supersample).mean(axis=(1, 3)).astype(np.float32)


@dataclass
class Sites:
    kind: list = field(default_factory=list)      # str per site
    p0: list = field(default_factory=list)        # (x, y) start
    p1: list = field(default_factory=list)        # (x, y) end
    ref: list = field(default_factory=list)       # nominal target length
    tag: list = field(default_factory=list)       # free text (seg index ...)

    def add(self, kind, p0, p1, ref, tag=""):
        self.kind.append(kind); self.p0.append(p0); self.p1.append(p1)
        self.ref.append(ref); self.tag.append(tag)

    def __len__(self):
        return len(self.kind)

    def idx(self, kind):
        return np.array([i for i, k in enumerate(self.kind) if k == kind], int)


def make_sites(clip: Clip, step_along: float = 12.0, end_keep: float = 30.0,
               lend_L0: float = 40.0, t2t_offsets=(-8.0, 0.0, 8.0)) -> Sites:
    S = Sites()
    L, p = clip.L, clip.pitch
    by_track = {}
    for i, s in enumerate(clip.segs):
        by_track.setdefault(s.track, []).append(i)
    for i, s in enumerate(clip.segs):
        xc = clip.xc(s.track)
        # pinch cuts across the line, bounded by mid-space on both sides
        nstep = int((s.length - 2 * end_keep) // step_along) + 1
        for j in range(max(nstep, 1)):
            y = (s.y0 + end_keep + j * step_along) % L
            if s.length < 2 * end_keep:
                y = (s.y0 + s.length / 2) % L
            S.add("pinch", (xc - p / 2, y), (xc + p / 2, y), s.width, f"seg{i}")
        # lateral bridge cuts to the right neighbour track
        k2 = (s.track + 1) % clip.tracks
        for i2 in by_track.get(k2, []):
            s2 = clip.segs[i2]
            for j in range(int(s.length // step_along) + 1):
                y = (s.y0 + j * step_along) % L
                if not (y_in_seg(y, s2, L) and y_in_seg(y, s, L)):
                    continue
                # keep away from both segments' ends (line-end corners)
                if min((y - s.y0) % L, (s.y0 + s.length - y) % L) < end_keep / 2:
                    continue
                if min((y - s2.y0) % L, (s2.y0 + s2.length - y) % L) < end_keep / 2:
                    continue
                x0 = xc
                x1 = x0 + p
                ref = p - s.width / 2 - s2.width / 2
                S.add("bridge", (x0, y), (x1, y), ref, f"seg{i}|seg{i2}")
    # tip-to-tip gaps and line ends along each track
    for k, ids in by_track.items():
        ids = sorted(ids, key=lambda i: clip.segs[i].y0)
        xc = clip.xc(k)
        for a, i in enumerate(ids):
            s = clip.segs[i]
            nxt = clip.segs[ids[(a + 1) % len(ids)]]
            end = s.y0 + s.length
            gap = (nxt.y0 - end) % L
            if len(ids) == 1 and gap < 1e-6:
                continue
            ext = 30.0
            for off in t2t_offsets:   # parallel cuts catch off-centre (S-bend) bridges
                S.add("t2t", (xc + off, end - ext), (xc + off, end + gap + ext), gap,
                      f"seg{i}->seg{ids[(a + 1) % len(ids)]}@{off:+.0f}")
            # line-end extent cuts: from end-L0 to the gap middle
            S.add("lend", (xc, end - lend_L0), (xc, end + gap / 2), lend_L0, f"seg{i}:top")
            S.add("lend", (xc, nxt.y0 + lend_L0), (xc, nxt.y0 - gap / 2), lend_L0,
                  f"seg{ids[(a + 1) % len(ids)]}:bot")
    # diagonal cuts: line-end corner -> nearest point of a neighbour-track segment
    for i, s in enumerate(clip.segs):
        if s.length >= L - 1e-6:
            continue
        xc = clip.xc(s.track)
        for ey, sgn in ((s.y0 + s.length, +1), (s.y0, -1)):
            for dk in (-1, +1):
                cx = xc + dk * s.width / 2
                for i2 in by_track.get((s.track + dk) % clip.tracks, []):
                    s2 = clip.segs[i2]
                    x2 = xc + dk * (p - s2.width / 2)
                    # nearest y on s2 (periodic)
                    rel = (ey - s2.y0) % L
                    if rel <= s2.length:
                        continue          # neighbour overlaps the end: lateral cut covers it
                    yb = s2.y0 if (L - rel) < (rel - s2.length) else s2.y0 + s2.length
                    dy = ((yb - ey + L / 2) % L) - L / 2
                    if dy * sgn < 0:
                        continue          # neighbour lies behind this end
                    dist = float(np.hypot(x2 - cx, dy))
                    if dist > 70.0:
                        continue
                    u = np.array([x2 - cx, dy]) / dist
                    a = np.array([cx, ey]) - 8 * u
                    b = np.array([x2, ey + dy]) + 8 * u
                    S.add("diag", tuple(a), tuple(b), dist, f"seg{i}~seg{i2}")
    return S


class SiteSampler:
    """Bilinear, periodic sampling of images along all site cutlines."""

    def __init__(self, sites: Sites, L: float, n: int, ds: float = 0.5):
        self.sites, self.L, self.n, self.ds = sites, L, n, ds
        p0 = np.array(sites.p0, float)
        p1 = np.array(sites.p1, float)
        lens = np.linalg.norm(p1 - p0, axis=1)
        self.nsamp = np.maximum(np.ceil(lens / ds).astype(int), 1)
        smax = self.nsamp.max()
        t = (np.arange(smax) + 0.5)[None, :] / self.nsamp[:, None]
        valid = np.arange(smax)[None, :] < self.nsamp[:, None]
        pts = p0[:, None, :] + t[..., None] * (p1 - p0)[:, None, :]
        self.step = torch.tensor(lens / self.nsamp, dtype=torch.float32)
        self.valid = torch.tensor(valid, dtype=torch.float32)
        d = L / n
        gx = pts[..., 0] / d - 0.5
        gy = pts[..., 1] / d - 0.5
        x0 = np.floor(gx); y0 = np.floor(gy)
        fx = gx - x0; fy = gy - y0
        x0 = x0.astype(int) % n; y0 = y0.astype(int) % n
        x1 = (x0 + 1) % n; y1 = (y0 + 1) % n
        self.ix = [torch.tensor(a.ravel()) for a in (y0 * n + x0, y0 * n + x1,
                                                     y1 * n + x0, y1 * n + x1)]
        self.wts = [torch.tensor(w.ravel(), dtype=torch.float32) for w in
                    ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)]
        self.shape = pts.shape[:2]
        self.kind = np.array(sites.kind)
        self.ref = torch.tensor(sites.ref, dtype=torch.float32)

    def sample(self, img: torch.Tensor) -> torch.Tensor:
        """img [..., n, n] -> [..., n_sites, n_samples]"""
        flat = img.reshape(*img.shape[:-2], -1)
        v = sum(flat[..., i] * w for i, w in zip(self.ix, self.wts))
        return v.reshape(*img.shape[:-2], *self.shape)

    def lengths(self, img, th, dose=1.0, alpha=None):
        """Printed (bright) length along every cut.  Hard if alpha is None."""
        v = dose * self.sample(img) - th
        on = (v > 0).float() if alpha is None else torch.sigmoid(alpha * v)
        return (on * self.valid).sum(-1) * self.step


def measure(lengths_bright: torch.Tensor, S: SiteSampler, lend_L0=40.0):
    """Convert bright lengths to the site's physical quantity.

    pinch: printed width.     bridge / t2t: printed gap = cut - bright.
    lend : printed extent beyond (end - L0) = bright length (from end-L0)."""
    cut = S.step * torch.tensor(S.nsamp, dtype=torch.float32)
    out = lengths_bright.clone()
    gapmask = torch.tensor(np.isin(S.kind, ["bridge", "t2t", "diag"]))
    out = torch.where(gapmask, cut - lengths_bright, out)
    return out
