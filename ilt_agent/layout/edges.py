"""Edge decomposition and DRC-style edge-pair measurement by ray casting.

Each polygon ring is reduced to its vertices (collinear points removed) and split into
edges.  For a CCW exterior ring the outward normal of edge p->q is (dy, -dx)/|pq|; holes
are CW after shapely.orient, so the same formula points out of the material there too.

width of an edge  : distance from the edge, along the inward normal, to the far side of
                    the same polygon (min over sample points)
space of an edge  : distance along the outward normal to the first boundary of any
                    polygon (including itself: then it is a notch)
Hit edges are classified so spaces can be typed: side-side, end-end (tip-to-tip),
end-side, notch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point

EPS = 1e-6


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class Edge:
    poly: int
    idx: int
    p: tuple
    q: tuple
    length: float
    orient: str            # 'H', 'V' or 'D' (diagonal / curved)
    normal: tuple          # outward unit normal
    convex_p: bool         # corner at p is convex (material angle < 180)
    convex_q: bool
    hole: bool = False
    ring: int = 0
    kind: str = "side"     # 'side' | 'end' (line end) | 'short' (jog / small step)
    width: float | None = None
    space: float | None = None
    space_to: int | None = None          # poly id hit by the space ray
    space_to_edge: int | None = None     # edge idx (in that poly) hit
    space_kind: str | None = None        # side-side | end-end | end-side | notch | none

    @property
    def mid(self):
        return ((self.p[0] + self.q[0]) / 2, (self.p[1] + self.q[1]) / 2)

    def as_dict(self):
        return {"poly": self.poly, "edge": self.idx, "kind": self.kind, "orient": self.orient,
                "p": [round(v, 1) for v in self.p], "q": [round(v, 1) for v in self.q],
                "length_nm": round(self.length, 1), "normal": [round(v, 2) for v in self.normal],
                "width_nm": None if self.width is None else round(self.width, 1),
                "space_nm": None if self.space is None else round(self.space, 1),
                "space_kind": self.space_kind, "space_to_poly": self.space_to}


def _clean(coords):
    pts = [tuple(c) for c in coords[:-1]]
    out = []
    n = len(pts)
    for i in range(n):
        a, b, c = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % n])
        if abs(cross2(b - a, c - b)) > EPS:
            out.append(tuple(b))
    return out


def ring_edges(coords, poly_id, start_idx=0, hole=False, ring=0):
    pts = _clean(list(coords))
    n = len(pts)
    edges = []
    for i in range(n):
        p, q = np.array(pts[i]), np.array(pts[(i + 1) % n])
        prev, nxt = np.array(pts[i - 1]), np.array(pts[(i + 2) % n])
        d = q - p
        L = float(np.hypot(*d))
        if L < EPS:
            continue
        nrm = (d[1] / L, -d[0] / L)
        o = "H" if abs(d[1]) < EPS else ("V" if abs(d[0]) < EPS else "D")
        conv_p = cross2(p - prev, q - p) > 0           # left turn on a CCW ring = convex
        conv_q = cross2(q - p, nxt - q) > 0
        edges.append(Edge(poly_id, start_idx + len(edges), tuple(p), tuple(q), L, o, nrm,
                          bool(conv_p), bool(conv_q), hole, ring))
    return edges


def polygon_edges(poly, poly_id):
    edges = ring_edges(poly.exterior.coords, poly_id)
    for k, ring in enumerate(poly.interiors, start=1):
        edges += ring_edges(ring.coords, poly_id, len(edges), hole=True, ring=k)
    return edges


def _first_hit(origin, direction, max_len, geoms):
    """Nearest intersection of a ray with polygon boundaries: (distance, geom index, point)."""
    o = np.asarray(origin)
    ray = LineString([tuple(o), tuple(o + max_len * np.asarray(direction))])
    best = (np.inf, None, None)
    for gi, g in geoms:
        inter = ray.intersection(g.boundary)
        if inter.is_empty:
            continue
        pts = getattr(inter, "geoms", [inter])
        for pt in pts:
            for c in (pt.coords if hasattr(pt, "coords") else []):
                dd = float(np.hypot(c[0] - o[0], c[1] - o[1]))
                if EPS * 10 < dd < best[0]:
                    best = (dd, gi, c)
    return best


def _hit_edge(edges_of_poly, pt):
    P = Point(pt)
    best, bd = None, np.inf
    for e in edges_of_poly:
        d = LineString([e.p, e.q]).distance(P)
        if d < bd:
            best, bd = e, d
    return best


def measure_edges(layer, edges_by_poly, max_width=200.0, max_space=200.0, samples=(0.25, 0.5, 0.75),
                  end_max_nm=60.0):
    """Fill width / space / kind for every edge.  edges_by_poly: {poly_id: [Edge]}."""
    polys = layer.polys
    # pass 1: width (inside own polygon) and end classification
    for pid, edges in edges_by_poly.items():
        for e in edges:
            nin = -np.asarray(e.normal)
            ws = []
            for t in samples:
                o = np.asarray(e.p) + t * (np.asarray(e.q) - np.asarray(e.p)) + nin * 1e-3
                d, _, _ = _first_hit(o, nin, max_width, [(pid, polys[pid])])
                ws.append(d)
            e.width = float(np.min(ws)) if np.isfinite(np.min(ws)) else None
            depth = float(np.max(ws)) if np.isfinite(np.max(ws)) else 0.0
            if e.convex_p and e.convex_q and e.length <= end_max_nm and depth >= 1.5 * e.length:
                e.kind = "end"
            elif e.length < 0.5 * min(end_max_nm, depth if depth > 0 else end_max_nm) and not (e.convex_p and e.convex_q):
                e.kind = "short"
    # pass 2: space along the outward normal
    for pid, edges in edges_by_poly.items():
        for e in edges:
            nout = np.asarray(e.normal)
            best = (np.inf, None, None)
            for t in samples:
                o = np.asarray(e.p) + t * (np.asarray(e.q) - np.asarray(e.p)) + nout * 1e-3
                ray = LineString([tuple(o), tuple(o + max_space * nout)])
                cand = [(i, polys[i]) for i in layer.query(ray)]
                hit = _first_hit(o, nout, max_space, cand)
                if hit[0] < best[0]:
                    best = hit
            if best[1] is None:
                e.space, e.space_kind = None, "none"
                continue
            e.space, e.space_to = float(best[0]), int(best[1])
            he = _hit_edge(edges_by_poly[e.space_to], best[2])
            e.space_to_edge = he.idx if he else None
            if e.space_to == pid:
                e.space_kind = "notch"
            else:
                a = "end" if e.kind == "end" else "side"
                b = "end" if (he is not None and he.kind == "end") else "side"
                e.space_kind = f"{min(a, b)}-{max(a, b)}"
    return edges_by_poly


def convex_corners(edges):
    """Convex corner points of a polygon (where both adjacent edges meet at a convex vertex)."""
    return [e.p for e in edges if e.convex_p]
