"""Vector polygon layers built from rasters (or given polygons), with a spatial index.

Raster -> polygon conversion is exact at pixel edges: every row is split into runs,
vertically identical runs are merged into rectangles, and the rectangles are unioned.
For Manhattan layouts this reproduces the drawn polygons up to the pixel grid; for
curvilinear masks/contours it gives the staircase outline (use geometry.contour_curvature
for smooth curvature).  Coordinates: nm, origin lower-left, x = column, y = row.
"""
from __future__ import annotations

import numpy as np
from shapely import STRtree
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union


def _runs(row):
    d = np.diff(np.r_[0, row.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def raster_to_polygons(binary, pixel_nm):
    b = np.asarray(binary, bool)
    rects, active = [], {}
    for r in range(b.shape[0]):
        new = {}
        for run in _runs(b[r]):
            new[run] = active.pop(run, r)
        for (c0, c1), r0 in active.items():
            rects.append(box(c0 * pixel_nm, r0 * pixel_nm, c1 * pixel_nm, r * pixel_nm))
        active = new
    for (c0, c1), r0 in active.items():
        rects.append(box(c0 * pixel_nm, r0 * pixel_nm, c1 * pixel_nm, b.shape[0] * pixel_nm))
    if not rects:
        return []
    u = unary_union(rects)
    polys = list(u.geoms) if isinstance(u, MultiPolygon) else [u]
    return [orient(p.simplify(0), 1.0) for p in polys if p.area > 0]


def rect_decomposition_count(binary):
    """Number of rectangles in the row-run/vertical-merge decomposition (VSB shot proxy)."""
    b = np.asarray(binary, bool)
    n, active = 0, set()
    for r in range(b.shape[0]):
        runs = set(_runs(b[r]))
        n += len(active - runs)
        active = runs
    return n + len(active)


class PolyLayer:
    """A list of oriented (CCW) polygons with ids, an STRtree and cached properties."""

    def __init__(self, polys, name="layer", frame=None):
        self.name = name
        self.polys = [orient(p, 1.0) for p in polys]
        self.tree = STRtree(self.polys) if self.polys else None
        self.frame = frame                                   # (x0, y0, x1, y1) of the window

    @classmethod
    def from_raster(cls, binary, pixel_nm, name="layer"):
        h, w = np.asarray(binary).shape
        return cls(raster_to_polygons(binary, pixel_nm), name, (0.0, 0.0, w * pixel_nm, h * pixel_nm))

    def __len__(self):
        return len(self.polys)

    def query(self, geom):
        """Ids of polygons whose envelope intersects geom."""
        if self.tree is None:
            return []
        return [int(i) for i in self.tree.query(geom)]

    def nearest(self, geom, exclude=()):
        best, bd = None, np.inf
        for i, p in enumerate(self.polys):
            if i in exclude:
                continue
            d = p.distance(geom)
            if d < bd:
                best, bd = i, d
        return best, bd

    def touches_frame(self, i, tol=1e-6):
        if self.frame is None:
            return False
        x0, y0, x1, y1 = self.polys[i].bounds
        fx0, fy0, fx1, fy1 = self.frame
        return x0 <= fx0 + tol or y0 <= fy0 + tol or x1 >= fx1 - tol or y1 >= fy1 - tol

    def ids_in(self, x0, y0, x1, y1):
        reg = box(x0, y0, x1, y1)
        return [i for i in self.query(reg) if self.polys[i].intersects(reg)]


def to_polygon(coords):
    return orient(Polygon(coords), 1.0)
