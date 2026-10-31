"""LayoutModel: cached, JSON-returning layout-reading and geometry tools over a bundle.

Groups (see CAPABILITIES):  structure | drc | patterns | mask | contours | query
All coordinates nm; ids: polygon ids of the target layer, gauge ids 'G###'.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, box

from .. import geometry as geo
from . import contours as ct
from . import drc
from . import masktools as mt
from . import patterns as pt
from . import structure as st
from .edges import measure_edges, polygon_edges
from .polylayer import PolyLayer, to_polygon

CAPABILITIES = {
    "structure": "layout_structure, list_polygons, describe_polygon, list_line_ends, pattern_context",
    "drc": "width_space_statistics, design_rule_check, corner_to_corner_spacing",
    "patterns": "pattern_catalog, find_similar_patterns",
    "mask": "mask_target_correspondence, mask_bias_statistics, line_end_treatment, assist_features, "
            "mask_complexity",
    "contours": "gauge_cd_table, process_window, line_end_pullback, corner_rounding, edge_epe",
    "query": "features_in_region, nearest_features, measure_distance, cutline_profile",
}


def _r(v, n=1):
    return None if v is None else round(float(v), n)


class LayoutModel:
    def __init__(self, bundle, spec=None):
        self.b = bundle
        self.spec = spec
        self._c = {}

    def _cached(self, k, fn):
        if k not in self._c:
            self._c[k] = fn()
        return self._c[k]

    # ------------------------------------------------------------ layers
    @property
    def frame(self):
        w, h = self.b.extent_nm
        return (0.0, 0.0, w, h)

    @property
    def target(self):
        def build():
            polys = self.b.meta.get("target_polygons")
            if polys:
                return PolyLayer([to_polygon(p) for p in polys], "target", self.frame)
            return PolyLayer.from_raster(self.b.target, self.b.pixel_nm, "target")
        return self._cached("target", build)

    @property
    def edges(self):
        def build():
            L = self.target
            ebp = {i: polygon_edges(p, i) for i, p in enumerate(L.polys)}
            return measure_edges(L, ebp)
        return self._cached("edges", build)

    @property
    def inventory(self):
        return self._cached("inv", lambda: st.polygon_inventory(self.target, self.edges))

    @property
    def lines(self):
        return self._cached("lines", lambda: st.lines_and_tracks(self.target, self.edges, self.inventory))

    @property
    def ends(self):
        halo = getattr(self.spec, "border_nm", 30.0) if self.spec is not None else 30.0
        return self._cached("ends", lambda: st.line_ends(self.target, self.edges, halo_nm=halo))

    @property
    def mask_bin(self):
        return self.b.mask > 0.5

    @property
    def prof(self):
        def build():
            f = {f"sdf:{c}": geo.signed_distance(v, self.b.pixel_nm) for c, v in self.b.printed.items()}
            if self.b.intensity is not None:
                f["image"] = self.b.intensity
            return ct.Profiler(self.b.pixel_nm, f)
        return self._cached("prof", build)

    @property
    def cond_params(self):
        return self._cached("cp", lambda: ct.parse_conditions(self.b.conditions, self.b.meta))

    # ========================================================= structure
    def layout_structure(self):
        inv, li = self.inventory, self.lines
        classes = {}
        for p in inv:
            classes[p["class"]] = classes.get(p["class"], 0) + 1
        widths = [l["width_nm"] for l in li["lines"]]
        tracks = sorted({l["track"] for l in li["lines"] if l["track"] is not None})
        ends = self.ends
        gk = {}
        for e in ends:
            gk[e["gap_kind"]] = gk.get(e["gap_kind"], 0) + 1
        return {"n_polygons": len(inv), "classes": classes, "orientation": li["orientation"],
                "orientation_purity": li["orientation_purity"], "pitch_nm": li["pitch_nm"],
                "track_offset_nm": li["track_offset_nm"], "tracks_used": tracks,
                "n_lines": len(li["lines"]),
                "line_width_nm": {"values": sorted({round(w, 1) for w in widths}),
                                  "median": _r(np.median(widths)) if widths else None},
                "n_line_ends": len(ends), "line_end_gap_kinds": gk,
                "polygons_cut_by_frame": sum(1 for p in inv if p["cut_by_frame"]),
                "density": round(float(self.b.target.mean()), 3),
                "source": "target_polygons from meta" if self.b.meta.get("target_polygons") else
                          "polygonized from raster (pixel-edge exact)"}

    def list_polygons(self, x0=None, y0=None, x1=None, y1=None, cls="", limit=40):
        ids = self.target.ids_in(x0, y0, x1, y1) if x0 is not None else range(len(self.inventory))
        rows = [self.inventory[i] for i in ids if not cls or self.inventory[i]["class"] == cls]
        return {"count": len(rows), "polygons": rows[:limit]}

    def describe_polygon(self, poly_id):
        inv = self.inventory[poly_id]
        es = self.edges[poly_id]
        line = next((l for l in self.lines["lines"] if l["id"] == poly_id), None)
        nb = {}
        for e in es:
            if e.space_to is not None and e.space_to != poly_id:
                nb[e.space_to] = min(nb.get(e.space_to, np.inf), e.space)
        return {**inv, "track": line["track"] if line else None,
                "edges": [e.as_dict() for e in es][:40],
                "neighbours": [{"poly": k, "min_space_nm": _r(v)} for k, v in sorted(nb.items(), key=lambda kv: kv[1])],
                "line_ends": [e for e in self.ends if e["poly"] == poly_id]}

    def list_line_ends(self, max_gap_nm=None, gap_kind="", limit=40):
        rows = []
        for e in self.ends:
            if max_gap_nm and (e["gap_nm"] is None or e["gap_nm"] > max_gap_nm):
                continue
            if gap_kind and e["gap_kind"] != gap_kind:
                continue
            ctx = st.neighbour_context(self.lines, e["x_nm"], e["y_nm"])
            rows.append({**e, "context": ctx.get("context"), "track": ctx.get("track")})
        rows.sort(key=lambda r: r["gap_nm"] if r["gap_nm"] is not None else 1e9)
        return {"count": len(rows), "line_ends": rows[:limit]}

    def pattern_context(self, x_nm, y_nm):
        P = Point(x_nm, y_nm)
        pid, d = self.target.nearest(P)
        inside = self.target.polys[pid].contains(P) if pid is not None else False
        return {"point_nm": [x_nm, y_nm], "inside_target": bool(inside), "nearest_poly": pid,
                "distance_to_nearest_nm": _r(d), "density": st.density_at(self.b.target, self.b.pixel_nm, x_nm, y_nm),
                "mask_density": st.density_at(self.mask_bin, self.b.pixel_nm, x_nm, y_nm, radii=(50.0, 100.0)),
                **st.neighbour_context(self.lines, x_nm, y_nm)}

    # =============================================================== drc
    def width_space_statistics(self):
        return drc.width_space_summary(self.edges)

    def design_rule_check(self, min_width=None, min_space=None, min_t2t=None, min_e2s=None, min_c2c=None,
                          min_area=None, min_edge=None, limit=30):
        return drc.rule_check(self.target, self.edges, min_width, min_space, min_t2t, min_e2s, min_c2c,
                              min_area, min_edge, limit)

    def corner_to_corner_spacing(self, max_nm=120.0, limit=20):
        r = drc.corner_to_corner(self.target, self.edges, max_nm)
        return {"count": len(r), "closest": r[:limit]}

    # ========================================================== patterns
    def _anchors(self, kind):
        px = self.b.pixel_nm
        if kind == "line_end":
            return [(int(e["y_nm"] / px), int(e["x_nm"] / px), f"end@poly{e['poly']}") for e in self.ends]
        if kind == "corner":
            return [(int(e.p[1] / px), int(e.p[0] / px), f"corner@poly{e.poly}")
                    for es in self.edges.values() for e in es if e.convex_p]
        raise ValueError("kind must be 'line_end' or 'corner'")

    def pattern_catalog(self, anchor="line_end", radius_nm=80.0, limit=15):
        px = self.b.pixel_nm
        rad = max(3, int(round(radius_nm / px)))
        groups = pt.catalog(self.b.target, self._anchors(anchor), rad)
        rows = [{"pattern_id": h, "count": len(v),
                 "members_nm": [[round((c + .5) * px, 1), round((r + .5) * px, 1)] for r, c, _ in v][:8]}
                for h, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
        return {"anchor": anchor, "radius_nm": radius_nm, "n_anchors": sum(r["count"] for r in rows),
                "n_unique_patterns": len(rows), "patterns": rows[:limit],
                "note": "fingerprints are invariant to 90-degree rotation and mirroring"}

    def find_similar_patterns(self, x_nm, y_nm, radius_nm=60.0, max_mismatch_frac=0.03, limit=20):
        px = self.b.pixel_nm
        rad = max(3, int(round(radius_nm / px)))
        r, c = int(y_nm / px), int(x_nm / px)
        q = pt.window(self.b.target, r, c, rad)
        hits = pt.search(self.b.target, q, max_mismatch_frac)
        return {"query_nm": [x_nm, y_nm], "radius_nm": radius_nm, "n_matches": len(hits),
                "matches": [{"x_nm": round((cc + .5) * px, 1), "y_nm": round((rr + .5) * px, 1),
                             "mismatch_frac": round(m, 4)} for rr, cc, m in hits[:limit]]}

    # ============================================================== mask
    def mask_target_correspondence(self):
        return self._cached("corr", lambda: mt.correspondence(self.b.target, self.mask_bin, self.b.pixel_nm))

    def mask_bias_statistics(self):
        an_samples = geo.target_edge_samples(self.b.target, self.b.pixel_nm, 4.0)
        P = an_samples[0]
        C = geo.target_corners(self.b.target, self.b.pixel_nm)
        ok = np.ones(len(P), bool)
        if len(C):
            d = np.min(np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1]), 1)
            ok &= d * self.b.pixel_nm > 12.0
        m = 30.0 / self.b.pixel_nm
        h, w = self.b.shape
        ok &= (P[:, 0] > m) & (P[:, 1] > m) & (P[:, 0] < h - m) & (P[:, 1] < w - m)
        bias, Pv = mt.bias_along_normals(self.b.target, self.mask_bin, self.b.pixel_nm, an_samples, ok)
        if not len(bias):
            return {"n": 0}
        return {"n_edge_samples": int(len(bias)), "bias_nm": {"min": _r(bias.min()), "p5": _r(np.percentile(bias, 5)),
                "median": _r(np.median(bias)), "p95": _r(np.percentile(bias, 95)), "max": _r(bias.max())},
                "fraction_negative": round(float((bias < 0).mean()), 3),
                "note": "positive = mask edge outside the drawn edge; corners and frame halo excluded",
                "largest_bias_at_nm": [[round((Pv[i, 1] + .5) * self.b.pixel_nm, 1), round((Pv[i, 0] + .5) * self.b.pixel_nm, 1),
                                        _r(bias[i])] for i in np.argsort(-bias)[:5]]}

    def line_end_treatment(self, limit=30):
        rows = mt.line_end_treatment(self.mask_bin, self.b.pixel_nm, self.ends)
        ext = [r["mask_extension_nm"] for r in rows]
        hr = [r["hammerhead_ratio"] for r in rows]
        return {"count": len(rows), "extension_nm": {"median": _r(np.median(ext)) if ext else None,
                                                     "max": _r(max(ext)) if ext else None},
                "hammerhead_ratio": {"median": _r(np.median(hr), 2) if hr else None, "max": _r(max(hr), 2) if hr else None},
                "line_ends": rows[:limit]}

    def assist_features(self, band_nm=15.0, limit=20):
        corr = self.mask_target_correspondence()
        app = mt.appendages(self.b.target, self.mask_bin, self.b.pixel_nm, band_nm)
        return {"detached_assists": {"count": len(corr["detached"]), "items": corr["detached"][:limit]},
                "attached_appendages_beyond_band": {"band_nm": band_nm, "count": len(app), "items": app[:limit]},
                "mask_shapes_joining_targets": {"count": len(corr["merging"]), "items": corr["merging"][:limit]}}

    def mask_complexity(self):
        return mt.complexity(self.mask_bin, self.b.pixel_nm)

    # ========================================================== contours
    def gauges(self):
        return self._cached("gauges", lambda: ct.make_gauges(self.lines, self.ends, self.edges, self.frame))

    def gauge_table(self):
        return self._cached("gtab", lambda: ct.measure_gauges(self.gauges(), self.prof, self.b.conditions,
                                                             self.b.nominal, self.b.intensity is not None))

    def _spec_min(self, kind):
        s = self.spec
        if s is None:
            return 0.0
        return s.w_min_nm if kind == "width" else s.s_min_nm

    def gauge_cd_table(self, kind="", worst_first=True, limit=25):
        rows = [ct.process_window(r, self.cond_params, self.b.nominal, self._spec_min(r["kind"]))
                for r in self.gauge_table() if not kind or r["kind"] == kind]
        if worst_first:
            rows.sort(key=lambda r: r["margin_to_spec_nm"])
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        return {"n_gauges": len(rows), "by_kind": kinds, "gauges": rows[:limit],
                "note": "CD per condition from sub-pixel zero crossings of the printed signed-distance field; "
                        "MEEF = dCD/d(mask CD) with mask CD = 2 x per-edge bias. Printed patterns are binary "
                        f"rasters ({self.b.pixel_nm:.2f} nm pixels): CD resolution is ~half a pixel, so "
                        "sensitivities smaller than that can read as 0; nominal_cd_from_image is exact."}

    def process_window(self, gauge_id):
        row = next((r for r in self.gauge_table() if r["id"] == gauge_id), None)
        if row is None:
            return {"error": f"unknown gauge {gauge_id}"}
        summ = ct.process_window(row, self.cond_params, self.b.nominal, self._spec_min(row["kind"]))
        table = [{**self.cond_params.get(c, {"condition": c}), "condition": c, "cd_nm": round(v, 2)}
                 for c, v in row["cd"].items()]
        return {"summary": summ, "table": table}

    def line_end_pullback(self, limit=25):
        rows = ct.pullback(self.ends, self.prof, self.b.conditions)
        rows.sort(key=lambda r: -r["max_pullback_nm"])
        return {"count": len(rows), "line_ends": [{k: v for k, v in r.items() if k != "pullback_nm"} for r in rows[:limit]],
                "note": "pull-back > 0: printed end short of the drawn end; < 0: printed beyond it"}

    def corner_rounding(self, condition="", limit=25):
        rows = ct.corner_rounding(self.edges, self.prof, condition or self.b.nominal, self.frame)
        cv = [r["equiv_radius_nm"] for r in rows if r["type"] == "convex"]
        cc = [r["equiv_radius_nm"] for r in rows if r["type"] == "concave"]
        rows.sort(key=lambda r: -r["equiv_radius_nm"])
        return {"condition": condition or self.b.nominal,
                "convex_radius_nm": {"n": len(cv), "median": _r(np.median(cv)) if cv else None, "max": _r(max(cv)) if cv else None},
                "concave_radius_nm": {"n": len(cc), "median": _r(np.median(cc)) if cc else None, "max": _r(max(cc)) if cc else None},
                "largest": rows[:limit]}

    def edge_epe(self, kind="", limit=25):
        rows = ct.edge_epe(self.edges, self.prof, self.b.conditions, self.b.nominal, self.frame)
        if kind:
            rows = [r for r in rows if r["kind"] == kind]
        rows.sort(key=lambda r: -max(abs(r["epe_min_nm"]), abs(r["epe_max_nm"])))
        return {"count": len(rows), "edges": rows[:limit],
                "note": "EPE > 0: printed edge outside the drawn edge"}

    # ============================================================= query
    def features_in_region(self, x0, y0, x1, y1, findings=None):
        reg = box(x0, y0, x1, y1)
        ids = self.target.ids_in(x0, y0, x1, y1)
        ends = [e for e in self.ends if x0 <= e["x_nm"] <= x1 and y0 <= e["y_nm"] <= y1]
        corr = self.mask_target_correspondence()
        det = [d for d in corr["detached"] if x0 <= d["x_nm"] <= x1 and y0 <= d["y_nm"] <= y1]
        mer = [d for d in corr["merging"] if x0 <= d["x_nm"] <= x1 and y0 <= d["y_nm"] <= y1]
        gs = [g for g in self.gauges() if reg.contains(Point(g["ref"]))]
        out = {"region_nm": [x0, y0, x1, y1], "polygons": [self.inventory[i] for i in ids][:20],
               "line_ends": ends[:20], "detached_assists": det, "mask_joins": mer,
               "gauges": [{"id": g["id"], "kind": g["kind"], "ref_nm": [round(v, 1) for v in g["ref"]]} for g in gs][:20]}
        if findings is not None:
            out["findings"] = [f for f in findings if x0 <= f["x_nm"] <= x1 and y0 <= f["y_nm"] <= y1][:20]
        return out

    def nearest_features(self, x_nm, y_nm, k=5):
        P = Point(x_nm, y_nm)
        ds = sorted(((p.distance(P), i) for i, p in enumerate(self.target.polys)))[:k]
        ends = sorted(self.ends, key=lambda e: np.hypot(e["x_nm"] - x_nm, e["y_nm"] - y_nm))[:k]
        return {"point_nm": [x_nm, y_nm],
                "polygons": [{"poly": i, "class": self.inventory[i]["class"], "distance_nm": _r(d)} for d, i in ds],
                "line_ends": [{**e, "distance_nm": _r(np.hypot(e["x_nm"] - x_nm, e["y_nm"] - y_nm))} for e in ends]}

    def measure_distance(self, x1=None, y1=None, x2=None, y2=None, poly_a=None, poly_b=None):
        if poly_a is not None and poly_b is not None:
            a, b = self.target.polys[poly_a], self.target.polys[poly_b]
            from shapely.ops import nearest_points
            pa, pb = nearest_points(a, b)
            return {"poly_a": poly_a, "poly_b": poly_b, "distance_nm": _r(a.distance(b), 2),
                    "closest_points_nm": [[_r(pa.x), _r(pa.y)], [_r(pb.x), _r(pb.y)]]}
        return {"distance_nm": _r(np.hypot(x2 - x1, y2 - y1), 2)}

    def cutline_profile(self, x1, y1, x2, y2, conditions=None):
        """Edges crossed by a cutline on target, mask and printed patterns; image min/max/ILS."""
        line = LineString([(x1, y1), (x2, y2)])
        L = line.length
        out = {"cutline_nm": [x1, y1, x2, y2], "length_nm": _r(L, 2)}

        def crossings(field_name):
            s, v = self.prof.profile(field_name, (x1, y1), (x2, y2)) if field_name in self.prof.fields else (None, None)
            if s is None:
                return None
            pos = v > 0                                   # boolean transitions: robust to exact zeros
            idx = np.nonzero(np.diff(pos))[0]
            return [round(float(s[i] + (s[i + 1] - s[i]) * v[i] / (v[i] - v[i + 1])), 2) if v[i] != v[i + 1]
                    else round(float(s[i]), 2) for i in idx]
        tsdf = geo.signed_distance(self.b.target, self.b.pixel_nm)
        msdf = geo.signed_distance(self.mask_bin, self.b.pixel_nm)
        self.prof.fields.setdefault("sdf:__target__", tsdf)
        self.prof.fields.setdefault("sdf:__mask__", msdf)
        out["target_edges_at_nm"] = crossings("sdf:__target__")
        out["mask_edges_at_nm"] = crossings("sdf:__mask__")
        conds = conditions or list(dict.fromkeys([self.b.nominal] + self.b.conditions[:0]))
        out["printed_edges_at_nm"] = {c: crossings(f"sdf:{c}") for c in conds}
        if "image" in self.prof.fields:
            s, v = self.prof.profile("image", (x1, y1), (x2, y2))
            gi = np.abs(np.gradient(v, s)) / np.maximum(v, 1e-6)
            idx = np.nonzero(np.diff(v > 1.0))[0]
            out["image"] = {"min": _r(v.min(), 3), "max": _r(v.max(), 3),
                            "threshold_crossings_nm": [round(float(s[i]), 2) for i in idx],
                            "ils_at_crossings_per_nm": [round(float(gi[i]), 4) for i in idx]}
        return out
