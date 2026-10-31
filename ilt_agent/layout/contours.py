"""Resist-contour metrology on the target geometry: automatic gauges, CD per condition,
process-window tables and sensitivities, line-end pull-back, corner rounding, EPE per edge.

Measurements use the bilinearly interpolated signed-distance field of each printed raster
(sub-pixel zero crossings, accuracy ~0.5 px).  When the bundle has the nominal resist image,
nominal CDs are also measured on the image itself (threshold 1.0, exact sub-pixel).
"""
from __future__ import annotations

import re

import numpy as np

from .. import geometry as geo

COND_RE = re.compile(r"d(?P<dose>[0-9.]+)_f(?P<focus>[+-]?[0-9.]+)_m(?P<mask>[+-]?[0-9.]+)")


def parse_conditions(names, meta=None):
    """{name: {'dose':, 'focus_nm':, 'mask_bias_nm':}} from names like d1.05_f-40_m+1, or meta['conditions']."""
    out = {}
    given = (meta or {}).get("conditions", {})
    for n in names:
        if n in given:
            out[n] = given[n]
            continue
        m = COND_RE.search(n)
        if m:
            out[n] = {"dose": float(m["dose"]), "focus_nm": float(m["focus"]), "mask_bias_nm": float(m["mask"])}
    return out


class Profiler:
    """Samples fields along a segment."""

    def __init__(self, pixel_nm, fields):
        self.px = pixel_nm
        self.fields = fields                                  # name -> 2D array (sdf or image)

    def profile(self, name, p0, p1, step=0.25):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.hypot(*(p1 - p0)))
        n = max(2, int(L / step) + 1)
        t = np.linspace(0, 1, n)
        pts = p0[None] + t[:, None] * (p1 - p0)[None]
        rc = np.stack([pts[:, 1] / self.px - 0.5, pts[:, 0] / self.px - 0.5], 1)
        f = self.fields[name]
        rc[:, 0] = np.clip(rc[:, 0], 0, f.shape[0] - 1.001)
        rc[:, 1] = np.clip(rc[:, 1], 0, f.shape[1] - 1.001)
        return t * L, geo.bilinear(f, rc)


def run_length(s, v, s_ref, positive=True):
    """Length of the run of (v > 0) [or v < 0] containing s_ref, with linear-interpolated ends.
    0 if s_ref is not inside such a run."""
    sign = 1 if positive else -1
    w = sign * v
    i = int(np.clip(np.searchsorted(s, s_ref), 1, len(s) - 1))
    if w[i] <= 0 and w[i - 1] <= 0:
        return 0.0
    lo = i if w[i] > 0 else i - 1
    hi = lo
    while lo > 0 and w[lo - 1] > 0:
        lo -= 1
    while hi < len(w) - 1 and w[hi + 1] > 0:
        hi += 1
    a = s[lo] if lo == 0 else s[lo - 1] + (s[lo] - s[lo - 1]) * (-w[lo - 1]) / (w[lo] - w[lo - 1])
    b = s[hi] if hi == len(w) - 1 else s[hi] + (s[hi + 1] - s[hi]) * w[hi] / (w[hi] - w[hi + 1])
    return float(b - a)


def make_gauges(lines_info, ends, edges_by_poly, frame, ext_nm=12.0, keep_from_end_nm=30.0, per_line=2):
    """Width gauges across lines, space gauges across side-side spaces, tip-to-tip gauges."""
    g = []
    ori = lines_info["orientation"]
    fx0, fy0, fx1, fy1 = frame
    for l in lines_info["lines"]:
        a, b = l["span_nm"]
        if b - a < 2 * keep_from_end_nm:
            continue
        for k in range(per_line):
            t = a + keep_from_end_nm + (k + 0.5) / per_line * (b - a - 2 * keep_from_end_nm)
            c, w = l["center_nm"], l["width_nm"]
            if ori == "V":
                p0, p1, ref = (c - w / 2 - ext_nm, t), (c + w / 2 + ext_nm, t), (c, t)
            else:
                p0, p1, ref = (t, c - w / 2 - ext_nm), (t, c + w / 2 + ext_nm), (t, c)
            g.append({"kind": "width", "poly": l["id"], "p0": p0, "p1": p1, "ref": ref, "design_nm": w})
    seen = set()
    for pid, es in edges_by_poly.items():
        for e in es:
            if e.space is None or e.space_kind not in ("side-side", "end-end"):
                continue
            key = tuple(sorted((pid, e.space_to))) + (round(e.mid[0] / 10), round(e.mid[1] / 10)) \
                if e.space_kind == "end-end" else (pid, e.idx)
            if key in seen:
                continue
            seen.add(key)
            if e.space_kind == "side-side" and e.length < 2 * keep_from_end_nm:
                continue
            m = np.asarray(e.mid); n = np.asarray(e.normal)
            p0 = tuple(map(float, m - n * ext_nm)); p1 = tuple(map(float, m + n * (e.space + ext_nm)))
            ref = tuple(map(float, m + n * e.space / 2))
            if not (fx0 < ref[0] < fx1 and fy0 < ref[1] < fy1):
                continue
            g.append({"kind": "space" if e.space_kind == "side-side" else "tip-to-tip", "poly": pid,
                      "partner": e.space_to, "p0": p0, "p1": p1, "ref": ref, "design_nm": e.space})
    # tip-to-tip gauges appear twice (one per facing end): keep one per pair
    uniq, out = set(), []
    for x in g:
        k = (x["kind"], round(x["ref"][0], 0), round(x["ref"][1], 0))
        if k not in uniq:
            uniq.add(k)
            out.append(x)
    for i, x in enumerate(out):
        x["id"] = f"G{i:03d}"
    return out


def measure_gauges(gauges, prof, conditions, nominal, has_image):
    res = []
    for g in gauges:
        s_ref = float(np.hypot(g["ref"][0] - g["p0"][0], g["ref"][1] - g["p0"][1]))
        pos = g["kind"] == "width"
        cd = {}
        for c in conditions:
            s, v = prof.profile(f"sdf:{c}", g["p0"], g["p1"])
            cd[c] = run_length(s, v, s_ref, positive=pos)
        row = {**{k: g[k] for k in ("id", "kind", "poly")}, "ref_nm": [round(float(v), 1) for v in g["ref"]],
               "design_nm": round(g["design_nm"], 1), "cd": cd}
        if has_image:
            s, v = prof.profile("image", g["p0"], g["p1"])
            row["nominal_cd_from_image"] = round(run_length(s, v - 1.0, s_ref, positive=pos), 2)
        res.append(row)
    return res


def process_window(row, cond_params, nominal, spec_min):
    """Summary of one gauge over the conditions: range, worst, margin, sensitivities."""
    cd = row["cd"]
    vals = np.array(list(cd.values()))
    worst_c = min(cd, key=cd.get)
    out = {"id": row["id"], "kind": row["kind"], "ref_nm": row["ref_nm"], "design_nm": row["design_nm"],
           "nominal_cd": round(cd.get(nominal, np.nan), 2), "min_cd": round(float(vals.min()), 2),
           "max_cd": round(float(vals.max()), 2), "range_nm": round(float(np.ptp(vals)), 2),
           "worst_condition": worst_c, "margin_to_spec_nm": round(float(vals.min() - spec_min), 2)}
    if "nominal_cd_from_image" in row:
        out["nominal_cd_from_image"] = row["nominal_cd_from_image"]
    P = cond_params
    if P and nominal in P:
        n = P[nominal]
        same = lambda c, **kw: all(abs(P[c][k] - (kw[k] if k in kw else n[k])) < 1e-9 for k in ("dose", "focus_nm", "mask_bias_nm"))
        dose_pts = [(P[c]["dose"], cd[c]) for c in cd if c in P and same(c, dose=P[c]["dose"])]
        if len({d for d, _ in dose_pts}) >= 2:
            d, v = np.array(dose_pts).T
            out["dCD_per_1pct_dose_nm"] = round(float(np.polyfit(d, v, 1)[0] * 0.01 * n["dose"]), 3)
        mask_pts = [(P[c]["mask_bias_nm"], cd[c]) for c in cd if c in P and same(c, mask_bias_nm=P[c]["mask_bias_nm"])]
        if len({m for m, _ in mask_pts}) >= 2:
            mb, v = np.array(mask_pts).T
            slope = np.polyfit(mb, v, 1)[0]                     # per nm of per-edge bias
            sign = 1 if row["kind"] == "width" else -1
            out["MEEF"] = round(float(sign * slope / 2.0), 3)   # mask CD changes by 2x per-edge bias
        foc = [(P[c]["focus_nm"], cd[c]) for c in cd if c in P and same(c, focus_nm=P[c]["focus_nm"])]
        if len({f for f, _ in foc}) >= 3:
            f, v = np.array(foc).T
            out["focus_curvature_nm_per_100nm2"] = round(float(np.polyfit(f / 100.0, v, 2)[0]), 3)
            out["focus_tilt_nm_per_100nm"] = round(float(np.polyfit(f / 100.0, v, 2)[1]), 3)
    return out


def pullback(ends, prof, conditions, depth_nm=40.0):
    """Printed line-end position relative to the drawn end, per condition (> 0: pulled back)."""
    out = []
    for e in ends:
        fx, fy = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.get(e["facing"], (0, 0))
        if (fx, fy) == (0, 0):
            continue
        gap = e["gap_nm"] if e["gap_nm"] else 80.0
        p0 = (e["x_nm"] - fx * depth_nm, e["y_nm"] - fy * depth_nm)
        p1 = (e["x_nm"] + fx * gap / 2, e["y_nm"] + fy * gap / 2)
        pb = {}
        for c in conditions:
            s, v = prof.profile(f"sdf:{c}", p0, p1)
            ext = run_length(s, v, 0.5, positive=True) if v[0] > 0 else 0.0
            pb[c] = round(depth_nm - ext, 2)
        vals = np.array(list(pb.values()))
        out.append({**{k: e[k] for k in ("poly", "x_nm", "y_nm", "facing", "gap_nm", "gap_kind")},
                    "pullback_nm": pb, "max_pullback_nm": round(float(vals.max()), 2),
                    "min_pullback_nm": round(float(vals.min()), 2),
                    "worst_condition": max(pb, key=pb.get)})
    return out


def corner_rounding(edges_by_poly, prof, condition, frame, max_nm=40.0):
    """Inset of the printed contour along the corner bisector; radius = inset / (sqrt2 - 1)."""
    out = []
    fx0, fy0, fx1, fy1 = frame
    for pid, es_all in edges_by_poly.items():
      for ring in sorted({e.ring for e in es_all}):
        es = [e for e in es_all if e.ring == ring]
        for i, e in enumerate(es):
            prev = es[i - 1]
            c = np.asarray(e.p)
            if not (fx0 + 2 < c[0] < fx1 - 2 and fy0 + 2 < c[1] < fy1 - 2):
                continue
            bis = np.asarray(prev.normal) + np.asarray(e.normal)
            if np.hypot(*bis) < 1e-6:
                continue
            bis = bis / np.hypot(*bis)
            if e.convex_p:
                p0, p1 = tuple(c + bis * 1e-3), tuple(c - bis * max_nm)          # inward
                s, v = prof.profile(f"sdf:{condition}", p0, p1)
                k = np.argmax(v > 0) if (v > 0).any() else None
                inset = float(s[k]) if k is not None else max_nm
                out.append({"poly": pid, "x_nm": round(c[0], 1), "y_nm": round(c[1], 1), "type": "convex",
                            "inset_nm": round(inset, 2), "equiv_radius_nm": round(inset / (np.sqrt(2) - 1), 1)})
            else:
                p0, p1 = tuple(c - bis * 1e-3), tuple(c + bis * max_nm)          # outward into the notch
                s, v = prof.profile(f"sdf:{condition}", p0, p1)
                k = np.argmax(v < 0) if (v < 0).any() else None
                fill = float(s[k]) if k is not None else max_nm
                out.append({"poly": pid, "x_nm": round(c[0], 1), "y_nm": round(c[1], 1), "type": "concave",
                            "fill_nm": round(fill, 2), "equiv_radius_nm": round(fill / (np.sqrt(2) - 1), 1)})
    return out


def edge_epe(edges_by_poly, prof, conditions, nominal, frame, samples=(0.25, 0.5, 0.75)):
    """EPE per target edge: sdf of the printed pattern at points on the drawn edge."""
    out = []
    fx0, fy0, fx1, fy1 = frame
    for pid, es in edges_by_poly.items():
        for e in es:
            m = e.mid
            if not (fx0 + 1 < m[0] < fx1 - 1 and fy0 + 1 < m[1] < fy1 - 1):
                continue
            pts = [np.asarray(e.p) + t * (np.asarray(e.q) - np.asarray(e.p)) for t in samples]
            vals = {}
            for c in conditions:
                f = prof.fields[f"sdf:{c}"]
                rc = np.array([[p[1] / prof.px - 0.5, p[0] / prof.px - 0.5] for p in pts])
                vals[c] = float(np.mean(geo.bilinear(f, rc)))
            v = np.array(list(vals.values()))
            out.append({"poly": pid, "edge": e.idx, "kind": e.kind, "x_nm": round(m[0], 1), "y_nm": round(m[1], 1),
                        "length_nm": round(e.length, 1), "epe_nominal_nm": round(vals[nominal], 2),
                        "epe_min_nm": round(float(v.min()), 2), "epe_max_nm": round(float(v.max()), 2),
                        "pv_band_nm": round(float(np.ptp(v)), 2)})
    return out
