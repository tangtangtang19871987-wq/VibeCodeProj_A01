"""DRC-style statistics and rule checks from measured edges."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point


def _stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "min": round(float(v.min()), 1), "p5": round(float(np.percentile(v, 5)), 1),
            "median": round(float(np.median(v)), 1), "max": round(float(v.max()), 1)}


def width_space_summary(edges_by_poly):
    allE = [e for es in edges_by_poly.values() for e in es]
    out = {"width_side_edges": _stats([e.width for e in allE if e.kind == "side"]),
           "line_end_widths": _stats([e.length for e in allE if e.kind == "end"])}
    for k in ("side-side", "end-end", "end-side", "notch"):
        out[f"space_{k}"] = _stats([e.space for e in allE if e.space_kind == k])
    return out


def corner_to_corner(layer, edges_by_poly, max_nm=120.0):
    """Distance from every convex corner to the nearest *other* polygon (diagonal spacing)."""
    res = []
    for pid, es in edges_by_poly.items():
        for e in es:
            if not e.convex_p:
                continue
            pt = Point(e.p)
            cand = [i for i in layer.query(pt.buffer(max_nm)) if i != pid]
            if not cand:
                continue
            j = min(cand, key=lambda i: layer.polys[i].distance(pt))
            d = layer.polys[j].distance(pt)
            if d <= max_nm:
                res.append({"poly": pid, "x_nm": round(e.p[0], 1), "y_nm": round(e.p[1], 1),
                            "to_poly": j, "distance_nm": round(float(d), 1)})
    return sorted(res, key=lambda r: r["distance_nm"])


def rule_check(layer, edges_by_poly, min_width=None, min_space=None, min_t2t=None, min_e2s=None,
               min_c2c=None, min_area=None, min_edge=None, limit=30):
    """Violations of the given rules (None = rule not checked)."""
    v = []
    for pid, es in edges_by_poly.items():
        cut = layer.touches_frame(pid)
        for e in es:
            m = e.mid
            if min_width and e.kind == "side" and e.width is not None and e.width < min_width:
                v.append(("width", pid, m, e.width, min_width))
            if e.space is not None:
                lim = {"side-side": min_space, "notch": min_space, "end-end": min_t2t or min_space,
                       "end-side": min_e2s or min_space}.get(e.space_kind)
                if lim and e.space < lim:
                    # report at the centre of the space so both facing edges map to one point
                    c = (m[0] + e.normal[0] * e.space / 2, m[1] + e.normal[1] * e.space / 2)
                    v.append((f"space:{e.space_kind}", pid, c, e.space, lim))
            if min_edge and e.length < min_edge and not cut:
                v.append(("edge_length", pid, m, e.length, min_edge))
    if min_c2c:
        for c in corner_to_corner(layer, edges_by_poly, max_nm=min_c2c):
            if c["distance_nm"] < min_c2c:
                v.append(("corner-corner", c["poly"], (c["x_nm"], c["y_nm"]), c["distance_nm"], min_c2c))
    if min_area:
        for i, p in enumerate(layer.polys):
            if p.area < min_area and not layer.touches_frame(i):
                v.append(("area", i, (p.centroid.x, p.centroid.y), p.area, min_area))
    # de-duplicate the two edges of one space (both sides report it)
    seen, out = set(), []
    for rule, pid, m, val, lim in sorted(v, key=lambda t: t[3] / t[4]):
        key = (rule, round(m[0] / 3), round(m[1] / 3), round(val, 0))
        if key in seen:
            continue
        seen.add(key)
        out.append({"rule": rule, "poly": pid, "x_nm": round(m[0], 1), "y_nm": round(m[1], 1),
                    "value": round(float(val), 1), "limit": lim})
    counts = {}
    for o in out:
        counts[o["rule"]] = counts.get(o["rule"], 0) + 1
    return {"counts": counts, "violations": out[:limit], "total": len(out)}
