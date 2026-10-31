"""Layout reading: polygon classification, dominant direction, pitch, tracks, line ends,
local pattern context.  Works on any PolyLayer + its measured edges.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def dominant_orientation(edges_by_poly):
    """'V' if lines run vertically (long V edges dominate), 'H' otherwise; with the ratio."""
    lh = sum(e.length for es in edges_by_poly.values() for e in es if e.orient == "H")
    lv = sum(e.length for es in edges_by_poly.values() for e in es if e.orient == "V")
    tot = lh + lv + 1e-9
    return ("V" if lv >= lh else "H"), float(max(lh, lv) / tot)


def classify_polygon(poly, edges):
    """line | L-shape | T/cross | pad | island | complex, from rectilinear shape descriptors."""
    n_conc = sum(1 for e in edges if not e.convex_p)
    n_v = len(edges)
    mrr = poly.minimum_rotated_rectangle
    xs, ys = mrr.exterior.coords.xy
    d = sorted({round(float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])), 3) for i in range(4)})
    w, l = (d[0], d[-1]) if len(d) > 1 else (d[0], d[0])
    aspect = l / max(w, 1e-9)
    fill = poly.area / max(mrr.area, 1e-9)
    if n_conc == 0:
        if aspect >= 2.0:
            return "line", w, l, aspect
        return ("pad" if poly.area > 2500 else "island"), w, l, aspect
    if n_conc == 1 and fill > 0.4:
        return "L-shape", w, l, aspect
    if n_conc in (2, 4) and n_v <= 12:
        return ("T-shape" if n_conc == 2 else "cross"), w, l, aspect
    return ("line" if aspect >= 4 and fill > 0.85 else "complex"), w, l, aspect


def polygon_inventory(layer, edges_by_poly):
    out = []
    for i, p in enumerate(layer.polys):
        es = edges_by_poly[i]
        cls, w, l, asp = classify_polygon(p, es)
        side_w = [e.width for e in es if e.kind == "side" and e.width]
        x0, y0, x1, y1 = p.bounds
        out.append({"id": i, "class": cls, "area_nm2": round(p.area, 1), "bbox_nm": [round(v, 1) for v in (x0, y0, x1, y1)],
                    "centroid_nm": [round(p.centroid.x, 1), round(p.centroid.y, 1)],
                    "width_nm": round(float(np.median(side_w)), 1) if side_w else round(w, 1),
                    "min_width_nm": round(float(np.min(side_w)), 1) if side_w else None,
                    "length_nm": round(l, 1), "aspect": round(asp, 2), "n_vertices": len(es),
                    "n_line_ends": sum(1 for e in es if e.kind == "end"),
                    "n_holes": len(p.interiors), "cut_by_frame": layer.touches_frame(i)})
    return out


def estimate_pitch(centers, min_sep=5.0):
    """Pitch from line-centre coordinates: the smallest frequent spacing (robust to empty tracks)."""
    c = np.sort(np.unique(np.round(np.asarray(centers), 1)))
    if len(c) < 2:
        return None, None
    d = np.diff(c)
    d = d[d > min_sep]
    if not len(d):
        return None, None
    base = float(np.percentile(d, 20))
    # refine: average of spacings within 15 % of a multiple of base, divided by that multiple
    est = []
    for v in d:
        k = max(1, round(v / base))
        if abs(v / k - base) <= 0.15 * base:
            est.append(v / k)
    pitch = float(np.mean(est)) if est else base
    offset = float(np.median(np.mod(c, pitch)))
    return pitch, offset


def lines_and_tracks(layer, edges_by_poly, inventory):
    """Lines along the dominant direction with track index, width, length, ends."""
    ori, ratio = dominant_orientation(edges_by_poly)
    lines = []
    for inv in inventory:
        if inv["class"] not in ("line", "L-shape", "T-shape", "complex"):
            continue
        p = layer.polys[inv["id"]]
        x0, y0, x1, y1 = p.bounds
        along = (y1 - y0) if ori == "V" else (x1 - x0)
        across = (x1 - x0) if ori == "V" else (y1 - y0)
        if along < 1.5 * across and inv["class"] == "line":
            continue                                         # a line in the other direction
        center = (x0 + x1) / 2 if ori == "V" else (y0 + y1) / 2
        lines.append({"id": inv["id"], "center_nm": round(center, 2), "width_nm": inv["width_nm"],
                      "length_nm": round(along, 1), "span_nm": [round(y0 if ori == "V" else x0, 1),
                                                                round(y1 if ori == "V" else x1, 1)],
                      "class": inv["class"], "cut_by_frame": inv["cut_by_frame"]})
    pitch, offset = estimate_pitch([l["center_nm"] for l in lines])
    for l in lines:
        l["track"] = int(round((l["center_nm"] - offset) / pitch)) if pitch else None
    return {"orientation": ori, "orientation_purity": round(ratio, 3),
            "pitch_nm": None if pitch is None else round(pitch, 2),
            "track_offset_nm": None if offset is None else round(offset, 2), "lines": lines}


def line_ends(layer, edges_by_poly, frame_margin=1.0, halo_nm=30.0):
    """Every line-end edge with its facing gap and partner type.  Ends inside the frame halo
    are skipped: near the window edge, fragments of cut polygons look like line ends."""
    out = []
    for pid, es in edges_by_poly.items():
        for e in es:
            if e.kind != "end":
                continue
            mx, my = e.mid
            fx0, fy0, fx1, fy1 = layer.frame or (-np.inf, -np.inf, np.inf, np.inf)
            on_frame = (min(abs(mx - fx0), abs(mx - fx1)) < frame_margin and e.orient == "V") or \
                       (min(abs(my - fy0), abs(my - fy1)) < frame_margin and e.orient == "H")
            if on_frame:
                continue                                     # cut by the window, not a real end
            if layer.frame and min(mx - fx0, fx1 - mx, my - fy0, fy1 - my) < halo_nm:
                continue
            facing = {(1, 0): "+x", (-1, 0): "-x", (0, 1): "+y", (0, -1): "-y"}.get(
                (int(round(e.normal[0])), int(round(e.normal[1]))), "diag")
            out.append({"poly": pid, "edge": e.idx, "x_nm": round(mx, 1), "y_nm": round(my, 1),
                        "facing": facing, "width_nm": round(e.length, 1),
                        "gap_nm": None if e.space is None else round(e.space, 1),
                        "gap_kind": e.space_kind, "partner_poly": e.space_to})
    return out


def density_at(binary, pixel_nm, x_nm, y_nm, radii=(50.0, 100.0, 200.0)):
    r = int(round(y_nm / pixel_nm - 0.5)); c = int(round(x_nm / pixel_nm - 0.5))
    out = {}
    for rad in radii:
        rp = max(1, int(round(rad / pixel_nm)))
        r0, r1 = max(0, r - rp), min(binary.shape[0], r + rp + 1)
        c0, c1 = max(0, c - rp), min(binary.shape[1], c + rp + 1)
        yy, xx = np.mgrid[r0:r1, c0:c1]
        disk = (yy - r) ** 2 + (xx - c) ** 2 <= rp * rp
        out[f"r{int(rad)}"] = round(float(binary[r0:r1, c0:c1][disk].mean()), 3)
    return out


def neighbour_context(lines_info, x_nm, y_nm):
    """For a point on/near a line: neighbours on the two adjacent tracks at that position."""
    ori, pitch = lines_info["orientation"], lines_info["pitch_nm"]
    if not pitch:
        return {"context": "unknown"}
    across, along = (x_nm, y_nm) if ori == "V" else (y_nm, x_nm)
    here = min(lines_info["lines"], key=lambda l: abs(l["center_nm"] - across), default=None)
    if here is None:
        return {"context": "empty"}
    t = here["track"]

    def present(track):
        for l in lines_info["lines"]:
            if l["track"] == track and l["span_nm"][0] <= along <= l["span_nm"][1]:
                return True
        return False
    left, right = present(t - 1), present(t + 1)
    ctx = "dense" if (left and right) else ("semi-isolated" if (left or right) else "isolated")
    return {"track": t, "line_poly": here["id"], "neighbour_minus": left, "neighbour_plus": right, "context": ctx}
