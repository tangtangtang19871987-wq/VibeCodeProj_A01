"""Mask reading: correspondence to the target, bias along edge normals, line-end treatment,
assist features (detached) and appendages (attached beyond a band), cross-target mask
connections, and complexity measures.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops

from .. import geometry as geo
from .polylayer import rect_decomposition_count


def correspondence(target, mask_bin, pixel_nm, frame_cut_ok=False):
    """Map mask components <-> target components."""
    lt = label(target, connectivity=1)
    lm = label(mask_bin, connectivity=1)
    frame = np.zeros(mask_bin.shape, bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    cut = set(np.unique(lm[frame])) - {0}
    m2t, t2m = {}, {}
    both = (lm > 0) & (lt > 0)
    for m, t in set(zip(lm[both].tolist(), lt[both].tolist())):
        m2t.setdefault(m, set()).add(t)
        t2m.setdefault(t, set()).add(m)
    out = {"n_mask_components": int(lm.max()), "n_target_components": int(lt.max()),
           "detached": [], "merging": [], "targets_without_mask": []}
    for reg in regionprops(lm):
        m = reg.label
        ts = m2t.get(m, set())
        r, c = reg.centroid
        if not ts and (m not in cut or frame_cut_ok):
            dist = ndi.distance_transform_edt(lt == 0)[lm == m].min() * pixel_nm
            out["detached"].append({"mask_comp": m, "x_nm": round((c + .5) * pixel_nm, 1),
                                    "y_nm": round((r + .5) * pixel_nm, 1), "area_nm2": round(reg.area * pixel_nm ** 2, 1),
                                    "width_nm": round(2 * ndi.distance_transform_edt(lm == m).max() * pixel_nm, 1),
                                    "length_nm": round(reg.axis_major_length * pixel_nm, 1),
                                    "distance_to_target_nm": round(float(dist), 1),
                                    "orientation_deg": round(float(np.degrees(reg.orientation)), 1)})
        if len(ts) >= 2:
            # where the mask joins two targets: mask pixel outside the targets, equidistant
            tl = sorted(ts)
            da = ndi.distance_transform_edt(lt != tl[0]); db = ndi.distance_transform_edt(lt != tl[1])
            score = np.where((lm == m) & (lt == 0), np.abs(da - db) + 0.5 * (da + db), np.inf)
            k = np.unravel_index(np.argmin(score), score.shape)
            # neck: thickest inscribed disk on the mid-surface between the two targets
            mid = (lm == m) & (lt == 0) & (np.abs(da - db) <= 1.0)
            edt = ndi.distance_transform_edt(lm == m)
            neck = (2 * edt[mid].max() - 1) * pixel_nm if mid.any() else 0.0
            out["merging"].append({"mask_comp": m, "targets": tl, "x_nm": round((k[1] + .5) * pixel_nm, 1),
                                   "y_nm": round((k[0] + .5) * pixel_nm, 1), "neck_width_nm": round(float(neck), 1),
                                   "target_gap_nm": round(float((da[k] + db[k]) * pixel_nm), 1)})
    for t in range(1, lt.max() + 1):
        if t not in t2m:
            r, c = np.argwhere(lt == t).mean(0)
            out["targets_without_mask"].append({"target": t, "x_nm": round((c + .5) * pixel_nm, 1),
                                                "y_nm": round((r + .5) * pixel_nm, 1)})
    return out


def bias_along_normals(target, mask_bin, pixel_nm, samples_PN, valid):
    """Mask edge displacement at target-edge samples (> 0: mask outside target = positive bias)."""
    P, N = samples_PN
    sd = geo.signed_distance(mask_bin, pixel_nm)
    return geo.bilinear(sd, P[valid]), P[valid]


def appendages(target, mask_bin, pixel_nm, band_nm=15.0, min_area_nm2=40.0):
    """Mask material farther than band_nm from any target but connected to a main mask shape."""
    far = ndi.distance_transform_edt(~target.astype(bool)) * pixel_nm > band_nm
    lm = label(mask_bin, connectivity=1)
    lt = label(target, connectivity=1)
    main = set(np.unique(lm[lt > 0])) - {0}
    cand = label(mask_bin & far & np.isin(lm, list(main)), connectivity=1)
    out = []
    for reg in regionprops(cand):
        if reg.area * pixel_nm ** 2 < min_area_nm2:
            continue
        r, c = reg.centroid
        out.append({"x_nm": round((c + .5) * pixel_nm, 1), "y_nm": round((r + .5) * pixel_nm, 1),
                    "area_nm2": round(reg.area * pixel_nm ** 2, 1),
                    "max_reach_from_target_nm": round(float((ndi.distance_transform_edt(~target.astype(bool))
                                                             [cand == reg.label]).max() * pixel_nm), 1)})
    return sorted(out, key=lambda d: -d["area_nm2"])


def _walk(sd, pixel_nm, x, y, dx, dy, max_nm=150.0):
    """Distance (nm) from (x, y) along (dx, dy) until the signed-distance field turns negative."""
    H, W = sd.shape
    step, t = pixel_nm / 4, 0.0
    while t < max_nm:
        qx, qy = x + dx * (t + step), y + dy * (t + step)
        rc = np.array([[qy / pixel_nm - 0.5, qx / pixel_nm - 0.5]])
        if not (0 <= rc[0, 0] < H - 1 and 0 <= rc[0, 1] < W - 1) or geo.bilinear(sd, rc)[0] < 0:
            break
        t += step
    return t


def line_end_treatment(mask_bin, pixel_nm, ends, probe_nm=(-8.0, 0.0, 8.0)):
    """For each target line end: mask extension beyond the end and mask width just inside /
    at / beyond the end edge (hammerhead ratio = widest probe / line width)."""
    sd = geo.signed_distance(mask_bin, pixel_nm)
    out = []
    for e in ends:
        fx, fy = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}.get(e["facing"], (0, 0))
        if (fx, fy) == (0, 0):
            continue
        x, y = e["x_nm"], e["y_nm"]
        ext = _walk(sd, pixel_nm, x, y, fx, fy, 120.0)
        px_, py_ = (0, 1) if fx else (1, 0)                    # perpendicular to the facing
        widths = {}
        for off in probe_nm:
            cx, cy = x + fx * off, y + fy * off
            w = _walk(sd, pixel_nm, cx, cy, px_, py_) + _walk(sd, pixel_nm, cx, cy, -px_, -py_)
            widths[f"mask_width_at_{off:+.0f}nm"] = round(w, 1)
        out.append({**{k: e[k] for k in ("poly", "x_nm", "y_nm", "facing", "width_nm")},
                    "mask_extension_nm": round(ext, 1), **widths,
                    "hammerhead_ratio": round(max(widths.values()) / max(e["width_nm"], 1e-6), 2)})
    return out


def complexity(mask_bin, pixel_nm):
    pts, k = geo.contour_curvature(mask_bin, pixel_nm)
    polys = geo.polygons(mask_bin, pixel_nm, simplify_nm=pixel_nm / 2)
    return {"rect_decomposition_count": int(rect_decomposition_count(mask_bin)),
            "polygons": len(polys),
            "vertices_simplified_total": int(sum(len(p.exterior.coords) for p in polys)),
            "perimeter_nm": round(float(sum(p.length for p in polys)), 1),
            "curvature_abs_p95_per_nm": round(float(np.percentile(np.abs(k), 95)), 4) if len(k) else None,
            "min_radius_nm_p5": round(float(1 / max(np.percentile(np.abs(k), 95), 1e-9)), 1) if len(k) else None,
            "note": "rect_decomposition_count = rectangles of a row-run decomposition at the pixel grid, a "
                    "proxy for VSB shot count; curvature from smoothed contours"}
