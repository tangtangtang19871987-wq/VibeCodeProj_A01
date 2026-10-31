"""Computational-geometry primitives on rasters (all lengths in nm).

Distances use exact Euclidean distance transforms; local width/space use the
medial axis (width = 2 x distance-to-boundary on the skeleton); EPE samples
the target boundary and reads the printed signed-distance field there.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import disk, medial_axis


# ---------------------------------------------------------------- basics
def signed_distance(binary, pixel_nm):
    """> 0 inside, < 0 outside, in nm (pixel-centre approximation)."""
    b = binary.astype(bool)
    inside = ndi.distance_transform_edt(b)
    outside = ndi.distance_transform_edt(~b)
    return (inside - outside + np.where(b, -0.5, 0.5)) * pixel_nm


def polygons(binary, pixel_nm, simplify_nm=0.0):
    """Raster -> shapely polygons (with holes) in nm coordinates."""
    img = binary.astype(np.uint8)
    contours, hier = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    hier = hier[0]
    for i, c in enumerate(contours):
        if hier[i][3] != -1 or len(c) < 3:
            continue                                    # holes handled with their parent
        shell = (c[:, 0, :] + 0.5) * pixel_nm
        holes = []
        j = hier[i][2]
        while j != -1:
            if len(contours[j]) >= 3:
                holes.append((contours[j][:, 0, :] + 0.5) * pixel_nm)
            j = hier[j][0]
        p = Polygon(shell, holes).buffer(0)
        if simplify_nm:
            p = p.simplify(simplify_nm)
        if not p.is_empty:
            out.append(p)
    return out


def polygon_stats(polys):
    if not polys:
        return {"count": 0}
    a = np.array([p.area for p in polys])
    per = np.array([p.length for p in polys])
    nv = np.array([len(p.exterior.coords) for p in polys])
    comp = per ** 2 / (4 * np.pi * np.maximum(a, 1e-9))         # 1 = circle
    return {"count": len(polys), "area_total_nm2": float(a.sum()), "area_min_nm2": float(a.min()),
            "area_median_nm2": float(np.median(a)), "perimeter_total_nm": float(per.sum()),
            "vertices_median": float(np.median(nv)), "compactness_median": float(np.median(comp))}


# ------------------------------------------------------ width / space map
def prune_skeleton(skel, n_iter):
    """Remove end-point branches up to n_iter pixels long (corner spurs of the medial
    axis).  Interior skeleton points, e.g. the axis through a real neck, are kept."""
    sk = skel.copy()
    k = np.ones((3, 3)); k[1, 1] = 0
    for _ in range(int(n_iter)):
        nb = ndi.convolve(sk.astype(np.uint8), k, mode="constant")
        ends = sk & (nb <= 1)
        if not ends.any():
            break
        sk &= ~ends
    return sk


def local_width(binary, pixel_nm, prune_nm=0.0):
    """Skeleton points (row, col) and local width (nm) of the True phase.

    prune_nm > 0 removes medial-axis spurs that run into convex corners: on a spur the
    distance to the boundary falls to zero, which would read as a false narrow width.
    A spur point is narrower than w only within ~w/sqrt(2) of the corner, so pruning
    w_min/sqrt(2) removes every false sub-w_min point of an ordinary corner."""
    skel, dist = medial_axis(binary.astype(bool), return_distance=True)
    if prune_nm > 0:
        skel = prune_skeleton(skel, np.ceil(prune_nm / (np.sqrt(2) * pixel_nm)) + 1)
    r, c = np.nonzero(skel)
    return r, c, 2.0 * dist[r, c] * pixel_nm


def violations_below(binary, pixel_nm, limit_nm, min_extent_px=1):
    """Cluster skeleton points whose local width < limit into hotspots (corner spurs pruned)."""
    r, c, w = local_width(binary, pixel_nm, prune_nm=limit_nm)
    bad = w < limit_nm
    if not bad.any():
        return []
    img = np.zeros(binary.shape, bool)
    img[r[bad], c[bad]] = True
    lab = label(ndi.binary_dilation(img, iterations=2))
    out = []
    for reg in regionprops(lab):
        sel = lab[r[bad], c[bad]] == reg.label
        if sel.sum() < min_extent_px:
            continue
        k = np.argmin(np.where(sel, w[bad], np.inf))
        out.append({"row": int(r[bad][k]), "col": int(c[bad][k]), "value_nm": float(w[bad][k]),
                    "extent_px": int(sel.sum())})
    return sorted(out, key=lambda d: d["value_nm"])


# ------------------------------------------------------------ topology
def connectivity(printed, target):
    """Compare printed vs target components.

    bridges: printed component touching >= 2 target components (k-1 bridges);
             location = printed pixel equidistant to the two target parts.
    opens:   target component covered by 0 (missing) or >= 2 (broken) printed parts.
    extras:  printed components touching no target (e.g. printing SRAF).
    Components cut by the frame are not judged for opens/extras: their other part is
    outside the window (e.g. the periodic neighbour), so their topology is unknown.
    """
    lt = label(target, connectivity=1)
    lp = label(printed, connectivity=1)
    core = target.astype(bool)
    pairs = np.unique(np.stack([lp[core], lt[core]], 1), axis=0)
    pairs = pairs[pairs[:, 0] > 0]
    bridges, opens, extras = [], [], []
    for pc in np.unique(pairs[:, 0]):
        tcs = np.unique(pairs[pairs[:, 0] == pc, 1])
        if len(tcs) < 2:
            continue
        comp = lp == pc
        da = ndi.distance_transform_edt(lt != tcs[0])
        for tb in tcs[1:]:
            db = ndi.distance_transform_edt(lt != tb)
            score = np.where(comp & ~core, np.abs(da - db) + 0.5 * (da + db), np.inf)
            k = np.unravel_index(np.argmin(score), score.shape)
            bridges.append({"row": int(k[0]), "col": int(k[1]), "targets": [int(tcs[0]), int(tb)]})
    for tc in np.unique(lt[lt > 0]):
        pcs = np.unique(pairs[pairs[:, 1] == tc, 0])
        if len(pcs) == 1:
            continue
        reg = np.argwhere(lt == tc)
        if len(pcs) == 0:
            rr, cc = reg.mean(0)
            opens.append({"row": int(rr), "col": int(cc), "kind": "missing", "target": int(tc)})
        else:
            # gap point: target pixel not printed, closest to two printed parts
            unp = (lt == tc) & ~printed
            if unp.any():
                d = [ndi.distance_transform_edt(lp != p) for p in pcs[:2]]
                score = np.where(unp, np.abs(d[0] - d[1]) + d[0], np.inf)
                k = np.unravel_index(np.argmin(score), score.shape)
            else:
                k = reg.mean(0).astype(int)
            opens.append({"row": int(k[0]), "col": int(k[1]), "kind": "broken", "target": int(tc),
                          "pieces": int(len(pcs))})
    frame = np.zeros(printed.shape, bool)
    frame[0, :] = frame[-1, :] = frame[:, 0] = frame[:, -1] = True
    cut_p = set(np.unique(lp[frame])) - {0}                     # components cut by the frame
    cut_t = set(np.unique(lt[frame])) - {0}
    opens = [o for o in opens if o["target"] not in cut_t]
    for pc in set(np.unique(lp[lp > 0])) - set(pairs[:, 0].tolist()) - cut_p:
        reg = np.argwhere(lp == pc)
        rr, cc = reg.mean(0)
        extras.append({"row": int(rr), "col": int(cc), "area_px": int(len(reg))})
    return {"bridges": bridges, "opens": opens, "extras": extras,
            "n_target": int(lt.max()), "n_printed": int(lp.max())}


# ------------------------------------------------------------------ EPE
def target_edge_samples(target, pixel_nm, step_nm=4.0):
    """Points on the target boundary every ~step_nm, with outward normals."""
    pts, nrm = [], []
    for cont in find_contours(np.pad(target.astype(float), 1), 0.5):
        cont = cont - 1.0                                        # (row, col) sub-pixel
        seg = np.diff(cont, axis=0)
        L = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        if L[-1] * pixel_nm < 2 * step_nm:
            continue
        s = np.arange(0, L[-1], step_nm / pixel_nm)
        r = np.interp(s, L, cont[:, 0]); c = np.interp(s, L, cont[:, 1])
        dr = np.gradient(r); dc = np.gradient(c)
        n = np.stack([dc, -dr], 1)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        pts.append(np.stack([r, c], 1)); nrm.append(n)
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    P, N = np.concatenate(pts), np.concatenate(nrm)
    # orient normals outward: stepping along +n must leave the target
    probe = np.clip(np.round(P + 1.5 * N).astype(int), 0, np.array(target.shape) - 1)
    flip = target[probe[:, 0], probe[:, 1]]
    N[flip] *= -1
    return P, N


def bilinear(img, rc):
    r, c = rc[:, 0], rc[:, 1]
    r0 = np.clip(np.floor(r).astype(int), 0, img.shape[0] - 2)
    c0 = np.clip(np.floor(c).astype(int), 0, img.shape[1] - 2)
    fr, fc = r - r0, c - c0
    return (img[r0, c0] * (1 - fr) * (1 - fc) + img[r0 + 1, c0] * fr * (1 - fc) +
            img[r0, c0 + 1] * (1 - fr) * fc + img[r0 + 1, c0 + 1] * fr * fc)


def epe(printed, target, pixel_nm, step_nm=4.0, samples=None):
    """Signed EPE at target-edge samples: > 0 printed edge outside target (bloat),
    < 0 inside (pull-back).  Value = printed signed distance at the target edge."""
    P, N = samples if samples is not None else target_edge_samples(target, pixel_nm, step_nm)
    if len(P) == 0:
        return P, np.zeros(0)
    sd = signed_distance(printed, pixel_nm)
    return P, bilinear(sd, P)


def target_corners(target, pixel_nm):
    """Polygon vertices of the target (rectilinear corners), as (row, col) array."""
    cs, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    pts = [c[:, 0, ::-1] for c in cs if len(c) >= 3]
    return np.concatenate(pts).astype(float) if pts else np.zeros((0, 2))


def separates_features(printed, target, row, col, half_width_px):
    """True if the printed shapes on either side of a narrow space at (row, col) belong to
    different target features (a real bridge risk) rather than one feature folding back
    on itself (a notch)."""
    lt = label(target, connectivity=1)
    _, idx = ndi.distance_transform_edt(lt == 0, return_indices=True)
    nearest = lt[idx[0], idx[1]]
    r = int(np.ceil(half_width_px)) + 2
    r0, r1 = max(0, row - r), min(target.shape[0], row + r + 1)
    c0, c1 = max(0, col - r), min(target.shape[1], col + r + 1)
    win = printed[r0:r1, c0:c1]
    labs = np.unique(nearest[r0:r1, c0:c1][win])
    return len(labs[labs > 0]) >= 2


# ------------------------------------------------------------- PV band
def pv_band(printed_dict):
    stack = np.stack(list(printed_dict.values()))
    union, inter = stack.any(0), stack.all(0)
    return union & ~inter, union, inter


# ----------------------------------------------------------------- MRC
def mrc(mask_binary, pixel_nm, min_width_nm, min_space_nm, min_area_nm2):
    b = mask_binary.astype(bool)
    rw = max(1, int(round(min_width_nm / 2 / pixel_nm)))
    rs = max(1, int(round(min_space_nm / 2 / pixel_nm)))
    opened = ndi.binary_opening(b, structure=disk(rw))
    closed = ndi.binary_closing(b, structure=disk(rs))
    width_v = b & ~opened
    space_v = closed & ~b
    lab = label(b, connectivity=1)
    small = [r for r in regionprops(lab) if r.area * pixel_nm ** 2 < min_area_nm2]

    def blobs(v):
        out = []
        for reg in regionprops(label(v)):
            rr, cc = reg.centroid
            out.append({"row": int(rr), "col": int(cc), "area_nm2": float(reg.area * pixel_nm ** 2)})
        return sorted(out, key=lambda d: -d["area_nm2"])
    return {"width": blobs(width_v), "space": blobs(space_v),
            "small_islands": [{"row": int(r.centroid[0]), "col": int(r.centroid[1]),
                               "area_nm2": float(r.area * pixel_nm ** 2)} for r in small]}


def contour_curvature(binary, pixel_nm, smooth_px=2.0):
    """Curvature (1/nm) along every contour of a binary image: (points[row,col], kappa)."""
    pts, ks = [], []
    for cont in find_contours(np.pad(binary.astype(float), 1), 0.5):
        cont = cont - 1.0
        if len(cont) < 9:
            continue
        r = ndi.gaussian_filter1d(cont[:, 0], smooth_px, mode="wrap")
        c = ndi.gaussian_filter1d(cont[:, 1], smooth_px, mode="wrap")
        dr, dc = np.gradient(r), np.gradient(c)
        ddr, ddc = np.gradient(dr), np.gradient(dc)
        k = (dc * ddr - dr * ddc) / np.maximum((dr ** 2 + dc ** 2) ** 1.5, 1e-9) / pixel_nm
        pts.append(np.stack([r, c], 1)); ks.append(k)
    if not pts:
        return np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(pts), np.concatenate(ks)
