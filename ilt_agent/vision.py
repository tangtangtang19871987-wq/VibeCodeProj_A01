"""Computer-vision layer: dense feature maps, tiled descriptors, anomaly scoring,
context signatures for hotspot clustering, and overlay rendering.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

COND_COLORS = ["#1f77b4", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def ils_map(intensity, pixel_nm):
    """Image log-slope |grad I| / I (1/nm) of the normalised resist image."""
    gy, gx = np.gradient(intensity.astype(float), pixel_nm)
    return np.hypot(gx, gy) / np.maximum(intensity, 1e-6)


def edge_band(binary, width_px=1):
    b = binary.astype(bool)
    return b ^ ndi.binary_erosion(b, iterations=width_px)


def tile_grid(shape, pixel_nm, tile_nm):
    t = max(4, int(round(tile_nm / pixel_nm)))
    rows = list(range(0, shape[0], t))
    cols = list(range(0, shape[1], t))
    return t, [(r, c, min(r + t, shape[0]), min(c + t, shape[1])) for r in rows for c in cols]


def robust_z(X):
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826
    scale = np.where(mad > 1e-9, mad, X.std(axis=0) + 1e-9)
    return (X - med) / scale


def anomaly_scores(X, seed=0):
    """Combine robust per-feature z-scores with an IsolationForest score.

    Returns (score, per-feature |z|).  Score in [0, 1]-ish: higher = more unusual."""
    Z = np.abs(robust_z(X))
    if len(X) >= 8:
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
        s_iso = -iso.score_samples(X)                         # larger = more anomalous
        s_iso = (s_iso - s_iso.min()) / (np.ptp(s_iso) + 1e-12)
    else:
        s_iso = np.zeros(len(X))
    s_z = np.clip(Z.max(1) / 6.0, 0, 1)
    return 0.5 * s_iso + 0.5 * s_z, Z


def context_signature(target, row, col, radius_px, size=16):
    """Down-sampled target window around (row, col): a pattern fingerprint."""
    pad = np.pad(target.astype(float), radius_px, mode="constant")
    win = pad[row:row + 2 * radius_px + 1, col:col + 2 * radius_px + 1]
    zoom = size / win.shape[0]
    return ndi.zoom(win, zoom, order=1)[:size, :size].ravel()


def cluster_signatures(sigs, eps=0.15):
    """DBSCAN on normalised signatures (cosine-like distance)."""
    if len(sigs) == 0:
        return np.zeros(0, int)
    S = np.asarray(sigs, float)
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-9)
    return DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit_predict(S)


def render(bundle, path, roi_px=None, conditions=None, markers=(), title="", show_mask=True,
           heat=None, heat_label=None):
    """PNG overlay: mask (grey), target outline (green), printed contours per condition,
    optional heat map and labelled markers [(row, col, label, color)]."""
    h, w = bundle.shape
    r0, c0, r1, c1 = roi_px or (0, 0, h, w)
    px = bundle.pixel_nm
    ext = [c0 * px, c1 * px, r0 * px, r1 * px]
    fig, ax = plt.subplots(figsize=(6.4, 6.4 * (r1 - r0) / max(c1 - c0, 1) + 0.4))
    if show_mask:
        ax.imshow(bundle.mask[r0:r1, c0:c1], origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    if heat is not None:
        im = ax.imshow(heat[r0:r1, c0:c1], origin="lower", extent=ext, cmap="magma", alpha=0.55)
        fig.colorbar(im, ax=ax, fraction=0.04, label=heat_label or "")
    ax.contour(bundle.target[r0:r1, c0:c1].astype(float), [0.5], origin="lower", extent=ext,
               colors="#2ca02c", linewidths=1.2)
    conds = conditions or [bundle.nominal]
    for k, name in enumerate(conds):
        ax.contour(bundle.printed[name][r0:r1, c0:c1].astype(float), [0.5], origin="lower", extent=ext,
                   colors=COND_COLORS[k % len(COND_COLORS)], linewidths=0.9)
        ax.plot([], [], color=COND_COLORS[k % len(COND_COLORS)], label=f"printed: {name}")
    ax.plot([], [], color="#2ca02c", label="target")
    for (r, c, lab, col) in markers:
        if r0 <= r < r1 and c0 <= c < c1:
            x, y = (c + 0.5) * px, (r + 0.5) * px
            ax.plot(x, y, "o", mfc="none", mec=col, ms=14, mew=2)
            ax.text(x, y, f" {lab}", color=col, fontsize=8, weight="bold")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return path
