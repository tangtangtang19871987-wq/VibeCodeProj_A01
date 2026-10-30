"""Shared experiment helpers: run + hard evaluation + JSON logging."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from clips import conflict_clip, dense_clip, random_clip  # noqa: E402
from ilt import Problem, Recipe, Spec, default_corners     # noqa: E402

FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SUMMARY_KEYS = ("n_hot", "n_hot_sites", "n_bridge_all", "n_pinch_all", "topo_bridge", "topo_open",
                "min_bridge_all", "min_pinch_all", "n_pinch", "n_bridge", "n_t2t", "n_diag",
                "n_lend", "sraf_print", "zone_bridge", "zone_pinch")

# the reference "good" recipe used as the starting point of most experiments
GOOD = dict(iters=300, w_fid=1.0, w_bridge=10.0, w_pinch=10.0, w_sraf=50.0)


def summarize(r: dict) -> dict:
    return {k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else int(r[k]))
            for k in SUMMARY_KEYS if k in r}


def run_eval(P: Problem, rc: Recipe, theta0=None, topo=True):
    t = time.time()
    theta, rho, hist = P.run(rc, theta0=theta0)
    r = P.evaluate(rho, topo=topo)
    r["runtime_s"] = time.time() - t
    return theta, rho, hist, r


def save_json(name, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=conv)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_good(seed, clip=None, recipe_kw=None, tag="good"):
    """Cached reference solution for a clip (theta, rho, eval dict, Problem)."""
    import torch
    clip = clip or random_clip(seed)
    P = Problem(clip)
    path = os.path.join(RES, f"cache_{tag}_{clip.name}.pt")
    if os.path.exists(path):
        theta = torch.load(path)
        rc = Recipe(**(recipe_kw or GOOD))
        rho = P.rho(theta, rc.R, rc.freeze, rc.x_frozen)
    else:
        rc = Recipe(**(recipe_kw or GOOD))
        theta, rho, _ = P.run(rc)
        torch.save(theta, path)
    return P, theta, rho, P.evaluate(rho)


def precise_lengths(P, I):
    """Per-site physical length with linearly interpolated crossings."""
    import torch
    from layout import measure  # noqa
    v = P.S.sample(I).numpy()
    step = P.S.step.numpy()
    gap = np.isin(P.kind, ["bridge", "t2t", "diag"])
    out = np.zeros(len(P.kind))
    for i in range(len(P.kind)):
        ns = P.S.nsamp[i]; y = v[i, :ns] - 1.0; h = step[i]
        on = y > 0
        tot = on.sum() * h
        for j in np.nonzero(np.diff(np.sign(y)))[0]:
            f = y[j] / (y[j] - y[j + 1])
            tot += (f - 0.5) * h if on[j] else (0.5 - f) * h
        out[i] = ns * h - tot if gap[i] else tot
    return out


def sensitivities(P, rho):
    """|dL/dD| (nm per 100 % dose) and MEEF (dL / d(mask CD)) for every site at nominal."""
    I = P.images(P.hard_mask(rho), [0])[0]
    dose = (precise_lengths(P, I * 1.01) - precise_lengths(P, I * 0.99)) / 0.02
    Ip = P.images(P.hard_mask(rho, 0.5), [0])[0]
    Im = P.images(P.hard_mask(rho, -0.5), [0])[0]
    # mask CD changes by 2 * 0.5 nm per 1 nm of bias step -> divide by 2 * 1.0
    meef = (precise_lengths(P, Ip) - precise_lengths(P, Im)) / 2.0
    return np.abs(dose), np.abs(meef)


def dense_mask_sites(P):
    """True for pinch sites whose two neighbour tracks both carry a line at that y."""
    from layout import y_in_seg
    clip = P.clip
    out = np.zeros(len(P.kind), bool)
    for i, (k, p0) in enumerate(zip(P.kind, P.sites.p0)):
        if k != "pinch":
            continue
        y = p0[1]
        tr = int(round((p0[0] + clip.pitch / 2) / clip.pitch - 0.5)) % clip.tracks
        nb = [(tr - 1) % clip.tracks, (tr + 1) % clip.tracks]
        has = [any(s.track == t and y_in_seg(y, s, clip.L) for s in clip.segs) for t in nb]
        out[i] = all(has)
    return out
