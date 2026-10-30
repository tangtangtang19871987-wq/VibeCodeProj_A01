"""E7  Frozen vs released SRAF, and hand-designed frozen SRAF.

Claims tested: (a) "with SRAF frozen the main feature is the only lever, so
released SRAF lowers hotspots"; (b) "SRAF helps semi-isolated edges (lower
1/ILS) but cannot help dense ones (no room)"; (c) "a well designed frozen
SRAF (here: grown by free ILT, then binarized and frozen) recovers most of
the benefit"; (d) rule-based pitch-matched bars help, too wide ones print.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import (FIG, GOOD, Problem, Recipe, dense_mask_sites, log, random_clip, run_eval,
                    save_json, sensitivities, summarize)
from ilt import disk_tf, fft_conv


def main_region(P, d=14.0):
    dil = fft_conv(P.T.double(), torch.tensor(disk_tf(P.g.n, P.g.dx, d)[0])).float().numpy()
    return dil > 0.5


def rule_bars(P, w_s, clearance=40.0, min_len=40.0):
    clip = P.clip; n = P.g.n; L = clip.L
    c = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(c, c, indexing="xy")
    out = np.zeros((n, n), np.float32)
    for k in range(clip.tracks):
        segs = sorted([s for s in clip.segs if s.track == k], key=lambda s: s.y0)
        xc = clip.xc(k)
        if not segs:
            runs = [(0.0, L, False)]
        else:
            runs = []
            for a, s in enumerate(segs):
                nx = segs[(a + 1) % len(segs)]
                st = s.y0 + s.length
                ln = (nx.y0 - st) % L
                runs.append((st, ln, True))
        for st, ln, bounded in runs:
            y0, y1 = (st + clearance, st + ln - clearance) if bounded else (st, st + ln)
            if y1 - y0 < min_len:
                continue
            ax = np.abs(((X - xc + L / 2) % L) - L / 2) <= w_s / 2
            t = (Y - y0) % L
            out[ax & (t <= y1 - y0)] = 1.0
    return out


rows = []
for seed in [1, 2, 3]:
    P = Problem(random_clip(seed))
    main = main_region(P)
    freeze = ~main
    dense = dense_mask_sites(P)
    semi = (P.kind == "pinch") & ~dense
    cfg = [("released (free)", None, None),
           ("frozen: none", freeze, np.zeros_like(main, np.float32)),
           ("frozen: rule bars 12 nm", freeze, rule_bars(P, 12.0)),
           ("frozen: rule bars 18 nm", freeze, rule_bars(P, 18.0)),
           ("frozen: rule bars 26 nm", freeze, rule_bars(P, 26.0))]
    theta_free = None
    for name, fz, xf in cfg:
        rc = Recipe(**GOOD, freeze=fz, x_frozen=xf)
        theta, rho, _, r = run_eval(P, rc)
        if fz is None:
            theta_free = theta
        ds, me = sensitivities(P, rho)
        row = dict(seed=seed, config=name, **summarize(r),
                   dense_min_pinch=float(r["margin"][dense].min()),
                   semi_min_pinch=float(r["margin"][semi].min()),
                   dense_med_dsens=float(np.median(ds[dense])), semi_med_dsens=float(np.median(ds[semi])),
                   dense_med_meef=float(np.median(me[dense])), semi_med_meef=float(np.median(me[semi])))
        rows.append(row)
        log(seed, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                         if k in ("n_hot", "n_bridge_all", "n_pinch_all", "sraf_print", "dense_min_pinch",
                                  "semi_min_pinch", "dense_med_dsens", "semi_med_dsens")})
        save_json("e07.json", rows)
    # grown -> binarized -> frozen, main re-optimized from scratch
    xg = (torch.sigmoid(theta_free) > 0.5).float().numpy()
    rc = Recipe(**GOOD, freeze=freeze, x_frozen=xg)
    theta, rho, _, r = run_eval(P, rc)
    ds, me = sensitivities(P, rho)
    row = dict(seed=seed, config="frozen: grown by free ILT", **summarize(r),
               dense_min_pinch=float(r["margin"][dense].min()), semi_min_pinch=float(r["margin"][semi].min()),
               dense_med_dsens=float(np.median(ds[dense])), semi_med_dsens=float(np.median(ds[semi])),
               dense_med_meef=float(np.median(me[dense])), semi_med_meef=float(np.median(me[semi])))
    rows.append(row)
    log(seed, row["config"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                             if k in ("n_hot", "n_bridge_all", "n_pinch_all", "sraf_print", "dense_min_pinch",
                                      "semi_min_pinch", "dense_med_dsens", "semi_med_dsens")})
    save_json("e07.json", rows)

names = list(dict.fromkeys(r["config"] for r in rows))
fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
x = np.arange(len(names))
for k, (key, lab) in enumerate((("n_bridge_all", "bridge-type"), ("n_pinch_all", "pinch-type"))):
    v = [np.mean([r[key] + (r["topo_bridge"] if "bridge" in key else r["topo_open"]) for r in rows if r["config"] == nm]) for nm in names]
    axs[0].bar(x + (k - 0.5) * 0.4, v, 0.4, label=lab, color=["#1f77b4", "#d62728"][k])
axs[0].plot(x, [np.mean([r["sraf_print"] for r in rows if r["config"] == nm]) for nm in names], "k^", label="SRAF prints")
axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7); axs[0].legend(fontsize=7)
axs[0].set_title("hotspots (mean of 3 clips)", fontsize=9)
for k, (a, b, lab) in enumerate((("dense_med_dsens", "semi_med_dsens", "median |dW/dD| (nm per 100% dose)"),
                                 ("dense_min_pinch", "semi_min_pinch", "worst pinch margin (nm)"))):
    ax = axs[1 + k]
    ax.bar(x - 0.2, [np.mean([r[a] for r in rows if r["config"] == nm]) for nm in names], 0.4, label="dense lines")
    ax.bar(x + 0.2, [np.mean([r[b] for r in rows if r["config"] == nm]) for nm in names], 0.4, label="semi-isolated lines")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7); ax.legend(fontsize=7)
    ax.set_title(lab, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e07_sraf_freedom.png"), dpi=130)
