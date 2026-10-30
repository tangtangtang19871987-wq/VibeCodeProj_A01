"""E4  Raising a bridge weight: global area term vs per-site linear vs per-site hinge.

Claim tested: "a global / always-active bridge penalty shrinks the mask
everywhere and creates pinch; a per-site hinge that switches off once a site
is safe does not".  For each formulation the pinch protection is held fixed
and only the bridge weight is swept.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, Problem, Recipe, log, random_clip, run_eval, save_json, summarize

FAM = {
    "global area (outer PV band)": (lambda w: dict(w_out=w, w_in=2.0), [0, 0.5, 2, 8, 32]),
    "per-site linear (always on)": (lambda w: dict(w_bridge=w, w_pinch=10.0, hinge="linear"), [0, 0.5, 2, 8, 32]),
    "per-site hinge":              (lambda w: dict(w_bridge=w, w_pinch=10.0), [0, 2, 10, 50, 250]),
}
rows = []
for seed in [1, 2, 3]:
    P = Problem(random_clip(seed))
    for fam, (mk, ws) in FAM.items():
        for w in ws:
            rc = Recipe(iters=300, w_fid=1.0, w_sraf=50.0, **mk(w))
            _, rho, hist, r = run_eval(P, rc)
            row = dict(seed=seed, family=fam, w=w, **summarize(r))
            rows.append(row)
            log(seed, fam, w, {k: row[k] for k in ("n_bridge_all", "n_pinch_all", "topo_bridge", "topo_open",
                                                  "min_bridge_all", "min_pinch_all")})
            save_json("e04.json", rows)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (fam, (mk, ws)) in zip(axs, FAM.items()):
    R = [r for r in rows if r["family"] == fam]
    for key, c, lab in (("bridge", "#1f77b4", "bridge-type hotspots"), ("pinch", "#d62728", "pinch-type hotspots")):
        y = np.array([[np.mean([r[f"n_{key}_all"] + (r["topo_bridge"] if key == "bridge" else r["topo_open"])
                                for r in R if r["w"] == w and r["seed"] == s]) for w in ws] for s in (1, 2, 3)])
        ax.plot(range(len(ws)), y.mean(0), "o-", c=c, label=lab)
        ax.fill_between(range(len(ws)), y.min(0), y.max(0), color=c, alpha=0.15)
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([str(w) for w in ws])
    ax.set_xlabel("bridge weight"); ax.set_ylabel("hotspots (mean, band = min..max over 3 clips)")
    ax.set_title(fam, fontsize=9); ax.legend(fontsize=7); ax.set_yscale("symlog")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e04_global_vs_local.png"), dpi=130)
