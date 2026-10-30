"""E10  Steepness of the soft threshold used to measure site lengths.

Claim tested: "too steep a sigmoid saturates (zero gradient away from the
edge), too soft a sigmoid mis-measures widths; both hurt convergence".
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, summarize

rows = []
for seed in (1, 2):
    P = Problem(random_clip(seed))
    low = P.lower.numpy(); isb = np.isin(P.kind, ["bridge", "t2t", "diag"])
    for a in (2, 6, 15, 60, 240):
        rc = Recipe(**{**GOOD, "alpha_site": float(a)})
        _, rho, hist, r = run_eval(P, rc)
        # soft (cost-side) vs hard (checker) worst margins: measurement bias of the relaxation
        q = hist[-1]["q"] - low
        rows.append(dict(seed=seed, alpha=a, soft_min_bridge=float(q[isb].min()), soft_min_pinch=float(q[~isb].min()),
                         soft_vs_hard_mae=float(np.mean(np.abs(q - r["margin"]))), **summarize(r)))
        log(seed, a, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()
                      if k in ("n_hot", "n_bridge_all", "n_pinch_all", "soft_vs_hard_mae", "min_bridge_all", "min_pinch_all")})
        save_json("e10.json", rows)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
A = sorted({r["alpha"] for r in rows})
for seed, m in ((1, "o"), (2, "s")):
    R = [r for r in rows if r["seed"] == seed]
    axs[0].plot(A, [r["n_hot"] for r in R], m + "-", label=f"clip {seed}: hotspots")
    axs[1].plot(A, [r["soft_vs_hard_mae"] for r in R], m + "-", label=f"clip {seed}")
axs[0].set_xscale("log"); axs[1].set_xscale("log"); axs[0].set_xlabel("site sigmoid steepness alpha")
axs[1].set_xlabel("alpha"); axs[0].legend(fontsize=7); axs[1].legend(fontsize=7)
axs[0].set_title("final hotspots", fontsize=9); axs[1].set_title("|soft site value - hard checker| (nm)", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e10_sigmoid_steepness.png"), dpi=130)
