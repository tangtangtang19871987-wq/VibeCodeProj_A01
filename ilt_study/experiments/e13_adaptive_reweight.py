"""E13  Residual hotspots: raise global weights vs hotspot-driven per-site weights.

Claim tested: "after a converged run, raising the global weights moves
violations around; multiplying the weights of only the violating sites
(checker -> cost loop) removes residuals with less collateral damage".
All strategies get the same extra iteration budget, warm-started.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Recipe, get_good, log, run_eval, save_json, summarize

WARM = {**GOOD, "iters": 100, "beta_sched": (16,), "lr": 0.05}
rows = []
for seed in range(1, 7):
    P, theta0, rho0, r0 = get_good(seed)
    base_bad = r0["margin"] < 0
    res = {"start": r0}
    _, _, _, res["continue same"] = run_eval(P, Recipe(**{**WARM, "iters": 200}), theta0=theta0)
    _, _, _, res["global x4"] = run_eval(P, Recipe(**{**WARM, "iters": 200, "w_bridge": 40.0, "w_pinch": 40.0}), theta0=theta0)
    th, sw, r = theta0, np.ones(len(P.kind)), r0
    for rnd in range(2):                                   # 2 rounds x 100 iters
        sw = sw * np.where(r["margin"] < 0.5, 4.0, 1.0)
        th, _, _, r = run_eval(P, Recipe(**{**WARM, "site_weights": sw}), theta0=th)
    res["per-site x4 (2 rounds)"] = r
    for name, rr in res.items():
        m = rr["margin"]
        rows.append(dict(seed=seed, strategy=name, collateral=int(((m < 0) & ~base_bad).sum()),
                         fixed=int(((m >= 0) & base_bad).sum()), **summarize(rr)))
        log(seed, name, {k: rows[-1][k] for k in ("n_hot", "n_bridge_all", "n_pinch_all", "topo_bridge", "collateral", "fixed")})
    save_json("e13.json", rows)
names = ["start", "continue same", "global x4", "per-site x4 (2 rounds)"]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(names))
ax.bar(x - 0.27, [np.sum([r["n_hot"] for r in rows if r["strategy"] == nm]) for nm in names], 0.27, label="total hotspots (6 clips)")
ax.bar(x, [np.sum([r["fixed"] for r in rows if r["strategy"] == nm]) for nm in names], 0.27, label="start-hotspots fixed")
ax.bar(x + 0.27, [np.sum([r["collateral"] for r in rows if r["strategy"] == nm]) for nm in names], 0.27, label="new (collateral) violations")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e13_adaptive_reweight.png"), dpi=130)
