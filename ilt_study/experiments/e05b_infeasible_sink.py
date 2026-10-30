"""E5b  Weight sweep and remedies at truly infeasible tip-to-tip gaps (38, 32 nm).

Fixes two problems of E5: (1) E5's gap selection treated a 0.0 margin as
infeasible (the chosen 50 nm gap is borderline-feasible); (2) its 'cap'
was applied to every site, which silences all large violations early in the
run.  Here the cap/minimax remedies act only on the infeasible pair, and
collateral damage is also measured on the *neighbouring* sites (<150 nm).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, conflict_clip, log, run_eval, save_json, summarize
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from e05_helpers import pair_ids, grad_share

rows = []
for gap in (38, 32):
    P = Problem(conflict_clip(t2t_gap=gap))
    t2t, lend = pair_ids(P)
    excl = t2t + lend
    c = np.mean([P.sites.p0[i] for i in excl], axis=0)
    L = P.clip.L
    d = np.array([np.hypot(*(((np.array(p) - c + L / 2) % L) - L / 2)) for p in P.sites.p0])
    near = (d < 150) & ~np.isin(np.arange(len(d)), excl)
    far = (d >= 150)
    cfgs = [("plain", w, {}) for w in (3, 10, 30, 100, 300)]
    cfgs += [("cap 2 nm (pair only)", w, {"cap": 2.0, "cap_sites": excl}) for w in (30, 300)]
    cfgs += [("minimax pair", w, {"balance_sites": [excl]}) for w in (30, 300)]
    for name, w, extra in cfgs:
        rc = Recipe(**{**GOOD, "w_bridge": w, "w_pinch": 10.0, **extra})
        theta, rho, _, r = run_eval(P, rc)
        m = r["margin"]
        row = dict(gap=gap, config=name, w_bridge=w, t2t=float(m[t2t].min()), lend=float(m[lend].min()),
                   near_hot=int((m[near] < 0).sum()), near_min=float(m[near].min()),
                   near_mean=float(np.mean(np.minimum(m[near], 5))),
                   far_hot=int((m[far] < 0).sum()), far_min=float(m[far].min()),
                   grad_share=grad_share(P, theta, Recipe(**{**GOOD, "w_bridge": w, "w_pinch": 10.0}), excl),
                   **summarize(r))
        rows.append(row)
        log(gap, name, w, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                           if k in ("t2t", "lend", "near_hot", "near_min", "near_mean", "far_hot", "grad_share", "topo_bridge")})
        save_json("e05b.json", rows)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for ax, gap in zip(axs, (38, 32)):
    R = [r for r in rows if r["gap"] == gap and r["config"] == "plain"]
    W = [r["w_bridge"] for r in R]
    ax.plot(W, [r["t2t"] for r in R], "o-", c="#17becf", label="pair: t2t margin")
    ax.plot(W, [r["lend"] for r in R], "o-", c="#ff7f0e", label="pair: line-end margin")
    ax.plot(W, [r["near_min"] for r in R], "s--", c="k", label="neighbours (<150 nm): worst margin")
    ax.plot(W, [r["near_mean"] for r in R], "^:", c="k", label="neighbours: mean margin (clipped at 5)")
    for r in rows:
        if r["gap"] == gap and r["config"] != "plain":
            ax.scatter([r["w_bridge"]], [r["near_min"]], marker="x" if "cap" in r["config"] else "+", s=60,
                       c="r" if "cap" in r["config"] else "g", label=f"{r['config']} w={r['w_bridge']}: neighbours worst")
    ax.set_xscale("log"); ax.axhline(0, c="gray", lw=0.6); ax.set_xlabel("bridge weight"); ax.set_ylabel("margin (nm)")
    ax.set_title(f"tip-to-tip gap {gap} nm (infeasible)", fontsize=9); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e05b_infeasible_sink.png"), dpi=130)
