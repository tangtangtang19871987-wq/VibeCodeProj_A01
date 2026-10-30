"""E5  An infeasible site acts as a gradient sink when its weight is raised.

Claim tested: "where bridge and pinch cannot both be met, raising the
weight only moves the violation between them, and because the site's
gradient never vanishes it drags neighbouring, feasible sites out of spec;
capping or minimax-balancing that site stops the collateral damage".

Step 1: shrink a tip-to-tip gap until the (t2t, line-end) pair has no
        simultaneous solution under the reference recipe.
Step 2: on that clip, sweep the bridge-class weight and record (a) the pair,
        (b) collateral hotspots on all other sites, (c) the pair's share of the
        total gradient.  Step 3: same with cap / minimax remedies.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, conflict_clip, log, run_eval, save_json, summarize
from ilt import site_values


def pair_ids(P):
    t2t = [i for i, t in enumerate(P.sites.tag) if t.startswith("seg3->seg4")]
    lend = [i for i, t in enumerate(P.sites.tag) if t in ("seg3:top", "seg4:bot")]
    return t2t, lend


def collateral(P, r, excl):
    m = r["margin"].copy()
    keep = np.ones(len(m), bool); keep[excl] = False
    return dict(coll_hot=int((m[keep] < 0).sum()), coll_min=float(m[keep].min()),
                coll_p5=float(np.percentile(m[keep], 5)))


def grad_share(P, theta, rc, excl):
    th = theta.clone().requires_grad_(True)
    q = site_values(P, th, rc)
    pen = F.relu(P.lower + rc.margin - q) ** 2
    isb = torch.tensor(np.isin(P.kind, ["bridge", "t2t", "diag"]), dtype=torch.float32)
    w = torch.where(isb > 0, torch.tensor(rc.w_bridge), torch.tensor(rc.w_pinch))
    pen = pen * w
    mask = torch.zeros(len(P.kind)); mask[excl] = 1
    g_pair = torch.autograd.grad((pen * mask).sum(), th, retain_graph=True)[0]
    g_rest = torch.autograd.grad((pen * (1 - mask)).sum(), th)[0]
    return float(g_pair.norm() / (g_pair.norm() + g_rest.norm() + 1e-12))


out = {"gap_scan": [], "weight": []}
# ---- step 1: feasibility vs gap -----------------------------------------
chosen = None
for gap in [60, 50, 44, 38, 32]:
    P = Problem(conflict_clip(t2t_gap=gap))
    t2t, lend = pair_ids(P)
    best = None
    for wb in (3, 10, 30, 100):          # several balances: is there any point with both >= 0?
        rc = Recipe(**{**GOOD, "w_bridge": wb, "w_pinch": 10.0})
        _, rho, _, r = run_eval(P, rc, topo=False)
        mt, ml = float(r["margin"][t2t].min()), float(r["margin"][lend].min())
        out["gap_scan"].append(dict(gap=gap, w_bridge=wb, t2t=mt, lend=ml))
        log(f"gap {gap} wb {wb}: t2t margin {mt:+.1f}  lend margin {ml:+.1f}")
        best = min(mt, ml) if best is None else max(best, min(mt, ml))
    if best < 0 and chosen is None:
        chosen = gap
log("first infeasible gap:", chosen)
out["chosen_gap"] = chosen
save_json("e05.json", out)

# ---- step 2/3: weight sweep and remedies on the infeasible clip ----------
P = Problem(conflict_clip(t2t_gap=chosen))
t2t, lend = pair_ids(P)
excl = t2t + lend
configs = [("plain", w, {}) for w in (3, 10, 30, 100, 300)]
configs += [("cap 2 nm", w, {"cap": 2.0}) for w in (30, 300)]
configs += [("minimax pair", w, {"balance_sites": [excl]}) for w in (30, 300)]
for name, w, extra in configs:
    rc = Recipe(**{**GOOD, "w_bridge": w, "w_pinch": 10.0, **extra})
    theta, rho, hist, r = run_eval(P, rc)
    row = dict(config=name, w_bridge=w, t2t=float(r["margin"][t2t].min()),
               lend=float(r["margin"][lend].min()), grad_share=grad_share(P, theta, Recipe(**{**GOOD, "w_bridge": w, "w_pinch": 10.0}), excl),
               **collateral(P, r, excl), **summarize(r))
    out["weight"].append(row)
    log(name, w, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()
                  if k in ("t2t", "lend", "grad_share", "coll_hot", "coll_min", "coll_p5", "topo_bridge", "topo_open")})
    save_json("e05.json", out)

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
G = sorted({d["gap"] for d in out["gap_scan"]})
for gap in G:
    D = [d for d in out["gap_scan"] if d["gap"] == gap]
    axs[0].plot([d["t2t"] for d in D], [d["lend"] for d in D], "o-", label=f"gap {gap} nm")
axs[0].axhline(0, c="gray", lw=0.6); axs[0].axvline(0, c="gray", lw=0.6)
axs[0].set_xlabel("t2t (bridge) margin"); axs[0].set_ylabel("line-end (pinch) margin")
axs[0].set_title("pair frontier vs gap (points: bridge weight 3..100)", fontsize=9); axs[0].legend(fontsize=7)
R = [d for d in out["weight"] if d["config"] == "plain"]
W = [d["w_bridge"] for d in R]
axs[1].plot(W, [d["t2t"] for d in R], "o-", c="#17becf", label="pair: t2t margin")
axs[1].plot(W, [d["lend"] for d in R], "o-", c="#ff7f0e", label="pair: line-end margin")
axs[1].plot(W, [d["coll_p5"] for d in R], "s--", c="k", label="other sites: 5th pct margin")
axs[1].plot(W, [d["coll_min"] for d in R], "^:", c="k", label="other sites: worst margin")
axs[1].set_xscale("log"); axs[1].axhline(0, c="gray", lw=0.6); axs[1].set_xlabel("bridge weight")
axs[1].set_ylabel("margin (nm)"); axs[1].legend(fontsize=7); axs[1].set_title(f"gap {chosen} nm: weight sweep", fontsize=9)
ax2 = axs[1].twinx(); ax2.plot(W, [d["grad_share"] for d in R], "x-", c="purple"); ax2.set_ylabel("pair share of gradient", color="purple")
names = [f"{d['config']}\nw={d['w_bridge']}" for d in out["weight"]]
axs[2].bar(range(len(names)), [d["coll_hot"] for d in out["weight"]], color="gray")
axs[2].set_xticks(range(len(names))); axs[2].set_xticklabels(names, fontsize=6, rotation=60)
axs[2].set_ylabel("collateral hotspots (sites outside the pair)"); axs[2].set_title("remedies", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e05_infeasible_sink.png"), dpi=130)
