"""E6  Is 'anti-aligned bridge/pinch gradients' a usable infeasibility signal?

Claim tested (from the earlier answer): "at an infeasible site the two
gradients point in opposite directions (cos ~ -1), so cos can flag it".
Concern: at ANY constrained optimum the active constraints' gradients are
anti-aligned (KKT), feasible or not.  We measure cos(grad m_bridge,
grad m_pinch) for adjacent site pairs at converged masks and compare pairs
that are (i) comfortably feasible, (ii) tight-but-feasible, (iii) violated.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, RES, Recipe, conflict_clip, get_good, log, save_json
from ilt import site_values


def pairs_for(P):
    pairs = []
    K, tags = P.kind, P.sites.tag
    for i, t in enumerate(tags):
        if K[i] == "t2t" and t.endswith("@+0"):
            a, b = t.split("@")[0].split("->")
            for j, t2 in enumerate(tags):
                if K[j] == "lend" and t2 in (f"{a}:top", f"{b}:bot"):
                    pairs.append((i, j, "t2t/line-end"))
    # lateral: each bridge site with the pinch site on its left line at the closest y
    pin = [j for j in range(len(K)) if K[j] == "pinch"]
    rng = np.random.default_rng(0)
    br = [i for i in range(len(K)) if K[i] == "bridge"]
    for i in rng.choice(br, size=min(25, len(br)), replace=False):
        x0, y0 = P.sites.p0[i]
        cand = [j for j in pin if abs(P.sites.p0[j][0] + P.clip.pitch / 2 - x0) < 1 and
                abs(P.sites.p0[j][1] - y0) < 13]
        if cand:
            pairs.append((int(i), cand[0], "lateral"))
    return pairs


rows = []
rc = Recipe(**GOOD)
gap = json.load(open(os.path.join(RES, "e05.json")))["chosen_gap"]
clips = [("rand1", dict(seed=1)), ("rand2", dict(seed=2)), ("rand3", dict(seed=3)),
         (f"conflict_g{gap}", dict(seed=None, clip=conflict_clip(t2t_gap=gap)))]
for name, kw in clips:
    P, theta, rho, r = get_good(kw.get("seed"), clip=kw.get("clip"))
    th = theta.clone().requires_grad_(True)
    q = site_values(P, th, rc)
    m = r["margin"]
    grads = {}

    def G(i):
        if i not in grads:
            grads[i] = torch.autograd.grad(q[i], th, retain_graph=True)[0].flatten()
        return grads[i]
    for i, j, typ in pairs_for(P):
        gi, gj = G(i), G(j)
        cos = float(gi @ gj / (gi.norm() * gj.norm() + 1e-20))
        mi, mj = float(m[i]), float(m[j])
        status = "violated" if min(mi, mj) < 0 else ("tight (<1.5 nm)" if min(mi, mj) < 1.5 else "comfortable")
        rows.append(dict(clip=name, type=typ, cos=cos, m_bridge=mi, m_pinch=mj, status=status))
    log(name, "pairs", len([r_ for r_ in rows if r_["clip"] == name]))
save_json("e06.json", rows)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
cols = {"comfortable": "#2ca02c", "tight (<1.5 nm)": "#ff7f0e", "violated": "#d62728"}
for ax, typ in zip(axs, ("lateral", "t2t/line-end")):
    for st, c in cols.items():
        v = [r_["cos"] for r_ in rows if r_["type"] == typ and r_["status"] == st]
        if v:
            ax.hist(v, bins=20, range=(-1, 1), alpha=0.5, color=c, label=f"{st} (n={len(v)}, median {np.median(v):+.2f})")
            log(typ, st, "n", len(v), "median cos", round(float(np.median(v)), 3))
    ax.set_xlabel("cos(grad bridge margin, grad pinch margin)"); ax.set_title(typ, fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e06_gradient_cosine.png"), dpi=130)
