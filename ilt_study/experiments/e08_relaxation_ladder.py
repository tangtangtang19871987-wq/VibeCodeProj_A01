"""E8  Relaxation ladder: which released freedom fixes each residual hotspot?

Claim tested: "running the same clip with progressively released freedoms
attributes each hotspot to the freedom that is its bottleneck".
L1 main-only (SRAF frozen = none), MRC proxy R=8
L2 + SRAF released
L3 + MRC relaxed (R=5)
L4 + corner set narrowed (dose +-4 %, mask +-0.75 nm) -- a spec decision,
   evaluated against the narrowed spec.
For every site violated at L1 we record the first level where it passes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, Spec, log, random_clip, run_eval, save_json, summarize
from ilt import disk_tf, fft_conv

narrow = Spec(doses=(0.96, 1.0, 1.04), biases=(-0.75, 0.0, 0.75))
rows, per_site = [], []
for seed in range(1, 7):
    clip = random_clip(seed)
    P = Problem(clip)
    Pn = Problem(clip, spec=narrow, imager=P.im)
    dil = fft_conv(P.T.double(), torch.tensor(disk_tf(P.g.n, P.g.dx, 14.0)[0])).float().numpy()
    freeze = dil <= 0.5
    levels = [("L1 main only", P, Recipe(**GOOD, freeze=freeze, x_frozen=np.zeros_like(dil, np.float32))),
              ("L2 +SRAF", P, Recipe(**GOOD)),
              ("L3 +MRC relaxed", P, Recipe(**{**GOOD, "R": 5.0})),
              ("L4 +narrow corners", Pn, Recipe(**{**GOOD, "R": 5.0}))]
    margins = []
    for name, PP, rc in levels:
        _, rho, _, r = run_eval(PP, rc)
        margins.append(r["margin"])
        rows.append(dict(seed=seed, level=name, **summarize(r)))
        log(seed, name, {k: rows[-1][k] for k in ("n_hot", "n_bridge_all", "n_pinch_all", "topo_bridge", "topo_open")})
    M = np.stack(margins)
    for i in np.nonzero(M[0] < 0)[0]:
        ok = np.nonzero(M[:, i] >= 0)[0]
        first = int(ok[ok > 0][0]) if np.any(ok > 0) else -1
        per_site.append(dict(seed=seed, site=int(i), kind=str(P.kind[i]), first_fix=first,
                             margins=M[:, i].tolist(), still_ok_after=bool(first >= 0 and np.all(M[first:, i] >= 0))))
    save_json("e08.json", {"levels": rows, "sites": per_site})

names = ["L1 main only", "L2 +SRAF", "L3 +MRC relaxed", "L4 +narrow corners"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
kinds = ["pinch", "lend", "bridge", "t2t", "diag"]
cnt = np.zeros((len(kinds), 4))
for s in per_site:
    cnt[kinds.index(s["kind"]), s["first_fix"] if s["first_fix"] > 0 else 0] += 1
bottom = np.zeros(4)
labels = ["never fixed", "fixed by +SRAF", "fixed by +MRC relax", "fixed by narrow corners"]
for k, kd in enumerate(kinds):
    axs[0].bar(labels, cnt[k], bottom=bottom, label=kd); bottom += cnt[k]
axs[0].set_ylabel("# sites violated at L1"); axs[0].legend(fontsize=7); axs[0].tick_params(axis="x", labelsize=7)
axs[0].set_title("bottleneck attribution (6 clips)", fontsize=9)
for key, c in (("n_bridge_all", "#1f77b4"), ("n_pinch_all", "#d62728")):
    y = np.array([[r[key] + (r["topo_bridge"] if "bridge" in key else r["topo_open"]) for r in rows if r["level"] == nm] for nm in names])
    axs[1].plot(range(4), y.mean(1), "o-", c=c, label=key.replace("n_", "").replace("_all", "") + " (mean)")
    axs[1].fill_between(range(4), y.min(1), y.max(1), color=c, alpha=0.15)
axs[1].set_xticks(range(4)); axs[1].set_xticklabels(names, fontsize=7); axs[1].legend(fontsize=7)
axs[1].set_yscale("symlog"); axs[1].set_title("hotspots per level", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e08_relaxation_ladder.png"), dpi=130)
