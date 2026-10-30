"""E7b  Separate 'serif / hammerhead freedom' from 'SRAF freedom'.

E7 froze everything farther than 14 nm from the target; that also forbids
line-end hammerheads larger than 14 nm.  Here no SRAF is allowed (frozen
pixels = 0) but the editable band around the target is varied:
8, 14, 25, 40 nm, plus fully free.  Breakdown by site type.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, summarize
from ilt import disk_tf, fft_conv

rows = []
for seed in (1, 2, 3):
    P = Problem(random_clip(seed))
    for d in (8.0, 14.0, 25.0, 40.0, None):
        if d is None:
            rc = Recipe(**GOOD); name = "free"
        else:
            dil = fft_conv(P.T.double(), torch.tensor(disk_tf(P.g.n, P.g.dx, d)[0])).float().numpy()
            rc = Recipe(**GOOD, freeze=dil <= 0.5, x_frozen=np.zeros_like(dil, np.float32)); name = f"band {d:.0f} nm"
        _, rho, _, r = run_eval(P, rc)
        rows.append(dict(seed=seed, config=name, **summarize(r)))
        log(seed, name, {k: rows[-1][k] for k in ("n_hot", "n_pinch", "n_lend", "n_bridge", "n_t2t", "topo_bridge", "sraf_print")})
        save_json("e07b.json", rows)
names = list(dict.fromkeys(r["config"] for r in rows))
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(names)); kinds = ["n_lend", "n_pinch", "n_t2t", "n_bridge"]
bottom = np.zeros(len(names))
for k in kinds:
    v = np.array([np.mean([r[k] for r in rows if r["config"] == nm]) for nm in names])
    ax.bar(x, v, bottom=bottom, label=k[2:]); bottom += v
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("hotspots (mean of 3 clips)")
ax.set_title("no SRAF: width of the editable band around the target", fontsize=9); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e07b_freedom_radius.png"), dpi=130)
