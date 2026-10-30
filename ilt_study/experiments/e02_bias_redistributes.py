"""E2  A global mask bias (or a dose-centre shift) only redistributes margin.

Claim tested: "biasing the mask moves margin from pinch to bridge; the
sum of the two worst margins is (nearly) bias-invariant, so bias cannot fix
both".  Starting from the converged reference ILT mask, we add a uniform
edge bias b (hard mask, exact sub-pixel offset) and, separately, shift the
dose centre, and re-run the full-factorial checker.
Sites are grouped by mechanism: lateral (bridge vs neighbouring pinch) and
along-track (t2t vs line-end pull-back).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, get_good, log, save_json
from ilt import Spec

out = {}
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for col, seed in enumerate([1, 3, 5]):
    P, theta, rho, r0 = get_good(seed)
    K = P.kind
    groups = {"lateral": (K == "bridge", K == "pinch"),
              "along-track": (K == "t2t", K == "lend")}
    res = {"bias": [], "dose": []}
    for b in np.arange(-4, 4.01, 0.5):
        r = P.evaluate(rho, mask_bias_fn=None, topo=False) if b == 0 else None
        # global bias: shift every corner's mask by b in addition to the corner bias
        sp = Spec(biases=tuple(x + b for x in P.spec.biases))
        r = P.evaluate(rho, spec=sp, topo=False)
        m = r["margin"]
        res["bias"].append({"b": b, **{f"{g}_{s}": float(m[msk].min()) for g, (bm, pm) in groups.items()
                                         for s, msk in (("bridge", bm), ("pinch", pm))}})
    for dc in np.arange(0.95, 1.051, 0.01):
        sp = Spec(doses=tuple(d * dc for d in P.spec.doses))
        m = P.evaluate(rho, spec=sp, topo=False)["margin"]
        res["dose"].append({"dc": dc, **{f"{g}_{s}": float(m[msk].min()) for g, (bm, pm) in groups.items()
                                           for s, msk in (("bridge", bm), ("pinch", pm))}})
    out[f"seed{seed}"] = res
    for row, (key, xk, xl) in enumerate((("bias", "b", "uniform mask bias (nm/edge)"),
                                         ("dose", "dc", "dose centre"))):
        ax = axs[row, col]
        X = [d[xk] for d in res[key]]
        for g, ls in (("lateral", "-"), ("along-track", "--")):
            mb = np.array([d[f"{g}_bridge"] for d in res[key]])
            mp = np.array([d[f"{g}_pinch"] for d in res[key]])
            ax.plot(X, mb, ls, c="#1f77b4", label=f"{g}: worst bridge-type")
            ax.plot(X, mp, ls, c="#d62728", label=f"{g}: worst pinch-type")
            ax.plot(X, mb + mp, ls, c="k", alpha=0.6, label=f"{g}: sum")
            log(f"seed{seed} {key} {g}: sum range {np.ptp(mb+mp):.2f} nm, "
                f"bridge range {np.ptp(mb):.2f}, pinch range {np.ptp(mp):.2f}")
        ax.axhline(0, c="gray", lw=0.6); ax.set_xlabel(xl); ax.set_ylabel("worst margin (nm)")
        ax.set_ylim(-20, 20); ax.set_title(f"seed {seed}", fontsize=9)
        if col == 0 and row == 0: ax.legend(fontsize=6)
save_json("e02.json", out)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e02_bias_redistributes.png"), dpi=120)
