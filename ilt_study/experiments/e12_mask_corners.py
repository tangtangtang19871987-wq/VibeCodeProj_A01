"""E12  Mask-CD-error corners inside the optimization.

Claim tested: "adding +-mask-bias corners makes ILT pick solutions that
are less sensitive to mask error (lower MEEF), which helps both bridge and
pinch under mask error".  Optimize without / with mask-bias corners; check
against +-1 nm and +-2 nm mask error.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, Spec, log, random_clip, run_eval, save_json, sensitivities, summarize

no_mask = [(1.0, 0, 0.0)] + [(d, f, 0.0) for f in (0, 1, 2) for d in (1.05, 0.95)]
rows = []
for seed in (1, 2, 3):
    P = Problem(random_clip(seed))
    lat = np.isin(P.kind, ["bridge", "pinch"])
    for name, corners in (("no mask corners", no_mask), ("with +-1 nm mask corners", None)):
        rc = Recipe(**{**GOOD, "corners_opt": corners})
        _, rho, _, r1 = run_eval(P, rc)
        r2 = P.evaluate(rho, spec=Spec(biases=(-2.0, 0.0, 2.0)))
        r0 = P.evaluate(rho, spec=Spec(biases=(0.0,)))
        ds, me = sensitivities(P, rho)
        rows.append(dict(seed=seed, config=name, med_meef=float(np.median(me[lat])), p90_meef=float(np.percentile(me[lat], 90)),
                         hot_mask0=r0["n_hot"], hot_mask1=r1["n_hot"], hot_mask2=r2["n_hot"],
                         minb_mask1=r1["min_bridge_all"], minp_mask1=r1["min_pinch_all"],
                         minb_mask2=r2["min_bridge_all"], minp_mask2=r2["min_pinch_all"], **summarize(r1)))
        log(seed, name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()
                         if k in ("med_meef", "p90_meef", "hot_mask0", "hot_mask1", "hot_mask2")})
        save_json("e12.json", rows)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
names = ["no mask corners", "with +-1 nm mask corners"]
for k, nm in enumerate(names):
    R = [r for r in rows if r["config"] == nm]
    axs[0].plot([0, 1, 2], [np.mean([r[f"hot_mask{i}"] for r in R]) for i in range(3)], "o-", label=nm)
    axs[1].bar(np.arange(3) + (k - 0.5) * 0.4, [r["med_meef"] for r in R], 0.4, label=nm)
axs[0].set_xticks([0, 1, 2]); axs[0].set_xticklabels(["no mask error", "+-1 nm", "+-2 nm"])
axs[0].set_ylabel("hotspots (mean of 3 clips)"); axs[0].legend(fontsize=7); axs[0].set_title("evaluated mask error", fontsize=9)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(["clip 1", "clip 2", "clip 3"]); axs[1].set_ylabel("median MEEF (lateral sites)")
axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e12_mask_corners.png"), dpi=130)
