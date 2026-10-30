"""E11  Adding an explicit contrast (log-slope) term to the cost.

Claim tested: "putting worst-corner contrast into the cost raises both
margins".  E1b predicts a catch: contrast bought by mask bias costs MEEF.
We add a worst-corner log-contrast term (line centre vs space centre on
every lateral cut) and track hotspots, worst margins, dose sensitivity and
MEEF.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, sensitivities, summarize

rows = []
for seed in (1, 2, 3):
    P = Problem(random_clip(seed))
    lat = np.isin(P.kind, ["bridge", "pinch"])
    for w in (0.0, 0.3, 3.0, 30.0):
        rc = Recipe(**{**GOOD, "w_ils": w})
        _, rho, _, r = run_eval(P, rc)
        ds, me = sensitivities(P, rho)
        rows.append(dict(seed=seed, w_ils=w, med_dsens=float(np.median(ds[lat])), med_meef=float(np.median(me[lat])),
                         p90_meef=float(np.percentile(me[lat], 90)), **summarize(r)))
        log(seed, w, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rows[-1].items()
                      if k in ("n_hot", "n_bridge_all", "n_pinch_all", "min_bridge_all", "min_pinch_all", "med_dsens", "med_meef")})
        save_json("e11.json", rows)
W = sorted({r["w_ils"] for r in rows})
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for key, ax, lab in (("n_hot", axs[0], "hotspots"), ("med_dsens", axs[1], "median |dL/dD| lateral sites (nm/100%)"),
                     ("med_meef", axs[2], "median MEEF lateral sites")):
    for seed, m in ((1, "o"), (2, "s"), (3, "^")):
        ax.plot(range(len(W)), [r[key] for r in rows if r["seed"] == seed], m + "-", label=f"clip {seed}")
    ax.set_xticks(range(len(W))); ax.set_xticklabels([str(w) for w in W]); ax.set_xlabel("contrast-term weight")
    ax.set_title(lab, fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e11_contrast_term.png"), dpi=130)
