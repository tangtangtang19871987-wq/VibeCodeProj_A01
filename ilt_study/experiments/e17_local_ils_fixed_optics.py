"""E17  With optics FIXED (binary mask, leaf source, fixed dose): can mask shape
alone raise the local image slope where the hotspots are?

Dense lines are bounded (E1/E1b: the mask only sets two diffraction-order
amplitudes).  Line ends / tip-to-tip / diagonal sites are not periodic, so the
mask can reshape their local spectrum.  We compare the per-site dose
sensitivity (= 1/ILS, E3) and MEEF of solutions whose editable band around
the target is 14 nm, 40 nm, or free, plus the free solution optimized with and
without mask-error corners (E12).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, sensitivities, summarize
from ilt import disk_tf, fft_conv

no_mask = [(1.0, 0, 0.0)] + [(d, f, 0.0) for f in (0, 1, 2) for d in (1.05, 0.95)]
KINDS = {"lateral (pinch+bridge)": ("pinch", "bridge"), "line-end": ("lend",), "tip-to-tip": ("t2t",),
         "diagonal": ("diag",)}
rows = []
for seed in (1, 2, 3):
    P = Problem(random_clip(seed))
    for name, d, corners in (("band 14 nm", 14.0, None), ("band 40 nm", 40.0, None),
                             ("free", None, None), ("free, no mask corners", None, no_mask)):
        if d is None:
            rc = Recipe(**{**GOOD, "corners_opt": corners})
        else:
            dil = fft_conv(P.T.double(), torch.tensor(disk_tf(P.g.n, P.g.dx, d)[0])).float().numpy()
            rc = Recipe(**GOOD, freeze=dil <= 0.5, x_frozen=np.zeros_like(dil, np.float32))
        _, rho, _, r = run_eval(P, rc, topo=False)
        ds, me = sensitivities(P, rho)
        row = dict(seed=seed, config=name, n_hot_sites=r["n_hot_sites"])
        for kn, ks in KINDS.items():
            sel = np.isin(P.kind, ks)
            row[f"{kn}|dsens"] = float(np.median(ds[sel]))
            row[f"{kn}|meef"] = float(np.median(me[sel]))
            row[f"{kn}|edge_var"] = float(np.median(0.05 * ds[sel] + 2.0 * me[sel]))
        rows.append(row)
        log(seed, name, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items() if k not in ("seed", "config")})
        save_json("e17.json", rows)

cfgs = list(dict.fromkeys(r["config"] for r in rows))
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
x = np.arange(len(KINDS))
for k, cfg in enumerate(cfgs):
    for ax, met, lab in zip(axs, ("dsens", "meef", "edge_var"),
                            ("median |dL/dD| (nm per 100% dose) = sum 1/ILS", "median MEEF",
                             "median length change for 5% dose + 1 nm/edge mask (nm)")):
        v = [np.mean([r[f"{kn}|{met}"] for r in rows if r["config"] == cfg]) for kn in KINDS]
        ax.bar(x + (k - 1.5) * 0.2, v, 0.2, label=cfg)
        ax.set_xticks(x); ax.set_xticklabels(list(KINDS), fontsize=8); ax.set_title(lab, fontsize=9)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e17_local_ils_fixed_optics.png"), dpi=130)
