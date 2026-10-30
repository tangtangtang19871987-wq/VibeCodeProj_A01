"""E7c  In the 25/40 nm editable bands of E7b: serif growth or detached islands?

Counts mask components that do not touch any target polygon (= assist
features) and the mask area outside the target, and plots the masks.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json
from ilt import disk_tf, fft_conv, label

rows = []
fig, axs = plt.subplots(2, 3, figsize=(13, 9))
for row, seed in enumerate((2, 3)):
    P = Problem(random_clip(seed))
    for col, d in enumerate((14.0, 40.0, None)):
        if d is None:
            rc = Recipe(**GOOD); name = "free"
        else:
            dil = fft_conv(P.T.double(), torch.tensor(disk_tf(P.g.n, P.g.dx, d)[0])).float().numpy()
            rc = Recipe(**GOOD, freeze=dil <= 0.5, x_frozen=np.zeros_like(dil, np.float32)); name = f"band {d:.0f} nm"
        _, rho, _, r = run_eval(P, rc, topo=False)
        m = (P.hard_mask(rho) > 0.5).numpy()
        lm = label(m)
        tgt = P.T.numpy() > 0.5
        touching = set(np.unique(lm[tgt])) - {0}
        islands = [c for c in np.unique(lm) if c and c not in touching]
        isl_area = float(sum((lm == c).sum() for c in islands) * P.g.dx ** 2)
        attached_extra = float(((m & ~tgt) & np.isin(lm, list(touching))).sum() * P.g.dx ** 2)
        rows.append(dict(seed=seed, config=name, n_islands=len(islands), island_area=isl_area,
                         attached_extra_area=attached_extra, n_hot_sites=r["n_hot_sites"]))
        log(rows[-1])
        ax = axs[row, col]
        ax.imshow(m.astype(float) + 0.0, origin="lower", extent=[0, P.clip.L] * 2, cmap="Greys", vmin=0, vmax=1.5)
        ax.contour(P.T.numpy(), levels=[0.5], origin="lower", extent=[0, P.clip.L] * 2, colors="g", linewidths=0.8)
        for c in islands:
            ax.contour((lm == c).astype(float), levels=[0.5], origin="lower", extent=[0, P.clip.L] * 2, colors="r", linewidths=1)
        ax.set_title(f"clip {seed}, {name}: {len(islands)} detached islands (red)", fontsize=8)
save_json("e07c.json", rows)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e07c_band_islands.png"), dpi=110)
