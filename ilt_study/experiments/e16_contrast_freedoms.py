"""E16  Freedoms that DO change the contrast budget: source trimming and attPSM.

E1/E11 showed that, with a binary mask, a fixed source and a fixed dose,
mask-only ILT cannot raise NILS.  Two-beam image contrast is
gamma = 2|a0 a1| / (|a0|^2 + |a1|^2): it rises when the 0th and 1st order
amplitudes are balanced, or when background-only source points are removed.
  * source trimmed to the two-beam region (every point passes 0th AND a 1st order)
  * attenuated PSM (pi-phase background, T = 6 %, 20 %): lowers a0 without
    shrinking the line, so contrast is not paid for with MEEF.
Part A: dense L/S, dose re-centred per mask width (E1b method), mask error
+-2 nm in the corners -> NILS, MEEF, worst margins.
Part B: full reference ILT recipe on three random clips for each variant.
Polarization is NOT testable here (scalar model).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Recipe, log, random_clip, run_eval, save_json, summarize
from ilt import Problem, calibrate_threshold
from litho import Grid, Imager, Optics

VARIANTS = {
    "binary, leaf source (baseline)": Optics(),
    "binary, two-beam source": Optics(two_beam_pitch=74.5),
    "attPSM 6%, leaf source": Optics(att_T=0.06),
    "attPSM 6%, two-beam source": Optics(att_T=0.06, two_beam_pitch=74.5),
    "attPSM 20%, two-beam source": Optics(att_T=0.20, two_beam_pitch=74.5),
}
g = Grid()
x = (np.arange(g.n) + 0.5) * g.dx
p = g.period_x
sub = 16
xs = ((np.arange(g.n * sub) + 0.5) * g.dx / sub) % p


def dense_scan(im):
    th = calibrate_threshold(im, g)

    def image_row(w):
        row = (np.abs(xs - p / 2) < w / 2).reshape(g.n, sub).mean(1)
        m = torch.tensor(np.tile(row, (g.n, 1)), dtype=torch.float32)
        return im.image(m, [0, 1, 2])[:, 0].numpy() / th

    def width(I, D):
        k = int(p / g.dx) + 2
        xx, v = x[:k], D * I[:k] - 1
        c = np.nonzero(np.diff(np.sign(v)))[0]
        if len(c) < 2:
            return 0.0 if v[k // 2] < 0 else p
        xe = [xx[i] - v[i] * (xx[i + 1] - xx[i]) / (v[i + 1] - v[i]) for i in c[:2]]
        return xe[1] - xe[0]
    rows = []
    for w in np.arange(18, 51, 2.0):
        I0 = image_row(w)
        lo, hi = 0.05, 8.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if width(I0[0], mid) < 37 else (lo, mid)
        D0 = 0.5 * (lo + hi)
        if abs(width(I0[0], D0) - 37) > 0.5:
            continue
        widths = [width(image_row(w + dw)[f], D0 * dd) for dw in (-2, 0, 2) for f in range(3) for dd in (0.95, 1, 1.05)]
        v = I0[0] * D0; xe = p / 2 + 18.5; i = np.searchsorted(x, xe)
        nils = 37 * abs((v[i] - v[i - 1]) / g.dx) / np.interp(xe, x, v)
        meef = (width(image_row(w + 1)[0], D0) - width(image_row(w - 1)[0], D0)) / 2
        rows.append(dict(mask_w=float(w), dose=float(D0), nils=float(nils), meef=float(meef),
                         pinch=float(min(widths) - 28), bridge=float(p - max(widths) - 28),
                         sum=float(min(widths) - 28 + p - max(widths) - 28),
                         worst=float(min(min(widths) - 28, p - max(widths) - 28))))
    return rows


out = {"dense": {}, "ilt": []}
for name, op in VARIANTS.items():
    im = Imager(g, op, keep=0.98)
    rows = dense_scan(im)
    out["dense"][name] = rows
    best = max(rows, key=lambda r: r["worst"])
    r37 = min(rows, key=lambda r: abs(r["mask_w"] - 38))
    log(f"{name}: src pts {len(im.src)} | best mask w={best['mask_w']:.0f}: NILS {best['nils']:.2f} MEEF {best['meef']:.2f} "
        f"pinch {best['pinch']:+.2f} bridge {best['bridge']:+.2f} sum {best['sum']:+.2f} | at w=38: NILS {r37['nils']:.2f} "
        f"MEEF {r37['meef']:.2f} sum {r37['sum']:+.2f}")
    save_json("e16.json", out)

for name in ("binary, leaf source (baseline)", "binary, two-beam source", "attPSM 6%, leaf source",
             "attPSM 6%, two-beam source"):
    im = Imager(g, VARIANTS[name], keep=0.98)
    for seed in (1, 2, 3):
        P = Problem(random_clip(seed), imager=im)
        _, rho, _, r = run_eval(P, Recipe(**GOOD))
        out["ilt"].append(dict(variant=name, seed=seed, **summarize(r)))
        log(name, seed, {k: out["ilt"][-1][k] for k in ("n_hot", "n_bridge_all", "n_pinch_all", "n_lend", "n_t2t",
                                                         "topo_bridge", "min_bridge_all", "min_pinch_all")})
        save_json("e16.json", out)

fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for name, rows in out["dense"].items():
    W = [r["mask_w"] for r in rows]
    axs[0].plot(W, [r["nils"] for r in rows], "o-", ms=3, label=name)
    axs[1].plot(W, [r["meef"] for r in rows], "o-", ms=3, label=name)
    axs[2].plot(W, [r["worst"] for r in rows], "o-", ms=3, label=name)
for a, t in zip(axs, ("NILS (dose re-centred to 37 nm)", "MEEF", "worse of pinch/bridge margin (nm), mask err +-2")):
    a.set_xlabel("mask line width (nm)"); a.set_title(t, fontsize=9)
axs[2].axhline(0, c="gray", lw=0.6); axs[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e16_contrast_freedoms.png"), dpi=130)
