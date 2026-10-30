"""E1b  Raising NILS with (mask bias + dose re-centering) on dense L/S.

Claim tested (user's intuition): "the fix is not moving the mask edge but
raising the local contrast".  Geometry fixed (37 nm line / pitch 74.5).
For each mask line width w we re-centre dose so the line prints 37 nm at
nominal, then evaluate the worst printed width (pinch) and worst space
(bridge) over dose +-5 %, focus +-40 nm, mask CD +-2 nm (+-1 per edge).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, log, save_json
from litho import Grid, Imager
from ilt import calibrate_threshold

g = Grid(); im = Imager(g, keep=0.98); th = calibrate_threshold(im, g)
sub = 16
xs = ((np.arange(g.n * sub) + 0.5) * g.dx / sub) % g.period_x
x = (np.arange(g.n) + 0.5) * g.dx
p = g.period_x


def image_row(w):
    row = (np.abs(xs - p / 2) < w / 2).reshape(g.n, sub).mean(1)
    m = torch.tensor(np.tile(row, (g.n, 1)), dtype=torch.float32)
    return im.image(m, [0, 1, 2])[:, 0].numpy() / th      # [F, n] normalised


def width(I, D):
    """printed width of the line centred at p/2 (first period)."""
    k = int(p / g.dx) + 2
    xx, v = x[:k], D * I[:k] - 1
    c = np.nonzero(np.diff(np.sign(v)))[0]
    if len(c) < 2:
        return 0.0 if v[k // 2] < 0 else p
    xe = [xx[i] - v[i] * (xx[i + 1] - xx[i]) / (v[i + 1] - v[i]) for i in c[:2]]
    return xe[1] - xe[0]


rows = []
for w, mask_err in [(w, me) for me in (2, 0) for w in np.arange(18, 47, 2.0)]:
    I0 = image_row(w)
    # dose-to-size at nominal focus (bisection)
    lo, hi = 0.3, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if width(I0[0], mid) < 37 else (lo, mid)
    D0 = 0.5 * (lo + hi)
    widths = []
    for dw in sorted({-mask_err, 0, mask_err}):
        Iw = image_row(w + dw)
        for f in range(3):
            for dd in (0.95, 1.0, 1.05):
                widths.append(width(Iw[f], D0 * dd))
    wmin, wmax = min(widths), max(widths)
    # NILS at nominal
    v = I0[0] * D0; k = int(p / g.dx) + 2
    xe = p / 2 + 18.5; i = np.searchsorted(x, xe)
    nils = 37 * abs((v[i] - v[i - 1]) / g.dx) / np.interp(xe, x, v)
    meef = (width(image_row(w + 1)[0], D0) - width(image_row(w - 1)[0], D0)) / 2
    rows.append(dict(mask_err=mask_err, mask_w=w, dose=D0, nils=nils, meef=meef, w_min=wmin, s_min=p - wmax,
                     pinch_margin=wmin - 28, bridge_margin=(p - wmax) - 28,
                     sum_margin=wmin - 28 + p - wmax - 28))
    log(f"maskerr={mask_err} w={w:.0f} dose={D0:.3f} NILS={nils:.3f} MEEF={meef:.2f} pinch m={wmin-28:+.2f} "
        f"bridge m={p-wmax-28:+.2f} sum={wmin-28+p-wmax-28:+.2f}")
save_json("e01b.json", rows)
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for a, me, ttl in ((ax[0], 2, "with mask CD error +-2 nm in corners"), (ax[1], 0, "without mask CD error")):
    R = [r for r in rows if r["mask_err"] == me]
    W = [r["mask_w"] for r in R]
    a.plot(W, [r["pinch_margin"] for r in R], "o-", c="#d62728", label="pinch margin")
    a.plot(W, [r["bridge_margin"] for r in R], "s-", c="#1f77b4", label="bridge margin")
    a.plot(W, [r["sum_margin"] for r in R], "k--", label="sum")
    a.axhline(0, c="gray", lw=0.6); a.set_xlabel("mask line width (nm); dose re-centred to 37 nm")
    a.set_ylabel("worst-corner margin (nm)"); a.legend(fontsize=7); a.set_title(ttl, fontsize=9)
R = [r for r in rows if r["mask_err"] == 2]; W = [r["mask_w"] for r in R]
ax2 = ax[2]; ax2.plot(W, [r["nils"] for r in R], "o-", label="NILS"); ax2.set_ylabel("NILS")
ax3 = ax2.twinx(); ax3.plot(W, [r["meef"] for r in R], "s--", c="tab:orange"); ax3.set_ylabel("MEEF", color="tab:orange")
ax2.set_xlabel("mask line width (nm)"); ax2.set_title("NILS (blue) rises, MEEF (orange) rises faster", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e01b_bias_dose_nils.png"), dpi=130)

# ---- decomposition of the printed-width range by factor ------------------
dec = []
for w in (20.0, 38.0):
    I0 = image_row(w)
    lo, hi = 0.3, 4.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if width(I0[0], mid) < 37 else (lo, mid)
    D0 = 0.5 * (lo + hi)
    rng_dose = width(I0[0], D0 * 1.05) - width(I0[0], D0 * 0.95)
    rng_focus = max(width(I0[f], D0) for f in range(3)) - min(width(I0[f], D0) for f in range(3))
    rng_mask = width(image_row(w + 2)[0], D0) - width(image_row(w - 2)[0], D0)
    dec.append(dict(mask_w=w, dose_range=rng_dose, focus_range=rng_focus, mask_range=rng_mask))
    log(f"decomp w={w}: CD range from dose+-5%={rng_dose:.2f}  focus+-40={rng_focus:.2f}  mask+-2={rng_mask:.2f} nm")
save_json("e01b_decomp.json", dec)
