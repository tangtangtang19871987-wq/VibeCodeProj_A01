"""E1  At pitch 74.5 nm the dense image is a pure cosine; NILS <= pi*gamma.

Claim tested: "a near-square-wave image is impossible at this pitch; the
achievable NILS is bounded, so ILT can only redistribute a fixed contrast
budget."  We (a) measure the harmonic content of dense L/S images for several
duty cycles, (b) let ILT *maximize* worst-corner log-contrast on the dense
clip and check that the image stays a cosine and NILS stays below pi*gamma.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG, Recipe, dense_clip, log, save_json, Problem
from litho import Grid, Imager


def row_metrics(I_row, dx, p, w, th=None):
    n = len(I_row)
    F = np.fft.rfft(I_row) / n
    per = int(round(n * dx / p))
    h = np.abs(F[[0, per, 2 * per, 3 * per]])
    gamma = 2 * h[1] / h[0]
    x = (np.arange(n) + 0.5) * dx
    # numeric NILS at the target edge of the line centred at p/2
    xe = p / 2 + w / 2
    i = np.searchsorted(x, xe)
    dI = (I_row[i] - I_row[i - 1]) / dx
    Ie = np.interp(xe, x, I_row)
    nils = w * abs(dI) / Ie
    pred = w * (2 * np.pi / p) * gamma * abs(np.sin(2 * np.pi * (xe - p / 2) / p)) / \
        (1 + gamma * np.cos(2 * np.pi * (xe - p / 2) / p))
    return h / h[0], gamma, nils, pred


def main():
    g = Grid()
    im = Imager(g, keep=0.98)
    out = {"duty": []}
    sub = 8
    xs = ((np.arange(g.n * sub) + 0.5) * g.dx / sub) % g.period_x
    for w in [20, 25, 30, 37, 45, 50]:
        row = (np.abs(xs - g.period_x / 2) < w / 2).reshape(g.n, sub).mean(1)
        m = torch.tensor(np.tile(row, (g.n, 1)), dtype=torch.float32)
        aer = im.aerial(m, [0])[0, 0].numpy()
        res = im.image(m, [0])[0, 0].numpy()
        ha, ga, na, pa = row_metrics(aer, g.dx, g.period_x, 37.0)
        hr, gr, nr, pr = row_metrics(res, g.dx, g.period_x, 37.0)
        out["duty"].append(dict(mask_w=w, harm_aerial=ha.tolist(), gamma_aerial=ga,
                                harm_resist=hr.tolist(), gamma_resist=gr,
                                nils37_resist=nr, nils37_pred_cos=pr))
        log(f"mask w={w}: harmonics(resist)={np.round(hr,4)} gamma={gr:.3f} "
            f"NILS@37={nr:.3f} cos-pred={pr:.3f} pi*gamma={np.pi*gr:.3f}")

    # (b) ILT maximizing worst-corner log-contrast on the dense clip
    P = Problem(dense_clip())
    rows = []
    for w_ils in [0.0, 1.0, 10.0]:
        rc = Recipe(iters=300, w_fid=1.0, w_bridge=10, w_pinch=10, w_ils=w_ils)
        th, rho, hist = P.run(rc)
        mh = P.hard_mask(rho)
        res = P.im.image(mh, [0])[0].numpy()
        r = P.evaluate(rho, topo=False)
        hr, gr, nr, pr = row_metrics(res[g.n // 2], g.dx, g.period_x, 37.0)
        rows.append(dict(w_ils=w_ils, harm=hr.tolist(), gamma=gr, nils=nr, pi_gamma=np.pi * gr,
                         min_bridge=r["min_bridge_all"], min_pinch=r["min_pinch_all"]))
        log(f"ILT w_ils={w_ils}: harmonics={np.round(hr,4)} gamma={gr:.3f} NILS={nr:.3f} "
            f"(pi*gamma={np.pi*gr:.3f}) min bridge/pinch margin={r['min_bridge_all']:.1f}/{r['min_pinch_all']:.1f}")
    out["ilt"] = rows
    # theoretical ceiling: resist blur MTF at the pitch frequency
    blur_mtf = float(np.exp(-2 * (np.pi * im.op.blur_nm / g.period_x) ** 2))
    out["blur_mtf"] = blur_mtf
    out["nils_ceiling"] = np.pi * blur_mtf
    save_json("e01.json", out)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ws = [d["mask_w"] for d in out["duty"]]
    ax[0].plot(ws, [d["harm_resist"][1] for d in out["duty"]], "o-", label="1st harmonic / DC")
    ax[0].plot(ws, [d["harm_resist"][2] for d in out["duty"]], "s-", label="2nd harmonic / DC")
    ax[0].plot(ws, [d["harm_resist"][3] for d in out["duty"]], "^-", label="3rd harmonic / DC")
    ax[0].set_xlabel("mask line width (nm), pitch 74.5"); ax[0].set_ylabel("relative amplitude")
    ax[0].legend(fontsize=7); ax[0].set_title("Dense L/S resist image: only DC + 1st harmonic", fontsize=9)
    ax[1].plot(ws, [d["nils37_resist"] for d in out["duty"]], "o-", label="NILS (numerical)")
    ax[1].plot(ws, [d["nils37_pred_cos"] for d in out["duty"]], "x--", label="pure-cosine prediction")
    ax[1].axhline(np.pi * blur_mtf, color="k", ls=":", label=r"ceiling $\pi\cdot$MTF$_{blur}$")
    for r in rows:
        ax[1].axhline(r["nils"], color="r", alpha=0.3)
    ax[1].text(ws[0], rows[-1]["nils"] + 0.03, "ILT max-contrast results", color="r", fontsize=7)
    ax[1].set_xlabel("mask line width (nm)"); ax[1].set_ylabel("NILS at 37 nm edge")
    ax[1].legend(fontsize=7); ax[1].set_title("NILS is bounded (no square wave possible)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e01_cosine_limit.png"), dpi=130)


if __name__ == "__main__":
    main()
