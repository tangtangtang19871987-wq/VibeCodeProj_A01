"""E3  Edge displacement per dose = (dD/D) / ILS, site by site.

Claim tested: "the margin a site loses to a process excursion scales as
1/ILS at its edges, so margin budgets are set by local image slope".
For every pinch / lateral-bridge / line-end / t2t site of the converged
reference masks we compare the measured d(length)/d(dose) at nominal with the
first-order prediction sum_edges 1/ILS_edge.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, get_good, log, save_json
from layout import measure

rows = []
for seed in [1, 2, 3]:
    P, theta, rho, r0 = get_good(seed)
    I = P.images(P.hard_mask(rho), [0])[0]
    v = P.S.sample(I).numpy()                        # [sites, samples] normalised image
    step = P.S.step.numpy()
    def precise_bright(D):
        """bright length with linear-interpolated crossings (no sample quantization)."""
        out = np.zeros(len(P.kind))
        for i in range(len(P.kind)):
            ns = P.S.nsamp[i]; y = D * v[i, :ns] - 1.0; h = step[i]
            on = y > 0
            tot = on.sum() * h
            for j in np.nonzero(np.diff(np.sign(y)))[0]:
                f = y[j] / (y[j] - y[j + 1])          # crossing fraction inside [j, j+1]
                # sample-count assigns the whole interval to one side; correct it
                tot += (f - 0.5) * h if on[j] else (0.5 - f) * h
            out[i] = tot
        return out
    gap = np.isin(P.kind, ["bridge", "t2t", "diag"])
    Bp, Bm = precise_bright(1.01), precise_bright(0.99)
    meas = np.where(gap, -(Bp - Bm), (Bp - Bm)) / 0.02  # nm per unit relative dose
    for i in range(len(P.kind)):
        ns = P.S.nsamp[i]
        y = v[i, :ns] - 1.0
        c = np.nonzero(np.diff(np.sign(y)))[0]
        if len(c) == 0:
            continue
        inv = 0.0
        for j in c:
            Ie = 1.0
            dI = (v[i, j + 1] - v[i, j]) / step[i]
            inv += 1.0 / max(abs(dI) / Ie, 1e-6)      # 1/ILS in nm
        # bright length grows with dose at every crossing; gaps shrink
        pred = inv if P.kind[i] in ("pinch", "lend") else -inv
        if P.kind[i] == "lend":
            pred = inv if len(c) == 1 else inv       # end crossing only inside the cut
        rows.append(dict(seed=seed, kind=str(P.kind[i]), meas=float(meas[i]), pred=float(pred),
                         n_edges=int(len(c)), ils_min=float(1 / max(inv / len(c), 1e-6))))
k = np.array([r["kind"] for r in rows]); m = np.array([r["meas"] for r in rows]); p = np.array([r["pred"] for r in rows])
ok = np.isfinite(m) & (np.abs(p) < 400)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cols = {"pinch": "#d62728", "lend": "#ff7f0e", "bridge": "#1f77b4", "t2t": "#17becf", "diag": "#9467bd"}
summary = {}
for kk, c in cols.items():
    s = ok & (k == kk)
    if s.sum() == 0: continue
    ax[0].scatter(p[s], m[s], s=6, c=c, label=kk, alpha=0.6)
    r = np.corrcoef(p[s], m[s])[0, 1] if s.sum() > 2 else np.nan
    slope = np.polyfit(p[s], m[s], 1)[0] if s.sum() > 2 else np.nan
    summary[kk] = dict(n=int(s.sum()), r=float(r), slope=float(slope),
                       median_abs_sens=float(np.median(np.abs(m[s]))))
    log(f"{kk}: n={s.sum()} corr={r:.3f} fit slope={slope:.3f} median |dL/dD|={np.median(np.abs(m[s])):.1f} nm/(100%)")
lim = np.percentile(np.abs(np.r_[p[ok], m[ok]]), 99)
ax[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
ax[0].set_xlabel("predicted  sum 1/ILS  (nm per 100% dose)"); ax[0].set_ylabel("measured dL/dD")
ax[0].legend(fontsize=7); ax[0].set_title("first-order dose sensitivity = 1/ILS", fontsize=9)
for kk, c in cols.items():
    s = ok & (k == kk)
    if s.sum(): ax[1].hist(np.abs(m[s]) * 0.05, bins=40, alpha=0.5, color=c, label=kk, range=(0, 15))
ax[1].set_xlabel("|length change| for a 5% dose step (nm)"); ax[1].legend(fontsize=7)
ax[1].set_title("dose sensitivity by site type", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e03_shift_vs_ils.png"), dpi=130)
save_json("e03.json", {"summary": summary, "rows": rows})
