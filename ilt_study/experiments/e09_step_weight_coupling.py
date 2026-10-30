"""E9  Weight x step-size coupling: plain gradient descent vs Adam.

Claim tested: "raising a weight is like raising the step size, so it can
make the optimizer oscillate / overshoot".  Expected nuance: this holds for
plain gradient descent (step = lr * weight * grad) but an adaptive optimizer
such as Adam normalizes the step, so for Adam only the *relative* weights
matter.  We scale both site weights by 1x, 10x, 100x.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, summarize

P = Problem(random_clip(1))
isb = np.isin(P.kind, ["bridge", "t2t", "diag"])
low = P.lower.numpy()
rows = []
# calibrate the plain-GD learning rate so that at weight scale x1 its first
# step matches Adam's first step (lr 0.1 per pixel, 95th percentile)
_th0 = None
_rc = Recipe(**{**GOOD, "iters": 1, "optimizer": "sgd", "lr": 1.0})
_x0 = torch.clamp(P.T, 0.02, 0.98); _th0 = torch.log(_x0 / (1 - _x0))
_th1, _, _ = P.run(_rc, theta0=_th0)
_step = (_th1 - _th0).abs().flatten()
SGD_LR = float(0.1 / torch.quantile(_step[_step > 0], 0.95))
log("calibrated SGD lr", SGD_LR)
for opt, lr in (("adam", 0.1), ("sgd", SGD_LR)):
    for s in (1, 10, 100):
        rc = Recipe(**{**GOOD, "w_bridge": 10.0 * s, "w_pinch": 10.0 * s, "w_fid": 1.0 * s,
                       "w_sraf": 50.0 * s, "optimizer": opt, "lr": lr, "log_every": 5})
        _, rho, hist, r = run_eval(P, rc)
        it = [h["it"] for h in hist]
        mb = [float((h["q"] - low)[isb].min()) for h in hist]
        mp = [float((h["q"] - low)[~isb].min()) for h in hist]
        rows.append(dict(opt=opt, scale=s, lr=lr, it=it, soft_min_bridge=mb, soft_min_pinch=mp,
                         loss=[h["loss"] / s for h in hist], **summarize(r)))
        log(opt, s, {k: rows[-1][k] for k in ("n_hot", "n_bridge_all", "n_pinch_all", "min_bridge_all", "min_pinch_all")})
        save_json("e09.json", rows)
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, opt in zip(axs, ("adam", "sgd")):
    for r, c in zip([r for r in rows if r["opt"] == opt], ("k", "tab:blue", "tab:red")):
        ax.plot(r["it"], r["soft_min_bridge"], "-", c=c, label=f"x{r['scale']}: worst bridge (soft)")
        ax.plot(r["it"], r["soft_min_pinch"], ":", c=c, label=f"x{r['scale']}: worst pinch (soft)")
    ax.axhline(0, c="gray", lw=0.6); ax.set_ylim(-30, 10); ax.set_xlabel("iteration")
    ax.set_title(f"{opt}: all weights scaled", fontsize=9); ax.legend(fontsize=6)
axs[0].set_ylabel("soft worst margin in the cost (nm)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e09_step_weight_coupling.png"), dpi=130)
