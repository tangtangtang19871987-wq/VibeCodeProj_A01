"""E14  Cost/checker mismatch.

Claims tested:
 (a) "a PV-band style cost can keep improving while the hotspot count does
     not" -- track the PV-band terms and the hard checker during a run;
 (b) "if the cost samples bridges only where the checker does not look
     (single central tip-to-tip cut) bridges escape around the cut" --
     single-cut cost vs triple-cut cost, both judged by the full checker.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import FIG, GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, summarize
from layout import make_sites

out = {"trace": [], "cuts": []}
# (a) PV-band cost trace vs checker
P = Problem(random_clip(2))
trace = []


def cb(it, theta):
    rho = P.rho(theta, 8.0)
    r = P.evaluate(rho)
    trace.append(dict(it=it, n_hot=r["n_hot"], n_bridge=r["n_bridge_all"] + r["topo_bridge"],
                      n_pinch=r["n_pinch_all"] + r["topo_open"]))


rc = Recipe(iters=400, w_fid=1.0, w_out=2.0, w_in=2.0, w_sraf=50.0, callback=cb, cb_every=25, log_every=25)
_, rho, hist, r = run_eval(P, rc)
pv = {h["it"]: h.get("out", 0) + h.get("in", 0) for h in hist}
for t in trace:
    t["pvband_cost"] = pv.get(t["it"], np.nan)
out["trace"] = trace
for t in trace:
    log("trace", t)

# (b) single vs triple t2t cut in the cost (checker always uses triple + topology)
for seed in (1, 2, 3, 4, 5, 6):
    clip = random_clip(seed)
    Pfull = Problem(clip)
    Psingle = Problem(clip, sites=make_sites(clip, t2t_offsets=(0.0,)), imager=Pfull.im)
    for name, PP in (("cost: single central t2t cut", Psingle), ("cost: three parallel t2t cuts", Pfull)):
        _, rho, _ = PP.run(Recipe(**GOOD))
        rr = Pfull.evaluate(rho)
        out["cuts"].append(dict(seed=seed, config=name, **summarize(rr)))
        log(seed, name, {k: out["cuts"][-1][k] for k in ("n_hot", "n_t2t", "topo_bridge")})
save_json("e14.json", out)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
it = [t["it"] for t in trace]
axs[0].plot(it, [t["pvband_cost"] for t in trace], "k-", label="PV-band cost (out+in)")
axs[0].set_yscale("log"); axs[0].set_xlabel("iteration"); axs[0].set_ylabel("PV-band cost")
a2 = axs[0].twinx()
a2.plot(it, [t["n_bridge"] for t in trace], "o-", c="#1f77b4", label="bridge hotspots")
a2.plot(it, [t["n_pinch"] for t in trace], "s-", c="#d62728", label="pinch hotspots")
a2.set_ylabel("hotspots"); a2.legend(fontsize=7, loc="upper right"); axs[0].legend(fontsize=7, loc="upper center")
axs[0].set_title("PV-band cost vs hard checker during one run", fontsize=9)
names = ["cost: single central t2t cut", "cost: three parallel t2t cuts"]
for k, nm in enumerate(names):
    R = [c for c in out["cuts"] if c["config"] == nm]
    axs[1].bar(np.arange(len(R)) + (k - 0.5) * 0.4, [c["n_t2t"] + c["topo_bridge"] for c in R], 0.4, label=nm)
axs[1].set_xlabel("clip"); axs[1].set_ylabel("t2t + topological bridges (full checker)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "e14_cost_checker_mismatch.png"), dpi=130)
