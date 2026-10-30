"""E15  Does minimax grouping of conflicting site pairs generalize to random clips?

E5b found that minimax over a (t2t, line-end) group reached a point the
weighted sum never reached.  Here every tip-to-tip gap of 6 random clips
gets its own minimax group (its three t2t cuts + the two facing line-end
sites), compared with the reference weighted-sum recipe.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from common import GOOD, Problem, Recipe, log, random_clip, run_eval, save_json, summarize


def groups(P):
    g = {}
    for i, t in enumerate(P.sites.tag):
        if P.kind[i] == "t2t":
            key = t.split("@")[0]
            g.setdefault(key, [])
            g[key].append(i)
    out = []
    for key, ids in g.items():
        a, b = key.split("->")
        lend = [j for j, t in enumerate(P.sites.tag) if P.kind[j] == "lend" and t in (f"{a}:top", f"{b}:bot")]
        out.append(ids + lend)
    return out


rows = []
for seed in range(1, 7):
    P = Problem(random_clip(seed))
    G = groups(P)
    gid = sorted({i for g in G for i in g})
    for name, extra in (("weighted sum", {}), ("minimax per t2t group", {"balance_sites": G})):
        for wb in (10.0, 100.0):
            _, rho, _, r = run_eval(P, Recipe(**{**GOOD, "w_bridge": wb, **extra}))
            m = r["margin"]
            rows.append(dict(seed=seed, config=name, w_bridge=wb, group_hot=int((m[gid] < 0).sum()),
                             group_min=float(m[gid].min()), other_hot=int((np.delete(m, gid) < 0).sum()),
                             **summarize(r)))
            log(seed, name, wb, {k: rows[-1][k] for k in ("n_hot", "group_hot", "group_min", "other_hot", "n_t2t", "n_lend", "topo_bridge")})
            save_json("e15.json", rows)
for name in ("weighted sum", "minimax per t2t group"):
    for wb in (10.0, 100.0):
        R = [r for r in rows if r["config"] == name and r["w_bridge"] == wb]
        log("TOTAL", name, wb, "n_hot", sum(r["n_hot"] for r in R), "group_hot", sum(r["group_hot"] for r in R),
            "other_hot", sum(r["other_hot"] for r in R), "topo_bridge", sum(r["topo_bridge"] for r in R))
