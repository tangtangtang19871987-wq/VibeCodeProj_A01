#!/usr/bin/env python3
"""X-22: the baseline any recipe-evaluation shortcut must beat -- how fast
does a plain warm-started full re-optimisation converge?

X-20/X-21 compared reduced models against a 60 s warm re-optimisation, but
never asked how much of those 60 s is needed. This records, for all 13 X-20
recipes, the printed L2 after 1, 2, 3, 5, 8, 12 and 20 Gauss-Newton outer
steps from the reference P0 (one continuous run per recipe, snapshotted via
the solver's on_iteration hook), with elapsed wall-clock. Recovery is
measured against X-20's 60 s warm result for the same recipe.

Usage::

    PYTHONPATH=src python scripts/rom_warmstart_curve.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

import rom_recipe_study as base
from gril.ilt.gauss_newton import solve_levenberg_marquardt

OUT = os.path.join(base.ROOT, "results/rom_warmstart")
CHECKPOINTS = (1, 2, 3, 5, 8, 12, 20)


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    target = base.load_target(base.CASE)
    p0 = torch.load(os.path.join(base.OUT, "p0.pt"))

    rows = []
    for rec in base.RECIPES:
        x20 = json.load(open(os.path.join(base.OUT, f"recipe_{rec['name'].replace(' ', '_')}.json")))
        lm, tgt = base.recipe_setup(litho, target, rec)
        snaps = {}

        def hook(it, p, elapsed):
            if it in CHECKPOINTS:
                snaps[it] = (p.clone(), elapsed)

        solve_levenberg_marquardt(tgt, lm, p0, iterations=max(CHECKPOINTS), beta=base.BETA, max_cg=10,
                                  on_iteration=hook)
        frozen, warm60 = x20["frozen"]["l2"], x20["warm"]["l2"]
        curve = {}
        for it, (p, elapsed) in sorted(snaps.items()):
            s, _ = base.score(lm, tgt, p)
            s["seconds"] = elapsed
            s["recovery"] = (frozen - s["l2"]) / (frozen - warm60)
            curve[str(it)] = s
        rows.append({"recipe": rec, "frozen_l2": frozen, "warm60_l2": warm60, "curve": curve})
        print(f"[{rec['name']:15s}] " + "  ".join(
            f"it{k}:{100 * v['recovery']:.0f}%/{v['seconds']:.1f}s" for k, v in curve.items()), flush=True)
    json.dump({"protocol": __doc__, "checkpoints": CHECKPOINTS, "rows": rows,
               "provenance": base.provenance()}, open(os.path.join(OUT, "curve.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
