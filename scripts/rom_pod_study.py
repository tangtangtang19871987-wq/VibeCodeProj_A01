#!/usr/bin/env python3
"""X-23: snapshot POD vs the operator basis for recipe changes.

X-21 showed the operator (tangent) basis at P0 misses most of the optimal
change because that change lives largely in J's null space at P0. A POD
basis built from *other recipes' actual optimal changes* contains those
null-space components, so it may succeed where the operator basis failed.

Snapshots are produced by the exact physics solver itself (no external
data), but by model-reduction convention POD is a snapshot/data-driven
method and is reported separately from the purely operator-based results.

For every recipe h (all 13 of X-20):
  * snapshots = warm re-optimised changes  dP_i = P_warm_i - P0
  * leave-one-out POD: basis from the other 12 recipes
  * same-family POD:   basis from the other recipes of the same knob only
                       (threshold / dose / bias) -- a one-knob sweep
  For each basis: (a) projection of the true dP_h (an upper bound) and
  (b) Gauss-Newton on the full model restricted to the POD span, with the
  reduced Jacobian J U recomputed every step (k JVPs, k <= 12).

Usage::

    PYTHONPATH=src python scripts/rom_pod_study.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

import rom_recipe_study as base
from gril.ilt.recipe_rom import pod_basis, solve_in_subspace

OUT = os.path.join(base.ROOT, "results/rom_pod")


def family(name: str) -> str:
    return name.split()[0]


def gn_in_span(lm, tgt, p0, u, iterations=15, damping=1.0):
    """Levenberg-Marquardt over P = P0 + U a with the FULL model and J U refreshed each step."""
    return solve_in_subspace(tgt, lm, p0, u, beta=base.BETA, iterations=iterations, damping=damping)


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    target = base.load_target(base.CASE)
    p0 = torch.load(os.path.join(base.OUT, "p0.pt"))

    warm = {}
    for rec in base.RECIPES:
        path = os.path.join(OUT, f"p_warm_{rec['name'].replace(' ', '_')}.pt")
        if not os.path.exists(path):
            lm, tgt = base.recipe_setup(litho, target, rec)
            p, _ = base.gn(lm, tgt, p0, base.WARM_BUDGET)
            torch.save(p, path)
            print(f"[truth] {rec['name']}", flush=True)
        warm[rec["name"]] = torch.load(path)

    rows = []
    for rec in base.RECIPES:
        h = rec["name"]
        lm, tgt = base.recipe_setup(litho, target, rec)
        truth, truth_print = base.score(lm, tgt, warm[h])
        frozen, _ = base.score(lm, tgt, p0, truth_print)
        gap = frozen["l2"] - truth["l2"]
        delta = (warm[h] - p0).reshape(-1)
        row = {"recipe": rec, "truth": truth, "frozen": frozen}
        bases = {
            "loo": [warm[n] - p0 for n in warm if n != h],
            "family": [warm[n] - p0 for n in warm if n != h and family(n) == family(h)],
        }
        for key, deltas in bases.items():
            u = pod_basis(deltas)
            proj = u @ (u.T @ delta)
            s, _ = base.score(lm, tgt, (p0.reshape(-1) + proj).reshape(p0.shape), truth_print)
            s["recovery"] = (frozen["l2"] - s["l2"]) / gap
            s["captured_norm_fraction"] = float(proj.norm() / delta.norm())
            s["dims"] = u.shape[1]
            row[f"{key}_project"] = s
            t0 = time.time()
            p = gn_in_span(lm, tgt, p0, u)
            s, _ = base.score(lm, tgt, p, truth_print)
            s["recovery"] = (frozen["l2"] - s["l2"]) / gap
            s["seconds"] = time.time() - t0
            s["dims"] = u.shape[1]
            row[f"{key}_gn"] = s
        rows.append(row)
        print(f"[{h:15s}] gap {gap:5.0f} | LOO proj {100 * row['loo_project']['recovery']:4.0f}% "
              f"gn {100 * row['loo_gn']['recovery']:4.0f}% ({row['loo_gn']['seconds']:.1f}s) | "
              f"family({row['family_gn']['dims']}) proj {100 * row['family_project']['recovery']:4.0f}% "
              f"gn {100 * row['family_gn']['recovery']:4.0f}% ({row['family_gn']['seconds']:.1f}s)", flush=True)
    json.dump({"protocol": __doc__, "rows": rows, "provenance": base.provenance()},
              open(os.path.join(OUT, "pod.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
