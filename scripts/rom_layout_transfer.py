#!/usr/bin/env python3
"""X-24: does a recipe-response POD basis transfer when the layout changes?

X-23's POD basis was built from optimal mask changes on ICCAD13 case 3.
Its vectors are pixel fields anchored to case 3's edges. This tests how far
that transfers:

  translation   exact by construction -- the optics is a circular convolution,
                so integer shifts commute with the model (checked numerically,
                7e-7); the basis just has to be shifted with the layout.
  mod_small     case 3 with one small shape moved by 4 px (16 nm)
  mod_large     case 3 with one shape removed and a new one added elsewhere
  case1         a different ICCAD13 clip

For each layout: a reference solve (modified layouts warm-start from case 3's
P0, as one would after an ECO; case 1 starts cold), then 6 recipes with
truth = 60 s warm re-optimisation from that layout's reference. Candidates:
  case3_pod      Gauss-Newton (full model) in the span of case 3's 13 snapshots
  case3_family   same, case 3's same-knob snapshots only
  own_loo        POD from this layout's own other 5 recipes (reference point:
                 what new snapshots on this layout would buy)
  warm12/warm20  12 / 20 full GN steps from the layout's reference
  operator_r     the purely operator-based reduced model (X-20's ReducedILT),
                 built at THIS layout's reference from its exact J^T J spectrum
                 -- needs no snapshots, so it has no transfer problem by
                 construction; the question is only its ceiling

Usage::

    PYTHONPATH=src python scripts/rom_layout_transfer.py
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
from gril.ilt.gauss_newton import ILTLeastSquares, solve_levenberg_marquardt
from gril.ilt.reduced_order import ReducedILT, gn_spectrum
from rom_pod_study import family, gn_in_span, pod_basis

OUT = os.path.join(base.ROOT, "results/rom_layout_transfer")
POD_DIR = os.path.join(base.ROOT, "results/rom_pod")
PICK = ["threshold x0.95", "threshold x1.05", "dose 0.97", "dose 1.03", "bias -1px", "bias +1px"]


load_target = base.load_target


def layouts(case3: torch.Tensor) -> dict[str, torch.Tensor]:
    """Edits by connected-component label: shapes' bounding boxes overlap in
    case 3, so editing by box would also cut into neighbours."""
    import scipy.ndimage as ndi

    lbl, n = ndi.label(case3.numpy())
    lbl = torch.tensor(lbl)
    sizes = {i: int((lbl == i).sum()) for i in range(1, n + 1)}
    small_id = min(sizes, key=lambda i: (sizes[i], -int(torch.nonzero(lbl == i)[:, 0].max())))
    small = case3.clone()
    shape = (lbl == small_id).float()
    small[shape > 0] = 0
    small = torch.maximum(small, torch.roll(shape, 4, dims=1))   # moved right by 4 px (16 nm)
    large = case3.clone()
    large[lbl == 6] = 0                                         # one shape removed
    large[145:160, 230:300] = 1                                 # new 15 x 70 px bar
    for name, t, expect in (("mod_small", small, n), ("mod_large", large, n)):
        got = ndi.label(t.numpy())[1]
        assert got == expect, f"{name}: {got} shapes, expected {expect} (merged or split by the edit)"
    return {"mod_small": small, "mod_large": large, "case1": load_target(1)}


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    case3 = load_target(3)
    p0_case3 = torch.load(os.path.join(base.OUT, "p0.pt"))
    snaps = {rec["name"]: torch.load(os.path.join(POD_DIR, f"p_warm_{rec['name'].replace(' ', '_')}.pt")) - p0_case3
             for rec in base.RECIPES}
    recipes = [r for r in base.RECIPES if r["name"] in PICK]

    for name, target in layouts(case3).items():
        path = os.path.join(OUT, f"{name}.json")
        if os.path.exists(path):
            continue
        start = p0_case3 if name.startswith("mod") else base.INIT_SCALE * (2 * target - 1)
        t0 = time.time()
        p_ref, _ = base.gn(litho, target, start, base.REF_BUDGET)
        ref_seconds = time.time() - t0

        torch.save(p_ref, os.path.join(OUT, f"{name}_p_ref.pt"))
        truths = {}
        for rec in recipes:
            lm, tgt = base.recipe_setup(litho, target, rec)
            truths[rec["name"]], _ = base.gn(lm, tgt, p_ref, base.WARM_BUDGET)
            torch.save(truths[rec["name"]], os.path.join(OUT, f"{name}_truth_{rec['name'].replace(' ', '_')}.pt"))

        t0 = time.time()
        model = ILTLeastSquares(target, litho, base.BETA)
        spec = gn_spectrum(model, model.state(p_ref), batch=16)
        with torch.no_grad():
            i_ref = litho.aerial_nominal(torch.sigmoid(base.BETA * p_ref))
        roms = {r: ReducedILT(spec, r, p_ref, i_ref, base.BASE.dose_nom, base.BETA) for r in (100, 200)}
        del spec
        operator_offline = time.time() - t0

        rows = []
        for rec in recipes:
            h = rec["name"]
            lm, tgt = base.recipe_setup(litho, target, rec)
            truth, truth_print = base.score(lm, tgt, truths[h])
            frozen, _ = base.score(lm, tgt, p_ref, truth_print)
            gap = frozen["l2"] - truth["l2"]
            row = {"recipe": rec, "truth": truth, "frozen": frozen}

            def record(key, p, seconds):
                s, _ = base.score(lm, tgt, p, truth_print)
                s["recovery"] = (frozen["l2"] - s["l2"]) / gap if gap else None
                s["seconds"] = seconds
                row[key] = s

            bases = {
                "case3_pod": list(snaps.values()),
                "case3_family": [d for n, d in snaps.items() if family(n) == family(h)],
                "own_loo": [truths[n] - p_ref for n in truths if n != h],
            }
            for key, deltas in bases.items():
                t0 = time.time()
                p = gn_in_span(lm, tgt, p_ref, pod_basis(deltas))
                record(key, p, time.time() - t0)

            for r, rom in roms.items():
                t0 = time.time()
                a, _ = rom.solve(lm.cfg.target_density, lm.cfg.print_steepness, lm.cfg.dose_nom, tgt)
                record(f"operator_r{r}", rom.params(a), time.time() - t0)

            snap = {}
            t0 = time.time()
            solve_levenberg_marquardt(tgt, lm, p_ref, iterations=20, beta=base.BETA, max_cg=10,
                                      on_iteration=lambda it, p, el: snap.__setitem__(it, (p.clone(), el)))
            for it in (12, 20):
                record(f"warm{it}", snap[it][0], snap[it][1])
            rows.append(row)
            print(f"[{name:9s} {h:15s}] gap {gap:5.0f} | case3_pod {100 * row['case3_pod']['recovery']:5.0f}% "
                  f"case3_family {100 * row['case3_family']['recovery']:5.0f}% own_loo {100 * row['own_loo']['recovery']:5.0f}% "
                  f"| operator r100/r200 {100 * row['operator_r100']['recovery']:4.0f}%/{100 * row['operator_r200']['recovery']:4.0f}% "
                  f"| warm12 {100 * row['warm12']['recovery']:4.0f}% warm20 {100 * row['warm20']['recovery']:4.0f}%",
                  flush=True)
        json.dump({"layout": name, "reference_seconds": ref_seconds, "operator_offline_seconds": operator_offline,
                   "rows": rows, "protocol": __doc__,
                   "provenance": base.provenance()}, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
