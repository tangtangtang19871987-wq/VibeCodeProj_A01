#!/usr/bin/env python3
"""X-25: why does case 3's recipe-response POD basis transfer to other layouts?

X-24 found the case-3 POD basis recovers a median 94% on a *different* clip
(case 1), contradicting the expectation that pixel fields anchored to case 3's
edges would be useless elsewhere. Hypothesis: what transfers is a global,
geometry-independent component -- e.g. a uniform shift of the level-set P,
i.e. a uniform mask bias. This tests snapshot-FREE bases defined from physics
or geometry at each layout's own reference, against random controls:

  const          P + c                          (1 dim)  uniform level-set offset
  normal         P + c |grad P_ref|             (1 dim)  uniform normal edge displacement
  const+normal                                  (2 dims)
  lowfreq9       the 9 lowest real Fourier modes (|k| <= 1 per axis): slowly varying bias
  random13       13 i.i.d. Gaussian fields                 (control)
  rand_band13    13 random band-limited fields             (control)
  case3_pod      case 3's 13 snapshots (as in X-24, for reference)

On case 3 itself and X-24's three layouts, 6 recipes each, Gauss-Newton on the
full model restricted to each span; recovery vs the same 60 s warm truths.

Usage::

    PYTHONPATH=src python scripts/rom_basis_physics.py
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
from gril.ilt.reduced_order import band_basis
from rom_layout_transfer import PICK, layouts, load_target
from rom_pod_study import gn_in_span, pod_basis

OUT = os.path.join(base.ROOT, "results/rom_basis_physics")
POD_DIR = os.path.join(base.ROOT, "results/rom_pod")
LT_DIR = os.path.join(base.ROOT, "results/rom_layout_transfer")


def grad_mag(p: torch.Tensor) -> torch.Tensor:
    gy, gx = torch.gradient(p)
    return (gx * gx + gy * gy).sqrt()


def bases_for(p_ref: torch.Tensor, case3_snaps: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    h, w = p_ref.shape
    g = torch.Generator().manual_seed(0)
    band = band_basis(h, w)
    rand_band = [(torch.randn(band.shape[0], generator=g) @ band.reshape(band.shape[0], -1)).view(h, w)
                 for _ in range(13)]
    return {
        "const": [torch.ones(h, w)],
        "normal": [grad_mag(p_ref)],
        "const+normal": [torch.ones(h, w), grad_mag(p_ref)],
        "lowfreq9": list(band_basis(h, w, half=1)),
        "random13": [torch.randn(h, w, generator=g) for _ in range(13)],
        "rand_band13": rand_band,
        "case3_pod": case3_snaps,
    }


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    case3 = load_target(3)
    p0 = torch.load(os.path.join(base.OUT, "p0.pt"))
    snaps = [torch.load(os.path.join(POD_DIR, f"p_warm_{r['name'].replace(' ', '_')}.pt")) - p0 for r in base.RECIPES]

    flat = torch.stack([s.reshape(-1) for s in snaps])
    dc_fraction = (flat.mean(1) ** 2 / (flat ** 2).mean(1)).tolist()
    lf = band_basis(*p0.shape, half=1).reshape(9, -1)
    lf = lf / lf.norm(dim=1, keepdim=True)
    lowfreq_fraction = (((flat @ lf.T) ** 2).sum(1) / (flat ** 2).sum(1)).tolist()
    print("case-3 snapshot energy in DC: median %.3f | in 9 lowest modes: median %.3f"
          % (sorted(dc_fraction)[6], sorted(lowfreq_fraction)[6]), flush=True)

    layout_set = {"case3": (case3, p0, {r["name"]: os.path.join(POD_DIR, f"p_warm_{r['name'].replace(' ', '_')}.pt")
                                        for r in base.RECIPES})}
    for name, target in layouts(case3).items():
        layout_set[name] = (target, torch.load(os.path.join(LT_DIR, f"{name}_p_ref.pt")),
                            {r: os.path.join(LT_DIR, f"{name}_truth_{r.replace(' ', '_')}.pt") for r in PICK})

    results = {"snapshot_dc_fraction": dc_fraction, "snapshot_lowfreq9_fraction": lowfreq_fraction}
    for name, (target, p_ref, truth_paths) in layout_set.items():
        bases = {k: pod_basis(v) for k, v in bases_for(p_ref, snaps).items()}
        rows = []
        for rec in [r for r in base.RECIPES if r["name"] in PICK]:
            lm, tgt = base.recipe_setup(litho, target, rec)
            truth, truth_print = base.score(lm, tgt, torch.load(truth_paths[rec["name"]]))
            frozen, _ = base.score(lm, tgt, p_ref, truth_print)
            gap = frozen["l2"] - truth["l2"]
            row = {"recipe": rec["name"], "gap": gap}
            for key, u in bases.items():
                t0 = time.time()
                s, _ = base.score(lm, tgt, gn_in_span(lm, tgt, p_ref, u), truth_print)
                row[key] = {"recovery": (frozen["l2"] - s["l2"]) / gap, "l2": s["l2"], "dims": u.shape[1],
                            "seconds": time.time() - t0}
            rows.append(row)
            print(f"[{name:9s} {rec['name']:15s}] " + " ".join(
                f"{k} {100 * row[k]['recovery']:4.0f}%" for k in bases), flush=True)
        results[name] = rows
        json.dump(results | {"protocol": __doc__, "provenance": base.provenance()},
                  open(os.path.join(OUT, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
