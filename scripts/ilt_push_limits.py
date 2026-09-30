#!/usr/bin/env python3
"""Push EPE further past X-17: how low can it go? (follow-up to F-LBFGS-03)

X-17 (iterations=100) already beats the paper's own EPE@15nm average, but
EPE@3nm still trails the paper's own no-generator ISPD25 reference on 9 of
10 cases (case 3 is the only one already ahead). This tests three further
levers on case 3 (the one case that's ahead, to find the true ceiling) and
case 6 (the largest remaining EPE@3nm gap, +28, despite EPE@15nm already
at 0 -- representative of where the real headroom is):

  E1. Just keep running L-BFGS longer (200, 300 iterations) -- is the
      100-150 plateau seen on case 3 in F-LBFGS-03 real, or does it
      resume with enough budget?
  E2. Two-stage hybrid: L-BFGS to convergence (100 it, cheap/fast), THEN
      warm-start Adam with the calibrated EPE-aware loss (F-EPE-01's
      method) for a further refinement pass. D1 (F-LBFGS-03) showed
      EPE-aware loss backfires when L-BFGS optimizes it from scratch;
      this tests whether Adam can clean up a good L-BFGS solution
      without that failure mode, since F-EPE-01 already showed Adam
      handles this loss term cleanly.
  E3. More L-BFGS curvature memory (lbfgs_history_size 10 -> 30) at the
      X-17 budget -- a cheap, single-variable check.

Usage::

    PYTHONPATH=src python scripts/ilt_push_limits.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from gril.data.glp import Design
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations, l2_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
BASE_LBFGS = dict(mask_steepness=8.0, init_scale=0.5, optimizer="lbfgs",
                   lbfgs_max_iter_per_step=1, lbfgs_history_size=10, lbfgs_line_search="strong_wolfe")


def score(litho, target, mask, label, seconds):
    with torch.no_grad():
        b_nom, _, _ = litho.binary(mask)
    ein, eout = epe_violations(b_nom, target, 15, 1.0)
    ein3, eout3 = epe_violations(b_nom, target, 3, 1.0)
    l2 = l2_loss(b_nom, target)
    print(
        f"{label:44s} L2={l2:7.0f}  EPE@15={ein + eout:3d}  EPE@3={ein3 + eout3:3d}  ({seconds:.0f}s)",
        flush=True,
    )
    return {"label": label, "l2": l2, "epe_at_15nm": ein + eout, "epe_at_3nm": ein3 + eout3, "seconds": seconds}


def e1_more_iterations(litho, target, iters_list=(200, 300)):
    out = []
    for it in iters_list:
        t0 = time.time()
        res = solve(target, litho, ILTConfig(iterations=it, step_size=1.0, **BASE_LBFGS))
        out.append(score(litho, target, res.mask, f"E1 L-BFGS {it} it", time.time() - t0))
    return out


def e2_hybrid(litho, target, weight_epe=2500.0, tol=3.0, adam_iters=50, adam_step=0.02):
    t0 = time.time()
    lbfgs_res = solve(target, litho, ILTConfig(iterations=100, step_size=1.0, **BASE_LBFGS))
    adam_cfg = ILTConfig(
        iterations=adam_iters, step_size=adam_step, mask_steepness=8.0,
        weight_epe=weight_epe, epe_tolerance_nm=tol, optimizer="adam",
    )
    adam_res = solve(target, litho, adam_cfg, init_params=lbfgs_res.params)
    total = time.time() - t0
    return score(
        litho, target, adam_res.mask,
        f"E2 L-BFGS(100)->Adam+EPE(w={weight_epe:.0f},tol={tol}) {adam_iters}it",
        total,
    )


def e3_more_history(litho, target, history_size=30, iterations=100):
    t0 = time.time()
    cfg_dict = dict(BASE_LBFGS)
    cfg_dict["lbfgs_history_size"] = history_size
    res = solve(target, litho, ILTConfig(iterations=iterations, step_size=1.0, **cfg_dict))
    return score(litho, target, res.mask, f"E3 L-BFGS history={history_size}, {iterations}it", time.time() - t0)


def run_case(case: int, litho) -> list[dict]:
    target = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{case}.glp").centred_raster(2048))
    print(f"=== case {case} ===", flush=True)
    records = []
    records.append(e1_more_iterations(litho, target))
    records.append(e2_hybrid(litho, target))
    records.append(e3_more_history(litho, target))
    # flatten E1's list
    flat = []
    for r in records:
        if isinstance(r, list):
            flat.extend(r)
        else:
            flat.append(r)
    return flat


def main() -> None:
    torch.set_num_threads(4)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())

    all_records = {}
    for case in (3, 6):
        all_records[str(case)] = run_case(case, litho)

    out_dir = os.path.join(ROOT, "results/ilt_push_limits")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    json.dump(all_records, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
