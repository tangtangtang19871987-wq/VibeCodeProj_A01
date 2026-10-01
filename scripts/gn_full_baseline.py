#!/usr/bin/env python3
"""X-19: Gauss-Newton (LM-CG) on all 10 ICCAD13 cases, wall-clock matched to X-17.

X-18 (scripts/optimizer_benchmark.py) found Gauss-Newton reaches the lowest
L2 of six optimizers on cases 3 and 6 at an equal 720 s budget. Two cases is
a small sample, so this runs it on all 10, giving each case EXACTLY the
wall-clock X-17's L-BFGS (iterations=100) used on that case (read from
results/iccad13_ilt_lbfgs100/caseN.json), from the same cold start. Output
uses run_iccad13's per-case JSON layout so existing tooling reads it.

Usage::

    PYTHONPATH=src python scripts/gn_full_baseline.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from gril.data.glp import Design
from gril.experiments.run_iccad13 import provenance, score_mask
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
REF_DIR = os.path.join(ROOT, "results/iccad13_ilt_lbfgs100")
OUT = os.path.join(ROOT, "results/iccad13_ilt_gn")
ILT = dict(optimizer="gauss_newton", gn_max_cg=10, mask_steepness=8.0, init_scale=0.5,
           iterations=100_000)  # gn_max_cg=10: X-18's tuning-stage selection


def main() -> None:
    torch.set_num_threads(4)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for case in range(1, 11):
        path = os.path.join(OUT, f"case{case}.json")
        if os.path.exists(path):
            results[case] = json.load(open(path))
            continue
        budget = json.load(open(os.path.join(REF_DIR, f"case{case}.json")))["seconds"]
        target = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{case}.glp").centred_raster(2048))
        res = solve(target, litho, ILTConfig(time_budget_s=budget, **ILT))
        rec = {
            "case": case, "budget_s": budget, "seconds": res.seconds,
            "iterations": res.iterations_run, "n_func_evals": res.n_func_evals,
            "scores": score_mask(res.mask, target, litho, [15, 3]),
            "loss_history": res.loss_history, "time_history": res.time_history,
        }
        json.dump(rec, open(path, "w"), indent=2)
        torch.save(res.mask.to(torch.uint8), os.path.join(OUT, f"mask{case}.pt"))
        results[case] = rec
        s = rec["scores"]
        print(f"[case {case}] L2 {s['l2']:.0f}  PVB {s['pvb']:.0f}  EPE@15 {s['epe@15nm']}  "
              f"EPE@3 {s['epe@3nm']}  it={res.iterations_run}  ({res.seconds:.0f}s / budget {budget:.0f}s)",
              flush=True)
    keys = results[1]["scores"].keys()
    summary = {
        "experiment_id": "iccad13_ilt_gn", "ilt": ILT, "budget_source": REF_DIR,
        "provenance": provenance(),
        "per_case": {str(c): r["scores"] for c, r in results.items()},
        "aggregate": {k: sum(r["scores"][k] for r in results.values()) / len(results) for k in keys},
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
