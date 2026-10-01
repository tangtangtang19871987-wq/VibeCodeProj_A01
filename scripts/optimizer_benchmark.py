#!/usr/bin/env python3
"""X-18: SOTA optimizers vs L-BFGS vs Gauss-Newton on ILT, wall-clock matched.

Protocol (fixed before any result was seen):

Stage "tune": ICCAD13 case 3 at 4x reduced resolution (512^2; the optics are
resolution-independent, see gril.experiments.infer.low_res_refine). Every
optimizer gets the SAME number of configurations (4) and the SAME wall-clock
budget per configuration. Selection is by final printed-image L2 -- never by
EPE, which is the comparison metric -- matching this project's standing tuning
discipline (configs/experiments/README.md).

Stage "full": each optimizer's selected configuration at full 2048^2
resolution on case 3 (hardest) and case 6 (largest remaining EPE@3nm gap to
the paper), with an identical wall-clock budget (default 720 s, ~ the per-case
cost of X-17's L-BFGS-100). Same cold start for every method (init_scale=0.5,
mask_steepness=8.0, as X-14/X-16/X-17).

Resumable: each finished run writes its own JSON and is skipped on restart.

Usage::

    PYTHONPATH=src python scripts/optimizer_benchmark.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn.functional as F

from gril.data.glp import Design
from gril.experiments.run_iccad13 import provenance
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations, l2_loss, pv_band

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
OUT = os.path.join(ROOT, "results/optimizer_benchmark")
COMMON = dict(mask_steepness=8.0, init_scale=0.5, iterations=100_000)

# Four configurations per optimizer -- the same tuning effort for each.
GRID: dict[str, list[dict]] = {
    "adam": [dict(step_size=s) for s in (0.05, 0.1, 0.2, 0.5)],
    "sf_adamw": [dict(step_size=s, sf_warmup_steps=10) for s in (0.05, 0.1, 0.2, 0.5)],
    "muon": [dict(step_size=s) for s in (0.02, 0.05, 0.1, 0.2)],
    "soap": [dict(step_size=s) for s in (0.05, 0.1, 0.2, 0.5)],
    "lbfgs": [dict(step_size=s, lbfgs_history_size=h) for s in (1.0, 0.5) for h in (10, 30)],
    "gauss_newton": [dict(gn_max_cg=5), dict(gn_max_cg=10), dict(gn_max_cg=20),
                     dict(gn_max_cg=10, gn_damping_init=0.1)],
}


def load_target(case: int, downsample: int = 1) -> torch.Tensor:
    t = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{case}.glp").centred_raster(2048))
    if downsample > 1:
        t = (F.avg_pool2d(t[None, None].float(), downsample)[0, 0] > 0.5).float()
    return t


def run_one(litho, target, optimizer, overrides, budget, pixel_nm, with_epe):
    cfg = ILTConfig(optimizer=optimizer, time_budget_s=budget, **COMMON, **overrides)
    res = solve(target, litho, cfg)
    with torch.no_grad():
        b_nom, b_max, b_min = litho.binary(res.mask)
    rec = {
        "optimizer": optimizer, "overrides": overrides, "budget_s": budget,
        "seconds": res.seconds, "iterations": res.iterations_run, "n_func_evals": res.n_func_evals,
        "final_objective": res.loss_history[-1] if res.loss_history else None,
        "l2": l2_loss(b_nom, target), "pvb": pv_band(b_max, b_min),
        # thinned curves: enough to plot loss-vs-wall-clock
        "loss_history": res.loss_history[:: max(1, len(res.loss_history) // 200)],
        "time_history": res.time_history[:: max(1, len(res.time_history) // 200)],
    }
    if with_epe:
        for tol in (15, 3):
            ein, eout = epe_violations(b_nom, target, tol, pixel_nm)
            rec[f"epe@{tol}nm"] = ein + eout
    return rec, res


def stage_tune(litho, budget: float) -> dict[str, dict]:
    target = load_target(3, downsample=4)
    best = {}
    for opt, configs in GRID.items():
        results = []
        for i, ov in enumerate(configs):
            path = os.path.join(OUT, "tune", f"{opt}_{i}.json")
            if os.path.exists(path):
                results.append(json.load(open(path)))
                continue
            rec, _ = run_one(litho, target, opt, ov, budget, pixel_nm=4.0, with_epe=False)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(rec, open(path, "w"), indent=2)
            results.append(rec)
            print(f"[tune] {opt:13s} {ov}  L2={rec['l2']:.0f}  obj={rec['final_objective']:.1f}  "
                  f"it={rec['iterations']}  ({rec['seconds']:.0f}s)", flush=True)
        best[opt] = min(results, key=lambda r: r["l2"])["overrides"]
    json.dump(best, open(os.path.join(OUT, "tune", "selected.json"), "w"), indent=2)
    return best


def stage_full(litho, best: dict[str, dict], budget: float, cases) -> None:
    for case in cases:
        target = load_target(case)
        for opt, ov in best.items():
            path = os.path.join(OUT, "full", f"case{case}_{opt}.json")
            if os.path.exists(path):
                continue
            rec, res = run_one(litho, target, opt, ov, budget, pixel_nm=1.0, with_epe=True)
            rec["case"] = case
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(rec, open(path, "w"), indent=2)
            torch.save(res.mask.to(torch.uint8), os.path.join(OUT, "full", f"case{case}_{opt}_mask.pt"))
            print(f"[full] case {case} {opt:13s} L2={rec['l2']:.0f}  PVB={rec['pvb']:.0f}  "
                  f"EPE@15={rec['epe@15nm']}  EPE@3={rec['epe@3nm']}  it={rec['iterations']}  "
                  f"evals={rec['n_func_evals']}  ({rec['seconds']:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "full", "all"], default="all")
    ap.add_argument("--budget-tune", type=float, default=32.0)  # ~210 evals at 512^2, matching the full-res regime
    ap.add_argument("--budget-full", type=float, default=720.0)
    ap.add_argument("--cases", type=int, nargs="+", default=[3, 6])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())
    os.makedirs(OUT, exist_ok=True)
    json.dump({"protocol": __doc__, "grid": GRID, "common": COMMON, "args": vars(args),
               "provenance": provenance()}, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    best = stage_tune(litho, args.budget_tune) if args.stage in ("tune", "all") else \
        json.load(open(os.path.join(OUT, "tune", "selected.json")))
    print("selected:", json.dumps(best), flush=True)
    if args.stage in ("full", "all"):
        stage_full(litho, best, args.budget_full, args.cases)


if __name__ == "__main__":
    main()
