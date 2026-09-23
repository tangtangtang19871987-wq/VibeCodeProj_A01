#!/usr/bin/env python3
"""Regenerate results/pvb_sweep/summary.json (F-PVB-01).

A deliberate small-scale, reduced-resolution calibration sweep -- like the
step_size/mask_steepness sweep in configs/experiments/README.md -- not a
full-resolution, full-benchmark experiment. Resumable: re-running skips any
weight already present in results/pvb_sweep/sweep_raw.json.

Usage::

    PYTHONPATH=src python scripts/pvb_weight_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn.functional as F

from gril.data.glp import Design
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"

CASES = [1, 3, 9]
WEIGHTS = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]


def main() -> None:
    torch.set_num_threads(2)  # leave headroom for other concurrent work
    litho = LithoModel(KERNEL_DIR, ProcessConfig())

    targets = {}
    for c in CASES:
        t = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{c}.glp").centred_raster(2048))
        td = F.avg_pool2d(t[None, None], 4)[0, 0]
        targets[c] = (td > 0.5).float()

    out_dir = os.path.join(ROOT, "results/pvb_sweep")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "sweep_raw.json")
    sweep = json.load(open(cache_path)) if os.path.exists(cache_path) else []
    done_weights = {r["weight_pvb"] for r in sweep}

    print(f"{'w_pvb':>8}  {'L2_sum':>8}  {'PVB_sum':>8}  {'seconds':>8}")
    for r in sweep:
        print(f"{r['weight_pvb']:8.3f}  {r['l2_sum']:8.0f}  {r['pvb_sum']:8.0f}  (cached)")

    for w in WEIGHTS:
        if w in done_weights:
            continue
        cfg = ILTConfig(iterations=60, step_size=0.2, mask_steepness=8.0, weight_pvb=w)
        l2_sum, pvb_sum = 0, 0
        t0 = time.time()
        per_case = {}
        for c in CASES:
            res = solve(targets[c], litho, cfg)
            s = evaluate(res.mask, targets[c], litho, tolerance=15)
            l2_sum += s.l2
            pvb_sum += s.pvb
            per_case[c] = {"l2": s.l2, "pvb": s.pvb, "epe": s.epe}
        dt = time.time() - t0
        record = {"weight_pvb": w, "l2_sum": l2_sum, "pvb_sum": pvb_sum, "seconds": dt, "per_case": per_case}
        sweep.append(record)
        json.dump(sweep, open(cache_path, "w"), indent=2)
        print(f"{w:8.3f}  {l2_sum:8.0f}  {pvb_sum:8.0f}  {dt:8.0f}")

    summary = {
        "experiment_id": "pvb_sweep",
        "description": (
            "weight_pvb sweep at reduced resolution (4x downsample) on cases "
            "1/3/9, iterations=60, BEFORE any full-scale run or paper comparison"
        ),
        "cases": CASES,
        "sweep": sweep,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
