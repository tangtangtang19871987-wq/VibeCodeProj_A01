#!/usr/bin/env python3
"""Regenerate results/epe_aware_case3/summary.json (F-EPE-01).

Measures the raw-magnitude ratio between the L2 term and the soft EPE-site
loss at the solver's actual starting point, then runs case 3 (full 2048
resolution, 100 iterations, the main baseline's own step_size/mask_steepness)
at a naive weight and at weights calibrated to that measured ratio, so the
"does the EPE-aware term work" question is answered at a weight where the
term can plausibly compete with L2 -- see docs/findings.md F-EPE-01 for why
weight_epe=2.0 alone would give a false negative.

Usage::

    PYTHONPATH=src python scripts/epe_weight_calibration.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from gril.data.glp import Design
from gril.ilt.epe_loss import build_epe_sites, soft_epe_loss
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig, resist
from gril.metrics.core import epe_violations, l2_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
CASE = 3
INIT_SCALE = 2.0
MASK_STEEPNESS = 8.0


def measure_magnitude_ratio(litho: LithoModel, target: torch.Tensor) -> dict:
    """Raw L2 vs raw soft_epe_loss at the solver's own cold start."""
    p0 = INIT_SCALE * (2 * target.float() - 1)
    mask = torch.sigmoid(p0 * MASK_STEEPNESS)
    z_nom = resist(litho.aerial_nominal(mask), litho.cfg)
    l2_raw = ((z_nom - target.float()) ** 2).sum().item()
    sites = build_epe_sites(target, tolerance=15.0, pixel_nm=1.0)
    n_sites = len(sites.inner_rows) + len(sites.outer_rows)
    epe_raw = soft_epe_loss(z_nom, sites).item()
    return {
        "raw_l2_sum_sq_err": l2_raw,
        "raw_soft_epe_loss_tol15nm": epe_raw,
        "n_epe_sites_tol15nm": n_sites,
        "ratio_l2_over_epe": l2_raw / epe_raw,
    }


def score(litho: LithoModel, target: torch.Tensor, cfg: ILTConfig, label: str) -> dict:
    t0 = time.time()
    res = solve(target, litho, cfg)
    dt = time.time() - t0
    with torch.no_grad():
        b_nom, _, _ = litho.binary(res.mask)
    ein, eout = epe_violations(b_nom, target, tolerance=15, pixel_nm=1.0)
    ein3, eout3 = epe_violations(b_nom, target, tolerance=3, pixel_nm=1.0)
    l2 = l2_loss(b_nom, target)
    print(
        f"{label:40s} L2={l2:.0f}  EPE@15={ein + eout}  EPE@3={ein3 + eout3}  "
        f"({dt:.0f}s, {res.n_func_evals} evals)",
        flush=True,
    )
    return {
        "label": label,
        "weight_epe": cfg.weight_epe,
        "epe_tolerance_nm": cfg.epe_tolerance_nm if cfg.weight_epe > 0 else None,
        "l2": l2,
        "epe_at_15nm": ein + eout,
        "epe_at_3nm": ein3 + eout3,
        "seconds": dt,
        "n_func_evals": res.n_func_evals,
    }


def main() -> None:
    torch.set_num_threads(4)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())
    target = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{CASE}.glp").centred_raster(2048))

    calib = measure_magnitude_ratio(litho, target)
    print(
        f"raw L2={calib['raw_l2_sum_sq_err']:.1f}  "
        f"raw soft_epe_loss={calib['raw_soft_epe_loss_tol15nm']:.1f}  "
        f"({calib['n_epe_sites_tol15nm']} sites)  "
        f"ratio={calib['ratio_l2_over_epe']:.1f}",
        flush=True,
    )

    runs = []
    base = dict(iterations=100, step_size=0.2, mask_steepness=MASK_STEEPNESS)
    runs.append(score(litho, target, ILTConfig(**base, weight_epe=0.0), "baseline (L2 only)"))
    runs.append(
        score(litho, target, ILTConfig(**base, weight_epe=2.0, epe_tolerance_nm=15.0),
              "EPE-aware (naive weight)")
    )
    runs.append(
        score(litho, target, ILTConfig(**base, weight_epe=2.0, epe_tolerance_nm=3.0),
              "EPE-aware (naive weight, strict tol)")
    )
    runs.append(
        score(litho, target, ILTConfig(**base, weight_epe=1000.0, epe_tolerance_nm=15.0),
              "EPE-aware (calibrated weight)")
    )
    runs.append(
        score(litho, target, ILTConfig(**base, weight_epe=3000.0, epe_tolerance_nm=15.0),
              "EPE-aware (calibrated weight, higher)")
    )

    out_dir = os.path.join(ROOT, "results/epe_aware_case3")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "experiment_id": "epe_aware_case3",
        "description": (
            "F-EPE-01: weight_epe calibration on ICCAD13 case 3, full 2048 "
            "resolution, 100 iterations, step_size=0.2, mask_steepness=8.0 "
            "(main-baseline constants). A naive guess (weight_epe=2.0) vs a "
            "magnitude-calibrated weight (weight_epe in {1000,3000})."
        ),
        "magnitude_calibration": {
            "measured_at": "case 3 solver start point (init_scale=2.0, mask_steepness=8.0 sigmoid init)",
            **calib,
        },
        "runs": runs,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
