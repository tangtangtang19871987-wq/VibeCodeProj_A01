#!/usr/bin/env python3
"""Three candidate directions for improving on the X-16 L-BFGS baseline,
tested first on case 3 (the hardest case, and the one with the most
remaining headroom on EPE@3nm even after X-16 matched the paper's EPE@15nm
exactly). All three reuse the exact starting point X-14/X-16 validated
(init_scale=0.5, mask_steepness=8.0) so any change is attributable to the
one mechanism under test, not a different baseline.

  D1. EPE-aware loss (F-EPE-01) STACKED on top of L-BFGS, targeting the
      3nm tolerance directly (the metric with the most headroom left),
      at a weight calibrated the same way F-EPE-01 calibrated it.
  D2. Coarse-to-fine (multi-resolution) L-BFGS: a short low-resolution
      solve for large-scale structure, upsampled as a warm start for a
      full-resolution L-BFGS refinement -- a different mechanism (search
      trajectory / avoiding local minima) than D1's objective-shaping.
  D3. Best-of-K multi-start L-BFGS on the REAL ICCAD13 benchmark (X-15's
      calibrated-diversity random-perturbation methodology, but with
      L-BFGS -- a much stronger per-candidate optimizer -- instead of the
      short Adam-based refine X-15 used, and on the real benchmark instead
      of synthetic layouts).

Usage::

    PYTHONPATH=src python scripts/ilt_improvement_directions.py --case 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn.functional as F

from gril.data.glp import Design
from gril.ilt.epe_loss import build_epe_sites, soft_epe_loss
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig, resist
from gril.metrics.core import epe_violations, l2_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
BASE = dict(mask_steepness=8.0, init_scale=0.5, optimizer="lbfgs",
            lbfgs_max_iter_per_step=1, lbfgs_history_size=10, lbfgs_line_search="strong_wolfe")


def score(litho, target, mask, label, seconds, extra=""):
    with torch.no_grad():
        b_nom, _, _ = litho.binary(mask)
    ein, eout = epe_violations(b_nom, target, 15, 1.0)
    ein3, eout3 = epe_violations(b_nom, target, 3, 1.0)
    l2 = l2_loss(b_nom, target)
    print(
        f"{label:32s} L2={l2:7.0f}  EPE@15={ein + eout:3d}  EPE@3={ein3 + eout3:3d}  "
        f"({seconds:.0f}s){extra}",
        flush=True,
    )
    return {"label": label, "l2": l2, "epe_at_15nm": ein + eout, "epe_at_3nm": ein3 + eout3, "seconds": seconds}


def d0_baseline(litho, target):
    t0 = time.time()
    res = solve(target, litho, ILTConfig(iterations=50, step_size=1.0, **BASE))
    return score(litho, target, res.mask, "D0 baseline (X-16, single-res L-BFGS)", time.time() - t0)


def d1_epe_aware(litho, target, weight_epe=3000.0, tol=3.0):
    t0 = time.time()
    res = solve(target, litho, ILTConfig(
        iterations=50, step_size=1.0, weight_epe=weight_epe, epe_tolerance_nm=tol, **BASE
    ))
    return score(litho, target, res.mask, f"D1 EPE-aware (w={weight_epe:.0f}, tol={tol}nm)", time.time() - t0)


def d2_coarse_to_fine(litho, target, downsample=4, coarse_iters=30, fine_iters=30):
    t0 = time.time()
    full_hw = target.shape[-2:]
    t_low = (F.avg_pool2d(target[None, None].float(), downsample) > 0.5).float()[0, 0]
    coarse_cfg = ILTConfig(iterations=coarse_iters, step_size=1.0, **BASE)
    coarse_res = solve(t_low, litho, coarse_cfg)
    # Warm-start the full-res field from the upsampled coarse CONTINUOUS mask
    # (bicubic), converted back to the sigmoid pre-image at the same steepness.
    coarse_soft = torch.sigmoid(BASE["mask_steepness"] * coarse_res.params)
    up = F.interpolate(coarse_soft[None, None], size=full_hw, mode="bicubic", align_corners=False)[0, 0]
    up = up.clamp(1e-4, 1 - 1e-4)
    init_params = (torch.log(up / (1 - up)) / BASE["mask_steepness"])
    fine_cfg = ILTConfig(iterations=fine_iters, step_size=1.0, **BASE)
    fine_res = solve(target, litho, fine_cfg, init_params=init_params)
    total_seconds = time.time() - t0
    return score(
        litho, target, fine_res.mask,
        f"D2 coarse-to-fine ({downsample}x, {coarse_iters}+{fine_iters} it)",
        total_seconds,
    )


def d3_multistart(litho, target, k=4, sigma=0.15, iterations=50, seed=0):
    t0 = time.time()
    rng = torch.Generator().manual_seed(seed)
    p0 = BASE["init_scale"] * (2.0 * target.float() - 1.0)
    results = []
    for i in range(k):
        noise = torch.randn(target.shape, generator=rng) * sigma if i > 0 else torch.zeros_like(p0)
        cfg = ILTConfig(iterations=iterations, step_size=1.0, **BASE)
        res = solve(target, litho, cfg, init_params=p0 + noise)
        with torch.no_grad():
            b_nom, _, _ = litho.binary(res.mask)
        ein, eout = epe_violations(b_nom, target, 15, 1.0)
        results.append((ein + eout, res.mask))
        print(f"    [candidate {i}] EPE@15={ein + eout}", flush=True)
    best_epe, best_mask = min(results, key=lambda r: r[0])
    return score(litho, target, best_mask, f"D3 best-of-{k} multistart L-BFGS", time.time() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    litho = LithoModel(KERNEL_DIR, ProcessConfig())
    target = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{args.case}.glp").centred_raster(2048))

    print(f"=== case {args.case} ===", flush=True)
    records = []
    records.append(d0_baseline(litho, target))
    records.append(d1_epe_aware(litho, target))
    records.append(d2_coarse_to_fine(litho, target))
    records.append(d3_multistart(litho, target))

    out_dir = os.path.join(ROOT, "results/ilt_improvement_directions")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"case{args.case}.json")
    json.dump({"case": args.case, "records": records}, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
