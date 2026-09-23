#!/usr/bin/env python3
"""How much does post-hoc MRC cleanup cost, and does it actually work?
(F-MRC-01)

Every mask this project has produced so far used `mrc_open_size=0` (off) --
the raw sigmoid-threshold output, with no manufacturability constraint of any
kind applied during or after optimization. This script quantifies exactly how
unmanufacturable that raw output is (isolated single-pixel debris, hundreds of
disconnected shapes) and measures the L2/EPE cost of cleaning it up with the
morphological opening already implemented in `gril.ilt.solver.morphological_open`
(`ILTConfig.mrc_open_size`), applied POST-HOC to the already-committed ICCAD13
masks -- no re-optimization needed, since opening is a pure post-processing
step (see `_binarize()`).

Usage::

    PYTHONPATH=src python scripts/mrc_cleanup_sweep.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import scipy.ndimage as ndi
import torch

from gril.data.glp import Design
from gril.experiments.run_iccad13 import provenance
from gril.ilt.solver import morphological_open
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations, l2_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
BASELINE_DIR = os.path.join(ROOT, "results/iccad13_ilt")
CASES = list(range(1, 11))
OPEN_SIZES = [0, 3, 5, 7, 9]
TINY_AREA_PX2 = 20  # smaller than a 4.5nm x 4.5nm square -- physically absurd debris


def mask_stats(mask_np: np.ndarray) -> dict:
    lbl, n = ndi.label(mask_np > 0.5, structure=np.ones((3, 3)))
    sizes = ndi.sum(mask_np > 0.5, lbl, range(1, n + 1)) if n > 0 else np.array([])
    tiny = int((sizes < TINY_AREA_PX2).sum()) if n > 0 else 0
    return {"n_components": int(n), "n_tiny_components": tiny}


def main() -> None:
    torch.set_num_threads(2)  # a running full-scale ILT job may be using the rest
    litho = LithoModel(KERNEL_DIR, ProcessConfig())

    per_case = {}
    for c in CASES:
        mask_path = os.path.join(BASELINE_DIR, f"mask{c}.pt")
        if not os.path.exists(mask_path):
            print(f"[case {c}] skip -- {mask_path} not found")
            continue
        mask = torch.load(mask_path, weights_only=False).float()
        target = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{c}.glp").centred_raster(2048))

        by_size = {}
        for size in OPEN_SIZES:
            m = morphological_open(mask, size) if size > 0 else mask
            with torch.no_grad():
                b_nom, _, _ = litho.binary(m)
            ein, eout = epe_violations(b_nom, target, 15, 1.0)
            ein3, eout3 = epe_violations(b_nom, target, 3, 1.0)
            l2 = l2_loss(b_nom, target)
            stats = mask_stats(m.numpy())
            by_size[str(size)] = {
                "l2": l2,
                "epe_at_15nm": ein + eout,
                "epe_at_3nm": ein3 + eout3,
                **stats,
            }
            print(
                f"[case {c:>2}] open_size={size:>2}  L2={l2:7.0f}  EPE@15={ein + eout:3d}  "
                f"EPE@3={ein3 + eout3:3d}  n_components={stats['n_components']:4d}  "
                f"tiny={stats['n_tiny_components']:4d}",
                flush=True,
            )
        per_case[str(c)] = by_size

    def mean(size: str, key: str) -> float:
        vals = [v[str(size)][key] for v in per_case.values()]
        return sum(vals) / len(vals)

    summary = {
        "experiment_id": "mrc_sweep",
        "description": (
            "F-MRC-01: post-hoc morphological-opening MRC cleanup applied to the "
            "already-committed results/iccad13_ilt masks (raw, mrc_open_size=0), "
            "across all 10 cases, at open_size in {0,3,5,7,9}. Quantifies how "
            "unmanufacturable the raw output is (isolated debris, disconnected "
            "component count) and the L2/EPE cost of cleaning it up."
        ),
        "cases": CASES,
        "open_sizes": OPEN_SIZES,
        "tiny_area_px2_threshold": TINY_AREA_PX2,
        "per_case": per_case,
        "aggregate": {
            str(size): {
                "l2_mean": mean(size, "l2"),
                "epe_at_15nm_mean": mean(size, "epe_at_15nm"),
                "epe_at_3nm_mean": mean(size, "epe_at_3nm"),
                "n_components_mean": mean(size, "n_components"),
                "n_tiny_components_mean": mean(size, "n_tiny_components"),
            }
            for size in OPEN_SIZES
        },
        "provenance": provenance(),
    }
    out_dir = os.path.join(ROOT, "results/mrc_sweep")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    json.dump(summary, open(summary_path, "w"), indent=2)
    print("wrote", summary_path)

    print("\n=== aggregate (mean over 10 cases) ===")
    for size in OPEN_SIZES:
        a = summary["aggregate"][str(size)]
        print(
            f"  open_size={size:>2}  L2={a['l2_mean']:8.0f}  EPE@15={a['epe_at_15nm_mean']:5.1f}  "
            f"EPE@3={a['epe_at_3nm_mean']:6.1f}  n_components={a['n_components_mean']:6.1f}  "
            f"tiny={a['n_tiny_components_mean']:6.1f}"
        )


if __name__ == "__main__":
    main()
