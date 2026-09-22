"""Run numerical ILT over the ICCAD13 benchmark and score it (CPU).

Produces one JSON per case plus a summary, under ``results/<experiment_id>/``.
Raw per-case numbers are always written -- never only a plot. Completed cases are
skipped on re-run, so a long CPU job can be resumed after interruption.

Usage::

    python -m gril.experiments.run_iccad13 --config configs/experiments/iccad13_ilt150.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

import torch
import yaml

from gril.data.glp import Design
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations, l2_loss, pv_band


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def provenance() -> dict:
    """Environment facts recorded with every result file."""
    return {
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),   # expected False: CPU-only host
        "device": "cpu",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
    }


def score_mask(mask, target, litho, tolerances, pixel_nm: float = 1.0) -> dict:
    """Score a binary mask; returns L2, PV Band, and EPE at each tolerance (nm)."""
    with torch.no_grad():
        b_nom, b_max, b_min = litho.binary(mask)
        out = {"l2": l2_loss(b_nom, target), "pvb": pv_band(b_max, b_min)}
        for tol in tolerances:
            ein, eout = epe_violations(b_nom, target, tol, pixel_nm)
            out[f"epe@{tol}nm"] = ein + eout
            out[f"epe_in@{tol}nm"] = ein
            out[f"epe_out@{tol}nm"] = eout
        return out


def run(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    exp_id = cfg["experiment_id"]
    out_dir = os.path.join(cfg.get("output_root", "results"), exp_id)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(cfg.get("seed", 0))
    torch.set_num_threads(cfg.get("threads", os.cpu_count() or 4))

    litho = LithoModel(cfg["kernel_dir"], ProcessConfig(**cfg.get("process", {})))
    ilt_raw = dict(cfg.get("ilt", {}))
    if "checkpoints" in ilt_raw:
        ilt_raw["checkpoints"] = tuple(ilt_raw["checkpoints"])
    ilt_cfg = ILTConfig(**ilt_raw)
    tolerances = cfg.get("epe_tolerances", [15, 3])
    canvas = cfg.get("canvas", 2048)

    results = {}
    for case in cfg["cases"]:
        case_file = os.path.join(out_dir, f"case{case}.json")
        if os.path.exists(case_file) and not cfg.get("overwrite", False):
            results[case] = json.load(open(case_file))
            print(f"[case {case}] cached", flush=True)
            continue

        glp = os.path.join(cfg["bench_dir"], f"M1_test{case}.glp")
        target = torch.tensor(Design.from_glp(glp).centred_raster(canvas))

        t0 = time.time()
        res = solve(target, litho, ilt_cfg)
        elapsed = time.time() - t0

        record = {
            "case": case,
            "scores": score_mask(res.mask, target, litho, tolerances),
            "no_opc": score_mask(target.clone(), target, litho, tolerances),
            "iterations": res.iterations_run,
            "seconds": elapsed,
            "seconds_per_iter": elapsed / max(res.iterations_run, 1),
            "final_ilt_loss": res.loss_history[-1] if res.loss_history else None,
            "loss_history": res.loss_history,
            # Iteration-budget curve (X-10): scores at each checkpoint.
            "budget_curve": {
                str(it): score_mask(m, target, litho, tolerances)
                for it, m in sorted(res.checkpoint_masks.items())
            },
        }
        with open(case_file, "w") as fh:
            json.dump(record, fh, indent=2)
        torch.save(res.mask.to(torch.uint8), os.path.join(out_dir, f"mask{case}.pt"))
        results[case] = record
        s = record["scores"]
        print(
            f"[case {case}] L2 {s['l2']:.0f}  PVB {s['pvb']:.0f}  "
            f"EPE@15 {s['epe@15nm']}  EPE@3 {s['epe@3nm']}  ({elapsed:.0f}s)",
            flush=True,
        )

    def mean(key: str) -> float:
        return sum(r["scores"][key] for r in results.values()) / len(results)

    summary = {
        "experiment_id": exp_id,
        "config": cfg,
        "provenance": provenance(),
        "per_case": {str(k): v["scores"] for k, v in results.items()},
        "aggregate": {
            k: mean(k) for k in results[cfg["cases"][0]]["scores"]
        },
        "total_seconds": sum(r["seconds"] for r in results.values()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    summary = run(args.config)
    agg = summary["aggregate"]
    print("\n=== aggregate ===")
    for k in sorted(agg):
        print(f"  {k:16s} {agg[k]:.1f}")


if __name__ == "__main__":
    main()
