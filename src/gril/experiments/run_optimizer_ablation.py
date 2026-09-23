"""X-14: fair Adam-vs-L-BFGS comparison on ICCAD13 (CPU).

Answers a question the raw iteration-budget curve (X-10) cannot: does a
different *optimizer family* -- not just a different iteration count -- close
any of the gap to the paper's numbers, or change the iteration/quality
trade-off the paper claims ("roughly half the iteration budget")?

Design notes, because a naive comparison here would be misleading
--------------------------------------------------------------------
1. **Same starting point for every optimizer, and NOT the ICCAD13 baseline's
   own starting point.** F-SAT-01 (docs/findings.md) found that the ICCAD13
   baseline's own constants (`init_scale=2.0`, `mask_steepness=8.0`) saturate
   the mask sigmoid so hard that L-BFGS (like plain SGD) makes zero measurable
   progress -- confirmed directly: 5 L-BFGS iterations at those exact constants
   moved the loss by 0.0%. Comparing "Adam at a workable start" against
   "L-BFGS at a stalled start" would not be a comparison of optimizers, it
   would be a comparison of which one tolerates a bad start. This ablation
   therefore uses a common, well-conditioned start (`init_scale=0.5`) for
   BOTH optimizers, and is a self-contained experiment rather than a
   continuation of `results/iccad13_ilt/` (whose numbers stay authoritative
   for the main ICCAD13 comparison in REPRODUCTION_REPORT.md).
2. **Compute is reported, never assumed equal.** Adam and L-BFGS iterations
   are not the same unit of work (REPRODUCTION_SPEC.md Sec. 8). Every record
   here carries `n_func_evals` and `seconds`, and the summary reports quality
   at matched (or close-to-matched) function-evaluation counts, not matched
   raw iteration counts.
3. **Cases are chosen for a reason, not arbitrarily**: case1 (paper-exact
   match on the Adam baseline -- does a better optimizer change an
   already-good result?), case3 (the hardest case, largest EPE gap to the
   paper -- does optimizer choice explain any of it?), case5 (one of only two
   cases where our Adam baseline missed the paper's EPE@15nm exactly -- does
   a different optimizer close that specific gap?).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import yaml

from gril.data.glp import Design
from gril.experiments.run_iccad13 import provenance, score_mask
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig


def run(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    exp_id = cfg["experiment_id"]
    out_dir = os.path.join(cfg.get("output_root", "results"), exp_id)
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(cfg.get("threads", os.cpu_count() or 4))

    litho = LithoModel(cfg["kernel_dir"], ProcessConfig(**cfg.get("process", {})))
    tolerances = cfg.get("epe_tolerances", [15, 3])
    canvas = cfg.get("canvas", 2048)
    shared = cfg["shared_ilt"]  # init_scale, mask_steepness: identical for every optimizer

    results: dict[str, dict] = {}
    for case in cfg["cases"]:
        glp = os.path.join(cfg["bench_dir"], f"M1_test{case}.glp")
        target = torch.tensor(Design.from_glp(glp).centred_raster(canvas))

        for opt_name, opt_overrides in cfg["optimizers"].items():
            key = f"case{case}_{opt_name}"
            case_file = os.path.join(out_dir, f"{key}.json")
            if os.path.exists(case_file) and not cfg.get("overwrite", False):
                results[key] = json.load(open(case_file))
                print(f"[{key}] cached", flush=True)
                continue

            ilt_cfg = ILTConfig(**{**shared, **opt_overrides})
            t0 = time.time()
            res = solve(target, litho, ilt_cfg)
            elapsed = time.time() - t0

            record = {
                "case": case,
                "optimizer": opt_name,
                "config": {**shared, **opt_overrides},
                "scores": score_mask(res.mask, target, litho, tolerances),
                "iterations_run": res.iterations_run,
                "n_func_evals": res.n_func_evals,
                "seconds": elapsed,
                "seconds_per_func_eval": elapsed / max(res.n_func_evals, 1),
                "final_loss": res.loss_history[-1] if res.loss_history else None,
                "initial_loss": res.loss_history[0] if res.loss_history else None,
            }
            with open(case_file, "w") as fh:
                json.dump(record, fh, indent=2)
            results[key] = record
            s = record["scores"]
            print(
                f"[{key}] L2 {s['l2']:.0f}  EPE@15 {s['epe@15nm']}  EPE@3 {s['epe@3nm']}  "
                f"evals {res.n_func_evals}  ({elapsed:.0f}s)",
                flush=True,
            )

    summary = {
        "experiment_id": exp_id,
        "config": cfg,
        "provenance": provenance(),
        "results": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def print_comparison(summary: dict) -> None:
    cfg = summary["config"]
    opt_names = list(cfg["optimizers"])
    print(f"\n{'case':6s} {'optimizer':10s} {'evals':>7s} {'seconds':>9s} "
          f"{'L2':>8s} {'EPE@15':>7s} {'EPE@3':>6s}")
    for case in cfg["cases"]:
        for opt in opt_names:
            r = summary["results"].get(f"case{case}_{opt}")
            if r is None:
                continue
            s = r["scores"]
            print(
                f"{case:<6d} {opt:10s} {r['n_func_evals']:7d} {r['seconds']:9.0f} "
                f"{s['l2']:8.0f} {s['epe@15nm']:7d} {s['epe@3nm']:6d}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    summary = run(args.config)
    print_comparison(summary)


if __name__ == "__main__":
    main()
