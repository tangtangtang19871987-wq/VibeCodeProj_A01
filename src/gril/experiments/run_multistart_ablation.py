"""X-15: does a LEARNED sampler beat dumb random multi-start? (the missing control)

The paper's central claim is that a trained generator's proposals, refined by
batched ILT, beat single-start ILT. It never asks the sharper question: how
much of that gain is from having K independent optimization trajectories AT
ALL (which a cheap random-perturbation multi-start would also give), versus
from the LEARNED prior placing those K starts somewhere better than random?
Neither this project's earlier experiments nor the paper itself run that
control. This script does.

Four arms, same designs, same per-candidate ILT budget, same selection rule
(best-of-K by EPE):

  A. single-start   -- one deterministic cold start, K=1, no sampling at all.
  B. random-K        -- K=8 independent random PERTURBATIONS of the same cold
                        start, no learning; best-of-K selected.
  C. PT (generator)  -- K=8 samples from the WGAN-GP-pretrained generator
                        (already trained and evaluated in train_scaled.yaml's
                        run; reused here, not retrained).
  D. PT+RL           -- K=8 samples from the GRPO-finetuned generator (same).

C and D are read from the ALREADY-COMPLETED results/train_scaled/summary.json
rather than re-run, since re-running them would not change their numbers
(deterministic given the same seed) and would cost real CPU time for nothing.

Design discipline, stated before any number was seen:
- Arm B's noise scale (`perturb_sigma`) is CALIBRATED (not guessed) to match
  arm C's measured sampling diversity (~0.0016) on the SAME validation
  designs, via a standalone script whose output is logged in
  docs/findings.md F-MULTISTART-01 -- so arm B gets a COMPARABLE amount of
  "spread" to arm C, and the comparison isolates "where is the spread placed"
  rather than "how much spread is there."
- All four arms use the exact same low_res_refine budget
  (configs/experiments/train_scaled.yaml's `refine:` block) and the exact
  same validation designs (regenerated deterministically from the same seed
  -- nothing here is re-fit or re-tuned after seeing a result.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import yaml

from gril.data.synthetic import layout_dataset
from gril.experiments.infer import low_res_refine
from gril.experiments.run_iccad13 import provenance
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations


def _diversity(masks: torch.Tensor) -> float:
    """Mean pairwise L1 distance between K binarised masks, (K, H, W)."""
    k = masks.shape[0]
    if k < 2:
        return 0.0
    flat = masks.flatten(1)
    pairs = [(flat[i] - flat[j]).abs().mean().item() for i in range(k) for j in range(i + 1, k)]
    return sum(pairs) / len(pairs)


def _score_candidates(
    masks: torch.Tensor, design: torch.Tensor, litho: LithoModel, tolerance: float, pixel_nm: float
) -> list[int]:
    """EPE (inner+outer) for each of K refined candidate masks."""
    scores = []
    for j in range(masks.shape[0]):
        with torch.no_grad():
            b_nom, _, _ = litho.binary(masks[j, 0])
        ein, eout = epe_violations(b_nom, design.view(*design.shape[-2:]), tolerance, pixel_nm)
        scores.append(ein + eout)
    return scores


def run(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = os.path.join(cfg.get("output_root", "results"), cfg["experiment_id"])
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    litho = LithoModel(cfg["kernel_dir"], ProcessConfig())
    size, pixel_nm = cfg["canvas"], cfg["pixel_nm"]
    K = cfg["group_size"]
    tol = cfg["epe_tolerance"]
    refine_kwargs = cfg["refine"]
    sigma = cfg["perturb_sigma"]
    init_scale = cfg["init_scale"]
    mask_steepness = cfg["mask_steepness"]

    # Deterministically regenerate the SAME validation split used to produce
    # the already-committed PT/PT+RL numbers this compares against.
    layouts = layout_dataset(
        cfg["n_train"] + cfg["n_val"], seed=cfg["seed"], size=size, pixel_nm=pixel_nm
    )
    val_designs = torch.tensor(layouts[cfg["n_train"] :])

    per_design_records = []
    rng = torch.Generator().manual_seed(cfg["seed"])
    for i, d in enumerate(val_designs):
        design = d.view(1, 1, size, size)
        p0 = init_scale * (2.0 * design - 1.0)

        # --- Arm A: single-start ---
        single_init = (torch.sigmoid(mask_steepness * p0) > 0.5).float()
        single_refined = low_res_refine(single_init, design, litho, **refine_kwargs)
        single_epe = _score_candidates(single_refined, design, litho, tol, pixel_nm)[0]

        # --- Arm B: random-K (no learning) ---
        noise = torch.randn(K, 1, size, size, generator=rng) * sigma
        random_init = (torch.sigmoid(mask_steepness * (p0 + noise)) > 0.5).float()
        random_diversity = _diversity(random_init[:, 0])
        random_refined = low_res_refine(random_init, design, litho, **refine_kwargs)
        random_scores = _score_candidates(random_refined, design, litho, tol, pixel_nm)

        record = {
            "design_index": i,
            "single_start_epe": single_epe,
            "random_k_best_epe": min(random_scores),
            "random_k_mean_epe": sum(random_scores) / len(random_scores),
            "random_k_diversity": random_diversity,
            "random_k_all_scores": random_scores,
        }
        per_design_records.append(record)
        print(
            f"[design {i}] single={single_epe}  random_best={min(random_scores)}  "
            f"random_mean={sum(random_scores)/len(random_scores):.1f}  "
            f"diversity={random_diversity:.5f}",
            flush=True,
        )

    def mean(key):
        return sum(r[key] for r in per_design_records) / len(per_design_records)

    # Pull the already-computed generator arms in for the comparison table,
    # rather than re-running them.
    generator_arms = {}
    train_summary_path = cfg.get("generator_summary_path")
    if train_summary_path and os.path.exists(train_summary_path):
        train_summary = json.load(open(train_summary_path))
        generator_arms = train_summary.get("metrics", {})

    summary = {
        "experiment_id": cfg["experiment_id"],
        "config": cfg,
        "provenance": provenance(),
        "per_design": per_design_records,
        "aggregate": {
            "single_start_epe_mean": mean("single_start_epe"),
            "random_k_best_epe_mean": mean("random_k_best_epe"),
            "random_k_sample_epe_mean": mean("random_k_mean_epe"),
            "random_k_diversity_mean": mean("random_k_diversity"),
        },
        "generator_arms_from": train_summary_path,
        "generator_arms": generator_arms,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def print_comparison(summary: dict) -> None:
    agg = summary["aggregate"]
    gen = summary.get("generator_arms", {})
    print("\n=== Four-arm comparison: best-of-K EPE (lower is better) ===")
    print(f"  A. single-start        : {agg['single_start_epe_mean']:.2f}")
    print(f"  B. random-K (no learn) : {agg['random_k_best_epe_mean']:.2f}  "
          f"(sample-mean {agg['random_k_sample_epe_mean']:.2f}, "
          f"diversity {agg['random_k_diversity_mean']:.5f})")
    if gen.get("PT"):
        print(f"  C. PT (generator)      : {gen['PT']['epe_best_mean']:.2f}  "
              f"(sample-mean {gen['PT']['epe_sample_mean']:.2f}, "
              f"diversity {gen['PT']['diversity']:.5f})")
    if gen.get("PT+RL"):
        print(f"  D. PT+RL (generator)   : {gen['PT+RL']['epe_best_mean']:.2f}  "
              f"(sample-mean {gen['PT+RL']['epe_sample_mean']:.2f}, "
              f"diversity {gen['PT+RL']['diversity']:.5f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    summary = run(args.config)
    print_comparison(summary)


if __name__ == "__main__":
    main()
