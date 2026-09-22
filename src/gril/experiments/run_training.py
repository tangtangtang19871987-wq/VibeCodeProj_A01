"""End-to-end two-stage training: WGAN-GP pretraining then GRPO finetuning.

Scaled to run on CPU. The method is implemented exactly as the paper specifies
(Eqs. 5-10); only the SCALE differs, and every deviation is recorded in the
output manifest so a larger host can re-run the same config unchanged.

Deviations from the paper's configuration, and why (see docs/gap_ledger.md):
* **Data**: seeded synthetic M1-like layouts instead of LithoBench MetalSet +
  ViaSet, which is unreachable from this environment (G-041).
* **Ground-truth masks**: produced by our own ILT solver, since LithoBench's
  reference masks are unavailable. The paper itself calls those masks
  sub-optimal (Sec. 3.1).
* **Resolution**: 256x256 at 8 nm/pixel instead of 2048x2048 at 1 nm/pixel. The
  physical canvas is identical (2048 nm), so the optical kernels remain valid --
  verified to <1% of peak in tests_gril/physics.
* **RL refinement downsample**: the paper uses 8x (2048->256). At a 256 base a
  further 8x would give a 32x32 grid, smaller than the 35x35 optical kernel,
  which is physically invalid (and now raises). We use 2x (256->128).
* **Epochs / dataset size**: far smaller. This host has 4 CPU cores and no GPU;
  the paper used 8x A100 (G-040).

CPU-only; no CUDA paths.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from gril.data.synthetic import layout_dataset
from gril.experiments.infer import low_res_refine
from gril.experiments.run_iccad13 import provenance
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import epe_violations
from gril.models.generator import Discriminator, GeneratorConfig, StyleAwareGenerator
from gril.train.grpo import GRPOConfig, finetune_step
from gril.train.pretrain import PretrainConfig, critic_step, generator_step


def build_ground_truth(layouts: np.ndarray, litho: LithoModel, iters: int, cache: str) -> torch.Tensor:
    """ILT-optimised masks used as the reconstruction target in Eq. 5."""
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    masks = []
    for i, lay in enumerate(layouts):
        t = torch.tensor(lay)
        res = solve(t, litho, ILTConfig(iterations=iters, step_size=0.2, mask_steepness=8.0))
        masks.append(res.mask)
        if (i + 1) % 10 == 0:
            print(f"  ground truth {i+1}/{len(layouts)}", flush=True)
    out = torch.stack(masks).unsqueeze(1)
    torch.save(out, cache)
    return out


def evaluate(generator, designs, litho, cfg, tolerance, refine_kwargs) -> dict:
    """Sample K per design, refine, and report best/mean EPE and sampler diversity."""
    generator.eval()
    best_epes, mean_epes, diversities = [], [], []
    rng = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for d in designs:
            design = d.view(1, 1, *d.shape[-2:])
            z = torch.randn(cfg["group_size"], cfg["latent_dim"], generator=rng)
            logits = generator(design.expand(cfg["group_size"], -1, -1, -1), z)
            init = (logits > 0.5).float()

            flat = init.flatten(1)
            k = flat.shape[0]
            pairs = [(flat[i] - flat[j]).abs().mean().item() for i in range(k) for j in range(i + 1, k)]
            diversities.append(sum(pairs) / max(len(pairs), 1))

            refined = low_res_refine(init, design, litho, **refine_kwargs)
            epes = []
            for j in range(k):
                b_nom, _, _ = litho.binary(refined[j, 0])
                ein, eout = epe_violations(b_nom, design.view(*d.shape[-2:]), tolerance)
                epes.append(ein + eout)
            best_epes.append(min(epes))
            mean_epes.append(sum(epes) / len(epes))
    generator.train()
    return {
        "epe_best_mean": float(np.mean(best_epes)),
        "epe_sample_mean": float(np.mean(mean_epes)),
        "diversity": float(np.mean(diversities)),
    }


def run(config_path: str) -> dict:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = os.path.join(cfg.get("output_root", "results"), cfg["experiment_id"])
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))

    litho = LithoModel(cfg["kernel_dir"], ProcessConfig())
    size, pixel_nm = cfg["canvas"], cfg["pixel_nm"]

    layouts = layout_dataset(cfg["n_train"] + cfg["n_val"], seed=cfg["seed"], size=size, pixel_nm=pixel_nm)
    train_lay, val_lay = layouts[: cfg["n_train"]], layouts[cfg["n_train"] :]

    print("Building ILT ground-truth masks ...", flush=True)
    gt = build_ground_truth(train_lay, litho, cfg["gt_iterations"], os.path.join(out_dir, "gt.pt"))
    designs = torch.tensor(train_lay).unsqueeze(1)
    val_designs = torch.tensor(val_lay)

    gcfg = GeneratorConfig(**cfg["generator"])
    gen = StyleAwareGenerator(gcfg)
    critic = Discriminator(**cfg.get("discriminator", {}))
    pcfg = PretrainConfig(**cfg["pretrain"])

    opt_g = torch.optim.Adam(gen.parameters(), lr=pcfg.lr_generator, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(critic.parameters(), lr=pcfg.lr_discriminator, betas=pcfg.betas_discriminator)

    refine_kwargs = cfg["refine"]
    eval_cfg = {"group_size": cfg["eval_group_size"], "latent_dim": gcfg.latent_dim}
    tol = cfg["epe_tolerance"]

    history = {"pretrain": [], "rl": []}
    n = designs.shape[0]
    bs = pcfg.batch_size

    # ---------------- Stage 1: WGAN-GP pretraining (paper Eq. 5) --------------
    # Resume support: a long CPU run can be interrupted (this one was, by a
    # container restart). If the stage-1 checkpoint is present, reuse it rather
    # than repeating ~20 minutes of pretraining.
    pt_ckpt = os.path.join(out_dir, "generator_pt.pt")
    resumed_pretrain = os.path.exists(pt_ckpt) and not cfg.get("overwrite", False)
    t0 = time.time()
    step = 0
    if resumed_pretrain:
        gen.load_state_dict(torch.load(pt_ckpt, weights_only=False))
        print(f"\n=== Stage 1: RESUMED from {pt_ckpt} (pretraining skipped) ===", flush=True)
    else:
        print("\n=== Stage 1: generative pretraining (WGAN-GP + reconstruction) ===", flush=True)
        for epoch in range(pcfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n - bs + 1, bs):
                idx = perm[i : i + bs]
                d_stats = critic_step(critic, gen, designs[idx], gt[idx], pcfg, opt_d)
                step += 1
                # Log EVERY step. Logging only on generator steps means that when
                # the dataset yields fewer than n_critic batches the history stays
                # empty and the printed metrics are the "missing" default, which
                # reads as NaN and looks exactly like divergence.
                record = {"epoch": epoch, "step": step, **d_stats}
                if step % pcfg.n_critic == 0:
                    record.update(generator_step(critic, gen, designs[idx], gt[idx], pcfg, opt_g))
                history["pretrain"].append(record)
            if step < pcfg.n_critic:
                print(
                    f"  WARNING: only {step} critic steps so far and n_critic="
                    f"{pcfg.n_critic}; the generator has not been updated yet.",
                    flush=True,
                )
            if (epoch + 1) % cfg["log_every"] == 0:
                last = history["pretrain"][-1] if history["pretrain"] else {}
                recs = [r for r in history["pretrain"] if "g_rec" in r]
                g_rec = recs[-1]["g_rec"] if recs else None
                print(
                    f"  epoch {epoch+1}/{pcfg.epochs} "
                    f"d_loss {last.get('d_loss', float('nan')):.3f} "
                    f"g_rec {'n/a' if g_rec is None else f'{g_rec:.4f}'} "
                    f"W {last.get('wasserstein', float('nan')):.3f}",
                    flush=True,
                )
        torch.save(gen.state_dict(), pt_ckpt)
    pretrain_seconds = time.time() - t0
    print("Evaluating PT model ...", flush=True)
    metrics_pt = evaluate(gen, val_designs, litho, eval_cfg, tol, refine_kwargs)
    print(f"  PT: {metrics_pt}", flush=True)

    # ---------------- Stage 2: GRPO finetuning (paper Eqs. 6-10) -------------
    print("\n=== Stage 2: GRPO reinforcement finetuning ===", flush=True)
    teacher = copy.deepcopy(gen).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    rcfg = GRPOConfig(**cfg["grpo"])
    opt_rl = torch.optim.Adam(gen.parameters(), lr=cfg["rl_lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_rl, T_max=max(cfg["rl_steps"], 1), eta_min=cfg["rl_lr_min"]
    )
    rng = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.time()
    for s in range(cfg["rl_steps"]):
        design = designs[s % n : s % n + 1]
        tgt2d = design.view(*design.shape[-2:])

        def refine(masks):
            return low_res_refine(masks, design, litho, **refine_kwargs)

        def reward_fn(refined):
            vals = []
            for j in range(refined.shape[0]):
                b_nom, _, _ = litho.binary(refined[j, 0])
                ein, eout = epe_violations(b_nom, tgt2d, rcfg.epe_tolerance)
                vals.append(-(ein + eout))
            return torch.tensor(vals, dtype=torch.float32)

        opt_rl.zero_grad(set_to_none=True)
        stats = finetune_step(gen, teacher, design, refine, reward_fn, rcfg, rng)
        stats["loss"].backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.get("grad_clip", 1.0))
        opt_rl.step()
        sched.step()
        rec = {k: v for k, v in stats.items() if k != "loss"}
        rec["step"] = s
        history["rl"].append(rec)
        if (s + 1) % cfg["log_every"] == 0:
            print(
                f"  step {s+1}/{cfg['rl_steps']} R {rec['reward_mean']:.2f} "
                f"(sd {rec['reward_std']:.2f})  A {rec['advantage_mean']:+.2f}  "
                f"pg {rec['loss_pg']:.4f}  imit {rec['loss_imit']:.4f}",
                flush=True,
            )
    rl_seconds = time.time() - t0

    torch.save(gen.state_dict(), os.path.join(out_dir, "generator_ptrl.pt"))
    print("Evaluating PT+RL model ...", flush=True)
    metrics_ptrl = evaluate(gen, val_designs, litho, eval_cfg, tol, refine_kwargs)
    print(f"  PT+RL: {metrics_ptrl}", flush=True)

    summary = {
        "experiment_id": cfg["experiment_id"],
        "config": cfg,
        "provenance": provenance(),
        "pretrain_seconds": pretrain_seconds,
        "pretrain_resumed_from_checkpoint": resumed_pretrain,
        "rl_seconds": rl_seconds,
        "metrics": {"PT": metrics_pt, "PT+RL": metrics_ptrl},
        "deviations_from_paper": [
            "synthetic layouts instead of LithoBench (G-041, unreachable)",
            "ground-truth masks from our own ILT solver",
            f"resolution {size}x{size} at {pixel_nm} nm/px (paper: 2048x2048 at 1 nm/px)",
            f"RL refine downsample {refine_kwargs.get('downsample')} (paper: 8; 8x here would be "
            "smaller than the 35x35 optical kernel and physically invalid)",
            "far fewer epochs and designs: 4 CPU cores, no GPU (G-040)",
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    s = run(args.config)
    print("\n=== PT vs PT+RL ===")
    for name, m in s["metrics"].items():
        print(f"  {name:6s} best-EPE {m['epe_best_mean']:.2f}  "
              f"sample-EPE {m['epe_sample_mean']:.2f}  diversity {m['diversity']:.5f}")


if __name__ == "__main__":
    main()
