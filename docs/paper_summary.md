# Paper Summary — arXiv 2602.19027

> **READ THIS FIRST.** The primary PDF/HTML of this paper was **never retrieved**
> in this session (arXiv and every mirror are blocked by the organization egress
> proxy — see `docs/source_inventory.md` for the 10 logged retrieval failures).
>
> Everything below is reconstructed from secondary sources. Each statement is
> tagged with its source ID and a confidence level:
> - **[A]** = stated in the abstract, corroborated across >=2 independent WebSearch
>   summaries of the arXiv abstract page (S1.1). Reliable at abstract granularity.
> - **[B]** = from the third-party LLM digest (S1.2). **Unverified.** Plausible but
>   could be hallucinated. Never treat as a paper fact.
> - **[C]** = my own inference from the surrounding literature, explicitly labeled
>   as an engineering assumption, not a paper claim.
>
> **No equation in this paper has been read.** Not one. Every formula in
> `REPRODUCTION_SPEC.md` is therefore a reconstruction from the named method
> (WGAN-GP, GRPO, AdaIN, MOSAIC-style ILT), not a transcription.

## Bibliographic facts [A]

- **Title:** Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning
- **Authors:** Haoyu Yang, Haoxing Ren (NVIDIA)
- **Venue:** 63rd Design Automation Conference (DAC'26). Also SPIE Advanced
  Lithography + Patterning 2026, presentation 13980-26.
- **arXiv:** 2602.19027v1, February 2026. Reference count reported as 22 [B].

## Problem statement [A]

ILT mask synthesis has a highly non-convex objective, so solvers stall in poor
local minima. Prior generative warm-starts train *deterministic* image-to-image
translators that imitate a sub-optimal mask dataset, which gives limited help in
escaping those traps during refinement.

## Core idea [A]

Reformulate mask synthesis as **conditional sampling** rather than deterministic
regression. A generator learns a *distribution* over masks conditioned on the
design and proposes **multiple candidates**.

Training is two-stage [A]:
1. **Pretrain** the generator with **WGAN** plus a **reconstruction loss**.
2. **Fine-tune** with **Group Relative Policy Optimization (GRPO)** using an
   **ILT-guided imitation loss**.

At inference [A]: sample a small batch of masks, run **fast batched ILT
refinement**, evaluate lithography metrics (EPE, process window), and **select
the best candidate**.

## Reported results [A]

- **>20% reduction in EPE violations** on ICCAD13 contest cases.
- **2-3x speedup** over the state-of-the-art numerical ILT solver.
- On **LithoBench**, reduced EPE violations under a **3 nm tolerance** and roughly
  doubled throughput versus a strong numerical ILT baseline [A].
- Achieved while using roughly **half the ILT iteration budget** [A/B].

> The **per-case tables (Table 1, Table 2) were not obtained.** Three targeted
> searches failed to surface any per-benchmark L2 / PVB / EPE / runtime numbers.
> This is the single hardest blocker for tier-L2 reproduction.

## Architecture, as described by the unverified digest [B]

**All of this is [B]. Treat as a hypothesis to be validated, not a specification.**

- **Style-aware U-Net generator** `G`. Inputs: design `Z` (full resolution) and a
  noise vector `z`. Output: mask logits `Y`.
- A **style mapping MLP** maps `z -> w`. Claimed `z`-dim = **256**.
- A **Style ResBlock at the coarse level (Level 2)** applies **AdaIN** with style
  code `w`. Style is injected **only at the coarse level**, so design topology
  (content, from `Z`) is preserved while geometry/style varies. The digest quotes
  this as section 3.2.1.
- Claimed rationale: features anchored to `Z` + style entering only via AdaIN
  keeps samples design-consistent and off the off-manifold artifacts of direct
  pixel-space sampling.

### Pretraining [A for the loss family, B for details]

`WGAN-GP` adversarial loss + `lambda_1 * l_rec(M_hat, M_gt)` against ground-truth masks.

### RL fine-tuning [B]

For each layout: sample `K` latents, generate `{Y_k}`, binarize
`M_k = 1[Y_k > 0.5]`, downsample, run a short low-resolution ILT refinement,
upsample to `M_k^ILT`.

- **Reward:** `R_k = -EPE(M_k^ILT, Z)` — negative EPE after refinement.
- **Advantage:** `A_k = R_k - R_k^T`, a **teacher-relative** baseline where the
  frozen pretrained generator `G_T` supplies `R_k^T` on the same latent. The
  digest claims this gives better-aligned credit assignment than a group-mean
  baseline and "anchors progress to a known-good policy".
- **Imitation loss:** distills `M_k^ILT` back into `G` via a smoothed L2 term.
- Claimed weights: `lambda_pg = 500`, `lambda_imit = 1`.
- Claimed RL ILT loop: **8x downsample**, **~100 iterations**, **K = 16**.
- `G_T` frozen; only `G` updated.

### Inference [A]

Sample `K` masks -> batched fast ILT -> evaluate EPE -> select best `M*`.

## Claimed ablation data point [B]

`StdContact` average EPE at the 3 nm threshold: **PT 9.0 -> PT+RL 6.5**. This is a
single number from an unverified digest and is the only per-experiment figure
recovered from any source.

## Limitations the digest attributes to the paper [B]

- Fast batched ILT solver details (algorithm, step size, convergence) are not
  fully specified in the paper.
- Lithography model parameters (SOCS coefficients, process window) are not fully
  specified. **This one is largely moot for ICCAD13**, because the contest fixes
  the optical model and the 24-kernel SOCS set is public (S2.2) [C].
- No ablation on the RL-loop ILT resolution / iteration count.
- Only EPE is used in the reward; multi-objective reward is future work.

## Relationship to the authors' prior work [C]

Haoyu Yang is the author of **GAN-OPC** (DAC'18, ILT-guided GAN mask synthesis),
**DAMO**, **ILILT** (ICML'24), and **LithoBench** (NeurIPS'23). This paper is the
direct successor to the GAN-OPC line, replacing the deterministic generator with a
conditional sampler and adding RL fine-tuning. That lineage is why the ICCAD13
24-kernel SOCS model and the L2/PVB/EPE protocol (S2.3-S2.5) are almost certainly
the exact evaluation stack used — but this is **inference, not a paper fact**.
