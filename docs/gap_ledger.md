# Gap Ledger

**Updated after the primary PDF was supplied.** G-001 is closed; most Tier-1 rows
are now resolved from the paper. What remains open is (a) what the paper itself
does not state, (b) one place where the paper **contradicts itself**, and (c) the
hard environment limits of this host.

**Confidence:** High = verified against a public artifact executed here, or read
verbatim from the paper. Medium = stated by the paper but under-specified.
Low = our reconstruction. **None = unknown.**

---

## CLOSED

| ID | Item | Resolution |
|---|---|---|
| **G-001** | Primary paper unobtainable | **CLOSED.** User supplied `2602.19027v1` (7 pp). All equations, both tables, and the full configuration are now in `docs/paper_summary.md`. |
| **G-002** | Table 1 / Table 2 numbers unknown | **CLOSED.** Both tables transcribed in full, per ICCAD13 case, including all baselines. |
| **G-011** | Latent dimension | **CLOSED.** `z`-dim = **256** (Sec. 4.1). |
| **G-013** | Policy log-probability definition | **CLOSED.** Eq. 8: pixels are independent Bernoulli with prob `sigmoid(Y_k)`; BCE is the surrogate for `-log P`; `M_k` is **detached**. Our reconstruction was correct. |
| **G-014** | Loss weights | **CLOSED.** `lambda_pg = 500`, `lambda_imit = 1` (Sec. 4.1). Imitation is **LpLoss with p=2** and **25x25 stride-1 average-pool smoothing**. |
| **G-016** | EPE tolerance | **CLOSED.** Table 1 uses **15 nm**; Table 2 is a **3 nm** stress test. The paper's stated rationale: 15 nm is "overly permissive even at the 45 nm node", and targeting 0 EPE gives vanishing/unstable policy gradients. Our decision to parameterize the tolerance and report both was right. |
| **G-017** | Ground-truth masks for pretraining | **CLOSED (as a fact).** LithoBench reference masks — which the paper itself calls **sub-optimal** (Sec. 3.1). Still **unobtainable here** (see G-041). |
| **G-018** | Dataset and split | **CLOSED.** Pretrain on MetalSet (14,824) + ViaSet (104,773); test on StdMetal (271), StdContact (165), ICCAD13 (10). **ICCAD13 is explicitly out-of-distribution.** |
| **G-019** | K, downsample, RL ILT steps | **CLOSED.** `K=16`, downsample factor **8**, **100** ILT iterations, upsample by custom low-res pooling + **bicubic**, binarize at 0.5. |

---

## OPEN — the paper does not say

### G-012 — **The paper contradicts itself on the RL baseline**
| | |
|---|---|
| **Issue** | Sec. 3.3.2 presents the **teacher-relative** baseline `A_k = R_k - R_k^T` (Eqs. 6-7) as the paper's contribution, and argues explicitly *against* the group mean (outlier sensitivity, weak advantages, rollout coupling). Sec. 4.1 then states the configuration used "a self-critical baseline given by **the group mean**." |
| **Why it matters** | These are different algorithms, and this is the paper's headline algorithmic contribution. Which one produced Tables 1-2 cannot be determined from the text. |
| **Current assumption** | Implement **both**, config-selected: `advantage: teacher_relative \| group_mean`. Default `teacher_relative` (what Sec. 3.3.2 argues for and what Eqs. 6-7 define). |
| **Sensitivity experiment** | **X-07** runs both under an identical budget and reports the difference — turning a paper defect into a measured result. |
| **Confidence** | **None** as to the paper's intent; **High** that both are implemented faithfully to their respective definitions. |

### G-015 — ILT solver internals still unspecified
| | |
|---|---|
| **Missing** | The paper never defines its ILT solver. It cites CurvyILT [4] (Yang & Ren, ISPD'25) for the solver *and* the morphological MRC handling, giving no mask parameterization, step size, optimizer, loss weights, or convergence rule. |
| **Sources searched** | The paper (nothing); no public code for [4] found. |
| **Current assumption** | MOSAIC/GAN-OPC/CurvyILT-lineage reconstruction in `src/gril/ilt/solver.py`: `M = sigmoid(beta_m * P)`, nominal-corner L2 vs the target, optional PV-band and total-variation terms. **Four optimizers implemented and tested** (Adam default, SGD, Nesterov, L-BFGS with strong-Wolfe line search); the committed ICCAD13 results use Adam, whose constants were tuned by our own sweep and logged, never tuned to match the paper's numbers. |
| **Sensitivity experiment** | X-02 step-size/steepness sweep (run); X-10 iteration-budget curve; **X-14 optimizer comparison** (Adam vs L-BFGS, wall-clock- and function-eval-matched, `configs/experiments/abl_optimizer.yaml`). |
| **Confidence** | **Medium** for the family (the paper's Eq. 1-2 physics is fixed and verified), **Low** for the constants. |
| **New finding (F-SAT-01)** | At the constants every ICCAD13 experiment uses (`init_scale=2.0`, `mask_steepness=8.0`), the sigmoid mask parameterization is saturated enough (`sigmoid(±16)`) that raw-gradient optimizers (SGD, Nesterov, L-BFGS) make no measurable progress from the standard cold start, while Adam's per-parameter normalization is insensitive to the tiny gradient magnitude and converges normally. This is why Adam is the default, now tested rather than assumed (`docs/findings.md` F-SAT-01). |
| **New finding (F-LBFGS-01)** | X-14 ran, on cases 1/3/5 at a well-conditioned start: L-BFGS matched/beat Adam's EPE@15nm (roughly halved it on the hardest case, 24 -> 13) and had lower L2, using 26-27% fewer function evaluations and 34-36% less wall-clock. EPE@3nm was mixed (worse on case 1). Evidence, on a 3-case sample, that a better-conditioned optimizer than the main baseline's Adam setup is available -- not yet run on all 10 cases. |
| **New finding (F-ANNEAL-01)** | Beta annealing (`ILTConfig.beta_init`) lets L-BFGS train from the main baseline's EXACT saturated constants (previously it stalled at 0.0% progress): verified L2 103 -> 12 on a test pattern. Fixes the underlying conditioning rather than routing around it with a different starting point. SGD's continuous loss also moves under annealing but did not flip any pixel in a short 20-iteration budget -- not a uniform fix for every optimizer's speed. |

### G-010 — Generator sizing
| **Missing** | Channel widths, number of Style ResBlocks `n`, MLP depth for `f_phi`, and the "optional head downsample" choice. Fig. 3 fixes the **topology** (3-level pyramid, AdaIN at Level 2 only, element-wise-addition fusion, Local ResBlocks, UpConv, optional final bicubic) but no sizes. |
|---|---|
| **Current assumption** | Topology exactly per Fig. 3; widths config-exposed. |
| **Confidence** | **High** on topology, **None** on sizing. |

### G-020 — Pretraining loss details
| **Missing** | `l_rec` is "e.g. l1 or l2" — never pinned. `lambda_1` (reconstruction) and `lambda_2` (gradient penalty) are **never given**. |
|---|---|
| **Current assumption** | `l_rec = l1`, `lambda_1 = 100`, `lambda_2 = 10` (the standard WGAN-GP value). All config-exposed. |
| **Confidence** | **Low.** |

### G-021 — MRC rule values
| **Missing** | Curvilinear MRC rules are cited to [17] (a Siemens technical report) only; no numeric width/space/curvature limits appear. |
|---|---|
| **Current assumption** | Morphological opening with a configurable structuring-element size; default off, reported when on. |
| **Confidence** | **None.** |

---

## OPEN — environment limits of this host (not paper gaps)

### G-040 — No GPU
The paper runs on **8x A100 80GB**, PyTorch 2.3.0 + CUDA 12.4, DDP/NCCL. This
host has **4 CPU cores, no GPU**, and the user has directed that no GPU/CUDA path
be used. Measured here: **3.80 s/iter** at 2048x2048, **0.015 s/iter** at 256x256.
Paper-scale pretraining (50 epochs x ~120k layouts at 2048^2) is **not reachable**.
Consequence: the "3x speedup" / "2x throughput" claims are **not evaluable**, and
the "Ours" rows of Tables 1-2 **cannot be reproduced**.

### G-041 — LithoBench unobtainable
MetalSet/ViaSet/StdMetal/StdContact are the paper's training and primary test
sets. The host is unreachable under this session's egress policy. Consequence:
all StdMetal / StdContact rows are **not reproducible here**; training data is
substituted by a documented, seeded synthetic generator, and that substitution is
stated wherever it affects a number.

---

## Standing rule

Resolving a gap by tuning until numbers match the paper is forbidden. Solver
constants were selected by a sweep on an objective (L2) that is **not** the
reported comparison metric, before any comparison to the paper's tables.
