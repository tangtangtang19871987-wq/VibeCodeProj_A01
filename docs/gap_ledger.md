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
| **New finding (F-MULTISTART-01)** | **X-15**, the control this ambiguity motivates asking for: at this project's training scale, a K=8 random multi-start (no learning, diversity-matched to the trained generator) exactly matches single-start ILT on every one of 8 held-out designs (115.00 EPE@3nm both), while BOTH trained-generator arms (PT 134.75, PT+RL 134.50) score *worse* than either. Whichever advantage definition Table 2 actually used, the generator this reproduction trained is not placing its K samples better than noise would, and is not beating a plain cold start either (`docs/findings.md` F-MULTISTART-01). Extends F-RL-01 (GRPO finetuning null result) to the pretraining stage too. |

### G-015 — ILT solver internals still unspecified
| | |
|---|---|
| **Missing** | The paper never defines its ILT solver. It cites CurvyILT [4] (Yang & Ren, ISPD'25) for the solver *and* the morphological MRC handling, giving no mask parameterization, step size, optimizer, loss weights, or convergence rule. |
| **Sources searched** | The paper (nothing); no public code for [4] found. |
| **Current assumption** | MOSAIC/GAN-OPC/CurvyILT-lineage reconstruction in `src/gril/ilt/solver.py`: `M = sigmoid(beta_m * P)`, nominal-corner L2 vs the target, optional PV-band and total-variation terms. **Four optimizers implemented and tested** (Adam default, SGD, Nesterov, L-BFGS with strong-Wolfe line search); the committed ICCAD13 results use Adam, whose constants were tuned by our own sweep and logged, never tuned to match the paper's numbers. |
| **Sensitivity experiment** | X-02 step-size/steepness sweep (run); X-10 iteration-budget curve; **X-14 optimizer comparison** (Adam vs L-BFGS, wall-clock- and function-eval-matched, `configs/experiments/abl_optimizer.yaml`). |
| **Confidence** | **Medium** for the family (the paper's Eq. 1-2 physics is fixed and verified), **Low** for the constants. |
| **New finding (F-SAT-01)** | At the constants every ICCAD13 experiment uses (`init_scale=2.0`, `mask_steepness=8.0`), the sigmoid mask parameterization is saturated enough (`sigmoid(±16)`) that raw-gradient optimizers (SGD, Nesterov, L-BFGS) make no measurable progress from the standard cold start, while Adam's per-parameter normalization is insensitive to the tiny gradient magnitude and converges normally. This is why Adam is the default, now tested rather than assumed (`docs/findings.md` F-SAT-01). |
| **New finding (F-LBFGS-01)** | X-14 ran, on cases 1/3/5 at a well-conditioned start: L-BFGS matched/beat Adam's EPE@15nm (roughly halved it on the hardest case, 24 -> 13) and had lower L2, using 26-27% fewer function evaluations and 34-36% less wall-clock. EPE@3nm was mixed (worse on case 1). Evidence, on a 3-case sample, that a better-conditioned optimizer than the main baseline's Adam setup is available. |
| **New finding (F-LBFGS-02)** | **X-16** extended this to all 10 cases: L-BFGS (same well-conditioned start X-14 validated) matches the paper's Table 1 "OURS" EPE@15nm column EXACTLY on 10/10 cases (mean 1.6, identical per-case), closing the main Adam baseline's case-3 gap (22->13) and case-5's known F-PERF-01 pixel (1->0) in one single-variable optimizer swap. Mean L2 also lower in 8/10 cases. Not adopted as a replacement for the main baseline (different starting point/budget, deliberate single-question test) -- see `docs/findings.md` F-LBFGS-02 for full scope caveats. |
| **New finding (F-LBFGS-03)** | User asked for the 3 most promising directions to improve on X-16, implemented and tested; the real lever wasn't any of the 3. D1 (EPE-aware loss stacked on L-BFGS) backfired (L2/EPE@15nm both got much worse). D2 (coarse-to-fine multi-resolution) looked like a 3-metric win but a matched-iteration-budget CONTROL beat it on EPE@15nm using no multi-resolution trick, refuting the "escapes bad local minima" hypothesis -- D2's real value is wall-clock efficiency, not a better optimum. D3 (best-of-4 multistart) gave a marginal, non-uniform gain for 4x the compute. The actual fix, found via that same control: L-BFGS was simply under-budgeted at 50 iterations. **X-17** (iterations=100, all 10 cases) gets mean EPE@15nm to **1.0 -- better than the paper's own reported 1.6** -- and cuts mean EPE@3nm from 55.5 to 46.6, though that still falls short of the paper's own no-generator ISPD25 reference (32.8). `docs/findings.md` F-LBFGS-03. |
| **New finding (F-ANNEAL-01)** | Beta annealing (`ILTConfig.beta_init`) lets L-BFGS train from the main baseline's EXACT saturated constants (previously it stalled at 0.0% progress): verified L2 103 -> 12 on a test pattern. Fixes the underlying conditioning rather than routing around it with a different starting point. SGD's continuous loss also moves under annealing but did not flip any pixel in a short 20-iteration budget -- not a uniform fix for every optimizer's speed. |
| **New finding (F-PVB-01)** | `weight_pvb` (the PV-band term, already in `ilt_loss`) produces a real, monotonic trade-off, measured directly on a 7-point sweep: PV Band down ~7.5% (11514 -> 10654) from `weight_pvb=0` to `1.0`, L2 up only ~3.8% over the same range, on cases 1/3/9. Confirms the term is doing physically sensible work; not adopted into the main baseline since the paper specifies no PV-band weight to reproduce against (`docs/findings.md` F-PVB-01). |
| **New finding (F-EPE-01)** | An independently-built differentiable EPE-site loss (`gril.ilt.epe_loss`), cross-checked to exactly reproduce the verified `epe_violations()` metric's per-side counts on a binary mask. A naive weight (`weight_epe=2.0`) had no measurable effect, root-caused to a ~2883x magnitude mismatch against the L2 term; at a calibrated weight (`weight_epe=3000`, measured to be the same order of magnitude as L2 at this resolution) it reduces case 3's EPE@15nm by 38% (26 -> 16) with L2 essentially unchanged from the L2-only baseline (`docs/findings.md` F-EPE-01). |

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
| **New finding (F-MRC-01)** | Every committed mask so far used the default (opening off): measured, this means **~211 connected components per case on average, ~80 of them smaller than a 4.5x4.5nm square** -- real, user-flagged, physically unmanufacturable debris, not a visualization artifact. A 10-case post-hoc sweep (`scripts/mrc_cleanup_sweep.py`) shows `mrc_open_size=5` removes essentially all of that debris (79.9 -> 0.3 mean tiny components) for a 0.33% mean L2 cost -- but ~123-135 components still remain even at the largest sizes tested, and case 3's EPE@15nm specifically gets WORSE under opening (22 -> 24-25). Opening is a cheap partial fix for the worst symptom, not a real MRC solution (`docs/findings.md` F-MRC-01). |

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
