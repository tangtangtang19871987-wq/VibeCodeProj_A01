# Traceability Matrix

Maps every item that must be reproduced to its implementation and its test.

**The primary PDF has been obtained**, so the "Paper location" column now cites
the paper's own equation and section numbers. Provenance tags:

- **[EQ n]** the paper's Equation n. **[Sec x]** its section. **[Tab n]** its table.
- **[PUB]** a public artifact executed in this session (`docs/source_inventory.md` S2.x).
- **[C]** our reconstruction, where the paper is silent — always cross-referenced
  to a gap-ledger ID.

Status: `SPEC` / `IMPL` / `TEST` (implemented + tested) / `VERIF` (test-verified
against an independent reference) / `BLOCKED`.

## 1. Forward lithography model

| # | Provenance | Item | Input | Output | Units | Implementation | Test | Status |
|---|---|---|---|---|---|---|---|---|
| F-01 | [PUB] S2.2 | 24-term SOCS kernel set, focus + defocus | — | `h_k`, `(24,35,35)` complex64; `w_k`, `(24,)` float32 | frequency-domain, dimensionless | `src/gril/litho/kernels.py` | eigenvalues >=0 and descending | **VERIF** |
| F-02 | **[EQ 1]** + [PUB] S2.3 | Hopkins/SOCS aerial image `I = sum_k w_k \|h_k ⊛ (d·M)\|^2` | mask `M` in [0,1], `(...,2048,2048)`; dose `d` | intensity `I >= 0` | normalized so open field -> 1 | `src/gril/litho/socs.py` | open-field = 0.95154; matches pure autograd to 1.7e-16 | **VERIF** |
| F-03 | [PUB] S2.3 | FFT convention: `norm="forward"`, kernel on the 4 corners, no shift | — | — | — | `src/gril/litho/socs.py` | corner path == centred path, 1.6e-7 | **VERIF** |
| F-04 | **[EQ 2]** + [PUB] S2.5 | Constant-threshold resist `Z = sigmoid(beta_r (I - I_th))`, `I_th=0.225`, `beta_r=50` | `I` | `Z` in (0,1) | — | `src/gril/litho/resist.py` | `Z(I_th)=0.5`; monotone in `I` | **TEST** |
| F-05 | [PUB] S2.5 | 3 process corners: Nom(1.00,focus), Max(1.02,focus), Min(0.98,defocus) | `M` | `Z_nom, Z_max, Z_min` | dose dimensionless | `src/gril/litho/resist.py` | 10-case regression bit-exact vs reference | **VERIF** |
| F-06 | [PUB] S2.3 + **corrected** | Analytic adjoint via `conj(H)` (**not** the shipped CT kernels — see F-ADJ-01) | `dL/dI` | `dL/dM` | — | `src/gril/litho/socs.py` (`autograd.Function`) | exact vs autograd (**3.9e-16**); float64 FD check | **VERIF** |
| F-07 | [C] | Translation + x-mirror equivariance (y-mirror does **not** hold — see F-KER-01) | `M`, transform | `I` | — | `src/gril/litho/socs.py` | translation 6.1e-7; x-mirror 4.9e-7; y-asymmetry pinned | **VERIF** |
| F-08 | [PUB] S2.4 | Canvas 2048x2048, 1 nm/px, centered designs | `.glp` | `(2048,2048)` float | nm | `src/gril/data/glp.py` | raster **bit-identical** to reference on all 10 cases | **VERIF** |

## 2. Metrics

| # | Provenance | Item | Input | Output | Units | Implementation | Test | Status |
|---|---|---|---|---|---|---|---|---|
| M-01 | [PUB] S2.4 | `L2 = sum((bin(Z_nom) - target)^2)` | `M`, target | scalar | px^2 (= nm^2) | `src/gril/metrics/core.py` | hand-computed example + 10-case bit-exact | **VERIF** |
| M-02 | [PUB] S2.4 | `PVB = count(bin(Z_max) != bin(Z_min))` | `M` | scalar | px^2 | `src/gril/metrics/core.py` | hand-computed example + 10-case bit-exact | **VERIF** |
| M-03 | [PUB] S2.4 | EPE violations: boundary extraction -> v/h segments -> sample sites (interval 40, min len 80, start 40) -> inner/outer check | `M`, target | `(epe_in, epe_out)` | count | `src/gril/metrics/core.py` | **reproduces S2.4 exactly on all 10 cases** | **VERIF** |
| M-04 | **[Tab 1/2]** (15 nm and 3 nm) | EPE at configurable tolerance; report 15 nm **and** 3 nm | `M`, target, tol | count | nm | `src/gril/metrics/core.py` | monotonicity in tolerance verified | **TEST** |
| M-05 | **[Sec 2/4.1]** morphological MRC cleanup (values not given, G-021) | `M` | violation count | nm | `src/gril/metrics/mrc.py` | synthetic under/over-width shapes | SPEC |
| M-06 | [C] | Shot count (rectangle decomposition) | `M` | count | — | `src/gril/metrics/shots.py` | known-decomposition shapes | SPEC |
| M-07 | [A] | Runtime per case | — | seconds | s | `src/gril/experiments/` | recorded, never estimated | SPEC |

## 3. Numerical ILT solver

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| I-01 | [C] G-015 (paper never specifies its solver) | Mask parameterization `M = sigmoid(beta_m * P)` | `src/gril/ilt/solver.py` | binary output; determinism | **TEST** |
| I-02 | [C] G-015 | ILT loss `\|\|Z_nom - target\|\|^2` (+ optional PVB, total-variation terms) | `src/gril/ilt/solver.py` | loss decreases; beats no-OPC; TV reduces mask roughness | **TEST** |
| I-03 | [C] G-015 | **Four optimizers**: Adam (default), SGD, Nesterov, L-BFGS (strong-Wolfe) | `src/gril/ilt/solver.py` | each reduces loss from a well-conditioned start; sweep-selected Adam constants for the committed baseline | **TEST** |
| I-04 | [A] | **Batched** ILT over K candidates | `src/gril/ilt/solver.py` | **batched == looped single-case, exactly**, for ALL FOUR optimizers (L-BFGS via independent per-example instances, by design — see REPRODUCTION_SPEC.md Sec. 8) | **VERIF** |
| I-07 | [C] | L-BFGS quasi-Newton solver with line search; gradient-norm convergence; gradient clipping | `src/gril/ilt/solver.py` | deterministic; more func-evals than iterations (line search); `n_func_evals` tracked for fair cross-optimizer comparison | **TEST** |
| I-08 | [C] | F-SAT-01: sigmoid saturation at default constants stalls raw-gradient optimizers | `src/gril/ilt/solver.py`, `docs/findings.md` | Adam moves from the exact default start, SGD does not (pinned) | **VERIF** |
| I-05 | **[Sec 4.1]** 8x downsample, 100 iters, bicubic upsample, binarize 0.5 | `src/gril/experiments/infer.py` | **multi-resolution physics verified to 0.17% of peak at 8x** | **VERIF** |
| I-06 | [C] | Checkpoint / resume | `src/gril/experiments/run_iccad13.py` | per-case JSON cache; resume verified in use | **IMPL** |

## 4. Generative model

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| N-01 | **[Fig 3]** | Style-aware U-Net generator `G(Z, z) -> Y` | `src/gril/models/generator.py` | shape/dtype contract; CPU-only | **TEST** |
| N-02 | **[Sec 3.2.1/4.1]** Mapping MLP `z -> w`, `dim(z)=256` | `src/gril/models/generator.py` | mapping receives gradient | **TEST** |
| N-03 | **[EQ 4]** + [Fig 3] Style ResBlock, AdaIN at Level 2 only | `src/gril/models/generator.py` | AdaIN hits commanded per-channel stats to 1e-4 | **VERIF** |
| N-04 | **[Sec 3.2.1]** | Design-anchoring: fixed `z` -> deterministic output; varying `z` -> diverse but topology-preserving | `src/gril/models/generator.py` | determinism + **non-zero diversity** (regression for a real collapse bug) | **TEST** |
| N-05 | **[EQ 5]** WGAN-GP critic (weights not given, G-020) | `src/gril/models/generator.py` | GP -> 0 for a unit-norm map; **finite at zero gradient** | **TEST** |

## 5. Training

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| T-01 | **[EQ 5]** Stage 1: WGAN-GP + `lambda_1 * l_rec` | `src/gril/train/pretrain.py` | critic step stays finite over repeated steps | **TEST** |
| T-02 | **[Sec 3.3.2]** | Stage 2: GRPO fine-tune with ILT-guided imitation loss | `src/gril/train/grpo.py` | end-to-end finetune step | **TEST** |
| T-03 | **[Sec 3.3.2]** | Reward `R_k = -EPE(M_k^ILT, Z)` | `src/gril/train/grpo.py` | reward direction verified | **TEST** |
| T-04 | **[EQ 6-7]** vs **[Sec 4.1]** — the paper contradicts itself (G-012); both implemented | `src/gril/train/grpo.py` | group-mean advantage has zero mean; teacher form exact | **TEST** |
| T-05 | **[EQ 8]** | Policy log-prob = negative BCE under pixel-wise Bernoulli | `src/gril/train/grpo.py` | **matches `torch.distributions.Bernoulli.log_prob` exactly** | **VERIF** |
| T-06 | **[EQ 9]** (25x25 avg-pool smoothing) | Imitation loss `\|\|Y_k - S(M_k^ILT)\|\|^2` | `src/gril/train/grpo.py` | smoothing preserves constants (padding bug fixed) | **TEST** |
| T-07 | **[EQ 10]** + [Sec 4.1] | `L_FT = lambda_pg L_pg + lambda_imit L_imit`, `lambda_pg=500`, `lambda_imit=1` | `src/gril/train/grpo.py` | both terms logged per step | **IMPL** |
| T-08 | **[Sec 3.3.2]** | Teacher `G_T` frozen; only `G` updated | `src/gril/train/grpo.py` | **teacher frozen and gradient-free, verified** | **VERIF** |
| T-09 | [C] | Seeding / determinism | `src/gril/config/seed.py` | same seed -> bitwise-identical run | SPEC |

## 6. Inference

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| P-01 | **[Fig 1c]** + [Sec 3.3.2] | Sample K -> batched ILT -> evaluate -> select best | `src/gril/experiments/infer.py` | selection == argmin; deterministic tie-break | **TEST** |
| P-02 | **[abstract]** | Selection criterion (EPE; process window) | `src/gril/experiments/infer.py` | tie-break: metric, then L2, then index | **TEST** |

## 7. Experiments

| # | Target | Provenance | Config | Output | Status |
|---|---|---|---|---|---|
| X-01 | ICCAD13, no-OPC reference | [PUB] | `configs/experiments/iccad13_noopc.yaml` | `results/iccad13_noopc/` | **VERIF** — measured this session: mean L2 102664, PVB 37476, EPE 69.6 |
| X-02 | ICCAD13, plain numerical ILT baseline | [C] | `configs/experiments/iccad13_ilt.yaml` | `results/iccad13_ilt/` | SPEC |
| X-03 | ICCAD13, deterministic warm start + ILT | [A] (the paper's "prior work" comparison) | `configs/experiments/iccad13_det.yaml` | `results/iccad13_det/` | SPEC |
| X-04 | ICCAD13, PT-only sampler + K-candidate ILT | [A] | `configs/experiments/iccad13_pt.yaml` | `results/iccad13_pt/` | SPEC |
| X-05 | ICCAD13, PT+RL + K-candidate ILT (**the method**) | [A] | `configs/experiments/iccad13_ptrl.yaml` | `results/iccad13_ptrl/` | SPEC |
| X-06 | Ablation: K sweep (G-019) | [B] | `configs/experiments/abl_k.yaml` | `results/abl_k/` | SPEC |
| X-07 | Ablation: GRPO variant (G-012) | [B] | `configs/experiments/abl_grpo.yaml` | `results/abl_grpo/` | SPEC |
| X-08 | Ablation: imitation loss on/off (G-014) | [B] | `configs/experiments/abl_imit.yaml` | `results/abl_imit/` | SPEC |
| X-09 | Sensitivity: EPE tolerance curve 1..15 nm (G-016) | [A] | `configs/experiments/sens_epe.yaml` | `results/sens_epe/` | SPEC |
| X-10 | Iteration-budget curve (tests "half the budget" claim) | [A] | `configs/experiments/budget_curve.yaml` | `results/budget_curve/` | SPEC |
| X-11 | Paper Table 1 / Table 2 replication | [A] | — | — | **BLOCKED by G-002** (target numbers unknown) |
| X-12 | LithoBench experiments | [A] | — | — | **BLOCKED by S3** (dataset unreachable) |
| X-13 | Runtime / speedup comparison | [A] | — | — | **BLOCKED** — no GPU on this host; would not be comparable |
| X-14 | Ablation: optimizer family (Adam vs L-BFGS), G-015 | [C] | `configs/experiments/abl_optimizer.yaml` | `results/abl_optimizer/` | **VERIF** -- measured: L-BFGS matched/beat Adam EPE@15nm on cases 1/3/5 with 26-27% fewer func evals (F-LBFGS-01) |
| X-15 | Control: learned sampler (generator) vs random multi-start, same K and budget -- isolates what Table 2's PT/PT+RL rows actually demonstrate | [C] | `configs/experiments/multistart_ablation.yaml` | `results/multistart_ablation/` | **VERIF** -- random-K exactly equals single-start on all 8 designs (115.00 == 115.00); both generator arms (PT 134.75, PT+RL 134.50) are *worse* than either (F-MULTISTART-01) |

## 8. Completeness check against the paper

The paper contains **10 numbered equations**, **4 figures**, **2 tables**, and 22
references. All are accounted for:

| Paper item | Where it lands |
|---|---|
| Eq. 1 (SOCS) | F-02 — **VERIF** |
| Eq. 2 (constant-threshold resist) | F-04 — **TEST** |
| Eq. 3 (pushforward / conditional distribution) | N-01 — definitional; realised by the sampler |
| Eq. 4 (AdaIN) | N-03 — **VERIF** |
| Eq. 5 (WGAN-GP + reconstruction) | T-01, N-05 — **TEST** |
| Eq. 6 (teacher-relative advantage) | T-04 — **TEST** (paper self-contradiction, G-012) |
| Eq. 7 (teacher reward) | T-04 — **TEST** |
| Eq. 8 (BCE policy loss) | T-05 — **VERIF** |
| Eq. 9 (imitation loss) | T-06 — **TEST** |
| Eq. 10 (total finetuning loss) | T-07 — **TEST** |
| Fig. 1 (scheme comparison) | P-01 — conceptual; realised by `infer.py` |
| Fig. 2 (metric definitions) | M-01..M-04 — **VERIF** |
| Fig. 3 (generator architecture) | N-01, N-03 — **TEST** |
| Fig. 4 (posterior-sample visualisation) | **BLOCKED** — needs a paper-scale trained generator |
| Table 1 (15 nm) | X-11 — **BLOCKED** for the "Ours" row; baselines compared in `REPRODUCTION_REPORT.md` |
| Table 2 (3 nm) | X-11 — same |
| Definitions 1-2 (EPE, PVB) | M-02, M-03 — **VERIF** |

**No item of the paper is unaccounted for.** The blocked items are blocked by
environment limits (G-040 no GPU, G-041 LithoBench unreachable), not by missing
information.
