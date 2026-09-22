# Traceability Matrix

Maps every item that must be reproduced to its implementation and its test.

**Critical caveat:** the "Paper location" column normally cites an equation or
section number. Because the primary text was never retrieved (G-001), that column
instead records the **best available provenance**, using the tags from
`docs/paper_summary.md`:

- **[A]** abstract-level, corroborated. **[B]** unverified LLM digest, with the
  section number *it* claims. **[C]** engineering reconstruction. **[PUB]** a
  public artifact executed in this session (`docs/source_inventory.md` S2.x).

**An item whose provenance is [B] or [C] is NOT "reproduced from the paper."** It
is reimplemented. Status values are:
`SPEC` (specified only) / `IMPL` (implemented) / `TEST` (implemented + tested) /
`VERIF` (test-verified against an independent reference) / `BLOCKED`.

## 1. Forward lithography model

| # | Provenance | Item | Input | Output | Units | Implementation | Test | Status |
|---|---|---|---|---|---|---|---|---|
| F-01 | [PUB] S2.2 | 24-term SOCS kernel set, focus + defocus | — | `h_k`, `(24,35,35)` complex64; `w_k`, `(24,)` float32 | frequency-domain, dimensionless | `src/gril/litho/kernels.py` | eigenvalues descending; Hermitian/PSD of implied TCC | SPEC |
| F-02 | [PUB] S2.3 | Hopkins/SOCS aerial image `I = sum_k w_k \|h_k ⊛ (d·M)\|^2` | mask `M` in [0,1], `(...,2048,2048)`; dose `d` | intensity `I >= 0` | normalized so open field -> 1 | `src/gril/litho/socs.py` | open-field = 0.95154 (+-1e-4); 400 nm square = 0.93562 | SPEC |
| F-03 | [PUB] S2.3 | FFT convention: `norm="forward"`, kernel on the 4 corners, no shift | — | — | — | `src/gril/litho/socs.py` | fast path == legacy fftshift path to 1e-6 | SPEC |
| F-04 | [PUB] S2.5 | Constant-threshold resist `Z = sigmoid(beta_r (I - I_th))`, `I_th=0.225`, `beta_r=50` | `I` | `Z` in (0,1) | — | `src/gril/litho/resist.py` | `Z(I_th)=0.5`; monotone in `I` | SPEC |
| F-05 | [PUB] S2.5 | 3 process corners: Nom(1.00,focus), Max(1.02,focus), Min(0.98,defocus) | `M` | `Z_nom, Z_max, Z_min` | dose dimensionless | `src/gril/litho/corners.py` | `I_max >= I_nom >= I_min` on a dense test pattern | SPEC |
| F-06 | [PUB] S2.3 | Analytic adjoint via CT kernels | `dL/dI` | `dL/dM` | — | `src/gril/litho/socs.py` (`autograd.Function`) | **finite-difference gradient check** vs float64 | SPEC |
| F-07 | [C] | Translation / mirror / rotation equivariance of the imaging operator | `M`, transform | `I` | — | `src/gril/litho/socs.py` | shifted mask -> shifted image (periodic) to 1e-5 | SPEC |
| F-08 | [PUB] S2.4 | Canvas 2048x2048, 1 nm/px, centered designs | `.glp` | `(2048,2048)` float | nm | `src/gril/data/glp.py` | bbox + area match GLP RECT/PGON records exactly | SPEC |

## 2. Metrics

| # | Provenance | Item | Input | Output | Units | Implementation | Test | Status |
|---|---|---|---|---|---|---|---|---|
| M-01 | [PUB] S2.4 | `L2 = sum((bin(Z_nom) - target)^2)` | `M`, target | scalar | px^2 (= nm^2) | `src/gril/metrics/l2.py` | hand-computed 8x8 example | SPEC |
| M-02 | [PUB] S2.4 | `PVB = count(bin(Z_max) != bin(Z_min))` | `M` | scalar | px^2 | `src/gril/metrics/pvband.py` | hand-computed example; PVB=0 for identical corners | SPEC |
| M-03 | [PUB] S2.4 | EPE violations: boundary extraction -> v/h segments -> sample sites (interval 40, min len 80, start 40) -> inner/outer check | `M`, target | `(epe_in, epe_out)` | count | `src/gril/metrics/epe.py` | **must reproduce S2.4's numbers exactly on all 10 ICCAD13 cases** | SPEC |
| M-04 | [A] + G-016 | EPE at configurable tolerance; report 15 nm **and** 3 nm | `M`, target, tol | count | nm | `src/gril/metrics/epe.py` | violations monotone non-increasing in tolerance | SPEC |
| M-05 | [C] | MRC (min width / min space) check | `M` | violation count | nm | `src/gril/metrics/mrc.py` | synthetic under/over-width shapes | SPEC |
| M-06 | [C] | Shot count (rectangle decomposition) | `M` | count | — | `src/gril/metrics/shots.py` | known-decomposition shapes | SPEC |
| M-07 | [A] | Runtime per case | — | seconds | s | `src/gril/experiments/` | recorded, never estimated | SPEC |

## 3. Numerical ILT solver

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| I-01 | [C] G-015 | Mask parameterization `M = sigmoid(beta_m * P)` | `src/gril/ilt/solver.py` | `M in (0,1)`; `beta_m -> inf` gives step | SPEC |
| I-02 | [C] G-015 | ILT loss `\|\|Z_nom - target\|\|^2` (+ optional PVB term, corner weights) | `src/gril/ilt/loss.py` | hand example; gradient check | SPEC |
| I-03 | [C] G-015 | Adam-based descent on `P`; configurable step/iters | `src/gril/ilt/solver.py` | monotone loss decrease on a smooth case | SPEC |
| I-04 | [A] | **Batched** ILT over K candidates | `src/gril/ilt/solver.py` | batched == looped single-case to 1e-6 | SPEC |
| I-05 | [A] G-019 | Low-resolution ILT loop used for the RL reward (downsample -> solve -> upsample) | `src/gril/ilt/lowres.py` | round-trip shape/dtype; reward correlation vs full-res | SPEC |
| I-06 | [C] | Checkpoint / resume | `src/gril/ilt/solver.py` | resumed run == uninterrupted run | SPEC |

## 4. Generative model

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| N-01 | [B] 3.2.1 | Style-aware U-Net generator `G(Z, z) -> Y` | `src/gril/models/generator.py` | shape/dtype/device contract | SPEC |
| N-02 | [B] 3.2.1 | Mapping MLP `z -> w`, `dim(z)=256` (G-011) | `src/gril/models/generator.py` | `w` shape; distinct `z` -> distinct `w` | SPEC |
| N-03 | [B] 3.2.1 | Style ResBlock with AdaIN at the coarse level only | `src/gril/models/styleblock.py` | AdaIN output has the commanded per-channel mean/std to 1e-5 | SPEC |
| N-04 | [B] 3.2.1 | Design-anchoring: fixed `z` -> deterministic output; varying `z` -> diverse but topology-preserving | `src/gril/models/generator.py` | same-seed determinism; **measured** pairwise diversity > 0 | SPEC |
| N-05 | [A]/[B] G-020 | WGAN-GP discriminator | `src/gril/models/discriminator.py` | GP term -> 0 for a 1-Lipschitz map | SPEC |

## 5. Training

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| T-01 | [A] | Stage 1 pretrain: WGAN(+GP) + `lambda_1 * l_rec` | `src/gril/train/pretrain.py` | loss decreases; deterministic under fixed seed | SPEC |
| T-02 | [A] | Stage 2: GRPO fine-tune with ILT-guided imitation loss | `src/gril/train/grpo.py` | — | SPEC |
| T-03 | [B] 3.3.2 | Reward `R_k = -EPE(M_k^ILT, Z)` | `src/gril/train/reward.py` | reward improves when mask improves | SPEC |
| T-04 | [B] 3.3.2 / G-012 | Advantage: `teacher_relative` (default) \| `group_normalized` | `src/gril/train/grpo.py` | group-normalized has zero mean, unit std | SPEC |
| T-05 | [B] / G-013 | Policy log-prob = negative BCE under pixel-wise Bernoulli | `src/gril/train/grpo.py` | matches `torch.distributions.Bernoulli.log_prob` | SPEC |
| T-06 | [B] 3.3.2 | Imitation loss `\|\|Y_k - S(M_k^ILT)\|\|^2` | `src/gril/train/grpo.py` | zero at `Y_k == S(M_k^ILT)` | SPEC |
| T-07 | [B] / G-014 | `L_FT = lambda_pg L_pg + lambda_imit L_imit`, `lambda_pg=500`, `lambda_imit=1` | `src/gril/train/grpo.py` | per-term gradient-norm ratio logged | SPEC |
| T-08 | [B] | Teacher `G_T` frozen; only `G` updated | `src/gril/train/grpo.py` | `G_T` params unchanged after a step | SPEC |
| T-09 | [C] | Seeding / determinism | `src/gril/config/seed.py` | same seed -> bitwise-identical run | SPEC |

## 6. Inference

| # | Provenance | Item | Implementation | Test | Status |
|---|---|---|---|---|---|
| P-01 | [A] | Sample K -> batched ILT -> evaluate -> select best | `src/gril/experiments/infer.py` | selected == argmin of the recorded metric | SPEC |
| P-02 | [A] | Selection criterion (EPE; process window) | `src/gril/experiments/infer.py` | tie-breaking deterministic | SPEC |

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

## 8. Items known to exist in the paper but not enumerable

Because the paper was never read, **the set of its equations, algorithm boxes,
figures and tables is itself unknown**. This matrix cannot claim completeness.
Reference count is reported as 22 [B]; figure and table counts are unknown beyond
"Table 1 and Table 2 exist" [B]. **Completing this matrix requires G-001.**
