# Paper Summary — arXiv 2602.19027v1

**Source status: PRIMARY SOURCE OBTAINED.** The user supplied the PDF
(`2602.19027v1`, 7 pages, 22 references, cs.LG, 22 Feb 2026). Gap **G-001 is
closed**. Everything below is read directly from the paper; equation numbers are
the paper's own. Where the paper is silent or self-contradictory, that is stated
explicitly rather than filled in.

- **Title:** Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning
- **Authors:** Haoyu Yang, Haoxing Ren (NVIDIA Corp., Austin, TX)
- **Venue:** DAC'26 (PDF carries a placeholder "Conference'17" template header)

## 1. Problem and claim

ILT is highly non-convex; solvers stall in poor local minima. Prior GenAI
warm-starts train deterministic image-to-image translators on sub-optimal
datasets (LithoBench is explicitly called sub-optimal, Sec. 3.1), which gives
little help escaping traps. The paper reformulates mask synthesis as
**conditional sampling**, generating multiple candidates, refining each with a
**batched ILT solver**, and selecting the best (Fig. 1c).

Headline results (abstract + Sec. 4): on LithoBench, reduced EPE violations at a
3 nm tolerance and roughly **2x throughput**; on ICCAD13, **>20% EPE improvement**
with **3x speedup** over the SOTA numerical solver.

## 2. Forward model (Sec. 2) — matches our implementation exactly

    I = sum_{i=1}^{k} alpha_i * || M (x) h_i ||^2                        (Eq. 1)

`alpha_i`, `h_i` are eigenvalues/eigenvectors of the transmission cross
coefficient matrix — i.e. **SOCS**, exactly what `src/gril/litho/socs.py`
implements. Resist is **constant thresholding** (Eq. 2):

    Z_ij = 1 if I_ij > I_th else 0

Metrics are **EPE violations** and **PV Band area** (Definitions 1 and 2).

**Important:** the paper adopts a **curvilinear mask assumption** with MRC rules
from ref. [17], and enforces **MRC-cleanness by morphological operations** as in
CurvyILT [4]. Sec. 4.1: "We perform morphological opening and cleanup to ensure
curvilinear MRC compliance before printability evaluation."

## 3. Generator (Sec. 3.2)

Mask distribution conditioned on the design:

    M ~ P_{M|Z} ≈ (G(Z, ·))_# N                                         (Eq. 3)

`Z` = Manhattan design vector, `q ~ N` = noise.

**Style-aware architecture (Fig. 3), three levels:**
- An **input pyramid** `{Z_0, Z_1, Z_2}` built by downsampling (optional head downsample).
- **Level 2 (coarsest)**: stride-2 downsamples -> **Style ResBlocks x n** -> UpConv -> "Low Mask".
- **Levels 1 and 0**: coarse-to-fine. Fuse the **upsampled output of the previous
  level** with **downsampled features at the current resolution** by
  **element-wise addition**, refine with **Local ResBlocks**, then UpConv.
- Optional final **bicubic** interpolation restores the original resolution.
- **Style ResBlock detail:** `x -> AdaIN(w) -> Conv -> AdaIN(w) -> Conv -> (+x) -> y`.

AdaIN (Eq. 4):

    AdaIN(x; w) = gamma(w) ⊙ (x - mu(x)) / (sigma(x) + eps) + beta(w)

`w = f_phi(z)`, `z ~ N(0, I)`, `f_phi` a lightweight MLP; `mu`, `sigma` are
per-channel statistics. Style is injected **only at Level 2**, so content comes
from `Z` and style from `w`.

## 4. Stage 1 — generative pretraining (Sec. 3.3.1)

    min_G max_D  E_{M~P_{M|Z}}[D(M,Z)] - E_{Z,q}[D(G(Z,q),Z)]
                 + lambda_1 E[ l_rec(G(Z,q), M) ]
                 - lambda_2 E[ (|| grad_{M_hat} D(M_hat, Z) ||_2 - 1)^2 ]   (Eq. 5)

WGAN with **gradient penalty**; `M_hat = eps*M + (1-eps)*G(Z,q)`, `eps ~ U(0,1)`;
`l_rec` is "e.g. l1 or l2" — **the paper does not say which** (see G-020).

## 5. Stage 2 — reinforcement finetuning (Sec. 3.3.2)

Draw `{q_k}_{k=1..K}`, generate logits `Y_k = G(Z, q_k)`, binarize
`M_k = 1[Y_k > 0.5]`. Each `M_k` goes through a **few-step, low-resolution ILT
loop**, is upsampled to full resolution as `M_k^ILT`, and

    R_k = -EPE(M_k^ILT, Z)

**Teacher-relative advantage** (Eqs. 6, 7) — the paper's stated contribution:

    A_k = R_k - R_k^T,   R_k^T = -EPE(M_{k,T}^ILT, Z),  M_{k,T}^ILT = G_T(Z, q_k)

`G_T` is the **frozen pretrained model**. The paper explicitly rejects the
original GRPO group-mean baseline `R_k - (1/K) sum_j R_j`, citing high reward
variance, outlier sensitivity of the group mean, weak advantages, and rollout
coupling.

**Policy loss** (Eq. 8) — pixels treated as independent Bernoulli with
probability `sigmoid(Y_k)`, BCE used as a surrogate for `-log P`:

    L_pg = -E_{q_k}[ A_k * ( -BCE(Y_k, M_k) ) ]

with `M_k` **detached** so gradients flow only through `Y_k`.

**Imitation loss** (Eq. 9):

    L_imit = E_{q_k}[ || Y_k - S(M_k^ILT) ||_2^2 ]

`S(·)` is "a mild low-pass smoothing ... e.g. **25x25 stride-1 average pooling**".

**Total** (Eq. 10): `L_FT = lambda_pg * L_pg + lambda_imit * L_imit`.

## 6. Experimental configuration (Sec. 4.1) — verbatim

| Item | Value |
|---|---|
| Datasets | LithoBench: MetalSet **14,824**; ViaSet **104,773**; StdMetal **271**; StdContact **165**. ICCAD13: **10**. |
| Split | **Pretrain on MetalSet + ViaSet; test on StdMetal, StdContact, ICCAD13.** ICCAD13 is explicitly **out-of-distribution** (no aligned training set). |
| Framework | PyTorch **2.3.0**, CUDA **12.4** |
| Hardware | **single DGX node, 8x A100 80GB**, DDP (NCCL) |
| Pretraining | **50 epochs**; discriminator **Adam(2e-4, betas 0.5/0.999)**; generator **Prodigy** + cosine annealing; **batch 16**, 16 workers |
| RL finetuning | **E=20 epochs**, **batch 8** (one design per step), **K=16**, **z-dim 256**, **lr 1e-4 (Prodigy)**, cosine to **1e-7** |
| Reward loop | short ILT at **downsample factor 8**, **100 iterations**, upsample via custom low-res pooling + **bicubic**, binarize at **0.5** |
| Imitation | LpLoss with **p=2**, **25x25** smoothing |
| Weights | **lambda_pg = 500**, **lambda_imit = 1** |
| EPE thresholds | **15 nm** (Table 1) and **3 nm** stress test (Table 2). Rationale given: 15 nm is "overly permissive even at the 45 nm node"; targeting 0 EPE gives vanishing/unstable policy gradients. |
| Iteration budget | **Ours 150 iterations vs CurvyILT 300 iterations** |

## 7. Results

### Table 1 — 15 nm EPE threshold

| Benchmark | DAC'22 EPE/PV | DAC'23 EPE/PV | ISPD'25-300it EPE/PV | **Ours-150it EPE/PV** |
|---|---|---|---|---|
| StdContact-Avg | – | 8.6 / 39997.0 | 3.8 / 36172.0 | **2.0 / 39186.4** |
| StdMetal-Avg | – | 0.0 / 24928.0 | 0.0 / 21631.0 | **0.0 / 21029.2** |
| ICCAD13-1 | 7 / 47015 | 3 / 47015 | 3 / 44447 | **3 / 47459** |
| ICCAD13-2 | 3 / 37555 | 0 / 37555 | 0 / 36914 | **0 / 33965** |
| ICCAD13-3 | 62 / 69361 | 22 / 69361 | 15 / 70580 | **13 / 74370** |
| ICCAD13-4 | 2 / 21514 | 0 / 21514 | 0 / 21584 | **0 / 21985** |
| ICCAD13-5 | 1 / 49683 | 0 / 49683 | 0 / 47870 | **0 / 47781** |
| ICCAD13-6 | 2 / 44127 | 0 / 44127 | 0 / 42288 | **0 / 42987** |
| ICCAD13-7 | 0 / 36961 | 0 / 36961 | 0 / 34389 | **0 / 36062** |
| ICCAD13-8 | 0 / 20985 | 0 / 20985 | 0 / 18649 | **0 / 18312** |
| ICCAD13-9 | 2 / 54948 | 0 / 54948 | 0 / 54387 | **0 / 52970** |
| ICCAD13-10 | 0 / 16581 | 0 / 16581 | 0 / 15014 | **0 / 14916** |
| ICCAD13-Avg | 7.9 / 39873 | 2.5 / 39873 | 1.8 / 38612.2 | **1.6 / 39080.7** |

### Table 2 — 3 nm stress test (ISPD'25 300 it vs ours 150 it)

| Benchmark | ISPD'25 EPE/PV | Ours (PT) EPE/PV | Ours (PT+RL) EPE/PV |
|---|---|---|---|
| StdContact-Avg | 40.4 / 32969.6 | 9.0 / 39160.4 | **6.5 / 39907.5** |
| StdMetal-Avg | 11.4 / 19083.8 | 7.2 / 21507.6 | **6.7 / 21029.2** |
| ICCAD13-1 | 51 / 44446 | 47 / 48098 | 47 / 47459 |
| ICCAD13-2 | 34 / 36940 | 39 / 40330 | 29 / 36948 |
| ICCAD13-3 | 107 / 70545 | 97 / 68814 | 86 / 74370 |
| ICCAD13-4 | 8 / 21577 | 7 / 24624 | 6 / 22647 |
| ICCAD13-5 | 29 / 47861 | 15 / 51175 | 16 / 50921 |
| ICCAD13-6 | 28 / 42287 | 21 / 44333 | 21 / 43866 |
| ICCAD13-7 | 7 / 34409 | 3 / 35633 | 1 / 37091 |
| ICCAD13-8 | 13 / 18644 | 10 / 19283 | 8 / 19496 |
| ICCAD13-9 | 49 / 54393 | 34 / 56981 | 38 / 55105 |
| ICCAD13-10 | 2 / 15013 | 0 / 15970 | 0 / 16256 |
| ICCAD13-Avg | 32.8 / 38611.5 | 27.3 / 40524.1 | **25.2 / 40415.9** |

Figure 4 shows one design (`HA_X1__1_0`): golden CurvyILT mask EPE=36 vs five
posterior samples at EPE 7, 5, 2, 6, 11 — i.e. sample quality varies widely and a
good *initial* mask does not guarantee the best *refined* mask (Sec. 4.4).

## 8. Where the paper is silent or inconsistent

These are carried into `docs/gap_ledger.md`; they are **not** filled in silently.

1. **Internal contradiction on the RL baseline.** Sec. 3.3.2 presents the
   teacher-relative baseline `A_k = R_k - R_k^T` as the contribution and argues
   at length against the group mean. Sec. 4.1 then states the configuration used
   "a self-critical baseline given by **the group mean**." These are different
   algorithms. (**G-012**, now a documented paper defect rather than a gap in our
   knowledge.)
2. **The ILT solver is never specified.** It is CurvyILT [4] (Yang & Ren, ISPD'25)
   by reference only: no step size, optimizer, mask parameterization, loss
   weights, or convergence rule appears in this paper. (**G-015**)
3. `l_rec` is "e.g. l1 or l2" — unspecified. `lambda_1`, `lambda_2` are never given. (**G-020**)
4. Generator channel widths, number of Style ResBlocks `n`, and MLP depth are not given. (**G-010**)
5. The **MRC rule values** are by reference to [17] only.
6. No ablation on the RL-loop resolution or iteration count.
7. The "3x speedup" and "2x throughput" claims are wall-clock on 8xA100 and are
   not decomposable from the paper.
