# Gap Ledger

Every piece of information that is missing, uncertain, or assumed. Nothing here is
silently resolved: each row states the assumption actually used in code, the
alternatives considered, and how sensitive the result is.

**Confidence scale:** High = verified against a public artifact executed in this
session. Medium = consistent across the literature/protocol, low risk. Low =
plausible guess. **None = we do not know and the code must expose it as a knob.**

---

## TIER-0 GAPS — these dominate everything else

### G-001 — The primary paper was never read
| | |
|---|---|
| **Missing** | The full text of arXiv 2602.19027: all equations, the algorithm boxes, Tables 1-2, all figures, every hyperparameter, the exact ILT solver. |
| **Why it matters** | This is the reproduction target. Without it there is no authoritative statement of *any* formula. Tier-L2 (numerical agreement with the paper) is **impossible**, because the target numbers are unknown. |
| **Sources searched** | arxiv.org (html+abs+pdf), export.arxiv.org, ar5iv, alphaxiv, Semantic Scholar API, HuggingFace papers, OpenReview, paperswithcode, dl.acm.org, spie.org, research.nvidia.com — **all HTTP 403 / EGRESS_BLOCKED**. Plus 4 WebSearch queries and a GitHub code/repo search for official code. Full log: `docs/source_inventory.md`. |
| **Current assumption** | Reconstruct the method from the abstract-level facts [A] plus standard, well-documented formulations of each named component (WGAN-GP, GRPO, AdaIN, MOSAIC-style gradient ILT). Label the result a **faithful reimplementation**, never an exact reproduction. |
| **Alternatives** | (a) User supplies the PDF — **collapses this gap and most of the ones below**. (b) Egress policy amended to allow arxiv.org. (c) Proceed as reimplementation. |
| **Sensitivity experiment** | Not applicable — this gap is binary. Once the PDF arrives, every [B]-tagged row below becomes checkable in a single pass. |
| **Confidence** | **None.** |

### G-002 — Paper's Table 1 / Table 2 numbers unknown
| | |
|---|---|
| **Missing** | Per-case L2, PV Band, EPE, runtime for the proposed method and all baselines. |
| **Why it matters** | `REPRODUCTION_REPORT.md` cannot have a "paper result" column. Every comparison degenerates to "our measured value" with no reference. |
| **Sources searched** | 3 targeted WebSearch queries naming Table 1, ICCAD13, and the baseline names; all returned prose only. |
| **Current assumption** | Report our own measured numbers, and compare against **baselines we implement and run ourselves** (no-OPC, plain ILT, deterministic-warm-start ILT) so that *relative* claims — the paper's ">20% EPE improvement" and "half the iteration budget" — are still testable in-house. |
| **Alternatives** | Wait for the PDF. |
| **Sensitivity experiment** | E-REL-01: measure relative EPE improvement of (PT+RL warm start + ILT) vs (plain ILT) under an identical iteration budget. This tests the paper's *claim shape* without needing its absolute numbers. |
| **Confidence** | **None.** |

---

## TIER-1 GAPS — method details

### G-010 — Generator architecture specifics
| | |
|---|---|
| **Missing** | Depth, channel widths, number of resolution levels, normalization, activation, up/downsampling operators, output head. Digest says "multi-scale path" with a Style ResBlock at "Level 2" [B] but gives no channel counts. |
| **Why it matters** | Determines capacity and whether the AdaIN-only-at-coarse-level claim behaves as described. |
| **Sources searched** | S1.2 only. No official code found. |
| **Current assumption** | A U-Net with 4 resolution levels, base width 64, doubling per level, GroupNorm elsewhere and AdaIN in the coarse Style ResBlocks, SiLU activation, bilinear upsample + 3x3 conv. **All exposed in config**, none hard-coded. |
| **Alternatives** | GAN-OPC's original generator topology; a plain conditional U-Net with concatenated noise. |
| **Sensitivity experiment** | E-ABL-ARCH: sweep base width and the level at which AdaIN is injected; measure sample diversity and post-ILT EPE. |
| **Confidence** | **Low.** |

### G-011 — Latent / style dimension
| **Missing** | `dim(z)`, `dim(w)`, mapping-MLP depth. | 
|---|---|
| **Current assumption** | `dim(z)=256` [B], `dim(w)=256`, 4-layer mapping MLP [C]. Config-exposed. |
| **Sensitivity experiment** | E-ABL-Z: sweep `dim(z)` in {32, 128, 256}; measure pairwise mask diversity. |
| **Confidence** | **Low** (the 256 is [B]). |

### G-012 — GRPO objective form
| | |
|---|---|
| **Missing** | The actual policy-gradient expression. GRPO normally uses a group-normalized advantage `A_k = (R_k - mean(R))/std(R)` with a PPO-style clipped ratio and a KL term to a reference policy. The digest instead describes a **teacher-relative** advantage `A_k = R_k - R_k^T` with a **BCE surrogate** for the log-probability and *no* clipping or KL term mentioned [B]. These are materially different algorithms. |
| **Why it matters** | This is the paper's central contribution. Getting it wrong means not reproducing the paper. |
| **Sources searched** | S1.2; S1.3 contradicts it (uses group_size=4 + ratio clipping). |
| **Current assumption** | Implement **both** behind one config switch: `advantage: {teacher_relative, group_normalized}` and `ratio_clip: {none, ppo}`. Default to the digest's description (`teacher_relative`, no clip) since it is the only source that claims to describe *this* paper, while making the standard-GRPO variant a one-line change. |
| **Alternatives** | Canonical GRPO (group-normalized + clip + KL); plain REINFORCE with baseline. |
| **Sensitivity experiment** | E-ABL-GRPO: train all three variants under an identical budget; compare final selected EPE and reward variance. This is a *designed* experiment, not a workaround. |
| **Confidence** | **Low.** |

### G-013 — Pixel-wise log-probability of a mask
| | |
|---|---|
| **Missing** | How a policy log-prob is defined over a megapixel binary mask. |
| **Why it matters** | Without it the policy gradient is undefined. |
| **Current assumption** | Treat each pixel as an independent Bernoulli with parameter `sigma(Y)`; `log pi(M|Z,z) = sum_pixels [M log sigma(Y) + (1-M) log(1-sigma(Y))]`, i.e. the negative BCE. This is consistent with the digest's "BCE surrogate" phrasing [B] and is the only standard construction that makes the stated reward-weighted objective well-typed [C]. Mean-reduce (not sum) to keep the scale sane, with the reduction config-exposed. |
| **Sensitivity experiment** | E-ABL-LOGP: sum vs mean reduction, and the effect of `lambda_pg` rescaling that this induces. |
| **Confidence** | **Low-Medium.** |

### G-014 — lambda_pg = 500, lambda_imit = 1
| **Why it matters** | A 500:1 ratio is extreme; it is only sane if the log-prob is mean-reduced. Strongly coupled to G-013. |
|---|---|
| **Current assumption** | Use [B]'s values as defaults *together with* mean-reduction, and verify empirically that neither term dominates. If one does, report that and sweep. |
| **Sensitivity experiment** | E-ABL-LAMBDA: 2-D sweep over `lambda_pg` x `lambda_imit`; record the ratio of gradient norms of the two terms. |
| **Confidence** | **Low.** |

### G-015 — The "fast batched ILT solver"
| | |
|---|---|
| **Missing** | Algorithm, parameterization, step size, momentum, schedule, stopping rule. The digest explicitly lists this as a paper limitation [B]. |
| **Why it matters** | It sets both the reward and the headline runtime claim. |
| **Current assumption** | MOSAIC/GAN-OPC-lineage gradient ILT: sigmoid-relaxed mask variable `M = sigma(beta_m * P)`, litho via 24-kernel SOCS, resist via `sigma(beta_r*(I - I_th))`, loss `||Z_resist - Z_target||_2^2` (+ optional PVB term), Adam. Exposed: `step_size`, `beta_m`, `beta_r`, `iters`, `corner weights`. This lineage is the documented public standard for ICCAD13 [C]. |
| **Sensitivity experiment** | E-ABL-ILT: step size and iteration-count sweep; report the iteration/quality curve, which is exactly what the "half the iteration budget" claim needs. |
| **Confidence** | **Medium** for the family, **Low** for the constants. |

### G-016 — EPE at "3 nm tolerance" vs ICCAD13's 15 nm
| | |
|---|---|
| **Missing** | Whether the paper redefines the EPE check, or only tightens the threshold. |
| **Why it matters** | The public ICCAD13 checker uses `EPE_CONSTRAINT=15` (S2.4). The paper reports EPE violations under a **3 nm** tolerance [A]. Numbers under the two settings are not comparable. |
| **Current assumption** | Keep the reference sampling scheme (interval 40, min segment 80, start offset 40) **exactly** as in S2.4 and expose the tolerance as a parameter, reporting **both** 15 nm and 3 nm. |
| **Alternatives** | The paper may sample EPE sites differently. Unknowable without G-001. |
| **Sensitivity experiment** | E-SENS-EPE: report the full EPE-violation-vs-tolerance curve for 1..15 nm, so any future threshold can be read off. |
| **Confidence** | **Medium** on the mechanism, **None** on matching the paper. |

### G-017 — Ground-truth masks for pretraining
| | |
|---|---|
| **Missing** | Which `M_gt` the reconstruction loss targets. LithoBench ships reference ILT masks; we cannot download it (S3). |
| **Current assumption** | Generate `M_gt` **in-house** by running our own converged ILT solver on each training layout, and document that these are self-produced, not the paper's. This mirrors how GAN-OPC built its training set [C]. |
| **Sensitivity experiment** | E-SENS-GT: vary the ILT budget used to make `M_gt` (50/200/500 iters) and measure the effect on pretraining quality. |
| **Confidence** | **Medium** as a method, **None** as a match to the paper's data. |

### G-018 — Training layouts / dataset split
| **Missing** | LithoBench splits, counts, augmentation, crop size. |
|---|---|
| **Current assumption** | ICCAD13's 10 cases are **evaluation only** (never trained on). Training uses a **synthetic M1-like layout generator** with a fixed seed, documented and checksummed, with a held-out validation split. |
| **Sensitivity experiment** | E-SENS-DATA: train-set size sweep. |
| **Confidence** | **None** as a match to the paper. |

### G-019 — K=16, 8x downsample, ~100 RL ILT steps
| **Current assumption** | Use [B]'s values as config defaults. On this CPU-only host the *executed* runs use smaller `K` and resolution; every result records what was actually used. |
|---|---|
| **Sensitivity experiment** | E-ABL-K: K in {1,2,4,8,16} vs final EPE and vs total compute — directly tests the paper's multi-candidate premise. |
| **Confidence** | **Low.** |

### G-020 — WGAN variant and discriminator
| **Missing** | WGAN vs WGAN-GP, `n_critic`, gradient-penalty weight, discriminator architecture. Abstract says "WGAN" [A]; digest says "WGAN-GP" [B]. |
|---|---|
| **Current assumption** | WGAN-GP (gp weight 10, `n_critic` 5) as the default, with weight-clipping WGAN available by config. |
| **Sensitivity experiment** | E-ABL-GAN: GP vs clipping, on pretraining stability. |
| **Confidence** | **Low-Medium.** |

---

## TIER-2 GAPS — resolved with high confidence from public artifacts

These are recorded for completeness; they are **not** open risks.

| ID | Item | Resolution | Confidence |
|---|---|---|---|
| G-030 | Optical model | 24-term SOCS, 35x35 complex kernels, focus + defocus, from S2.2. Verified: eigenvalues descending, open-field intensity 0.95154. | **High** |
| G-031 | Process corners | Nominal (dose 1.00, focus), Max (dose 1.02, focus), Min (dose 0.98, defocus) — from `config/lithoiccad13.txt` (S2.5). | **High** |
| G-032 | Resist model | Constant-threshold, `I_th = 0.225`, sigmoid steepness 50, print threshold 0.5 (S2.5). | **High** |
| G-033 | FFT convention | `norm="forward"` (1/N^2 on forward), kernel applied to the four FFT corners (low frequencies), no fftshift in the fast path. Verified against the legacy shifted path in S2.3. | **High** |
| G-034 | Canvas / sampling | 2048x2048 at 1 nm/pixel; designs centered. Verified by parsing all 10 GLP files. | **High** |
| G-035 | L2 and PV Band | `L2 = sum((binarize(printed_nom) - target)^2)`; `PVB = count(binary_max != binary_min)` (S2.4). | **High** |
| G-036 | Adjoint / gradient | Analytic adjoint using conjugate-transpose kernels, per S2.3, cross-checked in this repo by finite differences (`tests/physics`). | **High** |

---

## Standing rule

If an item in this ledger is later resolved by the primary PDF, the row must be
updated **and** the corresponding `REPRODUCTION_SPEC.md` section and traceability
matrix entry re-derived. Resolving a gap by tuning until numbers match the paper
is explicitly forbidden by this project's ground rules.
