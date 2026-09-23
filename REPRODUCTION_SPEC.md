# Reproduction Specification

Target: arXiv 2602.19027, *Pushing the Limits of Inverse Lithography with
Generative Reinforcement Learning* (Yang & Ren, NVIDIA, DAC'26).

> **Scope declaration.** The primary paper was not retrievable in this environment
> (G-001). This document therefore specifies a **faithful reimplementation**, not
> an exact reproduction. Sections sourced to a public artifact are marked **[PUB]**
> and are exact; sections marked **[A]/[B]/[C]** are reconstructions at decreasing
> confidence (see `docs/paper_summary.md`). Formulas below are **our** statements
> of the model; none is transcribed from the paper.

---

## 1. Physical model and approximations

Partially-coherent imaging of a thin (Kirchhoff) mask under scalar diffraction,
evaluated by **Sum Of Coherent Systems (SOCS)** — the standard truncation of the
Hopkins bilinear transmission cross-coefficient (TCC) operator. **[PUB]**

Hopkins: the aerial intensity is the bilinear form

    I(x) = ∫∫ TCC(f1,f2) M̂(f1) M̂*(f2) exp(i2π(f1-f2)·x) df1 df2

TCC is Hermitian positive semi-definite, so it admits an eigendecomposition
`TCC = Σ_k w_k φ_k φ_k^†` with `w_k ≥ 0` in descending order. SOCS truncates to
the first `K_soc` terms:

    I(x) = Σ_{k=1}^{K_soc} w_k · | (h_k ⊛ M)(x) |²                        (Eq. S-1)

with `K_soc = 24`. **[PUB]** `h_k` is the k-th coherent kernel; `w_k` the k-th
eigenvalue. Approximations accepted: scalar (non-vector) diffraction, thin-mask,
no mask 3-D effects, no resist diffusion/acid blur beyond the threshold model, no
aberrations beyond those baked into the supplied kernels.

**Truncation error is measured, not assumed:** with `K_soc = 24` the open-field
intensity is **0.95154** instead of 1.0, i.e. the truncation discards 4.85% of the
energy. This is a property of the supplied kernel set and is reported, never
silently renormalized away.

## 2. Coordinate system, units, sampling

| Quantity | Convention |
|---|---|
| Canvas | 2048 x 2048 pixels **[PUB]** |
| Pixel pitch | 1 nm/pixel; canvas = 2048 nm x 2048 nm **[PUB]** |
| Index order | `array[row, col]` = `[y, x]`, origin at top-left, consistent with the GLP raster and the reference checker |
| Design placement | Layout bounding box centered on the canvas **[PUB]** |
| Kernel support | 35 x 35 samples in the **frequency** domain, occupying the lowest spatial frequencies **[PUB]** |
| Frequency sampling | `Δf = 1/2048 nm⁻¹`; kernel spans `±17·Δf ≈ ±8.3e-3 nm⁻¹`, consistent with a 193 nm / NA 1.35 system (`2NA/λ ≈ 1.4e-2 nm⁻¹`) **[C]** |
| Mask values | `M ∈ [0,1]`, real. Binary mask = `{0,1}` **[PUB]** |
| Polygon fill | **Boundary-inclusive**: a `w x h` RECT rasterises to `(w+1) x (h+1)` pixels. Verified bit-identical to the reference rasteriser on synthetic rectangles and on all 10 contest layouts. Preserved deliberately — every published ICCAD13 number depends on it. **[PUB]** |
| Dose | dimensionless multiplier on the mask field **[PUB]** |

## 3. FFT convention, shift, normalization, padding, boundaries

This section is load-bearing and is **exactly** matched to the reference
implementation **[PUB]**; any deviation changes every downstream number.

1. Forward transform uses `torch.fft.fft2(·, norm="forward")`, i.e. the forward
   direction carries the `1/N²` factor and the inverse carries none.
2. **No `fftshift`.** The mask spectrum stays in standard FFT layout with DC at
   index `[0,0]`. The 35x35 kernel is applied to the **four corner blocks** of the
   spectrum, which are the low frequencies:

   ```
   out[:, :18,  :18 ] = F[:, :18,  :18 ] * h[:, -18:, -18:]
   out[:, :18,  -17:] = F[:, :18,  -17:] * h[:, -18:, :17 ]
   out[:, -17:, :18 ] = F[:, -17:, :18 ] * h[:, :17,  -18:]
   out[:, -17:, -17:] = F[:, -17:, -17:] * h[:, :17,  :17 ]
   ```
   with `knxh = 35//2 = 17`. All other frequencies are zeroed (the kernel is
   band-limited). An equivalent centered-and-shifted path exists and **must agree
   to 1e-6**; this equivalence is a required test (F-03).
3. **Padding: none.** The FFT is circular, so the canvas boundary is **periodic**.
   This is the reference behaviour and is what the contest scoring assumes. Edge
   artifacts are bounded because designs are centered with >=300 nm of margin —
   verified: the largest ICCAD13 bbox is 828 x 640 nm inside 2048 nm.
4. Inverse: `torch.fft.ifft2(·, norm="forward")`.
5. Intensity: `I = Σ_k w_k |E_k|²` with `w_k` real, `E_k` complex.

**Kernel normalization:** the supplied `w_k` are absolute eigenvalues; **no
per-kernel renormalization is applied**. The only sanity anchor is the measured
open-field value (0.95154).

## 4. TCC / SOCS kernel handling

- Kernels are loaded verbatim from the public artifact and transposed from the
  stored `(35,35,24)` layout to `(24,35,35)`. **[PUB]**
- Four sets exist: `focus`, `defocus`, and their conjugate transposes
  (`ct_focus`, `ct_defocus`). The CT sets exist to make the **adjoint** exact.
- Truncation: `K_soc` is configurable but defaults to 24. Required test: the
  reconstructed TCC slice is Hermitian, its eigenvalues are non-negative and
  descending, and intensity error vs `K_soc = 24` decreases monotonically in `K_soc`.

## 5. Adjoint / gradient

With `I = Σ_k w_k |h_k ⊛ (dM)|²`, the Wirtinger derivative w.r.t. a real mask is

    ∂L/∂M = 2d · Re{ Σ_k w_k · h_k^† ⊛ [ (h_k ⊛ dM) · (∂L/∂I) ] }        (Eq. S-2)

implemented with the CT kernel sets as a custom `torch.autograd.Function`. **[PUB]**

**This is verified, not trusted:** a central finite-difference check in float64
against random probe directions must agree to a relative error < 1e-4 (F-06).

## 6. Resist / process model

    Z(x) = sigmoid( β_r · ( I(x) - I_th ) ),   I_th = 0.225, β_r = 50    (Eq. S-3)
    binary print: 1[ Z >= 0.5 ]  ⟺  1[ I >= I_th ]                       **[PUB]**

Process corners **[PUB]**:

| Corner | Dose | Defocus |
|---|---|---|
| Nominal | 1.00 | no |
| Max | 1.02 | no |
| Min | 0.98 | yes |

## 7. Data representations

| Object | Type | Shape | Range |
|---|---|---|---|
| `target` | float32 | `(H,W)` or `(B,H,W)` | `{0,1}` |
| `params P` | float32 | `(B,H,W)` | ℝ |
| `mask M` | float32 | `(B,H,W)` | `[0,1]` |
| `aerial I` | float32 | `(B,3,H,W)` (corners) | `≥0` |
| `resist Z` | float32 | `(B,3,H,W)` | `(0,1)` |
| contour | derived | — | boundary of `1[Z≥0.5]` |

## 8. ILT loss, gradient, regularization, constraints **[C], G-015**

Mask parameterization (keeps the variable unconstrained and the map differentiable):

    M = sigmoid( β_m · P ),   β_m default 4                              (Eq. S-4)

Objective:

    L_ILT = Σ_c λ_c · || Z_c(M) - target ||²₂  +  λ_pvb·L_pvb  +  λ_tv·L_tv   (Eq. S-5)

with corner weights `λ_nom = 1`, `λ_max = λ_min = 0` by default (pure nominal-L2,
the MOSAIC-lineage default) and an optional process-window term
`L_pvb = ||Z_max - Z_min||²₂`. `L_tv` is an optional total-variation smoothness
regularizer, default off. **Every weight is config-exposed and logged.**

**Optimizers.** Four are implemented, all config-selected via `ILTConfig.optimizer`,
none dictated by the paper (G-015):

| Optimizer | Notes |
|---|---|
| `adam` (default) | Unchanged from the module's first version; every committed ICCAD13 result uses this and is reproduced bit-for-bit by the current code. |
| `sgd` | Plain or classical-momentum gradient descent (`momentum` field, default 0 = plain). |
| `nesterov` | Nesterov-accelerated SGD; requires `momentum > 0`, validated at construction. |
| `lbfgs` | Quasi-Newton L-BFGS with a strong-Wolfe line search (`torch.optim.LBFGS`). |

Stopping: fixed iteration count by default (so "iteration budget" comparisons
are exact), with optional early stop on relative loss improvement (`early_stop_rtol`)
or on gradient-norm convergence (`convergence_grad_tol`).

**Regularization.** An optional total-variation term
`L_tv = Σ |∂M/∂x| + |∂M/∂y|` (Eq. S-5's `λ_tv·L_tv`) penalizes mask-boundary
roughness in the continuous relaxation, a standard ILT-literature proxy for
lower mask complexity / e-beam shot count. Off by default (`weight_tv = 0`).
Optional gradient-norm clipping (`grad_clip`, default 0 = off) is available for
stabilization with any optimizer.

**Comparing optimizers fairly — a real subtlety, not a footnote.** "Iterations"
is not the same unit of compute across optimizers: Adam/SGD/Nesterov do exactly
one forward+backward pass per iteration, while each L-BFGS iteration runs a
line search that may evaluate the closure several times. `ILTResult.n_func_evals`
records the true number of forward/backward evaluations, and this project never
compares optimizers by raw iteration count without also reporting it.

**Batched L-BFGS runs per-example, not jointly.** `torch.optim.LBFGS` builds one
shared curvature (Hessian) approximation over the entire flattened tensor it is
given. Handed a `(B,H,W)` batch of otherwise-independent ILT problems directly,
it would silently couple their curvature estimates — the joint optimum still
coincides with the per-example optima (the objective is separable), but it
would break the `batched == looped single-case` equivalence this project's test
suite otherwise guarantees for every optimizer. `solve()` therefore runs one
independent `torch.optim.LBFGS` instance per batch element for that path, at
the cost of a Python-level loop rather than a fully vectorized kernel.

**A genuine numerical finding (F-SAT-01, `docs/findings.md`):** the defaults
`init_scale=2.0` and `mask_steepness=8.0` push `sigmoid(β_m·P_0)` to `sigmoid(±16)`
at every pixel — already almost exactly binary, with a correspondingly tiny
derivative (measured raw gradient magnitude ≈ 1.7e-6 at this starting point).
Adam's per-parameter step normalization is insensitive to this and converges
normally; SGD, Nesterov, and L-BFGS all use the gradient's actual magnitude and
make **no measurable progress** from this exact starting point at step sizes
that are reasonable for a well-conditioned problem. This is why every existing
ICCAD13 result uses Adam, and it is now a tested, documented reason rather than
an unexamined default: pinned by
`tests_gril/unit/test_ilt_optimizers.py::test_saturation_makes_raw_gradient_methods_stall`.
Anyone using `sgd`/`nesterov`/`lbfgs` should keep `init_scale * mask_steepness`
moderate or warm-start from something already close to a reasonable mask.

**Declared honestly:** the paper's actual solver is unknown (G-015). All of the
above — the optimizer choices, the regularization, and the specific constants —
are ours.

## 9. Generative model **[B], G-010/011**

    w = f_φ(z),        z ~ N(0, I_256)                                    (Eq. S-6)
    Y = G(Z_design, w)                                                    (Eq. S-7)
    M = 1[ sigmoid(Y) > 0.5 ]                                             (Eq. S-8)

`G` is a U-Net over the design; AdaIN modulation by `w` is applied **only** in the
coarse-level Style ResBlocks:

    AdaIN(f, w) = γ(w) · (f - μ(f))/σ(f) + β(w)                           (Eq. S-9)

with `μ, σ` per-channel instance statistics.

Stage-1 objective (WGAN-GP + reconstruction) **[A]/[B]**:

    L_D = E[D(M̂)] - E[D(M_gt)] + λ_gp · E[(||∇D(M̃)||₂ - 1)²]             (Eq. S-10)
    L_G = -E[D(M̂)] + λ_rec · || sigmoid(Y) - M_gt ||²₂                    (Eq. S-11)

`λ_gp = 10`, `n_critic = 5`, `λ_rec` config-exposed.

## 10. GRPO fine-tuning **[B], G-012/013/014**

For design `Z`, sample `K` latents `z_1..z_K`; produce `Y_k`, binarize to `M_k`;
run the low-resolution ILT loop to get `M_k^ILT`; reward:

    R_k = - EPE( M_k^ILT, Z )                                             (Eq. S-12)

Advantage — **two variants, config-selected, because the sources disagree** (G-012):

    (a) teacher-relative  A_k = R_k - R_k^T        (default, per [B])      (Eq. S-13a)
    (b) group-normalized  A_k = (R_k - mean R)/(std R + ε)   (canonical GRPO) (Eq. S-13b)

Policy log-probability under a pixel-wise Bernoulli policy (G-013):

    log π(M_k | Z, z_k) = Σ_x [ M_k log σ(Y_k) + (1-M_k) log(1-σ(Y_k)) ]  (Eq. S-14)

reduced by **mean** over pixels by default (`reduction` is config-exposed, and the
choice is coupled to `λ_pg` — see G-014).

    L_pg   = - (1/K) Σ_k A_k · log π(M_k | Z, z_k)                        (Eq. S-15)
    L_imit = (1/K) Σ_k || sigmoid(Y_k) - S(M_k^ILT) ||²₂                   (Eq. S-16)
    L_FT   = λ_pg · L_pg + λ_imit · L_imit,   λ_pg = 500, λ_imit = 1      (Eq. S-17)

`S(·)` is a smoothing operator on the refined mask. `G_T` (the frozen pretrained
generator) is used only to produce `R_k^T`; it receives no gradient.

An optional PPO-style clipped ratio and a KL-to-reference term are implemented but
**off by default**, since [B] describes neither.

## 11. Inference **[A]**

Sample `K` latents → generate `K` masks → **batched** ILT refinement → evaluate
metrics → select `M* = argmin_k EPE(M_k^ILT)`. Tie-break by L2, then by index, so
selection is deterministic.

## 12. Evaluation protocol **[PUB]**

- `L2 = Σ (1[Z_nom ≥ 0.5] - target)²` over the 2048² canvas, units nm².
- `PVB = #{ 1[Z_max ≥ 0.5] ≠ 1[Z_min ≥ 0.5] }`, units nm².
- **EPE**: extract the target boundary; split into vertical and horizontal
  segments; sample sites at interval **40**, starting **40** from each segment end,
  or the segment midpoint if the segment is shorter than **80**; at each site
  measure inward/outward displacement against tolerance `t`; count violations.
  Reference tolerance `t = 15` nm **[PUB]**; the paper reports `t = 3` nm **[A]**.
  **Both are reported, plus the full `t = 1..15` curve** (G-016).
- Runtime: wall-clock, recorded per case with the hardware string.

## 13. Tolerances for declaring agreement

| Comparison | Tolerance | Rationale |
|---|---|---|
| Our SOCS vs reference implementation, same input | **rel. 1e-6** | same math, float32 round-off only |
| Our EPE/L2/PVB vs reference checker, all 10 cases | **exact (bit-identical integers)** | integer counting; any mismatch is a bug |
| Analytic adjoint vs float64 finite difference | **rel. 1e-4** | FD truncation |
| batched vs single-case | **rel. 1e-6** | |
| float32 vs float64 forward | **rel. 1e-5** | reported, not enforced |
| Same seed, same host, re-run | **bitwise identical** | |
| **Our numbers vs the paper's** | **undefined — no target available** | G-002 |

That last row is the honest state of this reproduction and must not be papered over.

## 14. Known-unreproducible parts

1. **Every quantitative claim of the paper** (G-002) — target numbers unknown.
2. **LithoBench experiments** — dataset unreachable (S3).
3. **Runtime / "2-3x speedup"** — requires a GPU; this host has none.
4. **Exact architecture and hyperparameters** (G-010/011/019) — unknown.
5. **The exact GRPO variant** (G-012) — sources conflict; both implemented.
6. **The paper's ILT solver** (G-015) — described as unspecified even in the paper [B].

What *is* reproducible here: the full physics stack exactly **[PUB]**, the scoring
protocol exactly **[PUB]**, and the method's **claim shape** — that multi-candidate
sampling plus RL-tuned warm starts beats a deterministic warm start and plain ILT
at an equal iteration budget. That is an in-house, self-consistent test (X-06/X-10)
and is the primary scientific deliverable available under these constraints.
