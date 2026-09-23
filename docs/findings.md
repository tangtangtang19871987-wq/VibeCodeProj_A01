# Verification Findings

Issues discovered while validating the physics stack. Each was found by a test,
diagnosed to root cause, and resolved — none was hidden by loosening a tolerance.

---

## F-ADJ-01 — The public reference implementation's analytic adjoint is ~8.7% wrong

**Severity: high.** Affects any ILT result produced with the reference gradient.

### Symptom
A float64 central finite-difference check of the gradient of
`L = Σ w·I(M)` failed with relative errors of 1e-2 to 3e-1 across random
directions. Repeating in float64 with `eps = 1e-6` gave the same magnitudes, so
this was not finite-difference noise.

### Diagnosis
Three gradients were compared against a ground truth obtained by differentiating
a *pure-autograd* reimplementation of the identical forward op (PyTorch
differentiates FFT/complex ops exactly):

| Quantity | Relative error vs exact autograd |
|---|---|
| Forward intensity (ours vs autograd) | **1.7e-16** — forward was never in doubt |
| Ground-truth gradient vs finite differences | **7.7e-8** — the FD check is sound |
| **Our first analytic adjoint** | **8.48e-2** |
| **OpenILT `pylitho/exact.py` adjoint** | **8.73e-2** |

Both wrong, and wrong in nearly the same way (they differ from each other by only
2.2e-2), because ours was modelled on the reference's structure.

### Root cause
The forward operator is `A = IFFT ∘ diag(H_map) ∘ FFT`. With `A = W^H D (1/N²) W`,
its adjoint is

    A^H = (1/N²) W^H conj(D) W

i.e. **convolution with `conj(H)` under the identical corner placement**. The
reference instead uses the shipped `ct_*.pt` kernel tensors. Those are *not*
`conj(H)`: measured `max|h_ct − conj(h)| = 0.2567` against `max|h| = 0.1299`.
They are not `conj(flip(h))` or `flip(h)` either. They do agree with `conj(h)` at
the kernel centre (`h_ct[0,17,17] == conj(h[0,17,17])` exactly), which is
presumably why the error is a few percent rather than order-unity.

### Resolution
`src/gril/litho/socs.py` computes the adjoint with `conj(kernels)` and ignores
the shipped CT tensors. After the fix:

| Check | Result |
|---|---|
| Analytic adjoint vs exact autograd, dose ∈ {0.98, 1.00, 1.02}, focus + defocus | **3.7e-16 – 4.1e-16** |
| Analytic adjoint vs float64 finite differences | ≤ 6.6e-6 (FD cancellation-limited) |

Pinned by `tests_gril/physics/test_adjoint.py::test_analytic_adjoint_is_exact`.

### Consequence for this reproduction
Our ILT solver descends the true gradient. Any numerical comparison against
published results that used the reference gradient is therefore **not**
apples-to-apples, and that must be stated wherever such a comparison is made.
The forward model — and hence every *scored* metric — is unaffected, which is why
our L2/PVB/EPE still match the reference bit-for-bit.

---

## F-KER-01 — The published ICCAD13 kernels are column-symmetric but not row-symmetric

**Severity: medium (a correctness constraint on tests, not a bug).**

### Symptom
Mirror-equivariance tests passed for the x axis (4.9e-7) but failed for the y
axis (2.8e-2).

### Diagnosis
Not a bug in the imaging code, and **not** a SOCS truncation artifact: the error
is already 2.6e-2 with a single kernel (`K=1`) and stays flat across `K = 1..24`.
Inspecting the dominant kernel directly:

| Property | Measured |
|---|---|
| Symmetry about centre **column** 17 | **5.9e-8** (exact) |
| Symmetry about centre **row** 17 | **6.9e-3** (≈6.6% of peak) |
| Symmetry about row 16.5 / 17.5 | 1.8e-2 / 1.6e-2 (worse) |

Row-wise `|h|` sums are visibly unequal (`0.00092, 0.03041, 0.12957` vs
`0.13647, 0.03573, 0.00243`), while column-wise sums mirror exactly. The
asymmetry is a property of the distributed kernel artifact, about no axis.

### Resolution
y-mirror equivariance is **not** a valid invariant for this optical model, so
asserting it would be asserting something false. The suite now:
- asserts x-mirror equivariance strictly (`< 1e-5`);
- pins the kernel's column-symmetry / row-asymmetry directly;
- bounds the y-mirror error to `(1e-3, 5e-2)` so a change is caught loudly.

### Open question
Whether this reflects a genuinely asymmetric source, or an artifact of the
eigendecomposition grid used when the kernels were generated, cannot be settled
from the distributed artifact alone. Recorded rather than guessed.

---

## F-CASE4-01 — ICCAD13 test4 prints nothing without OPC (not a bug)

Its `L2` exactly equals its target area (84037), which looked like a metric bug.
Investigation: test4 is three bars of 64–65 nm CD; peak aerial intensity is
**0.2168**, just under the 0.225 resist threshold, so no pixel prints. Physically
correct and precisely why the case is in the benchmark. Pinned by
`tests_gril/regression/test_iccad13_baseline.py::test_case4_does_not_print_without_opc`.

---

## F-RAS-01 — Polygon rasterisation is boundary-inclusive

A `w × h` RECT rasterises to `(w+1) × (h+1)` pixels. Verified bit-identical to the
reference rasteriser on synthetic rectangles and on all ten contest layouts.
Preserved deliberately: every published ICCAD13 number depends on it. Documented
in `REPRODUCTION_SPEC.md` Sec. 2.

---

## F-ENV-01 — Pre-existing test suite was already uncollectable

The repository's top-level `__init__.py` does `from array_generator.core import ...`,
which only resolves when the checkout is installed under the name
`array_generator`. It is not, so **pytest could not collect any test in the
repository** — verified by running the pre-existing tests before adding anything.
Fixed non-invasively via `pytest.ini`'s `pythonpath` and a `.pkgroot/array_generator`
symlink. No pre-existing file was modified.


---

## F-UNIT-01 — EPE constants are in nanometres but measured in pixels

**Severity: high for any run below 1 nm/pixel. Found by inspecting a negative result.**

### Symptom
The scaled training run reported EPE at a "3 nm tolerance" on a 256x256 canvas.

### Diagnosis
`EPE_TOLERANCE_NM`, `EPE_CHECK_INTERVAL`, `MIN_EPE_CHECK_LENGTH` and
`EPE_CHECK_START_INTERVAL` are all defined in **nanometres**, but the EPE routine
applies them directly as **pixel** offsets. On the ICCAD13 canvas (2048 px at
1 nm/px) nm and px coincide exactly, so the missing conversion is invisible —
and all ICCAD13 numbers in this repository are unaffected and remain bit-exact
against the reference checker.

At the reduced training resolution (256 px at 8 nm/px over the same 2048 nm
field) they do not coincide. A "3 nm" tolerance was really **24 nm**, and the
site-sampling interval was 320 nm instead of 40 nm. The RL reward was therefore
measuring something far more permissive than intended.

### Resolution
`epe_violations()` and `evaluate()` now take `pixel_nm` and convert every
nm-valued constant into pixels, with a guard against a non-positive pitch. Three
regression tests pin it, including one that checks the same *physical* geometry
scores consistently at 1 nm/px and 4 nm/px.

### Lesson
This is exactly the failure mode that a unit-carrying API prevents: the constants
were documented in nm, and the code silently treated them as px. The ICCAD13
canvas made the two identical, which is the worst case for noticing.

---

## F-RL-01 — At this scale, GRPO finetuning is a null result, not an improvement

**Reported as measured, in two stages, because the first measurement had a bug.**

### Stage 1 result (pre F-UNIT-01 fix — since retracted)

An earlier run, before the EPE-units bug (F-UNIT-01) was found and fixed, showed
best-of-K EPE getting WORSE (29.25 -> 30.38) under an EPE tolerance that was
silently 8x too loose (effectively 24 nm instead of the intended 3 nm). That
number is **retracted as the headline result** because the tolerance it was
measured under was wrong, but the run is kept in `git log` for the record and
the mechanism it revealed (see below) still held after the fix.

### Stage 2 result (post F-UNIT-01 fix — current)

Same generator checkpoint, same 60-step GRPO run, re-executed with the corrected
per-pixel tolerance conversion (3 nm at 8 nm/px canvas -> 1 px, the tightest
tolerance representable at this resolution):

| Model | best-of-K EPE | sample-mean EPE | binarised diversity |
|---|---|---|---|
| PT (WGAN-GP only) | 134.75 | 135.59 | 0.00161 |
| PT + GRPO | 134.50 | 135.64 | 0.00133 |
| **Change** | **-0.19%** | **+0.04%** | **-17.4%** |

### Honest reading

This is a **null result, not a regression and not an improvement**. Best-of-K
EPE moved by 0.19%, well within run-to-run noise at `K=8`; sample-mean EPE moved
in the opposite direction by a comparable amount. Neither is a signal.

What *is* a signal, and reproduces the mechanism identified before the fix:
**diversity fell 17.4%** (0.00161 -> 0.00133). Reward trace: `reward_std == 0`
on 12 of 60 steps (20%) -- one designs-worth of group in five had every one of
its K=8 candidates score identically after refinement, at which point the
teacher-relative advantage is exactly zero and that step's policy-gradient term
vanishes (`pg -0.0000` is visible in the log at step 25). Reward itself is
noisy and non-monotonic across the run (best `-24.0` at step 3, worst `-195.4`,
ending at `-109.9`), consistent with a training signal too weak to consistently
overcome per-step noise at this scale.

### Mechanism (unchanged by the fix)

`L_pg = -A_k * log pi(M_k)` maximises the log-probability of the
**already-binarised** action whenever `A_k > 0`. That sharpens the per-pixel
Bernoulli distribution, which mechanically reduces sample diversity. Whether
this is offset by a genuine quality gain depends on how much real training
signal survives the group-reward collapse described above. At the paper's scale
(20 epochs, ~120k designs, K=16) there is evidently enough. At 60 steps on 64
synthetic layouts there is not: the diversity cost is visible and consistent
across both the pre-fix and post-fix runs; the quality benefit is not.

### Honest attribution

**Not evidence against the paper.** The run is ~4 orders of magnitude smaller
than the paper's along every axis that matters for RL sample efficiency (steps,
designs, group size), uses synthetic layouts rather than LithoBench, and a
reward computed on a CPU-only low-resolution proxy. What this experiment
establishes is narrower and still useful: (1) the GRPO implementation runs
end-to-end, is numerically stable, and its teacher stays frozen and
gradient-free, all verified independently by unit tests; (2) at this compute
budget it does not show the paper's reported improvement, in either direction;
(3) the diversity-reduction mechanism of the policy loss is real and
measurable, and would need to be outweighed by real reward signal at greater
scale for the paper's result to emerge -- which this experiment cannot supply.

The positive finding from the PT-only model stands independently of the RL
result: **best-of-K EPE beats the sample mean at every measurement in this
project** (both the pre-fix 29.25 vs 29.61 and the post-fix 134.75 vs 135.59),
which is the paper's core premise for sampling multiple candidates rather than
regressing a single mask, and it reproduces at this reduced scale.

No hyperparameter was adjusted between the two runs. The only change was the
EPE-unit bug fix in F-UNIT-01, applied uniformly to both PT and PT+RL
evaluation.


---

## F-SAT-01 — Default init_scale x mask_steepness saturates the mask sigmoid; only Adam escapes it

**Severity: informational.** Not a bug in any committed result (all ICCAD13
results use Adam, unaffected). Found while adding L-BFGS/SGD/Nesterov support
to the ILT solver and testing them fairly.

### Symptom
With the solver's default `init_scale=2.0` and the ICCAD13 experiment's
`mask_steepness=8.0`, plain SGD and L-BFGS made **zero measurable progress**
over dozens of iterations at step sizes that are perfectly reasonable for a
well-conditioned problem, while Adam converged normally from the identical
starting point.

### Diagnosis
`M = sigmoid(beta_m * P)` with `P_0 = init_scale * (2*target - 1)`. At the
defaults, `beta_m * P_0 = 8 * (+-2.0) = +-16` at every pixel. `sigmoid(+-16)`
is `0.9999999 / 1.1e-7` in float32 -- already almost exactly binary -- and its
derivative `sigmoid(x)(1-sigmoid(x))` is correspondingly tiny. Measured: raw
gradient magnitude at this starting point is **~1.7e-6**, several orders of
magnitude smaller than at a well-conditioned start (`init_scale=0.25` gives
gradients of order 1).

Adam's update rule divides each parameter's step by an estimate of that
parameter's own gradient RMS, so a *persistently tiny but nonzero* gradient
still produces an order-1 step -- Adam is effectively blind to the absolute
gradient scale. SGD, Nesterov, and L-BFGS all use the gradient's actual
magnitude directly (L-BFGS additionally builds a curvature estimate from it),
so at this starting point their steps are proportionally tiny too, and a
reasonable step size (or the L-BFGS default) does not move them measurably in
a small iteration budget.

### Resolution
Not "fixed" -- this is a real property of the sigmoid parameterisation at these
constants, not a defect. Two things were done:

1. `docs/findings.md` (this entry) and the `ILTConfig`/`solver.py` module
   docstrings now state it explicitly, so a future user of `optimizer="sgd"` or
   `"lbfgs"` is not left to rediscover it by watching a run silently fail to
   move.
2. `tests_gril/unit/test_ilt_optimizers.py::test_saturation_makes_raw_gradient_methods_stall`
   pins the phenomenon itself (Adam moves, SGD does not, from the exact
   configuration every ICCAD13 experiment uses) so it is caught if the
   underlying physics or parameterisation ever changes silently. All other
   optimizer tests use a deliberately well-conditioned `init_scale=0.5` starting
   point, which is the practical guidance this finding produces: keep
   `init_scale * mask_steepness` moderate (or warm-start from something already
   close to a reasonable mask) when using a raw-gradient optimizer.

### Consequence
None for existing results (all use Adam). It is the reason Adam remains this
project's default, and it is now a documented, tested reason rather than an
unexamined choice.
