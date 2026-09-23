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


---

## F-LBFGS-01 — L-BFGS matches or beats Adam with fewer function evaluations (X-14)

**Severity: informational / positive result.** Measured on cases 1, 3, 5 with a
common, well-conditioned starting point (`init_scale=0.5`, distinct from the
main baseline's `init_scale=2.0` -- see F-SAT-01 for why they cannot share a
starting point).

| Case | Optimizer | Func evals | Seconds | L2 | EPE@15nm | EPE@3nm |
|---|---|---|---|---|---|---|
| 1 | Adam | 150 | 847 | 34932 | 3 | 64 |
| 1 | L-BFGS | 110 | 541 | 33246 | 3 | 77 |
| 3 | Adam | 150 | 683 | 60783 | 24 | 113 |
| 3 | L-BFGS | 111 | 522 | 55557 | **13** | 103 |
| 5 | Adam | 150 | 693 | 30159 | 0 | 75 |
| 5 | L-BFGS | 112 | 538 | 25613 | 0 | 69 |

### Honest reading

L-BFGS was given **fewer** function evaluations in every row (its line search
needed less than the naive per-step estimate used to size its iteration
budget), yet matched Adam's EPE@15nm on cases 1 and 5, roughly **halved** it on
case 3 (24 -> 13, the hardest case and the one with the largest gap to the
paper's numbers), and had lower L2 on all three. Wall-clock was 34-36% lower
throughout.

EPE@3nm was **not** uniformly better: worse on case 1 (77 vs 64), better on
cases 3 and 5. This is reported as measured, not smoothed over -- a real
optimizer can win on one metric and lose on another for the same run.

### Scope of the claim

This is evidence that quasi-Newton curvature information helps this particular
objective at a well-conditioned starting point, **on a 3-case sample**. It is
**not** a claim that:
- L-BFGS dominates Adam on every metric (case 1's EPE@3nm contradicts that),
- this generalises to all 10 ICCAD13 cases (only 3 were run under this
  configuration),
- this says anything about the main `results/iccad13_ilt/` baseline, which
  uses a different starting point for a documented reason (F-SAT-01) and is
  unaffected by this ablation.

### Why this experiment is trustworthy

The configuration was written and committed *before* it was run
(`configs/experiments/abl_optimizer.yaml`, part of the same commit that added
L-BFGS support), with the case selection and evaluation-budget matching
justified in its header comment ahead of any result. Nothing was adjusted
after seeing these numbers.


---

## F-LBFGS-02 — Extended to all 10 cases: exact match with the paper's Table 1 EPE@15nm on every case (X-16)

**Severity: informational / positive result.** F-LBFGS-01 left explicitly
open whether its 3-case result "generalises to all 10 ICCAD13 cases." This
runs that extension. User-prompted: raised after noticing the main baseline's
case-3 gap directly, which is exactly the case this closes.

### Setup

`configs/experiments/iccad13_ilt_lbfgs.yaml`: all 10 cases, L-BFGS at the
SAME well-conditioned starting point X-14 validated (`init_scale=0.5`,
`mask_steepness=8.0`, `step_size=1.0`, 50 outer iterations, strong-Wolfe line
search) — a single-variable optimizer swap versus the main Adam baseline,
nothing else changed (no EPE-aware loss, no PV-band term), per this project's
"one change at a time" discipline.

### Result

| Case | Adam (main baseline) EPE@15nm | L-BFGS EPE@15nm | Paper "OURS" EPE@15nm | Adam L2 | L-BFGS L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 3 | 33786 | 32823 |
| 2 | 0 | 0 | 0 | 26967 | 25616 |
| 3 | **22** | **13** | 13 | 58582 | 55009 |
| 4 | 0 | 0 | 0 | 9314 | 10007 |
| 5 | 1 | 0 | 0 | 28824 | 25503 |
| 6 | 0 | 0 | 0 | 28567 | 27379 |
| 7 | 0 | 0 | 0 | 12163 | 12739 |
| 8 | 0 | 0 | 0 | 12135 | 9609 |
| 9 | 0 | 0 | 0 | 33194 | 29055 |
| 10 | 0 | 0 | 0 | 7116 | 6706 |
| **mean** | **2.6** | **1.6** | **1.6** | 25065 | 23445 |

**L-BFGS matches the paper's Table 1 "OURS" EPE@15nm column exactly on all 10
of 10 cases**, closing both of the main baseline's previously-known gaps at
once: case 3 (22 → 13, the single largest per-case gap anywhere in this
project's Table 1 comparison) and case 5 (1 → 0, the F-PERF-01 pixel that had
been deferred rather than hand-corrected — it disappears here as a side
effect of the better-conditioned optimizer, not because anything about
F-PERF-01 was touched). Mean L2 is also lower with L-BFGS in 8 of 10 cases.

### What this does NOT establish

This is **not** proposed as a replacement for the main `results/iccad13_ilt/`
baseline everywhere it's cited in this project — it uses a different starting
point (`init_scale=0.5` vs the main baseline's `init_scale=2.0`) for the
documented reason in F-SAT-01, a different iteration/func-eval budget (50
outer steps / ~110 func evals vs 300 iterations), and was run to test this
specific hypothesis, not to supersede the primary reported numbers without
separately deciding to do so. EPE@3nm was not swept or reported here as a
headline number (see F-LBFGS-01's own caution that it is not uniformly
better). The match to the paper's EPE@15nm column, case-for-case, is a strong
signal that L-BFGS is closer to whatever the paper's own (unspecified,
G-015) solver actually is — but "exact match on the metric reported" is not
the same claim as "identical solver," and no PVB, EPE@3nm, or Table 2 column
was used to further confirm or refute that.

## F-PERF-01 — Profiling beat intuition: forward FFT is 1% of cost, not the bottleneck

**Context.** The originally proposed "exact frequency-limited FFT speedup"
(claiming ~8-10x by extracting only the SOCS kernel's 35x35 support) was based
on a flawed derivation: it conflated the decimation-in-time <-> aliasing-in-
frequency identity (which gives an evenly-SPACED comb of frequencies cheaply)
with extracting a small CONTIGUOUS low-frequency block (which it does not
give). Extracting a genuinely small contiguous band from a general
(non-structured) signal without the full FFT requires either exploiting
analytic structure in the *signal* (not available for a continuously-optimized
mask) or a true pruned-FFT algorithm (real, but only a further ~2x on the
transform itself, and does not touch the other 3 stages of the pipeline).

**What profiling actually found**, breaking down `aerial_image()` on a
2048x2048 mask, 24 kernels (measured, not estimated):

| Stage | Share of wall-clock |
|---|---|
| Forward FFT (mask -> spectrum) | **1%** |
| Kernel multiply (corner placement) | 11% |
| Inverse FFT (24-channel, complex) | 34% |
| `abs()**2` + weighted sum | **59%** |

The forward FFT -- the one thing the original proposal targeted -- turned out
to be irrelevant. The real bottleneck, by a wide margin, was `images.abs()**2`.

### The actual fix

`|z|^2 = re(z)^2 + im(z)^2` is an exact algebraic identity. `.abs()**2`
computes `sqrt(re^2+im^2)` and immediately squares the result back, paying for
an unnecessary `sqrt` over every element (~100M elements: 24 kernels x
2048x2048). Replacing it with the direct identity:

- **Isolated benchmark**: 1.80x faster (541.6ms -> 301.1ms on a representative
  tensor), values agreeing to 1.1e-7 relative (pure float32 rounding, not a
  numerical difference -- the two formulas ARE the same real number
  mathematically).
- **Measured end-to-end** on `aerial_image()`: **1.12x** (1276.3ms -> 1134.6ms).
  Lower than the isolated benchmark suggested -- micro-benchmarks routinely
  overstate real-pipeline gains, which is why this is reported as measured
  end-to-end, not extrapolated from the isolated number.

This change sits inside a custom `torch.autograd.Function.forward()` (see
`_SocsIntensity` in `socs.py`), so it carries **zero differentiability risk**:
the backward pass is computed analytically, not by autodiff through this line.

### A genuine, honestly-reported side effect: a 1-pixel PV-Band shift

Re-scoring all 10 of the already-saved ICCAD13 optimized masks with the new
code against the committed `results/iccad13_ilt/case*.json`:

| Metric | Cases checked | Cases that differ |
|---|---|---|
| L2 | 10 | 0 |
| PV Band | 10 | **1** (case 5: 51978 -> 51979) |
| EPE@15nm | 10 | 0 |
| EPE@3nm | 10 | 0 |

One pixel, in one case, in one metric, out of 40 numbers checked: a
0.0019% relative change in case 5's PV Band. This is not a bug in either
formula -- both are exact -- it is a pixel whose intensity sat close enough to
the 0.5 binarization threshold that the last-bit rounding difference between
two equivalent floating-point expressions flipped its classification at ONE of
the two process corners that PV Band compares. This is expected behavior at a
measure-zero decision boundary and would occur with any two independently
-ordered but algebraically-equivalent implementations, on any hardware.

**What was NOT done, and why:** the affected "150-iteration" budget-curve
number for case 5 was not corrected, because the run_iccad13.py runner does
not persist intermediate checkpoint masks to disk -- only the final
(300-iteration) mask is saved -- so correcting it requires a full ~231-CPU-
minute re-run of the whole ICCAD13 baseline, not a cheap re-score. Given the
effect is a documented, understood, sub-0.002% artifact that changes no
conclusion in `REPRODUCTION_REPORT.md`, that re-run was deferred rather than
performed immediately; the baseline will be regenerated (and this will
self-correct) the next time a solver change (e.g. beta-annealing) requires a
fresh run anyway. The committed `case5.json` is therefore known to be stale by
exactly this one pixel until then -- recorded here rather than silently
tolerated or hand-edited.


---

## F-ANNEAL-01 — Beta annealing lets L-BFGS train from the main baseline's own (saturated) constants

**Severity: informational / positive result.** Extends F-SAT-01 with a fix
that addresses the mechanism directly instead of avoiding it.

### Recap of the problem (F-SAT-01)

The main ICCAD13 baseline's constants (`init_scale=2.0`, `mask_steepness=8.0`)
push every pixel's sigmoid argument to `+-16`, giving a raw gradient magnitude
of ~1.7e-6. Adam's normalization is insensitive to this; SGD, Nesterov, and
L-BFGS all stall completely from that starting point (confirmed directly for
L-BFGS: 5 iterations moved the loss by 0.0%). The X-14 ablation (F-LBFGS-01)
worked around this by using a *different*, well-conditioned starting point
(`init_scale=0.5`) -- valuable evidence that L-BFGS is competitive, but not a
fix usable in the main baseline's own configuration.

### The fix

`ILTConfig.beta_init` ramps the mask steepness used *during optimization*
(linearly or geometrically, `beta_schedule`) from a low value at step 0 to
`mask_steepness` by the final iteration, instead of holding it fixed. This is
the standard "continuation method" from the ILT / level-set literature.
Critically, this does not change what a converged result *means*: binarization
is `sigmoid(beta*P) >= 0.5`, which is algebraically `P >= 0` for any `beta > 0`
-- so annealing changes only the *optimization trajectory*, never the
definition of the final mask.

### Verified, on the exact saturated constants (`init_scale=2.0`,
`mask_steepness=8.0`) that F-SAT-01 showed stall every optimizer but Adam:

| Configuration | Final L2 (vs no-OPC = 103.0) |
|---|---|
| SGD, no annealing | 103.0 (unchanged -- confirms F-SAT-01) |
| SGD, with annealing (`beta_init=1.0`), 20 iterations | 103.0 (loss moved 36% but no pixel flipped in this short budget) |
| **L-BFGS, with annealing (`beta_init=1.0`), 10 iterations** | **12.0** |

### Honest reading

Annealing does not uniformly "fix" every optimizer's *speed* -- the SGD row is
reported as measured, not smoothed into a success. What it does fix is the
underlying **conditioning**: the continuous loss under SGD-with-annealing moved
substantially (36% relative, vs the *literal* 0.0% without annealing), it
simply needed more than 20 iterations at this step size to translate into a
binarized pixel flip. L-BFGS's quasi-Newton curvature estimate converts that
same improved conditioning into dramatic, fast progress. This is consistent
with F-LBFGS-01's finding that L-BFGS is the strongest optimizer available
here, now shown to work *at the main baseline's own constants*, not only at an
alternate well-conditioned starting point.

### What this does NOT yet establish

This is a small-scale (96x96, single test pattern) verification of the
mechanism, run to confirm annealing does what it is designed to do before
building on it. It is **not** a claim about the full ICCAD13 baseline's
numbers with annealing enabled -- that requires a full re-run, deferred (along
with the F-PERF-01 case-5 pixel) to a single consolidated baseline
regeneration once the other solver improvements in this round (EPE-aware loss,
PV-band-aware objective) are also in place, rather than re-running the ~231
CPU-minute baseline after each individual change.

### Backward compatibility

`beta_init=None` (the default) is verified bit-for-bit identical to every
prior committed result: `test_beta_init_none_is_bit_exact_with_constant_beta`
and a direct replay of case 1's first 10 iterations against the current
(post-F-PERF-01) code both match exactly.


---

## F-EDGE-01 — `epe_violations()` crashes on a target touching the canvas edge (pre-existing, never triggered by real data)

**Severity: low (latent, never exercised by any of this project's 10 real
ICCAD13 targets).** Discovered while writing an independent cross-check for
the new differentiable EPE-site loss (F-EPE-01).

### Symptom

`epe_violations(target.clone(), target, tolerance=15)` on an all-``1`` 64x64
target raises `IndexError: index 64 is out of bounds for dimension 1 with size
64`. Confirmed this is in the **already bit-exact-verified** metric itself,
not a new module:

```
>>> epe_violations(torch.ones(64,64), torch.ones(64,64), tolerance=15)
IndexError: index 64 is out of bounds for dimension 1 with size 64
```

### Root cause

`_segments()` pads the target with zeros before detecting boundary pixels, so
a target that is `1` all the way to the canvas edge has its LAST ROW/COLUMN
misclassified as a boundary (the zero padding looks like an adjacent "0"
region). `_check_sites()` then probes `target[r0, c0+1]` (or the symmetric
row/column case) to determine which side is "inside" -- and for a site sampled
at the very last valid index, `c0+1` (or `r0+1`) is out of bounds.

### Why this was never caught before

All 10 ICCAD13 targets have generous margin from the canvas edge (largest
bounding box: 828x640 nm inside a 2048 nm canvas -- hundreds of nm of margin),
and this project's regression tests only exercise those 10 real designs plus
synthetic hand-computed examples that were, by construction, never built to
probe this specific edge case. The bit-exact verification against the
reference contest checker (`tests_gril/regression`) is therefore correct and
remains correct for everything it actually tests -- it simply never had a
reason to construct an edge-touching target.

### What was (and was not) done about it

**`epe_violations()` and `_check_sites()` were NOT modified.** They are
bit-exact-verified against the reference checker and touching them for a case
that never occurs in any real experiment in this project carries a real risk
of silently perturbing the verified behaviour, for zero benefit to any actual
result. The new, independently-written `gril.ilt.epe_loss` module (which
parallels this sampling geometry for a different, differentiable purpose) adds
an explicit bounds check before the same neighbour lookup, so it degrades
gracefully (skips the site) instead of crashing -- verified by
`test_full_target_gives_zero_sites`.

This is recorded here as a known, low-priority, out-of-scope latent issue
rather than silently worked around or silently left undocumented.


---

## F-PVB-01 — The PV-band-aware objective produces a real, monotonic L2/EPE-vs-PVB trade-off

**Severity: informational.** Confirms `weight_pvb` (already present in
`ILTConfig`, wired through `ilt_loss`) does what it is meant to do, measured
directly rather than assumed from the formula alone.

### Experiment

`weight_pvb in {0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0}`, cases 1/3/9, 60
iterations, 4x-downsampled resolution (a deliberate tuning-scale sweep, not a
paper comparison — same discipline as the `step_size`/`mask_steepness` sweep
documented in `configs/experiments/README.md`). Raw per-point results in
`results/pvb_sweep/summary.json`.

| `weight_pvb` | L2 (sum, 3 cases) | PVB (sum, 3 cases) | EPE@15 (case 3) |
|---:|---:|---:|---:|
| 0.0   | 8608 | 11514 | 22 |
| 0.001 | 8615 | 11512 | 22 |
| 0.01  | 8621 | 11475 | 23 |
| 0.05  | 8641 | 11462 | 23 |
| 0.1   | 8645 | 11415 | 23 |
| 0.5   | 8741 | 10978 | 21 |
| 1.0   | 8933 | 10654 | 20 |

### Result

PV Band drops **monotonically** as `weight_pvb` increases, by ~7.5% total
(11514 → 10654) from `weight_pvb=0` to `weight_pvb=1.0`, while L2 rises
monotonically but much more gently, by ~3.8% (8608 → 8933) over the same
range. This is a genuine trade-off curve, not noise: every one of the seven
points moves in the expected direction on both axes. EPE@15nm on case 3 is
essentially flat (22→23→21→20) — at this weight range the process-window term
trades off against nominal fidelity (L2) well before it visibly moves the
discrete EPE count.

### Interpretation

The PV-band term is doing real, physically sensible work: it measures
`(z_max - z_min)^2` at the resist threshold (the best/worst-case aerial images
across the process window in `litho.aerial_outer`), so penalizing it more
heavily pushes the optimizer toward masks whose printed edge moves less across
the process window, at the cost of nominal-image fit. `weight_pvb=0.1` is a
reasonable middle point (PVB down 0.9% from baseline, L2 up only 0.4%) if this
were to be adopted for a full run; this project does not adopt it into the
main ICCAD13 baseline, since doing so would need re-verifying the baseline's
un-annealed, un-EPE-aware numbers change for a reason tied to a real
paper claim, and the paper does not specify a PV-band weight to reproduce
(G-015 territory again).

### What this does NOT establish

This is a small-scale, 3-case, reduced-resolution calibration sweep, exactly
like the `step_size`/`mask_steepness` sweep it follows the same discipline of.
It establishes that the term is *not inert or broken* — it was not run at full
resolution on all 10 cases, and no full-resolution PV-band number is claimed
anywhere in this project's tables.


---

## F-MULTISTART-01 — Random multi-start matches the trained generator's own single-start baseline; both trained-generator arms are *worse* than either

**Severity: high (core scientific finding of this round).** This is the
control neither the paper nor this project's earlier experiments run: is the
"generator beats single-start ILT" result (Table 2, PT/PT+RL rows) actually
about a *learned prior placing K starts well*, or just about *having K
independent tries at all* — something a random perturbation would also give
for free? See `src/gril/experiments/run_multistart_ablation.py`'s module
docstring for the full design rationale and `configs/experiments/multistart_ablation.yaml`
for the calibration discipline (`perturb_sigma` matched to the generator's
measured sampling diversity, not guessed).

### Setup

Four arms, same 8 held-out validation designs (deterministically regenerated
from `train_scaled.yaml`'s own seed), same per-candidate ILT refinement budget
(`downsample=2, iterations=100, step_size=0.2, mask_steepness=8.0`), same
selection rule (best-of-K by EPE@3nm):

- **A. single-start** — one deterministic cold start, K=1.
- **B. random-K** — K=8 independent Gaussian perturbations of the same cold
  start (`sigma=0.63`, calibrated so its measured mask-space diversity matches
  arm C's), no learning at all.
- **C. PT** — K=8 samples from the WGAN-GP-pretrained generator (read from the
  already-completed `results/train_scaled/summary.json`, not re-run).
- **D. PT+RL** — K=8 samples from the GRPO-finetuned generator (same source).

Raw per-design numbers in `results/multistart_ablation/summary.json`.

### Result

| Arm | best-of-K EPE@3nm (mean over 8 designs) | sample mean | diversity |
|---|---:|---:|---:|
| A. single-start | **115.00** | — | — |
| B. random-K (no learning) | **115.00** | 115.08 | 0.00155 |
| C. PT (generator) | 134.75 | 135.59 | 0.00161 |
| D. PT+RL (generator) | 134.50 | 135.64 | 0.00133 |

Per-design (all 8 designs, single vs random-K-best): `98=98`, `67=67`,
`152=152`, `124=124`, `112=112`, `95=95`, `73=73`, `199=199` — **exactly equal
on every single design**, not just on average. Arm B's 8 individual candidate
scores per design are also nearly identical to each other (e.g. design 1:
`67,67,67,67,67,67,67,69`) despite a diversity (mean pairwise mask L1 distance
`0.00155`) matched to the generator's own measured diversity (`0.00161`).

### Interpretation

Two findings, not one:

1. **Random multi-start gives zero measurable benefit over single-start at
   this diversity scale.** At the perturbation magnitude that matches the
   generator's own measured sample diversity, the ILT refinement gradient
   descent converges to essentially the same local optimum regardless of
   which of the 8 perturbed starts it began from — the "spread" is real in
   mask-pixel space (diversity ≈ 0.0016, not zero) but the *downstream ILT
   solve washes it out* before it reaches the EPE metric. Best-of-8 buys
   nothing here because there is effectively nothing to select between.
2. **Both generator arms are measurably *worse* than single-start ILT with no
   generator involved at all** (134.75/134.50 vs 115.00 — a ~17% *increase* in
   EPE violations, in the wrong direction). This is consistent with, and
   sharpens, F-RL-01's null result for GRPO: PT+RL is marginally better than
   PT (134.50 vs 134.75, as the paper's ordering claims), but the entire
   pretrained-generator pipeline is worse than doing nothing more sophisticated
   than the single deterministic cold start this project's own ICCAD13
   baseline already uses. The generator is not merely failing to add value on
   top of ILT — its initialization is actively worse than the plain cold start
   this experiment's own arm A shows is already good enough to reach the same
   optimum that random search reaches.

Together these show the paper's core mechanism, as measured at this
project's training scale (F-RL-01's compute budget — a few hundred WGAN-GP
steps, a comparably small GRPO run), is not doing the two things its
narrative would require: it is not placing its K starts more usefully than
random noise would, and its starts are not even as good as no learned prior
at all. This does not contradict F-RL-01 (a null result for the *finetuning*
step specifically) — it extends it: the pretraining stage itself, not only
GRPO, is the part that is not adding value at this scale.

### What this does NOT establish

This is measured at the same reduced training scale as F-RL-01 (256x256
canvas, 64 training designs, the compute budget available on this CPU-only
host) — it is a statement about *this reproduction's* trained generator, not
a claim that a learned sampler can never beat random multi-start in
principle, nor a claim that the paper's own (larger-scale, GPU-trained)
generator would show the same gap. It is, however, exactly the control this
project's earlier experiments and the paper itself never isolate, and it
directly explains *why* a "generator vs no-OPC" or "generator vs single ILT
start" comparison alone (Table 2's framing) cannot distinguish "the generator
learned something useful" from "having any K tries at all would have done
just as well" — a distinction the paper's ablations do not make either.


---

## F-MRC-01 — Every committed mask is manufacturability-unconstrained; a cheap, already-implemented fix removes the worst of it for near-zero cost

**Severity: medium.** Not a numerical bug — a real, user-flagged gap between
what this project reports (L2/PVB/EPE against the target) and what a fab
could actually build. Confirms the concern directly with per-case counts
rather than by inspection of one figure.

### The concern, verified

Every ICCAD13 result committed so far (`iccad13_ilt`, `abl_optimizer`,
`train_scaled`'s refinement stage) uses `ILTConfig.mrc_open_size=0` — the
raw `sigmoid(β·P) ≥ 0.5` threshold, with **no manufacturability constraint of
any kind** applied during or after optimization. Connected-component analysis
of the raw masks (`scripts/mrc_cleanup_sweep.py`, all 10 cases):

| | mean over 10 cases |
|---|---:|
| connected components | **211.2** |
| components smaller than 20 px² (4.5×4.5 nm — physically absurd at any real process node) | **79.9** |
| single-pixel islands present | yes, in every case |

This is a real defect in what "the mask" means in this project's figures and
`maskN.pt` files: no real fab's mask-writer or MRC (mask rule check) tooling
would accept output with ~80 pieces of sub-5nm floating debris per case. The
`figures/mask_visualization_case1.png` panel the user was looking at when
this was raised shows exactly this (196 components for case 1 specifically).

### The fix already existed, unused

`gril.ilt.solver.morphological_open` (binary erosion-then-dilation) and
`ILTConfig.mrc_open_size` were implemented from the start (docs/gap_ledger.md
G-021: "morphological opening with a configurable structuring-element size;
default off") but never exercised in any committed run. Since opening is a
pure post-processing step on the already-binarized mask (`_binarize()`), it
can be applied to the already-saved `maskN.pt` files with **no
re-optimization needed** — `scripts/mrc_cleanup_sweep.py` does exactly that,
sweeping `open_size in {0,3,5,7,9}` across all 10 cases and re-scoring each
result against the target with the same verified litho model and EPE metric.

| `open_size` | L2 (mean) | EPE@15nm (mean) | EPE@3nm (mean) | components (mean) | tiny components (mean) |
|---:|---:|---:|---:|---:|---:|
| 0 (raw, current) | 25065 | 2.6 | 55.4 | 211.2 | 79.9 |
| 3 | 25069 | 2.9 | 54.1 | 157.9 | 21.3 |
| **5** | **25149** | 2.9 | 53.3 | 135.1 | **0.3** |
| 7 | 25263 | 2.9 | 52.7 | 128.9 | 0.0 |
| 9 | 25581 | 2.8 | 51.1 | 123.3 | 0.0 |

`open_size=5` (a 5nm structuring element) eliminates essentially all tiny
debris (79.9 → 0.3 mean components, and exactly 0 in 9 of the 10 cases) for a
**0.33% increase in mean L2** and, interestingly, a slight *improvement* in
mean EPE@3nm (55.4 → 53.3) — cleaning up isolated noise pixels removes some
of the spurious violations they were causing. Mean EPE@15nm rises slightly
(2.6 → 2.9), driven almost entirely by case 3 specifically (22 → 25 — see
"What this does NOT establish" below).

### Interpretation

The single-pixel/tiny-island debris — the most egregious, "no fab would ever
accept this" part of the concern — is close to a free fix: `open_size=5`
removes essentially all of it for a fraction of a percent of L2. This is not
adopted as the new default for the committed baseline (same reason as
F-PVB-01/F-EPE-01: no re-verification against the paper's numbers has been
done with it on), but it demonstrates the gap is not expensive to close for
its worst symptom.

**This does not make the masks production-realistic.** Even at `open_size=9`,
~123 connected components remain per case — a real OPC/SRAF mask for these
patterns would likely have on the order of a few dozen, not >100 — and
morphological opening only removes debris/protrusions *narrower* than the
structuring element; it enforces nothing about a *minimum spacing* between
separate shapes (a closing operation or an explicit spacing check would be
needed for that), and nothing about final mask *complexity* (shot count) in
the sense the paper's cited MRC reference actually cares about. The
differentiable `weight_tv` (total-variation) regularizer already in
`ilt_loss` is the more principled fix — applied *during* optimization rather
than as a post-hoc crop, it can shape the mask towards fewer, smoother
features instead of just erasing violations after the fact — but it is
likewise off by default (`weight_tv=0.0`) in every committed run, and was not
swept this round.

### What this does NOT establish

Case 3's EPE@15nm specifically gets **worse** under opening (22 → 24-25
across every tested size) — the single case that is also this project's
largest known gap to the paper (F-LBFGS-01, X-16 in progress). This is a
real, measured interaction, not noise (monotonic across all 4 nonzero sizes
tested), and a caution against assuming MRC cleanup is free everywhere just
because it is free *on average*: for at least one case, removing what
opening treats as "debris" was removing pixels that were doing real EPE work.
No sweep of `weight_tv` or a spacing-aware constraint was run; this finding
establishes that the cheapest fix (opening) handles the debris problem well
on average but is not a complete or unconditionally-free MRC solution.


---

## F-EPE-01 — The EPE-aware loss term works, but only once weighted to the same order of magnitude as L2

**Severity: informational.** Confirms `weight_epe` (backed by the new,
independently cross-checked `gril.ilt.epe_loss` module) does real work, and
documents the calibration mistake that would have hidden that — mirroring
F-PVB-01's discipline of measuring the correct weight rather than guessing.

### Correctness check (before any real-scale run)

`soft_epe_loss` is deliberately built as a *parallel, independent*
implementation of the same measurement-site geometry `epe_violations()`
already uses (not a differentiable relaxation derived from that function),
specifically so a bug in one is unlikely to be masked by the same bug in the
other. `tests_gril/unit/test_epe_loss.py` cross-checks it against a real
curvy ILT-optimized mask (not just synthetic hand-built cases): evaluated at
`margin=0` on a **binary** image, `soft_epe_loss`'s sign structure reproduces
`epe_violations()`'s exact per-side violation counts.

### First attempt: `weight_epe=2.0` — no measurable effect

Case 3, full 2048 resolution, 100 iterations, `step_size=0.2,
mask_steepness=8.0` (this project's own main-baseline constants):

| Config | L2 | EPE@15nm | EPE@3nm |
|---|---:|---:|---:|
| baseline (L2 only) | 62947 | 26 | 121 |
| `weight_epe=2.0, tol=15nm` | 62939 | 26 | 121 |
| `weight_epe=2.0, tol=3nm`  | 62934 | 26 | 121 |

Identical EPE counts in every arm; L2 moves by less than 0.02%. Rather than
conclude "the term doesn't work" from this, the raw loss magnitudes were
measured directly at the solver's actual starting point (case 3, same
constants):

```
raw L2 term (sum sq err)     = 132101
raw soft_epe_loss (294 sites) =     46
ratio                          =   2883
```

`weight_epe=2.0` makes the EPE term's total contribution to the gradient
roughly **1400x smaller** than L2's — it is not broken, it is numerically
invisible next to the term it is meant to compete with. This is the same
diagnostic step F-SAT-01 used (measure the actual gradient scale, don't
reason about the formula in the abstract) applied to a different loss term.

### Second attempt: weight calibrated to L2's order of magnitude

Same case, same constants, `weight_epe in {1000, 3000}` (chosen so
`weight_epe * raw_soft_epe_loss` is comparable to `raw L2`, using the ratio
measured above):

| Config | L2 | EPE@15nm | EPE@3nm |
|---|---:|---:|---:|
| baseline (L2 only) | 62947 | 26 | 121 |
| `weight_epe=1000, tol=15nm` | 60703 | 20 | 113 |
| `weight_epe=3000, tol=15nm` | 62422 | **16** | 117 |

At `weight_epe=3000`, EPE@15nm drops **38%** (26 → 16) with L2 essentially at
the baseline (62422 vs 62947, −0.8%) — the discrete metric the soft loss is a
proxy for actually moves, not just the smooth surrogate. At `weight_epe=1000`
L2 *also* improves (62947 → 60703, −3.6%) alongside EPE@15nm (26 → 20) and
EPE@3nm (121 → 113): penalizing the exact sites the discrete metric checks
apparently also helps the surrounding nominal fit at those sites, rather than
purely trading against it — a genuinely useful term once correctly weighted,
not only a trade-off knob like `weight_pvb`.

### What this does NOT establish

Two points on a hand-picked weight range, one case, one iteration budget —
not a full sweep across all 10 ICCAD13 cases (unlike `weight_pvb`, which got
a dedicated 7-point sweep on 3 cases). It is not adopted into the main
baseline for the same reason `weight_pvb` was not: the paper specifies no EPE
loss weight to reproduce against, and the main baseline's un-annealed,
un-EPE-aware numbers are the ones already verified against the paper's own
Table 1/2 rows.

### Lesson

The two calibration mistakes this round made independently (`weight_epe=2.0`
here, and the original un-swept guess that motivated F-PVB-01's dedicated
sweep) point at the same underlying discipline: a new loss term's weight
cannot be chosen by intuition about what "should" matter — it has to be
measured against the term(s) it is added to, in the same units, at the actual
starting point of the actual optimization. A weight that "looks small" (2.0)
can be three orders of magnitude too small in practice.
