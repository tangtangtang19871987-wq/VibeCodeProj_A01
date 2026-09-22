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
