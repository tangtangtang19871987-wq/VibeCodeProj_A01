# Reproduction: *Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning*

Target paper: **arXiv 2602.19027** (Haoyu Yang, Haoxing Ren, NVIDIA — DAC'26).

> ### Status: honest summary, read before anything else
>
> **The paper itself could not be retrieved in this environment.** arxiv.org and
> all ten mirrors tried are blocked by the organization's egress proxy; every
> attempt is logged in [`docs/source_inventory.md`](docs/source_inventory.md).
> **No equation of the paper has been read.**
>
> What this repository therefore is:
> - a **verified, exact** implementation of the public **ICCAD 2013 lithography
>   model and scoring protocol** that the paper is benchmarked on, and
> - a **faithful reimplementation in progress** of the paper's method, built from
>   abstract-level facts plus standard formulations of each named component,
>   with every assumption logged in [`docs/gap_ledger.md`](docs/gap_ledger.md).
>
> It is **not** an exact reproduction, and cannot become one until the primary
> PDF is available. **If you can supply the PDF, that single input collapses most
> of the open gaps.**

## What is verified today

All numbers below were measured in this session and are pinned by tests.

| Component | Verification | Result |
|---|---|---|
| GLP parsing + rasterisation | vs reference rasteriser, all 10 cases | **bit-identical** |
| SOCS forward imaging | vs pure-autograd reimplementation | **1.7e-16** |
| Open-field normalisation | analytic clear-field limit | **0.95154** (24-term truncation loses 4.85%) |
| FFT convention | corner-indexed vs explicit fftshift | **1.6e-7** |
| Analytic adjoint | vs exact autograd, 3 doses x 2 focus conditions | **3.9e-16** |
| Translation equivariance | circular shift | **6.1e-7** |
| L2 / PV Band / EPE | vs reference contest checker, all 10 cases | **bit-exact** |
| Batched vs single | 3-case batch | **0.0** |

Plus two substantive findings — including an **~8.7% gradient error in the public
reference implementation's adjoint** — written up in [`docs/findings.md`](docs/findings.md).

### Measured no-OPC ICCAD13 baseline (mask = target, tolerance 15 nm)

| Case | L2 (nm²) | PV Band (nm²) | EPE |
|---|---|---|---|
| test1 | 116184 | 45874 | 86 |
| test2 | 117802 | 37036 | 84 |
| test3 | 160846 | 32646 | 125 |
| test4 | 84037 | 101 | 64 |
| test5 | 117516 | 59188 | 71 |
| test6 | 110523 | 50684 | 66 |
| test7 | 103219 | 54316 | 71 |
| test8 | 55012 | 19084 | 37 |
| test9 | 120211 | 60796 | 66 |
| test10 | 41291 | 15039 | 26 |
| **mean** | **102664** | **37476** | **69.6** |

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Fetch and checksum the public ICCAD13 assets (kernels + 10 layouts)
./scripts/fetch_data.sh

# 3. Run the test suite (49 tests: physics, unit, regression)
python -m pytest                 # ~40 s on 4 CPU cores
python -m pytest -m "not slow"   # ~4 s, skips full-canvas runs
```

## Layout

```
src/gril/          reproduction package
  litho/           SOCS kernels, Hopkins forward model, resist, process corners
  data/            ICCAD13 GLP layout parsing and rasterisation
  metrics/         L2, PV Band, EPE  (bit-exact vs the contest checker)
  ilt/ models/ train/ experiments/   (in progress)
tests_gril/        physics / unit / regression suites
docs/              source inventory, gap ledger, traceability matrix, findings
REPRODUCTION_SPEC.md    the spec, written before the code
```

## Relationship to the pre-existing contents of this repository

This checkout already contained an unrelated **Array Generator** module (SRAF
array synthesis). **No pre-existing file has been modified, moved, or deleted.**
All reproduction work lives in new paths. See
[`docs/environment_audit.md`](docs/environment_audit.md).

One non-invasive addition: `pytest.ini` + a `.pkgroot/array_generator` symlink
make the checkout importable under its real package name. Without it pytest could
not collect *any* test in the repository — a pre-existing condition, verified
before this work began (F-ENV-01).

## Reproduction tiers

| Tier | Status |
|---|---|
| **L1 — mathematical** | Physics stack + metrics **done and verified**; ILT solver, generator, GRPO **in progress** |
| **L2 — experimental** | **Blocked.** The paper's target numbers were never obtained (G-002); no GPU on this host |
| **L3 — engineering** | Spec, gap ledger, traceability, checksummed data, tests, one-command runs **in place** |
