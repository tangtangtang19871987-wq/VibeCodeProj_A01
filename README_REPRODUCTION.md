# Reproduction: *Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning*

Target paper: **arXiv 2602.19027v1** — Haoyu Yang, Haoxing Ren (NVIDIA), DAC'26.

A verifiable, auditable reproduction of the paper's physics, metrics, and method,
with an explicit account of what could and could not be reproduced here.

---

## Honest status in one table

| Tier | What it means | Status |
|---|---|---|
| **L1 — mathematical** | Forward model, adjoint, ILT, generator, GRPO objective all correct and tested | **Done.** 90+ tests; physics verified to machine precision |
| **L2 — experimental** | The paper's tables reproduced within a tolerance | **Partial, and bounded by the environment.** Our independent ILT baseline on ICCAD13 is measured and compared per case; the paper's *"Ours"* rows need LithoBench + 8× A100 and are **not reproduced** |
| **L3 — engineering** | Clean-env install, config-driven runs, checksums, one command | **Done.** |

The paper PDF was supplied by the user; every equation, both result tables, and
the full experimental configuration are read from the primary source. See
`docs/paper_summary.md`. Remaining unknowns are what the **paper itself does not
state** — tracked in `docs/gap_ledger.md`.

**This host is CPU-only** (4 cores, no GPU). Nothing in this repository requires
CUDA. The paper used 8× A100; that gap is the binding constraint on L2 and is
recorded rather than glossed over.

---

## Verified physics and metrics

Every number here was measured in this session and is pinned by a test.

| Check | Result |
|---|---|
| SOCS forward imaging vs independent pure-autograd implementation | **1.7e-16** |
| Analytic adjoint vs exact autograd (3 doses × 2 focus conditions) | **3.9e-16** |
| L2 / PV Band / EPE vs the reference contest checker, all 10 cases | **bit-exact** |
| GLP rasterisation vs reference, all 10 cases | **bit-identical** |
| Open-field normalisation | 0.95154 (24-term SOCS truncation loses 4.85%) |
| Corner-indexed FFT vs explicit fftshift | 1.6e-7 |
| Translation equivariance / x-mirror | 6.1e-7 / 4.9e-7 |
| 8× multi-resolution consistency (validates the paper's low-res reward loop) | **0.17% of peak** |
| Policy log-prob (Eq. 8) vs `torch.distributions.Bernoulli` | **exact** |

### Substantive findings

Written up in `docs/findings.md`:

1. **The public reference implementation's gradient is ~8.7% wrong** (F-ADJ-01).
   OpenILT's `pylitho/exact.py` builds its backward pass from the shipped `ct_*`
   kernels, which are not the adjoint of its own forward operator. The correct
   adjoint of `IFFT∘diag(H)∘FFT` is `conj(H)`. My first implementation inherited
   the same error; a float64 finite-difference check caught it.
2. **The published ICCAD13 kernels are column-symmetric but not row-symmetric**
   (F-KER-01), so y-mirror equivariance is not a valid invariant for this model.
3. **The paper contradicts itself on its central algorithm** (G-012): Sec. 3.3.2
   defines and argues for a teacher-relative advantage (Eqs. 6–7), while Sec. 4.1
   says the runs used the group mean. Both are implemented and selectable.

---

## Quick start

```bash
pip install -r requirements.txt     # pinned, CPU-only
./scripts/fetch_data.sh             # public ICCAD13 assets + checksum verify
python -m pytest                    # full suite
python -m pytest -m "not slow"      # fast subset

# Reproduce the ICCAD13 ILT baseline (CPU; ~15 min/case, resumable)
PYTHONPATH=src python -m gril.experiments.run_iccad13 \
    --config configs/experiments/iccad13_ilt.yaml

# Two-stage training: WGAN-GP pretrain -> GRPO finetune
PYTHONPATH=src python -m gril.experiments.run_training \
    --config configs/experiments/train_scaled.yaml

# Regenerate the comparison against the paper's tables
python scripts/make_report.py       # -> REPRODUCTION_REPORT.md
```

## Layout

```
src/gril/
  litho/        SOCS kernels, Hopkins forward model + exact adjoint, resist, corners
  data/         ICCAD13 GLP parsing; seeded synthetic layouts
  metrics/      L2, PV Band, EPE  (bit-exact vs the contest checker)
  ilt/          gradient ILT solver, morphological MRC, iteration checkpoints
  models/       style-aware generator (Fig. 3, Eq. 4) + WGAN-GP critic
  train/        Eq. 5 pretraining; Eqs. 6-10 GRPO finetuning
  experiments/  config-driven runners, sampling + selection
tests_gril/     physics / unit / regression
docs/           paper summary, source inventory, gap ledger, traceability, findings, tiers
REPRODUCTION_SPEC.md     the spec, written before the code
REPRODUCTION_REPORT.md   generated: our numbers vs the paper's tables
```

## Key documents

| File | What it answers |
|---|---|
| `docs/paper_summary.md` | What the paper says, equation by equation, plus where it is silent or self-contradictory |
| `docs/gap_ledger.md` | Every unknown, the assumption used, alternatives, and the sensitivity experiment |
| `docs/traceability_matrix.md` | Every equation/figure/table mapped to code and a test, with a completeness check |
| `docs/findings.md` | Bugs and physics issues found during validation, with root causes |
| `docs/REPRODUCTION_TIERS.md` | Per-claim map of reproducible vs not, and why |
| `REPRODUCTION_SPEC.md` | Physics conventions: FFT normalisation, units, corners, tolerances |

## Relationship to the pre-existing repository contents

This checkout already contained an unrelated **Array Generator** module (SRAF
array synthesis). **No pre-existing file has been modified, moved, or deleted.**
All reproduction work lives in new paths.

One non-invasive addition: `pytest.ini` plus a `.pkgroot/array_generator` symlink
make the checkout importable under its real package name. Without it pytest could
not collect *any* test in the repository — a pre-existing condition verified
before this work began (`docs/findings.md` F-ENV-01).
