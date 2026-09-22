# Environment Audit

## Measured host environment (this session)

| Property | Measured value | Command |
|---|---|---|
| OS | Linux 6.18.44-fc-v37, Ubuntu-based container | `uname -r` |
| CPU | **4 logical cores** | `nproc` |
| RAM | **15 GiB** | `free -g` |
| GPU | **NONE.** `nvidia-smi` not found; `torch.cuda.is_available() == False` | `nvidia-smi`, torch probe |
| Disk | 252 G volume, **30 G available** | `df -h /home/user` |
| Python | 3.11.15 | `python3 -V` |
| PyTorch | 2.9.1+cu128, **running CPU-only** | `torch.__version__` |
| NumPy / SciPy / OpenCV | 2.4.6 / installed / 5.0.0 (headless) | import probe |
| Network | Filtered egress proxy. `github.com`, `raw.githubusercontent.com`, PyPI reachable. arXiv/ACM/SPIE/HuggingFace/SemanticScholar/OpenReview **403**. | see `docs/source_inventory.md` |

## Measured performance baseline

Full ICCAD13 evaluation (2048x2048, 24 SOCS kernels, 3 process corners, L2 + PV
Band + EPE) on this CPU: **~6.5 s per case**, measured across all 10 cases.

A single forward+adjoint ILT step at 2048x2048 costs roughly one forward
evaluation. A 100-iteration ILT run is therefore on the order of **5-10 minutes
per case per candidate** on this host.

## Gap versus the paper's implied environment

The paper reports GPU-accelerated batched ILT and a 2-3x speedup over numerical
ILT solvers (S1.1). Its training stage, per S1.2, uses `K=16` candidates per
design with a ~100-step ILT loop inside the RL reward, over a LithoBench-scale
dataset (tens of thousands of layouts). Taking those figures at face value, one
RL epoch requires on the order of 10^6 ILT iterations.

**On 4 CPU cores with no GPU, the paper's training scale is not reachable — not
by a constant factor, but by roughly 3-4 orders of magnitude.** A straightforward
estimate: 16 candidates x 100 ILT steps x ~10^4 designs at CPU speed is several
CPU-years.

### What this means for the reproduction tiers

This is recorded here so it is never silently glossed over later:

| Tier | Definition | Feasibility on this host |
|---|---|---|
| **L1 — mathematical reproduction** | Forward model, adjoint/gradient, ILT optimizer, GRPO objective, metrics all correct and test-verified | **Feasible in full.** This is where the effort goes. |
| **L2 — experimental reproduction** | Paper's Tables/Figures reproduced within a declared tolerance | **Not feasible at paper scale**, and additionally blocked because the paper's target numbers were never obtained (S3). Achievable substitute: a scaled-down, fully-specified experiment on ICCAD13 at reduced resolution with the *same* protocol, reported as its own measured result — never presented as matching the paper. |
| **L3 — engineering reproduction** | Clean-env install, config-driven experiments, tests, checksums, one-command runs | **Feasible in full.** |

### Mitigations adopted

1. Physics runs at 2048x2048 for final scoring (matching the contest protocol),
   but training/RL loops run at reduced resolution with an explicit, configurable
   `resolution` field — the paper itself uses a downsampled ILT loop for the
   reward (S1.2), so this direction is faithful, only the magnitude differs.
2. Every experiment config records `hardware`, wall-clock `runtime`, and the
   resolution actually used, so a GPU host can re-run the identical config at
   full scale without code changes.
3. Runtime comparisons against the paper's "2-3x speedup" claim are reported as
   **not evaluable on this host** rather than estimated.

## Pre-existing repository content

`/home/user/VibeCodeProj_A01` already contained an unrelated **Array Generator**
module (SRAF consistency-check array synthesis): `core.py`, `constructors.py`,
`templates.py`, `exporters.py`, `visualizers.py`, `examples/`, `tests/`,
`README.md`, `IMPLEMENTATION_SUMMARY.md` (initial commit `c3ae545`).

**No pre-existing file has been modified, moved, or deleted.** All reproduction
work is added under new paths: `src/gril/`, `configs/`, `docs/`, `results/`,
`scripts/`, and new subdirectories `tests/unit`, `tests/physics`,
`tests/regression`. The pre-existing top-level `tests/*.py` files are untouched.
