# Data

This project uses two categories of data: public physics/benchmark assets
(fetched, checksummed, never modified) and synthetic training data (generated
in-process, deterministic, documented as a substitute for an unreachable
dataset). Neither is committed to git in bulk -- see below for what to run.

## Public assets (fetched)

`scripts/fetch_data.sh` clones a **pinned commit**
(`dabb97c6ca3dfd159362e48273c436444c77353b`) of the Apache-2.0 licensed
[OpenILT](https://github.com/OpenOPC/OpenILT) project and copies out exactly
three things into this directory:

| Path (after fetch) | Contents | Provenance |
|---|---|---|
| `kernel/kernels/*.pt` | 24-term SOCS optical kernels (focus/defocus + conjugate-transpose variants), `(35,35,24)` complex64 | `docs/source_inventory.md` S2.2 |
| `kernel/scales/*.pt` | Per-kernel eigenvalues (`w_k`), `(24,)` float32, descending | same |
| `benchmark/ICCAD2013/M1_test{1..10}.glp` | The 10 public ICCAD 2013 CAD Contest Problem-1 layouts | `docs/source_inventory.md` S2.1 |
| `config/lithoiccad13.txt` | The contest's process configuration (target density, print threshold/steepness, dose corners) | `docs/source_inventory.md` S2.5 |

**Run it:**

```bash
./scripts/fetch_data.sh
```

**Verify it:** the script itself runs `sha256sum -c CHECKSUMS.sha256` after
fetching and will fail loudly on any mismatch. To re-verify later without
re-fetching:

```bash
cd data && sha256sum -c CHECKSUMS.sha256
```

`CHECKSUMS.sha256` (committed to git, unlike the fetched files themselves) is
the authoritative manifest -- if OpenILT's upstream commit ever changes
(it shouldn't; the fetch script pins an exact commit hash), the checksums here
are what this project's own numbers were produced against.

None of these files are committed to git directly: they are a third party's
public artifact, correctly obtained by fetching and verifying rather than by
vendoring a copy, and `.gitignore` excludes `data/.openilt/`, `data/kernel/`,
`data/benchmark/`, `data/config/`.

## Synthetic training data (generated, not fetched)

The paper's own training set (LithoBench's MetalSet + ViaSet) is **not
reachable** from this environment (`docs/gap_ledger.md` G-041). Training
experiments (`configs/experiments/train_scaled.yaml` and similar) substitute a
seeded, deterministic synthetic M1-style layout generator
(`src/gril/data/synthetic.py`) instead. This is generated in-process at run
time from a fixed seed -- there is nothing to fetch or check into git, and
every training result explicitly states that it uses synthetic data, never
presenting it as LithoBench.

## Experiment outputs

Raw per-case results (`results/<experiment_id>/`) are a separate matter from
input data -- see `results/README.md` for what is tracked in git there and
why.
