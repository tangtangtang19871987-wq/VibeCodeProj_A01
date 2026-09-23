# Results

Every experiment writes **raw numerical results**, never only a plot or an image.

## What is tracked in git

| Path | Contents |
|---|---|
| `<experiment_id>/caseN.json` | Per-case raw scores: L2, PV Band, EPE at every tolerance, the full ILT loss history, wall-clock time, seconds/iteration, and the iteration-budget curve |
| `<experiment_id>/summary.json` | Aggregates, the complete resolved config, and provenance (git commit, torch version, platform, CPU count, thread count) |
| `<experiment_id>/history.json` | Per-step training traces (losses, rewards, advantages) |
| `<experiment_id>/maskN.pt` | The final optimised masks — the actual deliverable of an ILT run, stored as `uint8` |

## What is NOT tracked, and how to regenerate it

| Path | Why | Regenerate with |
|---|---|---|
| `<experiment_id>/gt.pt` | ~16 MB of ILT-generated ground-truth masks, fully determined by the config's seed and `gt_iterations` | delete it and re-run the training config |
| `<experiment_id>/generator_*.pt` | Model checkpoints; large and reproducible from the config | re-run the training config |

Runs are **resumable**: a completed `caseN.json` is reused rather than
recomputed, so an interrupted long CPU job can be restarted with the same
command.

## Reproducing

```bash
PYTHONPATH=src python -m gril.experiments.run_iccad13 --config configs/experiments/iccad13_ilt.yaml
PYTHONPATH=src python -m gril.experiments.run_training --config configs/experiments/train_scaled.yaml
PYTHONPATH=src python -m gril.experiments.run_optimizer_ablation --config configs/experiments/abl_optimizer.yaml
PYTHONPATH=src python -m gril.experiments.run_multistart_ablation --config configs/experiments/multistart_ablation.yaml
PYTHONPATH=src python scripts/pvb_weight_sweep.py          # -> results/pvb_sweep/summary.json (F-PVB-01)
PYTHONPATH=src python scripts/epe_weight_calibration.py    # -> results/epe_aware_case3/summary.json (F-EPE-01)
PYTHONPATH=src python scripts/mrc_cleanup_sweep.py          # -> results/mrc_sweep/summary.json (F-MRC-01)
python scripts/make_report.py     # -> REPRODUCTION_REPORT.md
python scripts/make_figures.py    # -> figures/*.png
```

`pvb_weight_sweep.py` and `epe_weight_calibration.py` are smaller, standalone
calibration scripts (not full `gril.experiments` runners with a YAML config)
since each answers one specific "does this loss weight do anything, and at
what scale" question rather than reproducing a paper table — see
`docs/findings.md` F-PVB-01 and F-EPE-01. `pvb_weight_sweep.py` is resumable
the same way the main runners are (`results/pvb_sweep/sweep_raw.json` caches
completed weights).

## A note on running multiple experiments concurrently

Every `solve()` call uses PyTorch's default (multi-threaded) CPU backend.
Launching two or more of these runners in parallel on a CPU-only host
oversubscribes the machine badly -- observed directly in this project: three
concurrent runs on 4 cores drove load average past 11, each run slowing to a
fraction of its solo speed. Either run experiments sequentially, or set
`torch.set_num_threads(N)` per process (e.g. `N=2` for two concurrent runs on
a 4-core host) before calling `solve()`.
