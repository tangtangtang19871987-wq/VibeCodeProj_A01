# Experiment Configurations

Each file is a complete, runnable specification. Every run records its git
commit, environment, wall-clock time, and all parameters into
`results/<experiment_id>/summary.json`, and writes raw per-case numbers — never
only a plot.

| Config | Paper reference | What it answers |
|---|---|---|
| `iccad13_ilt.yaml` | Tables 1 & 2, ICCAD13 rows | Our independent numerical-ILT baseline on all 10 cases, scored at 15 nm and 3 nm, with an iteration-budget curve at 25/50/100/150/200/300 |
| `train_scaled.yaml` | Sec. 3.3, Eqs. 5–10; Table 2 PT vs PT+RL | End-to-end two-stage training: WGAN-GP pretraining then GRPO finetuning |
| `abl_grpo_teacher_relative.yaml` | Eqs. 6–7 | **X-07a.** The advantage the paper *defines and argues for* in Sec. 3.3.2 |
| `abl_grpo_group_mean.yaml` | Sec. 4.1 | **X-07b.** The advantage Sec. 4.1 *says was actually used*. The paper contradicts itself (G-012); running both is the only way to say anything defensible about which matters. |
| `abl_optimizer.yaml` | G-015 (solver unspecified) | **X-14.** Fair Adam vs L-BFGS comparison on cases 1/3/5, at a common well-conditioned starting point and matched (recorded, not assumed) function-evaluation counts. See its header comment and `docs/findings.md` F-SAT-01 for why it does NOT reuse `iccad13_ilt.yaml`'s constants. |

## Reproducing the iteration-budget claim

The paper states its method reaches better quality "using roughly half of the
iteration budget" (150 vs the CurvyILT baseline's 300). `iccad13_ilt.yaml`
records scores at seven budgets in a single run, so the quality-vs-iterations
trade-off of *our* solver under *identical physics* can be read directly. That
does not verify the paper's claim about *its* solver — which is unspecified
(G-015) — but it does establish what the claim would have to beat.

## Note on tuning discipline

The ILT solver's `step_size` and `mask_steepness` were selected by a sweep over
{0.02, 0.05, 0.1, 0.2, 0.5, 1.0} × {1, 4, 8} on four cases at reduced
resolution, optimising **L2** — deliberately not one of the metrics used to
compare against the paper. Best: `step_size=0.2`, `mask_steepness=8.0`
(L2 7124 vs 24918 for no-OPC). No constant was adjusted after seeing a
comparison against the paper's tables.

## Note on comparing optimizers

`solver.py` also implements SGD, Nesterov, and L-BFGS (`ILTConfig.optimizer`).
"Iterations" is not the same unit of compute across optimizers -- L-BFGS's
strong-Wolfe line search evaluates the loss/gradient several times per outer
step, Adam/SGD/Nesterov exactly once -- so `ILTResult.n_func_evals` is recorded
on every run and is the number any comparison in this project actually cites.
Separately, F-SAT-01 (`docs/findings.md`) found that `iccad13_ilt.yaml`'s own
constants (`init_scale=2.0`, `mask_steepness=8.0`) saturate the mask sigmoid
badly enough that L-BFGS (like plain SGD) makes no measurable progress from
that starting point -- confirmed directly, not assumed. `abl_optimizer.yaml`
therefore uses its own common, well-conditioned starting point for every
optimizer it compares, rather than reusing the main baseline's.
