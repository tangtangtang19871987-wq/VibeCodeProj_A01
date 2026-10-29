# 实验脚本索引

`scripts/` 中每个脚本对应 `docs/traceability_matrix.md` 的一个实验编号（X-nn）
和 `docs/findings.md` 的一条结论（F-xxx）。可复用的算法放在 `src/gril/`，脚本只
负责实验协议：选哪些 case 和 recipe、给多少时间预算、怎么打分、结果写到哪里。

运行前提：

- 已执行 `bash scripts/fetch_data.sh`，或设置了 `GRIL_KERNEL_DIR` / `GRIL_BENCH_DIR`，
  见 `src/gril/paths.py`。
- 在仓库根目录运行，并设置 `PYTHONPATH=src`。

"耗时"一列是在本项目 4 核 CPU 上的量级，没有 GPU。

## 论文复现与 ILT 求解器（配置驱动的 runner 在 `src/gril/experiments/`）

| 实验 | 命令 | 输出 | 依赖 | 结论 | 耗时 |
|---|---|---|---|---|---|
| X-01/02 | `python -m gril.experiments.run_iccad13 --config configs/experiments/iccad13_ilt.yaml` | `results/iccad13_ilt/` | — | Adam 主基线 | 约 1–2 h |
| X-14 | `python -m gril.experiments.run_optimizer_ablation --config configs/experiments/abl_optimizer.yaml` | `results/abl_optimizer/` | — | F-LBFGS-01 | 约 1 h |
| X-15 | `python -m gril.experiments.run_multistart_ablation --config configs/experiments/multistart_ablation.yaml` | `results/multistart_ablation/` | 先跑 `train_scaled` | F-MULTISTART-01 | 小时级 |
| 训练 | `python -m gril.experiments.run_training --config configs/experiments/train_scaled.yaml` | `results/train_scaled/` | — | F-RL-01 | 小时级 |
| X-16 | `python -m gril.experiments.run_iccad13 --config configs/experiments/iccad13_ilt_lbfgs.yaml` | `results/iccad13_ilt_lbfgs/` | — | F-LBFGS-02 | 约 1 h |
| X-17 | `python -m gril.experiments.run_iccad13 --config configs/experiments/iccad13_ilt_lbfgs100.yaml` | `results/iccad13_ilt_lbfgs100/` | — | F-LBFGS-03 | 约 2 h |

## 单问题脚本

| 实验 | 脚本 | 输出 | 依赖 | 结论 |
|---|---|---|---|---|
| X-17 方向测试 | `ilt_improvement_directions.py --case 3` | `results/ilt_improvement_directions/` | — | F-LBFGS-03 |
| X-17 后续 | `ilt_push_limits.py` | `results/ilt_push_limits/`（容器重启中断，只有部分结果） | — | F-OPT-01 背景 |
| PVB 权重 | `pvb_weight_sweep.py` | `results/pvb_sweep/` | — | F-PVB-01 |
| EPE 权重 | `epe_weight_calibration.py` | `results/epe_aware_case3/` | — | F-EPE-01 |
| MRC 清理 | `mrc_cleanup_sweep.py` | `results/mrc_sweep/` | X-01 的掩模 | F-MRC-01 |
| X-18 | `optimizer_benchmark.py --stage all` | `results/optimizer_benchmark/` | — | F-OPT-01（约 3 h） |
| X-19 | `gn_full_baseline.py` | `results/iccad13_ilt_gn/` | X-17 的逐 case 耗时 | F-OPT-01（约 2 h） |

## 物理降阶 / recipe 快速验证（512²，ICCAD13 case 3 及其变体）

脚本之间有数据依赖，必须按下面的顺序运行。每个脚本都会跳过已经存在的结果文件，
可以断点续跑。

```
rom_recipe_study.py (X-20) ──> results/rom_study/p0.pt  (参考解)
   ├─> rom_diagnostics.py     (X-21)
   ├─> rom_warmstart_curve.py (X-22)
   └─> rom_pod_study.py       (X-23) ──> results/rom_pod/p_warm_*.pt  (case 3 真值)
          └─> rom_layout_transfer.py (X-24) ──> results/rom_layout_transfer/*_p_ref.pt, *_truth_*.pt
                 ├─> rom_basis_physics.py    (X-25)
                 └─> rom_ift_normal_field.py (X-26)
```

| 实验 | 脚本 | 问题 | 结论 | 耗时 |
|---|---|---|---|---|
| X-20 | `rom_recipe_study.py` | 线性化一次的算子降阶能否预测新 recipe | F-ROM-01：不能（25–33%） | 约 1 h |
| X-21 | `rom_diagnostics.py` | 瓶颈在子空间还是线性化 | F-ROM-02：在子空间 | 约 30 min，峰值内存约 8 GB |
| X-22 | `rom_warmstart_curve.py` | 热启动完整 GN 的收敛曲线（基线） | F-ROM-02 | 约 5 min |
| X-23 | `rom_pod_study.py` | 快照 POD | F-ROM-03：95%，2.8 s | 约 20 min |
| X-24 | `rom_layout_transfer.py` | 版图变化后的迁移 | F-ROM-04 | 约 1 h |
| X-25 | `rom_basis_physics.py` | 不用快照的几何基 | F-ROM-04 | 约 20 min |
| X-26 | `rom_ift_normal_field.py` | 隐函数预测 vs 法向位移场 | F-ROM-05：位移场 95–99% | 约 1 h |

`rom_recipe_study.py` 是这一组脚本的公共模块，其他脚本以 `base` 导入它。它定义了：

- 常量 `CASE`、`DOWNSAMPLE`、`BETA`、各项时间预算；
- 13 个 recipe 的列表 `RECIPES`；
- 函数 `load_target`、`recipe_setup`、`score`、`gn`。

算法本身在 `gril.ilt.recipe_rom`（POD、子空间 LM、法向位移场）和
`gril.ilt.reduced_order`（精确的 J^T J 谱、算子降阶）中。

## 报告与图

| 脚本 | 作用 |
|---|---|
| `make_report.py` | 从 `results/` 生成 `REPRODUCTION_REPORT.md`，论文表格作为数据内嵌在脚本中 |
| `make_figures.py` | 生成 `figures/*.png` |
| `fetch_data.sh` | 拉取 ICCAD13 核与版图，并校验 `data/CHECKSUMS.sha256` |
