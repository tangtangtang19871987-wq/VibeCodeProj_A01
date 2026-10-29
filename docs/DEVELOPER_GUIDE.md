# 开发者指南

本文面向要在这个仓库上继续开发、复现或扩展实验的人。项目背景与结论见
`docs/PROJECT_SUMMARY_zh.md`，复现状态见 `README_REPRODUCTION.md`，实验索引见
`scripts/README.md`。

## 1. 仓库里有两个项目

| 位置 | 内容 | 状态 |
|---|---|---|
| 根目录 `*.py`、`tests/`、`README.md` | 仓库原有的 `array_generator`（SRAF 检查用的周期阵列生成器） | 原样保留，没有修改 |
| `src/gril/`、`tests_gril/`、`scripts/`、`configs/`、`docs/`、`results/` | arXiv 2602.19027 的复现与延伸研究，即本指南的对象 | 主体 |

`pytest.ini` 只收集 `tests_gril/`。根目录的 `__init__.py` 依赖安装名
`array_generator`，所以用 `.pkgroot/array_generator` 符号链接加 `pythonpath` 让它能
被导入（F-ENV-01）。

## 2. 环境

```bash
python -m pip install -r requirements.txt   # 版本已固定；torch 2.9.1
bash scripts/fetch_data.sh                   # ICCAD13 核与版图 -> data/，带 sha256 校验
python -m pytest                             # 在仓库根目录运行
```

- **只需要 CPU。** 任何代码都不依赖 CUDA。如果 `download.pytorch.org` 不可达，可以直接
  从 PyPI 安装 `torch==2.9.1`。PyPI 上的是 CUDA 版，体积大，但在 CPU 上也能正常运行。
- **数据位置**由 `gril.paths` 统一解析，按以下顺序查找：
  1. 显式给出的路径；
  2. 环境变量 `GRIL_KERNEL_DIR` / `GRIL_BENCH_DIR`；
  3. `data/`；
  4. 开发时用过的旧路径 `/home/user/openopc/openilt`。

  YAML 配置里写 `kernel_dir: auto` 即走上述自动查找。不要在代码里硬编码数据路径。
- **线程数**：每个求解进程默认占满所有核。同时跑多个实验会严重超额订阅（实测负载 11+，
  4 核机器）。要么串行跑，要么每个进程调用 `torch.set_num_threads(N)`。

## 3. 测试

| 目录 | 内容 |
|---|---|
| `tests_gril/physics/` | SOCS 正向成像、伴随梯度、对称性；与纯 autograd 实现和有限差分交叉验证 |
| `tests_gril/regression/` | 10 个 ICCAD13 case 的无 OPC 基线，与参考检查器逐位一致 |
| `tests_gril/unit/` | 求解器、优化器、度量、EPE 损失、生成器、GRPO、配置校验、降阶模块 |

- 缺少数据时，依赖数据的测试会自动 skip，并提示运行 `fetch_data.sh`。
- `@pytest.mark.slow` 标记 2048² 全画布测试，可用 `-m "not slow"` 跳过。
- 测试纪律：容差从不放宽到让失败"消失"。每个失败都追到根因，写进 `docs/findings.md`。

## 4. 包结构

| 模块 | 职责 | 关键对象 |
|---|---|---|
| `gril.paths` | 数据路径解析 | `kernel_dir()`、`bench_dir()`、`glp_path(case)` |
| `gril.data.glp` | GLP 版图解析与光栅化（边界包含，与参考实现逐位一致） | `Design.from_glp(...).centred_raster(2048)` |
| `gril.litho.kernels` | 24 项 SOCS 核（焦面 / 离焦） | `load_kernel_set`、`KernelSet` |
| `gril.litho.socs` | Hopkins/SOCS 成像，自定义伴随（conj(H)） | `aerial_image`、`convolve` |
| `gril.litho.resist` | 常阈值 sigmoid 光刻胶、三个工艺角 | `ProcessConfig`（frozen）、`LithoModel` |
| `gril.metrics.core` | L2、PVB、EPE（与 ICCAD13 检查器逐位一致） | `l2_loss`、`pv_band`、`epe_violations`、`evaluate` |
| `gril.ilt.solver` | ILT 主入口；Adam、SGD、Nesterov、L-BFGS、SF-AdamW、Muon、SOAP、Gauss-Newton | `ILTConfig`、`solve`、`ilt_loss` |
| `gril.ilt.gauss_newton` | 无矩阵 Levenberg-Marquardt；解析 JVP/VJP | `ILTLeastSquares`、`solve_levenberg_marquardt` |
| `gril.ilt.optimizers` | 深度学习类优化器的 ILT 版本 | `ScheduleFreeAdamW`、`Muon`、`SOAP` |
| `gril.ilt.epe_loss` | 可微 EPE 采样点损失（独立实现，与度量交叉验证） | `build_epe_sites`、`soft_epe_loss` |
| `gril.ilt.reduced_order` | J 的精确低秩结构、J^T J 谱、算子降阶（已被证伪，保留作记录） | `band_basis`、`gn_spectrum`、`ReducedILT` |
| `gril.ilt.recipe_rom` | recipe 变化后的快速重优化 | `apply_recipe`、`pod_basis`、`solve_in_subspace`、`NormalDisplacementField`、`solve_in_normal_field` |
| `gril.models.generator` | 风格感知 U-Net 生成器、WGAN-GP 判别器 | `StyleAwareGenerator`、`Discriminator` |
| `gril.train.pretrain` / `grpo` | 第一阶段 WGAN-GP，第二阶段 GRPO（两种 advantage 都实现了，G-012） | `critic_step`、`finetune_step` |
| `gril.experiments.*` | 配置驱动的 runner，可断点续跑 | `run_iccad13`、`run_training`、…… |
| `gril.config.schema` | pydantic 配置校验，拼错的键会直接报错 | `load_and_validate` |

## 5. 物理与数值约定（违反这些约定的 bug 都真实发生过）

1. **画布与像素**：ICCAD13 是 2048² 像素、1 nm/px，所以 nm 和 px 恰好相等。
   **降采样后两者不再相等**：EPE 常数是 nm，必须换算成像素（F-UNIT-01）。降阶实验用
   512²、4 nm/px（`PIXEL_NM = 4.0`）。
2. **FFT 约定**：`norm="forward"`，35×35 的核放在频谱四个角上，不做 shift（F-03）。
3. **伴随**：`IFFT∘diag(H)∘FFT` 的伴随是 conj(H) 卷积。仓库附带的 `ct_*` 核**不是**
   伴随，公开参考实现用了它，梯度错约 8.7%（F-ADJ-01）。改动成像代码后，必须重跑
   `tests_gril/physics/test_adjoint.py`。
4. **对称性**：核在列方向对称、行方向不对称。y 方向镜像等变性**不成立**，不要写这样的
   测试（F-KER-01）。
5. **掩模参数化**：`M = sigmoid(β·P)`，初值 `P0 = init_scale·(2·target − 1)`。主基线的
   `init_scale=2, β=8` 会让 sigmoid 饱和，只有 Adam 推得动（F-SAT-01）。对 L-BFGS 和
   GN，要么用 `init_scale=0.5`，要么对 β 退火（`beta_init`，F-ANNEAL-01）。
6. **Gauss-Newton 只支持纯名义 L2 目标。** 设置了 `weight_pvb`、`weight_tv`、`weight_epe`
   或 `beta_init` 会直接报错，不会被悄悄忽略。
7. **精度**：physics 测试用 float64。实验用 float32。但 `gn_spectrum` 的 Gram 矩阵必须
   用 float64 分块累加：float32 会扭曲谱的尾部，不分块会内存溢出。
8. **打分只用完整模型**：任何降阶或代理模型都不能给自己打分。掩模先二值化（`P ≥ 0`），
   再用完整模型成像、计算度量。

## 6. 常用 API

**ILT 求解：**

```python
from gril.data.glp import Design
from gril.ilt.solver import ILTConfig, solve
from gril.litho.resist import LithoModel, ProcessConfig
from gril.paths import glp_path, kernel_dir
import torch

litho = LithoModel(kernel_dir(), ProcessConfig())
target = torch.tensor(Design.from_glp(glp_path(3)).centred_raster(2048))
res = solve(target, litho, ILTConfig(optimizer="gauss_newton", init_scale=0.5, mask_steepness=8.0,
                                     iterations=10_000, time_budget_s=600))
mask = res.mask        # 二值掩模；完整字段见 ILTResult
```

**recipe 变化后的快速重优化**（X-26 的方法；不需要快照）：

```python
from gril.ilt.recipe_rom import NormalDisplacementField, apply_recipe, solve_in_normal_field

lm, tgt = apply_recipe(litho, target, litho.cfg, threshold=0.95 * litho.cfg.target_density, dose=1.0, bias=0)
field = NormalDisplacementField(p_ref, spacing=16)          # 16 px 约等于光学分辨率（在 512²、4 nm/px 下）
p_new, _ = solve_in_normal_field(tgt, lm, p_ref, field, beta=8.0, iterations=5)
```

`spacing` 要和光学分辨率 λ/(2NA) ≈ 71 nm 相当。网格更粗时，恢复率上限约 91%
（F-ROM-05）。

## 7. 新增一个实验

1. **先写协议，再看结果。** 在脚本的 docstring 里写清楚：问题、case 和 recipe、预算、
   对照组、打分方式、成功判据。这些都要在运行前固定下来。
2. **公平性**：
   - 优化器按墙钟时间匹配（`time_budget_s`）；
   - 调参只看 L2，**从不看用来比较的指标**（EPE）；
   - 每次只改一个变量；
   - 一定要有对照组，例如随机多起点、单分辨率、冻结掩模。
3. **结果格式**：
   - 写原始 JSON，带上 `provenance()`（git commit、torch 版本、线程数）；
   - 可断点续跑：已有结果直接跳过；
   - 大文件（真值掩模、检查点）能复现就不要提交。
4. **代码位置**：可复用的算法放进 `src/gril/` 并配测试；实验协议留在 `scripts/` 或
   `configs/`。
5. **登记**：
   - `docs/traceability_matrix.md` 加一行 X-nn；
   - `docs/findings.md` 加一条 F-xxx：设置、结果表、解读、**这个结果不能说明什么**；
   - 在 `scripts/README.md` 和 `results/README.md` 加上运行命令。
6. **负结果照样记录。** 被证伪的假设要明确写出来，例如 F-ROM-05 的路线 A；被认为平凡的
   结论要降级，例如 F-ROM-04。

## 8. 长任务

- 用 `nohup ... > log.txt 2>&1 &` 在后台运行，脚本里 `print(..., flush=True)`，每完成一个
  单元就写一次 JSON。
- 容器可能重启。所以中间结果要落盘，脚本也要能跳过已完成的部分。

## 9. 已知遗留

- `scripts/ilt_push_limits.py` 的结果不完整（容器重启中断），只有
  `results_partial.json`。
- `tests_gril/unit/test_ilt_annealing.py` 里有一个未使用的局部变量（pyflakes 提示），
  不影响测试。
- 法向位移场在 b 空间里每步耗时约为全空间的 3 倍（CG 收敛慢），还没有加预条件。
