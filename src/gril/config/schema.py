"""Pydantic schemas for experiment YAML configs.

Every config-driven runner (`gril.experiments.run_iccad13`,
`run_training`, `run_optimizer_ablation`) validates its loaded YAML against a
schema here BEFORE using it, so a typo'd key, a wrong type, or an unknown
field fails fast with a clear message at startup instead of a cryptic
KeyError or silently-wrong behaviour partway through a multi-hour CPU run.

Design choice: validation is a GATE, not a rewrite of the runners' internals.
Each `load_and_validate_*` function returns the validated model AND the
original plain dict (`.model_dump()`-compatible), so existing runner code
that does `cfg["ilt"]`, `cfg.get(...)` etc. keeps working unchanged -- only
the loading step changes. `extra="forbid"` on every model means an unknown key
(e.g. a typo like `interations` instead of `iterations`) is rejected rather
than silently ignored.

`ILTConfigSchema` mirrors `gril.ilt.solver.ILTConfig` field-for-field; a
dedicated test (`tests_gril/unit/test_config_schema.py`) checks the two stay
in sync, so adding a field to one without the other is caught immediately
rather than letting the schema quietly go stale.
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    """Base class: reject unknown fields, disallow silent typos."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------- shared blocks

class ProcessConfigSchema(_Strict):
    """Mirrors `gril.litho.resist.ProcessConfig`."""

    target_density: float = 0.225
    print_thresh: float = 0.5
    print_steepness: float = 50.0
    dose_nom: float = 1.00
    dose_max: float = 1.02
    dose_min: float = 0.98
    num_kernels: int = Field(24, gt=0)


class ILTConfigSchema(_Strict):
    """Mirrors `gril.ilt.solver.ILTConfig` field-for-field.

    See `test_ilt_config_schema_matches_dataclass` for the drift check.
    """

    iterations: int = Field(150, gt=0)
    step_size: float = Field(1.0, gt=0)
    mask_steepness: float = Field(4.0, gt=0)
    weight_nominal: float = Field(1.0, ge=0)
    weight_pvb: float = Field(0.0, ge=0)
    weight_tv: float = Field(0.0, ge=0)
    init_scale: float = 2.0
    optimizer: Literal["adam", "sgd", "nesterov", "lbfgs"] = "adam"
    momentum: float = Field(0.0, ge=0, lt=1)
    grad_clip: float = Field(0.0, ge=0)
    early_stop_rtol: float = Field(0.0, ge=0)
    convergence_grad_tol: float = Field(0.0, ge=0)
    mrc_open_size: int = Field(0, ge=0)
    checkpoints: tuple[int, ...] = ()
    weight_epe: float = Field(0.0, ge=0)
    epe_tolerance_nm: float = Field(15.0, gt=0)
    epe_margin: float = Field(0.0, ge=0)
    beta_init: float | None = Field(None, gt=0)
    beta_schedule: Literal["linear", "exponential"] = "linear"
    lbfgs_history_size: int = Field(10, gt=0)
    lbfgs_max_iter_per_step: int = Field(1, gt=0)
    lbfgs_line_search: str | None = "strong_wolfe"
    seed: int = 0

    @field_validator("checkpoints", mode="before")
    @classmethod
    def _coerce_checkpoints(cls, v):
        return tuple(v) if v is not None else ()


# ------------------------------------------------------------ ICCAD13 runner

class ICCAD13ExperimentSchema(_Strict):
    """`configs/experiments/iccad13_ilt.yaml` and similar."""

    experiment_id: str
    paper_reference: str = ""
    seed: int = 0
    threads: int = Field(4, gt=0)
    canvas: int = Field(2048, gt=0)
    kernel_dir: str
    bench_dir: str
    cases: list[int]
    epe_tolerances: list[float] = [15, 3]
    process: ProcessConfigSchema = Field(default_factory=ProcessConfigSchema)
    ilt: ILTConfigSchema = Field(default_factory=ILTConfigSchema)
    output_root: str = "results"
    overwrite: bool = False

    @field_validator("cases")
    @classmethod
    def _cases_nonempty(cls, v):
        if not v:
            raise ValueError("cases must be a non-empty list of ICCAD13 case numbers")
        return v


# --------------------------------------------------------- optimizer ablation

class OptimizerAblationSchema(_Strict):
    """`configs/experiments/abl_optimizer.yaml`."""

    experiment_id: str
    paper_reference: str = ""
    threads: int = Field(4, gt=0)
    canvas: int = Field(2048, gt=0)
    kernel_dir: str
    bench_dir: str
    cases: list[int]
    epe_tolerances: list[float] = [15, 3]
    process: ProcessConfigSchema = Field(default_factory=ProcessConfigSchema)
    shared_ilt: dict = Field(default_factory=dict)
    optimizers: dict[str, dict] = Field(..., min_length=1)
    output_root: str = "results"
    overwrite: bool = False

    @field_validator("optimizers")
    @classmethod
    def _each_optimizer_is_valid_ilt_config(cls, v):
        # Each entry must be a subset of ILTConfigSchema's fields; validated
        # by actually constructing one (catches typos in optimizer-specific
        # overrides too, not just the shared block).
        for name, overrides in v.items():
            try:
                ILTConfigSchema(**overrides)
            except Exception as exc:  # re-raise with which entry failed
                raise ValueError(f"optimizers.{name}: {exc}") from exc
        return v


# ------------------------------------------------------------ training runner

class GeneratorConfigSchemaBlock(_Strict):
    latent_dim: int = Field(256, gt=0)
    style_dim: int = Field(256, gt=0)
    base_channels: int = Field(32, gt=0)
    n_style_blocks: int = Field(4, gt=0)
    n_local_blocks: int = Field(2, gt=0)
    mapping_layers: int = Field(4, gt=0)
    head_downsample: int = Field(1, gt=0)
    style_scale: float = Field(1.0, gt=0)


class DiscriminatorConfigSchemaBlock(_Strict):
    base_channels: int = Field(32, gt=0)
    levels: int = Field(4, gt=0)


class PretrainConfigSchemaBlock(_Strict):
    epochs: int = Field(50, gt=0)
    batch_size: int = Field(16, gt=0)
    lr_discriminator: float = Field(2e-4, gt=0)
    betas_discriminator: tuple[float, float] = (0.5, 0.999)
    lr_generator: float = Field(1e-4, gt=0)
    n_critic: int = Field(5, gt=0)
    lambda_rec: float = Field(100.0, ge=0)
    lambda_gp: float = Field(10.0, ge=0)
    rec_loss: Literal["l1", "l2"] = "l1"
    latent_dim: int = Field(256, gt=0)


class RefineConfigSchemaBlock(_Strict):
    downsample: int = Field(8, gt=0)
    iterations: int = Field(100, gt=0)
    step_size: float = Field(0.2, gt=0)
    mask_steepness: float = Field(8.0, gt=0)


class GRPOConfigSchemaBlock(_Strict):
    group_size: int = Field(16, gt=0)
    latent_dim: int = Field(256, gt=0)
    lambda_pg: float = Field(500.0, ge=0)
    lambda_imit: float = Field(1.0, ge=0)
    smoothing_kernel: int = Field(25, gt=0)
    advantage: Literal["teacher_relative", "group_mean"] = "teacher_relative"
    logprob_reduction: Literal["mean", "sum"] = "mean"
    epe_tolerance: float = Field(3.0, gt=0)


class TrainingExperimentSchema(_Strict):
    """`configs/experiments/train_scaled.yaml`."""

    experiment_id: str
    paper_reference: str = ""
    seed: int = 0
    threads: int = Field(4, gt=0)
    kernel_dir: str
    canvas: int = Field(256, gt=0)
    pixel_nm: float = Field(8.0, gt=0)
    n_train: int = Field(..., gt=0)
    n_val: int = Field(..., gt=0)
    gt_iterations: int = Field(100, gt=0)
    epe_tolerance: float = Field(3.0, gt=0)
    eval_group_size: int = Field(8, gt=0)
    log_every: int = Field(5, gt=0)
    grad_clip: float = Field(1.0, ge=0)
    generator: GeneratorConfigSchemaBlock = Field(default_factory=GeneratorConfigSchemaBlock)
    discriminator: DiscriminatorConfigSchemaBlock = Field(default_factory=DiscriminatorConfigSchemaBlock)
    pretrain: PretrainConfigSchemaBlock = Field(default_factory=PretrainConfigSchemaBlock)
    refine: RefineConfigSchemaBlock = Field(default_factory=RefineConfigSchemaBlock)
    grpo: GRPOConfigSchemaBlock = Field(default_factory=GRPOConfigSchemaBlock)
    rl_steps: int = Field(..., gt=0)
    rl_lr: float = Field(..., gt=0)
    rl_lr_min: float = Field(..., gt=0)
    output_root: str = "results"
    overwrite: bool = False


# ------------------------------------------------------------------- loaders

def load_and_validate(path: str, schema: type[BaseModel]) -> tuple[BaseModel, dict]:
    """Load a YAML config and validate it against `schema`.

    Returns
    -------
    (validated_model, raw_dict)
        `raw_dict` is the original parsed YAML (validated, but still a plain
        dict) -- pass THIS to the existing runner code, which indexes with
        `cfg["key"]` throughout. `validated_model` is available for callers
        that want typed attribute access instead.

    Raises
    ------
    pydantic.ValidationError
        With a clear, field-by-field message, before any experiment work runs.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    validated = schema(**raw)
    return validated, raw
