"""Tests for gril.config.schema: pydantic validation of experiment configs.

The central regression guard here is drift detection: ILTConfigSchema mirrors
gril.ilt.solver.ILTConfig field-for-field, by hand, so it is easy for the two
to silently diverge when ILTConfig gains a new field. test_ilt_config_schema_
matches_dataclass compares the actual field sets and fails loudly if they
disagree, rather than letting the schema quietly go stale.
"""
import dataclasses

import pytest
from pydantic import ValidationError

from gril.config.schema import (
    ICCAD13ExperimentSchema,
    ILTConfigSchema,
    OptimizerAblationSchema,
    TrainingExperimentSchema,
    load_and_validate,
)
from gril.ilt.solver import ILTConfig


# -------------------------------------------------------- drift detection

def test_ilt_config_schema_matches_dataclass_field_names():
    dataclass_fields = {f.name for f in dataclasses.fields(ILTConfig)}
    schema_fields = set(ILTConfigSchema.model_fields.keys())
    missing_from_schema = dataclass_fields - schema_fields
    extra_in_schema = schema_fields - dataclass_fields
    assert not missing_from_schema, (
        f"ILTConfig gained field(s) not mirrored in ILTConfigSchema: {missing_from_schema}"
    )
    assert not extra_in_schema, (
        f"ILTConfigSchema has field(s) not in ILTConfig: {extra_in_schema}"
    )


def test_ilt_config_schema_defaults_match_dataclass_defaults():
    """Catch a subtler drift: same field names, but the default VALUE changed
    in one place and not the other.
    """
    schema_defaults = ILTConfigSchema().model_dump()
    dataclass_defaults = dataclasses.asdict(ILTConfig())
    # checkpoints: dataclass uses tuple, schema also coerces to tuple.
    for key in dataclass_defaults:
        assert schema_defaults[key] == dataclass_defaults[key], (
            f"default for {key!r} differs: schema={schema_defaults[key]!r} "
            f"dataclass={dataclass_defaults[key]!r}"
        )


def test_ilt_config_schema_values_actually_construct_an_ilt_config():
    """A validated schema instance's fields must be directly usable to build
    a real ILTConfig -- the whole point of validating before running.
    """
    schema = ILTConfigSchema(iterations=50, optimizer="lbfgs", beta_init=1.0)
    cfg = ILTConfig(**schema.model_dump())
    assert cfg.iterations == 50
    assert cfg.optimizer == "lbfgs"
    assert cfg.beta_init == 1.0


# ------------------------------------------------------------ strictness

def test_unknown_top_level_key_rejected():
    with pytest.raises(ValidationError, match="typo_key"):
        ICCAD13ExperimentSchema(
            experiment_id="x", kernel_dir="k", bench_dir="b", cases=[1], typo_key=1,
        )


def test_unknown_ilt_key_rejected():
    with pytest.raises(ValidationError):
        ILTConfigSchema(iterations=10, interations=20)  # typo of "iterations"


def test_bad_optimizer_name_rejected():
    with pytest.raises(ValidationError):
        ILTConfigSchema(optimizer="rmsprop")


def test_empty_cases_list_rejected():
    with pytest.raises(ValidationError, match="non-empty"):
        ICCAD13ExperimentSchema(
            experiment_id="x", kernel_dir="k", bench_dir="b", cases=[],
        )


def test_negative_iterations_rejected():
    with pytest.raises(ValidationError):
        ILTConfigSchema(iterations=-5)


def test_negative_step_size_rejected():
    with pytest.raises(ValidationError):
        ILTConfigSchema(step_size=-0.1)


# ------------------------------------------------------ real config files

@pytest.mark.parametrize(
    "path,schema",
    [
        ("configs/experiments/iccad13_ilt.yaml", ICCAD13ExperimentSchema),
        ("configs/experiments/train_scaled.yaml", TrainingExperimentSchema),
        ("configs/experiments/abl_optimizer.yaml", OptimizerAblationSchema),
        ("configs/experiments/abl_grpo_teacher_relative.yaml", TrainingExperimentSchema),
        ("configs/experiments/abl_grpo_group_mean.yaml", TrainingExperimentSchema),
    ],
)
def test_every_committed_config_validates(path, schema):
    """Every config actually checked into this repo must pass its schema --
    this is what would have caught a typo in a committed YAML file.
    """
    import os

    if not os.path.exists(path):
        pytest.skip(f"{path} not present in this checkout")
    validated, raw = load_and_validate(path, schema)
    assert validated.experiment_id == raw["experiment_id"]


def test_load_and_validate_raises_on_missing_required_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("experiment_id: x\nkernel_dir: k\n")  # missing bench_dir, cases
    with pytest.raises(ValidationError):
        load_and_validate(str(bad), ICCAD13ExperimentSchema)


def test_optimizer_ablation_rejects_typo_in_per_optimizer_override():
    with pytest.raises(ValidationError, match="optimizers.lbfgs"):
        OptimizerAblationSchema(
            experiment_id="x",
            kernel_dir="k",
            bench_dir="b",
            cases=[1],
            optimizers={"lbfgs": {"iteratons": 10}},  # typo
        )
