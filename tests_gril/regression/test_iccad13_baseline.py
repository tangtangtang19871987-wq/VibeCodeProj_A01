"""Regression test: the no-OPC ICCAD13 baseline must never drift.

These numbers were produced by this repository AND independently verified to be
bit-identical to the public reference checker (OpenILT ``pyilt/evaluation.py``)
on all ten contest cases. They pin the entire stack: GLP parsing, rasterisation,
SOCS imaging, the three process corners, and all three metrics.

Full-canvas (2048x2048) evaluation costs ~6.5 s per case on 4 CPU cores, so the
whole module is marked ``slow``.
"""
import os

import pytest
import torch

from gril.data.glp import Design
from gril.litho.resist import LithoModel, ProcessConfig
from gril.metrics.core import evaluate

pytestmark = pytest.mark.slow

#: case -> (L2, PVBand, EPE) at the contest tolerance of 15 nm, mask == target.
EXPECTED = {
    1: (116184, 45874, 86),
    2: (117802, 37036, 84),
    3: (160846, 32646, 125),
    4: (84037, 101, 64),
    5: (117516, 59188, 71),
    6: (110523, 50684, 66),
    7: (103219, 54316, 71),
    8: (55012, 19084, 37),
    9: (120211, 60796, 66),
    10: (41291, 15039, 26),
}


@pytest.fixture(scope="module")
def litho(kernel_dir):
    return LithoModel(kernel_dir, ProcessConfig())


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_no_opc_baseline(case, bench_dir, litho):
    target = torch.tensor(
        Design.from_glp(os.path.join(bench_dir, f"M1_test{case}.glp")).centred_raster(2048)
    )
    scores = evaluate(target.clone(), target, litho, tolerance=15)
    exp_l2, exp_pvb, exp_epe = EXPECTED[case]
    assert scores.l2 == exp_l2
    assert scores.pvb == exp_pvb
    assert scores.epe == exp_epe


def test_case4_does_not_print_without_opc(bench_dir, litho):
    """test4's 64-65 nm bars peak below the 0.225 resist threshold.

    Physically correct, and the reason its L2 exactly equals the target area:
    nothing prints at all. Pinned so a future normalisation change is noticed.
    """
    target = torch.tensor(
        Design.from_glp(os.path.join(bench_dir, "M1_test4.glp")).centred_raster(2048)
    )
    i_nom, _, _ = litho.aerial(target)
    assert i_nom.max().item() < litho.cfg.target_density
    assert i_nom.max().item() == pytest.approx(0.2168, abs=2e-3)
    b_nom, _, _ = litho.binary(target)
    assert b_nom.sum().item() == 0
