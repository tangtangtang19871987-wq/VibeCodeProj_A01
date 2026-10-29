"""Shared fixtures. Physics tests need the public ICCAD13 assets."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _kernel_dir() -> str | None:
    from gril.paths import find_kernel_dir

    return find_kernel_dir()


def _bench_dir() -> str | None:
    from gril.paths import find_bench_dir

    return find_bench_dir()


requires_assets = pytest.mark.skipif(
    _kernel_dir() is None or _bench_dir() is None,
    reason="Public ICCAD13 assets missing; run scripts/fetch_data.sh",
)


@pytest.fixture(scope="session")
def kernel_dir() -> str:
    d = _kernel_dir()
    if d is None:
        pytest.skip("kernels missing; run scripts/fetch_data.sh")
    return d


@pytest.fixture(scope="session")
def bench_dir() -> str:
    d = _bench_dir()
    if d is None:
        pytest.skip("benchmarks missing; run scripts/fetch_data.sh")
    return d


@pytest.fixture(scope="session")
def focus_kernels(kernel_dir):
    from gril.litho.kernels import load_kernel_set

    return load_kernel_set(kernel_dir, defocus=False)
