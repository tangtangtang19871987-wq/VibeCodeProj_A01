"""Where the public ICCAD13 assets (SOCS kernels, M1 layouts) live.

Resolution order, first hit wins:

1. an explicit path (a config value or argument) that exists;
2. the environment variables ``GRIL_KERNEL_DIR`` / ``GRIL_BENCH_DIR``;
3. ``<repo>/data/``, which ``scripts/fetch_data.sh`` populates (checksummed);
4. ``/home/user/openopc/openilt/``, the checkout used while this project was
   developed (kept so that old results' recorded paths still resolve).

A config value of ``auto`` (or empty) skips step 1. Nothing is ever silently
substituted for a path that was given explicitly but does not exist *and* has
no fallback: :func:`kernel_dir` / :func:`bench_dir` raise with the fix
(``scripts/fetch_data.sh``) in the message.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LEGACY = "/home/user/openopc/openilt"


def _first(candidates, ok) -> str | None:
    for c in candidates:
        if c and ok(c):
            return c
    return None


def _is_kernel_dir(d: str) -> bool:
    return os.path.isdir(os.path.join(d, "kernels")) and os.path.isdir(os.path.join(d, "scales"))


def _is_bench_dir(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "M1_test1.glp"))


def _explicit(configured: str | None) -> str | None:
    return None if configured in (None, "", "auto") else configured


def find_kernel_dir(configured: str | None = None) -> str | None:
    return _first((_explicit(configured), os.environ.get("GRIL_KERNEL_DIR"),
                   os.path.join(ROOT, "data", "kernel"), os.path.join(_LEGACY, "kernel")), _is_kernel_dir)


def find_bench_dir(configured: str | None = None) -> str | None:
    return _first((_explicit(configured), os.environ.get("GRIL_BENCH_DIR"),
                   os.path.join(ROOT, "data", "benchmark", "ICCAD2013"),
                   os.path.join(_LEGACY, "benchmark", "ICCAD2013")), _is_bench_dir)


def kernel_dir(configured: str | None = None) -> str:
    d = find_kernel_dir(configured)
    if d is None:
        raise FileNotFoundError("ICCAD13 SOCS kernels not found; run scripts/fetch_data.sh "
                                "or set GRIL_KERNEL_DIR")
    return d


def bench_dir(configured: str | None = None) -> str:
    d = find_bench_dir(configured)
    if d is None:
        raise FileNotFoundError("ICCAD13 layouts not found; run scripts/fetch_data.sh or set GRIL_BENCH_DIR")
    return d


def glp_path(case: int, configured: str | None = None) -> str:
    return os.path.join(bench_dir(configured), f"M1_test{case}.glp")
