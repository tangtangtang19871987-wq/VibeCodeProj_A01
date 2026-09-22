"""Seeded synthetic M1-like layout generator.

**Why this exists.** The paper pretrains on LithoBench MetalSet (14,824) and
ViaSet (104,773). LithoBench is unreachable under this session's egress policy
(G-041), so training data is substituted by this generator. It is *not* a
reproduction of LithoBench and any number derived from it is reported as such.

Layouts are Manhattan rectangles on a fixed pitch with CDs in the 45-90 nm range,
matching the ICCAD13 M1 style. Deterministic given a seed.
"""

from __future__ import annotations

import numpy as np


def random_layout(
    size: int = 256,
    pixel_nm: float = 8.0,
    rng: np.random.Generator | None = None,
    n_shapes: tuple[int, int] = (3, 10),
    cd_nm: tuple[float, float] = (45.0, 90.0),
    length_nm: tuple[float, float] = (120.0, 700.0),
    margin_nm: float = 250.0,
) -> np.ndarray:
    """Generate one binary layout.

    Parameters
    ----------
    size:
        Canvas edge in pixels.
    pixel_nm:
        Physical pixel pitch, nm. ``size * pixel_nm`` should equal 2048 nm so the
        optical kernels remain valid (see the multi-resolution physics test).

    Returns
    -------
    np.ndarray
        ``(size, size)`` float32 in ``{0,1}``.
    """
    rng = rng or np.random.default_rng(0)
    canvas = np.zeros((size, size), dtype=np.float32)
    margin = int(margin_nm / pixel_nm)
    lo, hi = margin, size - margin
    if hi <= lo:
        raise ValueError("margin too large for canvas")

    for _ in range(int(rng.integers(*n_shapes))):
        cd = max(1, int(rng.uniform(*cd_nm) / pixel_nm))
        length = max(1, int(rng.uniform(*length_nm) / pixel_nm))
        horizontal = bool(rng.integers(0, 2))
        h, w = (cd, length) if horizontal else (length, cd)
        if h >= hi - lo or w >= hi - lo:
            continue
        y = int(rng.integers(lo, hi - h))
        x = int(rng.integers(lo, hi - w))
        canvas[y : y + h, x : x + w] = 1.0
    return canvas


def layout_dataset(n: int, seed: int = 0, **kwargs) -> np.ndarray:
    """Deterministic stack of ``n`` layouts, shape ``(n, size, size)``."""
    rng = np.random.default_rng(seed)
    return np.stack([random_layout(rng=rng, **kwargs) for _ in range(n)])
