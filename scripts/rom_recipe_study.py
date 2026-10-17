#!/usr/bin/env python3
"""X-20: physics-based reduced-order model for fast ILT recipe evaluation.

Question: after one full ILT run at a reference recipe, can the optimised
result for a *changed* recipe (resist threshold, dose, target bias) be
predicted from an exact-physics reduced model, without re-running ILT?

Setup (fixed before any result): ICCAD13 case 3 at 512^2 (4 nm pixels; the
optics is resolution-independent). Reference P0 = Gauss-Newton cold start,
120 s, nominal recipe. 13 recipe perturbations. For each, candidate masks:

  frozen      P0 unchanged, printed under the new recipe (no re-optimisation)
  one_step    one full-space Gauss-Newton step from P0 under the new recipe
              (= first-order implicit-function sensitivity of the optimum)
  rom_r       ReducedILT with the top-r eigenvectors of J^T J at P0, optics
              linearised once, recipe physics exact; online = matvecs only
  warm        full GN re-optimisation from P0, 60 s           (reference truth)
  cold        full GN from scratch under the new recipe, 120 s (what a fresh
              ILT run would give; warm-vs-cold spread = irreducible ambiguity)

Every candidate is binarised and printed with the FULL forward model; the
reduced model is never used to score itself.

Usage::

    PYTHONPATH=src python scripts/rom_recipe_study.py
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
import torch.nn.functional as F

from gril.data.glp import Design
from gril.experiments.run_iccad13 import provenance
from gril.ilt.gauss_newton import ILTLeastSquares, solve_levenberg_marquardt
from gril.ilt.reduced_order import ReducedILT, gn_spectrum
from gril.litho.resist import LithoModel, ProcessConfig, resist
from gril.metrics.core import epe_violations, l2_loss, pv_band

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = "/home/user/openopc/openilt/benchmark/ICCAD2013"
KERNEL_DIR = "/home/user/openopc/openilt/kernel"
OUT = os.path.join(ROOT, "results/rom_study")
CASE, DOWNSAMPLE, PIXEL_NM = 3, 4, 4.0
BETA, INIT_SCALE = 8.0, 0.5
REF_BUDGET, WARM_BUDGET, COLD_BUDGET = 120.0, 60.0, 120.0
RANKS = [25, 50, 100, 200, 400]

BASE = ProcessConfig()
RECIPES = (
    [dict(name=f"threshold x{f}", threshold=BASE.target_density * f) for f in (0.95, 0.98, 1.02, 1.05, 1.10)]
    + [dict(name=f"dose {d}", dose=d) for d in (0.97, 0.99, 1.01, 1.03)]
    + [dict(name=f"bias {b:+d}px", bias=b) for b in (-2, -1, 1, 2)]
)


def biased(target: torch.Tensor, b: int) -> torch.Tensor:
    t = target[None, None]
    for _ in range(abs(b)):
        t = F.max_pool2d(t, 3, 1, 1) if b > 0 else -F.max_pool2d(-t, 3, 1, 1)
    return t[0, 0]


def recipe_setup(litho: LithoModel, target: torch.Tensor, rec: dict):
    d = rec.get("dose", 1.0)
    cfg = dataclasses.replace(BASE, target_density=rec.get("threshold", BASE.target_density),
                              dose_nom=d * BASE.dose_nom, dose_max=d * BASE.dose_max, dose_min=d * BASE.dose_min)
    lm = copy.copy(litho)
    lm.cfg = cfg
    return lm, biased(target, rec.get("bias", 0))


def score(lm: LithoModel, target: torch.Tensor, params: torch.Tensor, ref_print=None) -> dict:
    mask = (params >= 0).float()
    with torch.no_grad():
        b_nom, b_max, b_min = lm.binary(mask)
    ein, eout = epe_violations(b_nom, target, 15, PIXEL_NM)
    out = {"l2": l2_loss(b_nom, target), "pvb": pv_band(b_max, b_min), "epe@15nm": ein + eout}
    if ref_print is not None:
        out["xor_vs_warm"] = int((b_nom != ref_print).sum())
    return out, b_nom


def gn(lm, target, p_init, budget, iterations=100_000):
    p, hist, ran, work, _ = solve_levenberg_marquardt(target, lm, p_init, iterations=iterations, beta=BETA,
                                                      max_cg=10, time_budget_s=budget)
    return p, ran


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = LithoModel(KERNEL_DIR, BASE)
    full = torch.tensor(Design.from_glp(f"{BENCH_DIR}/M1_test{CASE}.glp").centred_raster(2048))
    target = (F.avg_pool2d(full[None, None].float(), DOWNSAMPLE)[0, 0] > 0.5).float()

    p0_path = os.path.join(OUT, "p0.pt")
    if os.path.exists(p0_path):
        p0 = torch.load(p0_path)
    else:
        t0 = time.time()
        p0, ran = gn(litho, target, INIT_SCALE * (2 * target - 1), REF_BUDGET)
        torch.save(p0, p0_path)
        print(f"[ref] GN cold {ran} it in {time.time() - t0:.0f}s", flush=True)

    model = ILTLeastSquares(target, litho, BETA)
    st = model.state(p0)
    t0 = time.time()
    spec = gn_spectrum(model, st, batch=16)
    t_spec = time.time() - t0
    lam = spec.eigenvalues.double()
    cum = (torch.cumsum(lam, 0) / lam.sum()).tolist()
    n_pix = target.numel()
    structure = {
        "pixels": n_pix, "band_dim": spec.w_mat.shape[1], "dropped": spec.dropped,
        "offline_seconds": t_spec, "f_ref": st.f,
        "trace_fraction": {str(r): cum[r - 1] for r in [1, 5, 10, 25, 50, 100, 200, 400, 800] if r <= len(cum)},
        "dims_for": {str(q): int((torch.tensor(cum) < q).sum()) + 1 for q in (0.9, 0.99, 0.999)},
        "eigenvalues_head": lam[:50].tolist(),
        "eigenvalues_all": lam.tolist(),
    }
    print(f"[structure] offline {t_spec:.0f}s  dims for 90/99/99.9% of trace: {structure['dims_for']}", flush=True)
    json.dump(structure, open(os.path.join(OUT, "structure.json"), "w"), indent=2)

    i0 = _intensity(litho, p0)
    roms = {r: ReducedILT(spec, r, p0, i0, BASE.dose_nom, BETA) for r in RANKS}

    # Control: continue optimising under the UNCHANGED recipe (how much does "truth" drift by itself?)
    ctrl_path = os.path.join(OUT, "control.json")
    if not os.path.exists(ctrl_path):
        p_ctrl, _ = gn(litho, target, p0, WARM_BUDGET)
        s0, b0 = score(litho, target, p0)
        s1, _ = score(litho, target, p_ctrl, ref_print=b0)
        json.dump({"reference": s0, "warm_same_recipe": s1}, open(ctrl_path, "w"), indent=2)

    for rec in RECIPES:
        path = os.path.join(OUT, f"recipe_{rec['name'].replace(' ', '_')}.json")
        if os.path.exists(path):
            continue
        lm, tgt = recipe_setup(litho, target, rec)
        row = {"recipe": rec}
        t0 = time.time()
        p_warm, _ = gn(lm, tgt, p0, WARM_BUDGET)
        row["warm"], warm_print = score(lm, tgt, p_warm)
        row["warm"]["seconds"] = time.time() - t0
        t0 = time.time()
        p_cold, _ = gn(lm, tgt, INIT_SCALE * (2 * tgt - 1), COLD_BUDGET)
        row["cold"], _ = score(lm, tgt, p_cold, warm_print)
        row["cold"]["seconds"] = time.time() - t0
        row["frozen"], _ = score(lm, tgt, p0, warm_print)
        t0 = time.time()
        p_one, _ = gn(lm, tgt, p0, 0.0, iterations=1)
        row["one_step"], _ = score(lm, tgt, p_one, warm_print)
        row["one_step"]["seconds"] = time.time() - t0
        for r, rom in roms.items():
            t0 = time.time()
            a, _ = rom.solve(lm.cfg.target_density, lm.cfg.print_steepness, lm.cfg.dose_nom, tgt)
            dt = time.time() - t0
            row[f"rom_{r}"], _ = score(lm, tgt, rom.params(a), warm_print)
            row[f"rom_{r}"]["seconds"] = dt
        json.dump(row, open(path, "w"), indent=2)
        print(f"[{rec['name']:15s}] L2 frozen {row['frozen']['l2']:.0f} | one {row['one_step']['l2']:.0f} | "
              + " ".join(f"r{r} {row[f'rom_{r}']['l2']:.0f}" for r in RANKS)
              + f" | warm {row['warm']['l2']:.0f} cold {row['cold']['l2']:.0f}", flush=True)

    json.dump({"protocol": __doc__, "recipes": RECIPES, "ranks": RANKS, "provenance": provenance()},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)


def _intensity(litho: LithoModel, p: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return litho.aerial_nominal(torch.sigmoid(BETA * p))


if __name__ == "__main__":
    main()
