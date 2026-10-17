#!/usr/bin/env python3
"""X-21: why does the single-linearisation reduced model recover only part
of the gap? Separates the two candidate causes on X-20's recipes.

  D1 subspace adequacy  Project the TRUE optimal change (warm re-optimisation
                        minus P0) onto the top-r eigenvectors / the whole
                        1225-dim band range, and print the projected mask.
                        If this is close to truth, the subspace is not the
                        bottleneck.
  D2 chord ROM          Same r-dim subspace and frozen optics Jacobian, but
                        the residual is evaluated with the FULL forward model
                        and every step must lower the TRUE objective (an
                        implicit trust region). Online cost: one FFT forward
                        per step.
  D4 iterated ROM       Re-linearise: recompute the exact spectrum at the
                        current point, solve the fully-reduced problem, repeat
                        (3 outer steps). Measures how fast relinearisation
                        closes the gap, and at what offline cost.

Reference P0 and protocol constants are shared with scripts/rom_recipe_study.py.

Usage::

    PYTHONPATH=src python scripts/rom_diagnostics.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

import rom_recipe_study as base
from gril.ilt.gauss_newton import ILTLeastSquares
from gril.ilt.reduced_order import ReducedILT, gn_spectrum

OUT = os.path.join(base.ROOT, "results/rom_diagnostics")
PICK = ["threshold x0.95", "threshold x1.05", "dose 0.97", "bias -1px", "bias +1px"]
RANKS = [25, 100, 400]


def chord_rom(lm, tgt, v, k, p0, iterations=20, damping=1.0):
    """Frozen K_r, but true residual and true-objective step acceptance."""
    r = v.shape[1]
    model = ILTLeastSquares(tgt, lm, base.BETA)
    scale = (lm.cfg.dose_nom / base.BASE.dose_nom) ** 2
    a = torch.zeros(r)
    st = model.state(p0)
    lam, nu = damping, 2.0
    eye = torch.eye(r)
    for _ in range(iterations):
        jac = (st.s_z.reshape(-1) * scale).unsqueeze(1) * k
        g = jac.T @ st.r.reshape(-1)
        step = torch.linalg.solve(jac.T @ jac + lam * eye, -g)
        pred = -(2 * float(g @ step) + float((jac @ step).pow(2).sum()))
        trial = model.state((p0.reshape(-1) + v @ (a + step)).reshape(p0.shape))
        rho = (st.f - trial.f) / pred if pred > 0 else -1.0
        if rho > 1e-4:
            a, st = a + step, trial
            lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
            nu = 2.0
        else:
            lam *= nu
            nu *= 2.0
    return st.p


def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    full = torch.tensor(base.Design.from_glp(f"{base.BENCH_DIR}/M1_test{base.CASE}.glp").centred_raster(2048))
    target = (base.F.avg_pool2d(full[None, None].float(), base.DOWNSAMPLE)[0, 0] > 0.5).float()
    p0 = torch.load(os.path.join(base.OUT, "p0.pt"))
    model0 = ILTLeastSquares(target, litho, base.BETA)
    spec0 = gn_spectrum(model0, model0.state(p0), batch=16)
    q_all = spec0.vectors(spec0.eigenvalues.numel())
    vk = {r: (spec0.vectors(r), spec0.k_reduced(r)) for r in RANKS}
    del spec0  # keep only what is used, so D4's fresh spectra fit in memory

    for rec in [r for r in base.RECIPES if r["name"] in PICK]:
        path = os.path.join(OUT, f"{rec['name'].replace(' ', '_')}.json")
        if os.path.exists(path):
            continue
        lm, tgt = base.recipe_setup(litho, target, rec)
        p_warm, _ = base.gn(lm, tgt, p0, base.WARM_BUDGET)
        row = {"recipe": rec}
        row["warm"], warm_print = base.score(lm, tgt, p_warm)
        row["frozen"], _ = base.score(lm, tgt, p0, warm_print)
        delta = (p_warm - p0).reshape(-1)

        for r in RANKS + ["band"]:
            q = q_all if r == "band" else vk[r][0]
            proj = q @ (q.T @ delta)
            s, _ = base.score(lm, tgt, (p0.reshape(-1) + proj).reshape(p0.shape), warm_print)
            s["captured_norm_fraction"] = float(proj.norm() / delta.norm())
            row[f"D1_project_{r}"] = s

        for r in RANKS:
            t0 = time.time()
            p = chord_rom(lm, tgt, *vk[r], p0)
            s, _ = base.score(lm, tgt, p, warm_print)
            s["seconds"] = time.time() - t0
            row[f"D2_chord_{r}"] = s

        p, t_total = p0.clone(), 0.0
        for outer in range(3):
            t0 = time.time()
            m = ILTLeastSquares(tgt, lm, base.BETA)
            st = m.state(p)
            spec = gn_spectrum(m, st, batch=16)
            with torch.no_grad():
                i_cur = lm.aerial_nominal(torch.sigmoid(base.BETA * p))
            rom = ReducedILT(spec, 100, p, i_cur, lm.cfg.dose_nom, base.BETA)
            a, _ = rom.solve(lm.cfg.target_density, lm.cfg.print_steepness, lm.cfg.dose_nom, tgt)
            p = rom.params(a)
            del spec, rom  # each spectrum is ~2.6 GB at 512^2; never hold two at once
            t_total += time.time() - t0
            s, _ = base.score(lm, tgt, p, warm_print)
            s["cumulative_seconds"] = t_total
            row[f"D4_iterated_{outer + 1}"] = s

        json.dump(row, open(path, "w"), indent=2)
        print(f"[{rec['name']:15s}] frozen {row['frozen']['l2']:.0f} warm {row['warm']['l2']:.0f} | "
              f"proj r25/100/400/band " + "/".join(f"{row[f'D1_project_{r}']['l2']:.0f}" for r in RANKS + ['band'])
              + " | chord " + "/".join(f"{row[f'D2_chord_{r}']['l2']:.0f}" for r in RANKS)
              + " | iter " + "/".join(f"{row[f'D4_iterated_{i}']['l2']:.0f}" for i in (1, 2, 3)), flush=True)


if __name__ == "__main__":
    main()
