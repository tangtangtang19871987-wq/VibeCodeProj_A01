#!/usr/bin/env python3
"""X-26: two snapshot-free routes to the recipe response, tested separately.

The literature review found no ILT/OPC work on either of these. Both use only
the reference solution and the exact model -- no snapshots of other recipes.

Route A -- implicit-function-theorem (IFT) / Newton predictor.
  The reference P* satisfies grad F(P*; theta0) = 0. To first order the new
  optimum is  P*(theta) = P* - H^-1 [grad F(P*; theta) - grad F(P*; theta0)].
  X-21's operator ROM failed because the optimal change lies largely in J's
  null space at P*, where the Gauss-Newton matrix J^T J is zero. The FULL
  Hessian adds the residual-curvature term sum_i r_i grad^2 r_i, which is not
  confined to range(J^T); the hypothesis is that it supplies the missing
  null-space motion. Variants (Hessian-vector products by double backward,
  validated against finite differences to 1e-8):
    ift_gn       H = 2 J^T J at (P*, theta0), rhs = grad difference
    ift_full     H = full Hessian at (P*, theta0), rhs = grad difference
    newton_full  H = full Hessian at (P*, theta), rhs = grad F(P*; theta);
                 1 and 3 steps (the 3-step version is a corrector)
  Each linear solve is CG on (H + lam I), stopping at negative curvature,
  with lam from {0.1, 1, 10} and step length from {0.25, ..., 4} chosen by
  the NEW recipe's own objective (never by the truth).

Route B -- spatially varying normal edge displacement.
  In level-set terms an edge moved by b(x) along its normal is
  dP = b(x) |grad P*|. X-25's single mode was b = const (87%). Here b is a
  bilinear field on a coarse grid of spacing s (px at 512^2, 4 nm/px):
  s = global (1 dim), 64, 32, 16, 8 px (81 ... 4225 dims). Gauss-Newton on the
  full model in that span (matrix-free CG in b-space, cost per CG step
  independent of the dimension), with recovery checkpoints at 1..20 steps,
  against full-space warm GN at the same steps. Also the projection ceiling:
  the best b-field fit to the true change.

Same 24 (layout, recipe) pairs, truths and recovery metric as X-25.

Usage::

    PYTHONPATH=src python scripts/rom_ift_normal_field.py
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
from gril.ilt.gauss_newton import ILTLeastSquares, solve_levenberg_marquardt
from gril.ilt.recipe_rom import NormalDisplacementField, solve_in_normal_field
from gril.litho.socs import convolve
from rom_layout_transfer import PICK, layouts, load_target

OUT = os.path.join(base.ROOT, "results/rom_ift_normal")
POD_DIR = os.path.join(base.ROOT, "results/rom_pod")
LT_DIR = os.path.join(base.ROOT, "results/rom_layout_transfer")
SPACINGS = ["global", 64, 32, 16, 8]
CHECKPOINTS = (1, 2, 3, 5, 8, 12, 20)
LAMS = (0.1, 1.0, 10.0)
ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
CG_ITERS = 30


# ---------------------------------------------------------------- route A

def objective(model: ILTLeastSquares, p: torch.Tensor) -> torch.Tensor:
    m = torch.sigmoid(model.beta * p)
    e = convolve((model.dose * m).to(model.cdtype).unsqueeze(0), model.kern, model.n)[0]
    z = torch.sigmoid(model.gamma * ((model.w * (e.real ** 2 + e.imag ** 2)).sum(0) - model.theta))
    return ((z - model.t) ** 2).sum()


class Curvature:
    """Full Hessian (double backward) and Gauss-Newton products of F at one point."""

    def __init__(self, model: ILTLeastSquares, p: torch.Tensor):
        self.model, self.st = model, model.state(p)
        self.p = p.detach().clone().requires_grad_()
        self.g = torch.autograd.grad(objective(model, self.p), self.p, create_graph=True)[0]

    def full(self, v):
        return torch.autograd.grad(self.g, self.p, v, retain_graph=True)[0]

    def gn(self, v):
        return 2.0 * self.model.vjp(self.st, self.model.jvp(self.st, v))


def cg(apply_h, b, lam, iters=CG_ITERS):
    x, r = torch.zeros_like(b), b.clone()
    d, rs = r.clone(), float((b * b).sum())
    for _ in range(iters):
        hd = apply_h(d) + lam * d
        dhd = float((d * hd).sum())
        if dhd <= 0:                       # negative curvature: keep the iterate so far
            break
        a = rs / dhd
        x += a * d
        r -= a * hd
        rs_new = float((r * r).sum())
        if rs_new ** 0.5 < 1e-3 * float(b.norm()):
            break
        d, rs = r + (rs_new / rs) * d, rs_new
    return x


def predict(model_new, p, apply_h, rhs, lams=LAMS):
    """min over (lam, alpha) of F_new(p - alpha (H + lam I)^-1 rhs); returns (p, lam)."""
    best = (model_new.state(p).f, p, None)
    for lam in lams:
        step = cg(apply_h, -rhs.detach(), lam)
        for a in ALPHAS:
            q = p + a * step
            f = model_new.state(q).f
            if f < best[0]:
                best = (f, q, lam)
    return best[1], best[2]


def route_a(model_old, model_new, p_ref):
    out = {}
    g_new = model_new.gradient(model_new.state(p_ref))
    g_old = model_old.gradient(model_old.state(p_ref))
    t0 = time.time()
    c_old = Curvature(model_old, p_ref)
    out["ift_gn"] = (predict(model_new, p_ref, c_old.gn, g_new - g_old)[0], time.time() - t0)
    t0 = time.time()
    out["ift_full"] = (predict(model_new, p_ref, c_old.full, g_new - g_old)[0], time.time() - t0)
    del c_old
    t0 = time.time()
    p, lam = p_ref, None
    for k in range(3):
        c_new = Curvature(model_new, p)
        p, lam_k = predict(model_new, p, c_new.full, model_new.gradient(c_new.st),
                           LAMS if lam is None else (lam,))
        lam = lam or lam_k
        del c_new
        if k == 0:
            out["newton_full_1"] = (p, time.time() - t0)
    out["newton_full_3"] = (p, time.time() - t0)
    return out


# ---------------------------------------------------------------- route B

def lm_in_field(lm, tgt, p_ref, nf: NormalDisplacementField, iterations=max(CHECKPOINTS)):
    """Levenberg-Marquardt in b-space; returns {step: (p, seconds)}."""
    return solve_in_normal_field(tgt, lm, p_ref, nf, beta=base.BETA, iterations=iterations,
                                 checkpoints=CHECKPOINTS)[1]


# ---------------------------------------------------------------- driver

def main() -> None:
    torch.set_num_threads(4)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    litho = base.LithoModel(base.KERNEL_DIR, base.BASE)
    case3 = load_target(3)
    layout_set = {"case3": (case3, torch.load(os.path.join(base.OUT, "p0.pt")),
                            {r: os.path.join(POD_DIR, f"p_warm_{r.replace(' ', '_')}.pt") for r in PICK})}
    for name, target in layouts(case3).items():
        layout_set[name] = (target, torch.load(os.path.join(LT_DIR, f"{name}_p_ref.pt")),
                            {r: os.path.join(LT_DIR, f"{name}_truth_{r.replace(' ', '_')}.pt") for r in PICK})

    for name, (target, p_ref, truth_paths) in layout_set.items():
        model_old = ILTLeastSquares(target, litho, base.BETA)
        fields = {s: NormalDisplacementField(p_ref, s) for s in SPACINGS}
        rows = results.setdefault(name, [])
        done = {r["recipe"] for r in rows}
        for rec in [r for r in base.RECIPES if r["name"] in PICK and r["name"] not in done]:
            lm, tgt = base.recipe_setup(litho, target, rec)
            p_truth = torch.load(truth_paths[rec["name"]])
            truth, truth_print = base.score(lm, tgt, p_truth)
            frozen, _ = base.score(lm, tgt, p_ref, truth_print)
            gap = frozen["l2"] - truth["l2"]
            row = {"recipe": rec["name"], "gap": gap}

            def rec_of(p, seconds=None, **extra):
                s, _ = base.score(lm, tgt, p, truth_print)
                return {"recovery": (frozen["l2"] - s["l2"]) / gap, "l2": s["l2"], "seconds": seconds} | extra

            delta = p_truth - p_ref
            for s, nf in fields.items():
                proj = nf.project(delta)
                row[f"proj_{s}"] = rec_of(p_ref + proj, dims=nf.dims,
                                          captured=float((proj * delta).sum() / (delta * delta).sum()))
                row[f"field_{s}"] = {k: rec_of(p, sec) for k, (p, sec) in lm_in_field(lm, tgt, p_ref, nf).items()}

            snap = {}
            solve_levenberg_marquardt(tgt, lm, p_ref, iterations=max(CHECKPOINTS), beta=base.BETA, max_cg=10,
                                      on_iteration=lambda it, p, el: snap.__setitem__(it, (p.clone(), el)))
            row["warm_full"] = {k: rec_of(*snap[k]) for k in CHECKPOINTS}

            model_new = ILTLeastSquares(tgt, lm, base.BETA)
            for key, (p, sec) in route_a(model_old, model_new, p_ref).items():
                row[key] = rec_of(p.detach(), sec)

            rows.append(row)
            r = lambda d: f"{100 * d['recovery']:4.0f}%"
            print(f"[{name:9s} {rec['name']:15s}] IFT gn {r(row['ift_gn'])} full {r(row['ift_full'])} "
                  f"newton1 {r(row['newton_full_1'])} newton3 {r(row['newton_full_3'])} "
                  f"({row['newton_full_3']['seconds']:.0f}s) | field@5/20: "
                  + " ".join(f"{s}:{r(row[f'field_{s}'][5])}/{r(row[f'field_{s}'][20])}" for s in SPACINGS)
                  + f" | warm {r(row['warm_full'][5])}/{r(row['warm_full'][20])} | proj "
                  + " ".join(f"{s}:{r(row[f'proj_{s}'])}" for s in SPACINGS), flush=True)
            json.dump(results | {"protocol": __doc__, "provenance": base.provenance()}, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
