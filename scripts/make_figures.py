#!/usr/bin/env python3
"""Generate all report figures from raw per-case JSON in results/.

Colors: the validated categorical pair from the project's dataviz reference
palette (blue #2a78d6 / orange #eb6834; CVD Delta E 9.1, normal-vision Delta E
19.6, both clear the skill's floors). Used in fixed order across every figure
-- series identity never repaints when a figure's set of series changes.

Every figure is generated from a raw JSON file already committed under
results/ -- nothing here computes a new number, only visualizes ones already
reported in REPRODUCTION_REPORT.md and docs/findings.md.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

# Validated categorical pair (references/palette.md), fixed order throughout.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a897f"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c8c0",
    "axes.grid": True,
    "grid.color": "#e8e7e0",
    "grid.linewidth": 0.6,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig1_iteration_budget_curve():
    """Mean EPE@15nm and EPE@3nm vs ILT iteration count, across all 10 ICCAD13
    cases. Addresses the paper's "roughly half the iteration budget" claim.
    """
    iters = [25, 50, 100, 150, 200, 250, 300]
    epe15, epe3 = [], []
    for it in iters:
        vals15, vals3 = [], []
        for case in range(1, 11):
            d = json.load(open(os.path.join(ROOT, f"results/iccad13_ilt/case{case}.json")))
            bc = d["budget_curve"][str(it)]
            vals15.append(bc["epe@15nm"])
            vals3.append(bc["epe@3nm"])
        epe15.append(np.mean(vals15))
        epe3.append(np.mean(vals3))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iters, epe15, "-o", color=BLUE, linewidth=2, markersize=5, label="EPE @ 15 nm")
    ax.plot(iters, epe3, "-o", color=ORANGE, linewidth=2, markersize=5, label="EPE @ 3 nm")
    ax.set_xlabel("ILT iterations")
    ax.set_ylabel("mean EPE violations (10 ICCAD13 cases)")
    ax.set_title("Iteration-budget curve: our numerical ILT solver")
    ax.axvline(150, color=GRAY, linewidth=1, linestyle="--", alpha=0.7)
    ax.text(152, ax.get_ylim()[1] * 0.92, "150 it\n(paper's \"Ours\" budget)",
            fontsize=8.5, color=GRAY, va="top")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "iteration_budget_curve.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/iteration_budget_curve.png")


def fig2_optimizer_comparison():
    """Adam vs L-BFGS, per case, EPE@15nm (X-14 / F-LBFGS-01)."""
    s = json.load(open(os.path.join(ROOT, "results/abl_optimizer/summary.json")))
    cases = s["config"]["cases"]
    adam_vals = [s["results"][f"case{c}_adam"]["scores"]["epe@15nm"] for c in cases]
    lbfgs_vals = [s["results"][f"case{c}_lbfgs"]["scores"]["epe@15nm"] for c in cases]

    x = np.arange(len(cases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, adam_vals, width, color=BLUE, label="Adam (150 evals)")
    ax.bar(x + width / 2, lbfgs_vals, width, color=ORANGE, label="L-BFGS (~110 evals)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in cases])
    ax.set_ylabel("EPE @ 15 nm (lower is better)")
    ax.set_title("Optimizer comparison at a well-conditioned start (X-14)")
    ax.legend(frameon=False)
    for i, (a, l) in enumerate(zip(adam_vals, lbfgs_vals)):
        ax.annotate(str(a), (x[i] - width / 2, a), ha="center", va="bottom", fontsize=8.5, color=BLUE)
        ax.annotate(str(l), (x[i] + width / 2, l), ha="center", va="bottom", fontsize=8.5, color=ORANGE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "optimizer_comparison.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/optimizer_comparison.png")


def fig3_mask_visualization(case: int = 1):
    """Target vs optimized-mask overlay for one representative case, in the
    spirit of the paper's own Fig. 4 (a target/mask/printed-image panel).
    """
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch

    from gril.data.glp import Design

    mask_path = os.path.join(ROOT, f"results/iccad13_ilt/mask{case}.pt")
    glp_path = f"/home/user/openopc/openilt/benchmark/ICCAD2013/M1_test{case}.glp"
    if not (os.path.exists(mask_path) and os.path.exists(glp_path)):
        print(f"skip fig3: mask{case}.pt or the GLP benchmark file is not present")
        return

    mask = torch.load(mask_path, weights_only=False).float().numpy()
    target = Design.from_glp(glp_path).centred_raster(2048)

    # Crop to the design's bounding box (plus margin) -- the full 2048x2048
    # canvas is mostly empty and would waste the figure on blank space.
    ys, xs = np.nonzero(target)
    m = 60
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, 2048)
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, 2048)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        (target[y0:y1, x0:x1], "Target design", "Greys"),
        (mask[y0:y1, x0:x1], "ILT-optimised mask (300 it)", "Greys"),
        (None, "Overlay: target edge vs mask", None),
    ]
    for ax, (img, title, cmap) in zip(axes[:2], panels[:2]):
        ax.imshow(img, cmap=cmap, origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2]
    ax.imshow(np.zeros_like(target[y0:y1, x0:x1]), cmap="Greys", vmin=0, vmax=1, alpha=0)
    ax.contour(target[y0:y1, x0:x1], levels=[0.5], colors=[BLUE], linewidths=1.6)
    ax.contour(mask[y0:y1, x0:x1], levels=[0.5], colors=[ORANGE], linewidths=1.2, linestyles="dashed")
    ax.set_title("Overlay: target edge vs mask", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.6, label="target"),
        Line2D([0], [0], color=ORANGE, lw=1.2, ls="--", label="ILT mask"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")

    fig.suptitle(f"ICCAD13 case {case}: target vs ILT-optimised mask (Adam, 300 it, weight_pvb=0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mask_visualization_case{case}.png"), dpi=150)
    plt.close(fig)
    print(f"wrote figures/mask_visualization_case{case}.png")


def fig4_pvb_sweep():
    """PV Band vs weight_pvb, if that sweep's raw output has been saved."""
    path = os.path.join(ROOT, "results/pvb_sweep/summary.json")
    if not os.path.exists(path):
        print("skip fig4: results/pvb_sweep/summary.json not present")
        return
    s = json.load(open(path))
    weights = [r["weight_pvb"] for r in s["sweep"]]
    pvb = [r["pvb_sum"] for r in s["sweep"]]
    l2 = [r["l2_sum"] for r in s["sweep"]]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax.twinx()
    ax.plot(weights, pvb, "-o", color=BLUE, label="PV Band (sum, 3 cases)")
    ax2.plot(weights, l2, "-o", color=ORANGE, label="L2 (sum, 3 cases)")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlabel("weight_pvb")
    ax.set_ylabel("PV Band", color=BLUE)
    ax2.set_ylabel("L2", color=ORANGE)
    ax.set_title("Process-window-aware objective: weight_pvb sweep")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pvb_sweep.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/pvb_sweep.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_iteration_budget_curve()
    fig2_optimizer_comparison()
    fig3_mask_visualization(case=1)
    fig4_pvb_sweep()
