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


def fig2b_lbfgs_full_baseline():
    """Adam (main baseline) vs L-BFGS (X-16) vs paper, all 10 cases, EPE@15nm."""
    path = os.path.join(ROOT, "results/iccad13_ilt_lbfgs/summary.json")
    if not os.path.exists(path):
        print("skip fig2b: results/iccad13_ilt_lbfgs/summary.json not present")
        return
    lbfgs = json.load(open(path))
    cases = sorted(int(c) for c in lbfgs["per_case"])
    adam_vals, lbfgs_vals = [], []
    for c in cases:
        d = json.load(open(os.path.join(ROOT, f"results/iccad13_ilt/case{c}.json")))
        adam_vals.append(d["scores"]["epe@15nm"])
        lbfgs_vals.append(lbfgs["per_case"][str(c)]["epe@15nm"])
    paper_vals = [3, 0, 13, 0, 0, 0, 0, 0, 0, 0]  # Table 1 "OURS" column

    x = np.arange(len(cases))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, adam_vals, width, color=BLUE, label="Adam (main baseline, 300 it)")
    ax.bar(x + width / 2, lbfgs_vals, width, color=ORANGE, label="L-BFGS (X-16, ~110 evals)")
    ax.scatter(x, paper_vals, marker="_", s=420, linewidths=2.2, color=GRAY, zorder=3,
               label="paper Table 1 \"OURS\"")
    ax.set_xticks(x)
    ax.set_xticklabels([f"case {c}" for c in cases])
    ax.set_ylabel("EPE @ 15 nm (lower is better)")
    ax.set_title("L-BFGS matches the paper's EPE@15nm on all 10 cases (F-LBFGS-02)")
    ax.legend(frameon=False, fontsize=8.5)
    for i, l in enumerate(lbfgs_vals):
        ax.annotate(str(l), (x[i] + width / 2, l), ha="center", va="bottom", fontsize=8.5, color=ORANGE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lbfgs_full_baseline.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/lbfgs_full_baseline.png")


def fig2c_lbfgs_iteration_budget():
    """Adam -> X-16 -> X-17 progression vs paper references (F-LBFGS-03)."""
    path17 = os.path.join(ROOT, "results/iccad13_ilt_lbfgs100/summary.json")
    if not os.path.exists(path17):
        print("skip fig2c: results/iccad13_ilt_lbfgs100/summary.json not present")
        return

    def mean_of(exp_id, key):
        vals = [
            json.load(open(os.path.join(ROOT, f"results/{exp_id}/case{c}.json")))["scores"][key]
            for c in range(1, 11)
        ]
        return sum(vals) / len(vals)

    labels = ["Adam\n(main baseline)", "X-16\nL-BFGS, 50 it", "X-17\nL-BFGS, 100 it"]
    epe15 = [mean_of(e, "epe@15nm") for e in ("iccad13_ilt", "iccad13_ilt_lbfgs", "iccad13_ilt_lbfgs100")]
    epe3 = [mean_of(e, "epe@3nm") for e in ("iccad13_ilt", "iccad13_ilt_lbfgs", "iccad13_ilt_lbfgs100")]
    paper_epe15, paper_epe3_ispd25 = 1.6, 32.8

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ax = axes[0]
    ax.bar(x, epe15, width=0.55, color=BLUE)
    ax.axhline(paper_epe15, color=ORANGE, linewidth=1.8, linestyle="--")
    ax.text(len(labels) - 0.4, paper_epe15 + 0.05, "paper OURS (1.6)", color=ORANGE, fontsize=8.5, va="bottom")
    ax.set_ylabel("mean EPE @ 15 nm")
    ax.set_title("EPE@15nm: X-17 beats the paper's own average", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    for i, v in enumerate(epe15):
        ax.annotate(f"{v:.1f}", (x[i], v), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.bar(x, epe3, width=0.55, color=BLUE)
    ax.axhline(paper_epe3_ispd25, color=ORANGE, linewidth=1.8, linestyle="--")
    ax.text(len(labels) - 0.4, paper_epe3_ispd25 + 1, "paper ISPD25, no gen. (32.8)", color=ORANGE, fontsize=8.5, va="bottom")
    ax.set_ylabel("mean EPE @ 3 nm")
    ax.set_title("EPE@3nm: improves, gap to paper remains", fontsize=10.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    for i, v in enumerate(epe3):
        ax.annotate(f"{v:.1f}", (x[i], v), ha="center", va="bottom", fontsize=9)

    fig.suptitle("Iteration-budget effect found while testing 3 improvement directions (F-LBFGS-03)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "lbfgs_iteration_budget.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/lbfgs_iteration_budget.png")


def fig3_mask_visualization(case: int = 1):
    """Target / mask / mask-overlay / RESIST-overlay for one representative
    case, in the spirit of the paper's own Fig. 4.

    The mask is NOT supposed to look like the target -- it carries SRAFs
    (sub-resolution assist features) that shape the optical image but never
    print themselves, so a target-vs-mask overlay always looks like a poor
    match by construction. The panel that actually validates ILT quality is
    target vs the SIMULATED RESIST (the litho model's binarized nominal
    printed image) -- that is what score_mask()'s L2/EPE numbers are computed
    from, and it is included here explicitly so the two are never conflated.
    """
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch

    from gril.data.glp import Design
    from gril.litho.resist import LithoModel, ProcessConfig

    mask_path = os.path.join(ROOT, f"results/iccad13_ilt/mask{case}.pt")
    glp_path = f"/home/user/openopc/openilt/benchmark/ICCAD2013/M1_test{case}.glp"
    kernel_dir = "/home/user/openopc/openilt/kernel"
    if not (os.path.exists(mask_path) and os.path.exists(glp_path)):
        print(f"skip fig3: mask{case}.pt or the GLP benchmark file is not present")
        return

    mask_t = torch.load(mask_path, weights_only=False).float()
    target_np = Design.from_glp(glp_path).centred_raster(2048)
    target_t = torch.tensor(target_np)
    litho = LithoModel(kernel_dir, ProcessConfig())
    with torch.no_grad():
        resist_nom, _, _ = litho.binary(mask_t)
    mask = mask_t.numpy()
    resist = resist_nom.float().numpy()
    target = target_np
    pixel_match = float((resist_nom == target_t).float().mean())

    # Crop to the design's bounding box (plus margin) -- the full 2048x2048
    # canvas is mostly empty and would waste the figure on blank space.
    ys, xs = np.nonzero(target)
    m = 60
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, 2048)
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, 2048)
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for ax, (img, title) in zip(
        axes[:2],
        [(target[y0:y1, x0:x1], "Target design"), (mask[y0:y1, x0:x1], "ILT-optimised mask (300 it)\nincludes non-printing SRAFs")],
    ):
        ax.imshow(img, cmap="Greys", origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2]
    ax.imshow(np.zeros_like(target[y0:y1, x0:x1]), cmap="Greys", vmin=0, vmax=1, alpha=0)
    ax.contour(target[y0:y1, x0:x1], levels=[0.5], colors=[BLUE], linewidths=1.6)
    ax.contour(mask[y0:y1, x0:x1], levels=[0.5], colors=[ORANGE], linewidths=1.2, linestyles="dashed")
    ax.set_title("target vs MASK\n(expected to differ -- SRAFs don't print)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.6, label="target"),
            Line2D([0], [0], color=ORANGE, lw=1.2, ls="--", label="ILT mask"),
        ],
        frameon=False, fontsize=8.5, loc="upper right",
    )

    ax = axes[3]
    ax.imshow(np.zeros_like(target[y0:y1, x0:x1]), cmap="Greys", vmin=0, vmax=1, alpha=0)
    ax.contour(target[y0:y1, x0:x1], levels=[0.5], colors=[BLUE], linewidths=1.6)
    ax.contour(resist[y0:y1, x0:x1], levels=[0.5], colors=[ORANGE], linewidths=1.2, linestyles="dashed")
    ax.set_title(f"target vs printed RESIST\n(what score_mask() measures; {pixel_match:.1%} pixel match)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.6, label="target"),
            Line2D([0], [0], color=ORANGE, lw=1.2, ls="--", label="simulated resist"),
        ],
        frameon=False, fontsize=8.5, loc="upper right",
    )

    fig.suptitle(f"ICCAD13 case {case}: mask vs target look different by design; the printed resist is what has to match (Adam, 300 it, weight_pvb=0)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mask_visualization_case{case}.png"), dpi=150)
    plt.close(fig)
    print(f"wrote figures/mask_visualization_case{case}.png")


def fig4_pvb_sweep():
    """PV Band vs weight_pvb, if that sweep's raw output has been saved.

    Two side-by-side panels sharing an x-axis rather than one dual-y-axis
    plot: PV Band and L2 are different units at different scales, and a
    twin-axis chart makes their trade-off look however the two scales are
    chosen to make it look. Indexing both to their weight_pvb=0 baseline
    (F-PVB-01's own framing: ~7.5% PVB reduction vs ~3.8% L2 increase) reads
    the actual trade-off directly, with no scale-choice degree of freedom.
    """
    path = os.path.join(ROOT, "results/pvb_sweep/summary.json")
    if not os.path.exists(path):
        print("skip fig4: results/pvb_sweep/summary.json not present")
        return
    s = json.load(open(path))
    weights = [r["weight_pvb"] for r in s["sweep"]]
    pvb = np.array([r["pvb_sum"] for r in s["sweep"]])
    l2 = np.array([r["l2_sum"] for r in s["sweep"]])
    pvb_pct = 100.0 * (pvb / pvb[0] - 1.0)
    l2_pct = 100.0 * (l2 / l2[0] - 1.0)

    # Categorical x positions, not a log/symlog numeric axis: weight_pvb=0
    # has no logarithm, and the swept values are not meant to be read as
    # evenly-spaced on any continuous scale.
    x = np.arange(len(weights))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, pvb_pct, "-o", color=BLUE, linewidth=2, markersize=5,
            label="PV Band (sum, 3 cases)")
    ax.plot(x, l2_pct, "-o", color=ORANGE, linewidth=2, markersize=5,
            label="L2 (sum, 3 cases)")
    ax.axhline(0, color=GRAY, linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in weights])
    ax.set_xlabel("weight_pvb")
    ax.set_ylabel("% change from weight_pvb=0 baseline")
    ax.set_title("Process-window-aware objective: weight_pvb sweep")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pvb_sweep.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/pvb_sweep.png")


def fig5_multistart_comparison():
    """Four-arm best-of-K EPE@3nm comparison (X-15 / F-MULTISTART-01)."""
    path = os.path.join(ROOT, "results/multistart_ablation/summary.json")
    if not os.path.exists(path):
        print("skip fig5: results/multistart_ablation/summary.json not present")
        return
    s = json.load(open(path))
    agg = s["aggregate"]
    gen = s.get("generator_arms", {})
    labels = ["single-start\n(K=1)", "random-K\n(no learning)", "PT\n(generator)", "PT+RL\n(generator)"]
    values = [
        agg["single_start_epe_mean"],
        agg["random_k_best_epe_mean"],
        gen.get("PT", {}).get("epe_best_mean"),
        gen.get("PT+RL", {}).get("epe_best_mean"),
    ]
    if any(v is None for v in values):
        print("skip fig5: generator_arms missing from summary.json")
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(x, values, width=0.55, color=BLUE)
    # The two arms that involve no learned prior at all are the best result;
    # mark them, rather than relying on a second hue, to keep this a single
    # series (one legend-free color) with a secondary encoding for the point
    # of the figure.
    for i in (0, 1):
        bars[i].set_color(ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("best-of-K EPE @ 3 nm (mean, 8 held-out designs)")
    ax.set_title("Learned sampler vs random multi-start (X-15)")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.1f}", (x[i], v), ha="center", va="bottom", fontsize=9.5)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=ORANGE, label="no learned prior"),
            Patch(color=BLUE, label="trained generator"),
        ],
        frameon=False, loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multistart_comparison.png"), dpi=150)
    plt.close(fig)
    print("wrote figures/multistart_comparison.png")


def fig6_mrc_cleanup(case: int = 1):
    """Raw mask vs MRC-opened mask, zoomed on a debris-heavy region (F-MRC-01)."""
    import sys

    sys.path.insert(0, os.path.join(ROOT, "src"))
    import torch

    from gril.ilt.solver import morphological_open

    mask_path = os.path.join(ROOT, f"results/iccad13_ilt/mask{case}.pt")
    if not os.path.exists(mask_path):
        print(f"skip fig6: mask{case}.pt not present")
        return
    mask = torch.load(mask_path, weights_only=False).float()
    cleaned = morphological_open(mask, 5)
    mask_np, cleaned_np = mask.numpy(), cleaned.numpy()

    # The full design bounding box is dominated by large, intentional SRAF
    # rings that look similar before/after opening -- the debris this finding
    # is about is small (<10px^2) isolated specks, easy to miss at that zoom.
    # Center the crop on the densest cluster of tiny components instead, so
    # the removal is actually visible.
    import scipy.ndimage as ndi

    lbl, n = ndi.label(mask_np > 0.5, structure=np.ones((3, 3)))
    sizes = ndi.sum(mask_np > 0.5, lbl, range(1, n + 1)) if n > 0 else np.array([])
    tiny_ids = np.where(sizes < 10)[0] + 1
    if len(tiny_ids) > 0:
        centroids = np.array(ndi.center_of_mass(mask_np > 0.5, lbl, tiny_ids))
        # densest 240x240 window: for each tiny centroid, count how many
        # OTHER tiny centroids fall within 120px of it, and center on the
        # winner -- a simple, robust "most crowded" pick.
        w = 120
        counts = ((np.abs(centroids[:, None, :] - centroids[None, :, :]) < w).all(axis=2)).sum(axis=1)
        cy, cx = centroids[np.argmax(counts)]
        y0, y1 = max(int(cy - w), 0), min(int(cy + w), 2048)
        x0, x1 = max(int(cx - w), 0), min(int(cx + w), 2048)
    else:
        ys, xs = np.nonzero(mask_np > 0.5)
        m = 60
        y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, 2048)
        x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, 2048)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, (img, title) in zip(
        axes,
        [
            (mask_np[y0:y1, x0:x1], "raw (mrc_open_size=0, every committed run)"),
            (cleaned_np[y0:y1, x0:x1], "mrc_open_size=5 (post-hoc cleanup)"),
        ],
    ):
        ax.imshow(img, cmap="Greys", origin="upper", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"ICCAD13 case {case}: MRC cleanup removes isolated sub-5nm debris "
        "for ~0.3% mean L2 cost (F-MRC-01)",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"mrc_cleanup_case{case}.png"), dpi=150)
    plt.close(fig)
    print(f"wrote figures/mrc_cleanup_case{case}.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_iteration_budget_curve()
    fig2_optimizer_comparison()
    fig2b_lbfgs_full_baseline()
    fig2c_lbfgs_iteration_budget()
    fig3_mask_visualization(case=1)
    fig4_pvb_sweep()
    fig5_multistart_comparison()
    fig6_mrc_cleanup(case=1)
