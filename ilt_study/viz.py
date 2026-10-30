"""Plot helpers: mask, target outline, printed contours, hotspot markers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

KCOL = {"pinch": "#d62728", "lend": "#ff7f0e", "bridge": "#1f77b4", "t2t": "#17becf"}


@torch.no_grad()
def plot_solution(P, rho, ax, title="", corners=((1.0, 0, 0.0), (1.05, 1, 1.0), (0.95, 1, -1.0)),
                  margins=None, show_mask=True):
    n, L = P.g.n, P.clip.L
    ext = [0, L, 0, L]
    m = P.hard_mask(rho) if rho is not None else P.T
    if show_mask:
        ax.imshow(m.numpy(), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    ax.contour(P.T.numpy(), levels=[0.5], origin="lower", extent=ext, colors="g", linewidths=0.8)
    cols = ["k", "#1f77b4", "#d62728"]
    for (D, f, b), c in zip(corners, cols):
        I = P.images(P.hard_mask(rho, b) if rho is not None else P.T, [f])[0]
        ax.contour((D * I).numpy(), levels=[1.0], origin="lower", extent=ext, colors=c, linewidths=0.7)
    if margins is not None:
        bad = np.nonzero(margins < 0)[0]
        for i in bad:
            p0, p1 = P.sites.p0[i], P.sites.p1[i]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=KCOL[P.kind[i]], lw=2.0, alpha=0.9)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8); ax.tick_params(labelsize=6)
