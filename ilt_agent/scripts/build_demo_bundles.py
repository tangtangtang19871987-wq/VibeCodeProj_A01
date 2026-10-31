"""Build demo LayoutBundles from the ILT study simulator (ilt_study/).

rand2_fidelity : ILT with an image-fidelity-only cost (many bridge/pinch problems)
rand2_good     : the study's reference recipe (few residual problems)
Each bundle holds the target, the hard mask, the printed pattern at all 27 corners
(dose 0.95/1/1.05 x focus 0/-40/+40 nm x mask bias -1/0/+1 nm) and the nominal image.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "ilt_study"))
sys.path.insert(0, REPO)

from clips import random_clip            # noqa: E402
from ilt import Problem, Recipe       # noqa: E402
from common import GOOD                  # noqa: E402

from ilt_agent.bundle import LayoutBundle  # noqa: E402

FOCUS_NM = {0: 0, 1: -40, 2: 40}


@torch.no_grad()
def bundle_from(P, rho, meta):
    printed = {}
    for b in P.spec.biases:
        m = P.hard_mask(rho, b)
        I = P.images(m, P.spec.focus_idx)
        for D in P.spec.doses:
            for j, f in enumerate(P.spec.focus_idx):
                printed[f"d{D:.2f}_f{FOCUS_NM[f]:+d}_m{b:+.0f}"] = (D * I[j] > 1).numpy()
    nominal = "d1.00_f+0_m+0"
    inten = P.images(P.hard_mask(rho), [0])[0].numpy()
    meta = {**meta, "spec": {"w_min_nm": P.spec.w_min, "s_min_nm": P.spec.s_min, "epe_tol_nm": 5.0},
            "simulator": "ilt_study scalar SOCS, 193i NA1.35, leaf X-dipole, pitch 74.5 nm",
            "note": "periodic clip; analysis treats the frame as non-periodic"}
    meta["target_polygons"] = clip_rectangles(P.clip)
    return LayoutBundle(P.T.numpy() > 0.5, P.hard_mask(rho).numpy(), printed, P.g.dx, nominal, inten, meta)


def clip_rectangles(clip):
    """Exact drawn rectangles (nm) of a periodic clip, split where a segment wraps in y."""
    rects = []
    for s in clip.segs:
        x0, x1 = clip.xc(s.track) - s.width / 2, clip.xc(s.track) + s.width / 2
        y0, y1 = s.y0, s.y0 + s.length
        pieces = [(y0, min(y1, clip.L))] + ([(0.0, y1 - clip.L)] if y1 > clip.L else [])
        for a, b in pieces:
            if b - a > 1e-6:
                rects.append([[x0, a], [x1, a], [x1, b], [x0, b]])
    return rects


def main():
    out = os.path.join(os.path.dirname(HERE), "demo")
    os.makedirs(out, exist_ok=True)
    P = Problem(random_clip(2))
    for name, kw in (("rand2_fidelity", dict(iters=300, w_fid=1.0, w_sraf=50.0)), ("rand2_good", GOOD)):
        _, rho, _ = P.run(Recipe(**kw))
        r = P.evaluate(rho)
        meta = {"recipe": name, "study_checker": {k: r[k] for k in ("n_hot", "n_bridge_all", "n_pinch_all",
                                                                  "topo_bridge", "topo_open", "sraf_print")}}
        b = bundle_from(P, rho, meta)
        path = os.path.join(out, f"{name}.npz")
        b.save(path)
        print(name, path, meta["study_checker"])


if __name__ == "__main__":
    main()
