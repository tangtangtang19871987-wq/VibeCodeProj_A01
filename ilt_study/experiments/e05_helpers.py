import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from ilt import site_values


def pair_ids(P):
    t2t = [i for i, t in enumerate(P.sites.tag) if t.startswith("seg3->seg4")]
    lend = [i for i, t in enumerate(P.sites.tag) if t in ("seg3:top", "seg4:bot")]
    return t2t, lend


def collateral(P, r, excl):
    m = r["margin"].copy()
    keep = np.ones(len(m), bool); keep[excl] = False
    return dict(coll_hot=int((m[keep] < 0).sum()), coll_min=float(m[keep].min()),
                coll_p5=float(np.percentile(m[keep], 5)))


def grad_share(P, theta, rc, excl):
    th = theta.clone().requires_grad_(True)
    q = site_values(P, th, rc)
    pen = F.relu(P.lower + rc.margin - q) ** 2
    isb = torch.tensor(np.isin(P.kind, ["bridge", "t2t", "diag"]), dtype=torch.float32)
    w = torch.where(isb > 0, torch.tensor(rc.w_bridge), torch.tensor(rc.w_pinch))
    pen = pen * w
    mask = torch.zeros(len(P.kind)); mask[excl] = 1
    g_pair = torch.autograd.grad((pen * mask).sum(), th, retain_graph=True)[0]
    g_rest = torch.autograd.grad((pen * (1 - mask)).sum(), th)[0]
    return float(g_pair.norm() / (g_pair.norm() + g_rest.norm() + 1e-12))


