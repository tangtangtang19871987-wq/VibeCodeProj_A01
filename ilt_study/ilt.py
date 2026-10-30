"""Pixel ILT with filter+projection parameterization, corners, and pluggable costs.

theta --sigmoid--> x (density) --filter(R)--> rho --projection(beta, eta)--> mask
  * R (filter radius) is the MRC proxy: larger R => larger minimum feature.
  * frozen region: x is fixed there (pre-placed, frozen SRAF).
  * mask-bias corner: rho_b = rho + b * |grad rho| (detached), i.e. the
    iso-contour rho=eta moves outward by ~b nm.

Evaluation always uses a *hard* mask (sub-pixel anti-aliased from the
signed distance (rho-eta)/|grad rho|) at the full corner factorial, so the
soft relaxations used in the cost never leak into reported numbers.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from layout import Clip, SiteSampler, Sites, make_sites, measure, rasterize
from litho import Grid, Imager
from zones import make_zones


# --------------------------------------------------------------------------
# helpers
def fft_conv(x, tf):
    return torch.fft.ifft2(torch.fft.fft2(x) * tf).real


def cone_tf(n, d, R):
    c = (np.arange(n) - n // 2) * d
    X, Y = np.meshgrid(c, c)
    k = np.maximum(0, R - np.hypot(X, Y))
    if k.sum() == 0:
        k[n // 2, n // 2] = 1
    k = np.fft.ifftshift(k / k.sum())
    return torch.tensor(np.fft.fft2(k))


def grad_mag(rho, d):
    gx = (torch.roll(rho, -1, -1) - torch.roll(rho, 1, -1)) / (2 * d)
    gy = (torch.roll(rho, -1, -2) - torch.roll(rho, 1, -2)) / (2 * d)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def project(rho, beta, eta=0.5):
    a = math.tanh(beta * eta)
    return (a + torch.tanh(beta * (rho - eta))) / (a + math.tanh(beta * (1 - eta)))


def softmin(x, tau, dim=0):
    return -torch.logsumexp(-tau * x, dim=dim) / tau


def disk_tf(n, d, r):
    c = (np.arange(n) - n // 2) * d
    X, Y = np.meshgrid(c, c)
    k = (np.hypot(X, Y) <= r).astype(float)
    return np.fft.fft2(np.fft.ifftshift(k)), k.sum()


def mrc_violation_area(mhard, d, wmin, smin):
    """Area (nm^2) removed by opening (width<wmin) + added by closing (space<smin)."""
    b = (mhard > 0.5).astype(float)

    def erode(a, r):
        tf, s = disk_tf(a.shape[0], d, r)
        return (np.real(np.fft.ifft2(np.fft.fft2(a) * tf)) > s - 0.5).astype(float)

    def dilate(a, r):
        tf, s = disk_tf(a.shape[0], d, r)
        return (np.real(np.fft.ifft2(np.fft.fft2(a) * tf)) > 0.5).astype(float)

    opened = dilate(erode(b, wmin / 2), wmin / 2)
    closed = erode(dilate(b, smin / 2), smin / 2)
    return float(((b - opened) > 0).sum() + ((closed - b) > 0).sum()) * d * d


def label(binary: np.ndarray) -> np.ndarray:
    """4-connected periodic component labels (0 = background)."""
    n = binary.shape[0]
    lab = np.where(binary, np.arange(binary.size).reshape(binary.shape) + 1, 0)
    big = binary.size + 10
    while True:
        cur = np.where(binary, lab, big)
        nb = np.minimum.reduce([cur, np.roll(cur, 1, 0), np.roll(cur, -1, 0),
                                np.roll(cur, 1, 1), np.roll(cur, -1, 1)])
        new = np.where(binary, nb, 0)
        if np.array_equal(new, lab):
            return lab
        lab = new


def topology_errors(printed: np.ndarray, target: np.ndarray):
    """(#bridges, #opens): a printed component touching >1 target component
    counts (k-1) bridges; a target component covered by !=1 printed component
    (0 = missing, >=2 = broken) counts as an open."""
    lt = label(target > 0.5)
    lp = label(printed)
    core = target > 0.5
    pairs = np.unique(np.stack([lp[core], lt[core]], 1), axis=0)
    pairs = pairs[(pairs[:, 0] > 0)]
    bridges = 0
    for pc in np.unique(pairs[:, 0]):
        k = len(np.unique(pairs[pairs[:, 0] == pc, 1]))
        bridges += k - 1
    opens = 0
    for tc in np.unique(lt[lt > 0]):
        k = len(np.unique(pairs[pairs[:, 1] == tc, 0]))
        opens += int(k != 1)
    extra = len(set(np.unique(lp[lp > 0])) - set(pairs[:, 0]))
    return bridges, opens, extra


# --------------------------------------------------------------------------
@dataclass
class Spec:
    w_min: float = 28.0
    s_min: float = 28.0
    s_min_t2t: float = 30.0
    pb_max: float = 15.0              # allowed line-end pull-back
    lend_L0: float = 40.0
    doses: tuple = (0.95, 1.0, 1.05)
    focus_idx: tuple = (0, 1, 2)
    biases: tuple = (-1.0, 0.0, 1.0)  # mask CD error (nm, per edge)

    def lower(self, kind):
        return {"pinch": self.w_min, "bridge": self.s_min, "t2t": self.s_min_t2t,
                "diag": self.s_min, "lend": self.lend_L0 - self.pb_max}[kind]


@dataclass
class Recipe:
    iters: int = 300
    lr: float = 0.1
    R: float = 8.0                    # filter radius (MRC proxy)
    beta_sched: tuple = (2, 4, 8, 16)
    alpha: float = 15.0               # resist sigmoid steepness (normalized image)
    tau: float = 2.0                  # softmin sharpness over corners (1/nm)
    margin: float = 1.0               # hinge target margin (nm)
    w_fid: float = 1.0
    w_out: float = 0.0                # global 'outer PV band' (area-type bridge)
    w_in: float = 0.0                 # global 'inner PV band' (area-type pinch)
    w_bridge: float = 0.0             # site hinge, bridge + t2t
    w_pinch: float = 0.0              # site hinge, pinch + lend
    w_sraf: float = 0.0               # SRAF printing
    w_zout: float = 0.0               # keep-out zone (localized bridge, all bridge corners)
    w_zin: float = 0.0                # keep-in zone (localized pinch, all pinch corners)
    w_ils: float = 0.0                # log-contrast at lateral sites
    hinge: bool | str = True          # True hinge^2 | False quadratic tracking | 'linear' push
    site_weights: np.ndarray | None = None   # per-site multipliers
    cap: float | None = None          # cap per-site violation (nm)
    balance_sites: list | None = None # groups of site ids to minimax-balance
    corners_opt: list | None = None   # (dose, fidx, bias); None => default set
    freeze: np.ndarray | None = None  # bool [n,n] True = frozen pixel
    x_frozen: np.ndarray | None = None
    optimizer: str = "adam"           # "adam" | "sgd" (plain gradient descent)
    alpha_site: float | None = None   # sigmoid steepness for site lengths (default alpha)
    callback: object = None           # f(it, theta) called every cb_every iterations
    cb_every: int = 50
    log_every: int = 10
    record_sites: list = field(default_factory=list)  # site ids to trace


class Problem:
    def __init__(self, clip: Clip, n=256, spec: Spec | None = None, imager=None,
                 sites: Sites | None = None):
        self.clip = clip
        self.g = Grid(n=n, period_x=clip.pitch, tracks=clip.tracks)
        assert abs(self.g.lx - clip.L) < 1e-6, "clip must be square & match grid"
        self.im = imager or Imager(self.g, keep=0.98)
        self.spec = spec or Spec()
        self.T = torch.tensor(rasterize(clip, n))
        self.sites = sites or make_sites(clip, lend_L0=self.spec.lend_L0)
        self.S = SiteSampler(self.sites, clip.L, n)
        self.kind = np.array(self.sites.kind)
        self.lower = torch.tensor([self.spec.lower(k) for k in self.kind], dtype=torch.float32)
        self.th = calibrate_threshold(self.im, self.g)
        ko, ki = make_zones(clip, n, s_min=self.spec.s_min, w_min=self.spec.w_min,
                            pb_max=self.spec.pb_max)
        self.keep_out = torch.tensor(ko, dtype=torch.float32)
        self.keep_in = torch.tensor(ki, dtype=torch.float32)
        d = self.g.dx
        # "far" region: > 10 nm from any target edge -> any print there is spurious
        dil = fft_conv(self.T.double(), torch.tensor(disk_tf(n, d, 10.0)[0])).float()
        far = (dil < 0.5).numpy()
        # exclude tip-to-tip gaps (line-end overshoot is judged by t2t sites)
        c = (np.arange(n) + 0.5) * d
        X, Y = np.meshgrid(c, c, indexing="xy")
        for k, p0, p1 in zip(self.kind, self.sites.p0, self.sites.p1):
            if k != "t2t":
                continue
            xc = p0[0]
            y0, y1 = p0[1], p1[1]
            dxw = np.abs(((X - xc + clip.L / 2) % clip.L) - clip.L / 2) < clip.pitch / 2 - 8
            inside = ((Y - y0) % clip.L) < (y1 - y0)
            far &= ~(dxw & inside)
        self.far = torch.tensor(far, dtype=torch.float32)
        self.near = 1 - self.far

    # ---------------- forward model ---------------------------------------
    def rho(self, theta, R, freeze=None, x_frozen=None):
        x = torch.sigmoid(theta)
        if freeze is not None:
            fz = torch.tensor(freeze, dtype=torch.float32)
            x = x * (1 - fz) + torch.tensor(x_frozen, dtype=torch.float32) * fz
        return fft_conv(x.to(torch.complex64), cone_tf(self.g.n, self.g.dx, R).to(torch.complex64))

    def hard_mask(self, rho, bias=0.0, eta=0.5):
        sd = (rho - eta) / grad_mag(rho, self.g.dx)
        return torch.clamp(0.5 + (sd + bias) / self.g.dx, 0, 1)

    def images(self, mask, fidx):
        return self.im.image(mask, fidx) / self.th            # normalized: print > 1/dose

    # ---------------- evaluation (hard, full factorial) --------------------
    @torch.no_grad()
    def evaluate(self, rho=None, mask=None, spec: Spec | None = None, mask_bias_fn=None,
                 topo=True):
        """Return dict with per-site worst-corner values and margins."""
        sp = spec or self.spec
        vals = []
        topo_b = topo_o = extra = 0
        zo = np.zeros((self.g.n, self.g.n), bool)
        zi = np.zeros((self.g.n, self.g.n), bool)
        for b in sp.biases:
            m = self.hard_mask(rho, b) if mask is None else (
                mask if mask_bias_fn is None else mask_bias_fn(mask, b))
            I = self.images(m, sp.focus_idx)                  # [F,n,n]
            for D in sp.doses:
                Lb = self.S.lengths(I, 1.0, dose=D)            # [F,sites]
                vals.append(measure(Lb, self.S))
                pr = (D * I > 1)                                 # [F,n,n]
                zo |= (pr & (self.keep_out > 0)).any(0).numpy()
                zi |= ((~pr) & (self.keep_in > 0)).any(0).numpy()
                if topo:
                    for j in range(I.shape[0]):
                        tb, to, ex = topology_errors(pr[j].numpy(), self.T.numpy())
                        topo_b, topo_o = max(topo_b, tb), max(topo_o, to)
                        extra = max(extra, ex)
        V = torch.cat(vals, 0)                                  # [C,sites]
        worst = V.min(0).values
        margin = worst - self.lower
        out = {"worst": worst.numpy(), "margin": margin.numpy(),
               "topo_bridge": topo_b, "topo_open": topo_o}
        for k in ("pinch", "bridge", "t2t", "diag", "lend"):
            ids = self.kind == k
            if ids.any():
                mk = margin.numpy()[ids]
                out[f"min_{k}"] = float(mk.min())
                out[f"n_{k}"] = int((mk < 0).sum())
        out["n_bridge_all"] = out.get("n_bridge", 0) + out.get("n_t2t", 0) + out.get("n_diag", 0)
        m_np = margin.numpy()
        bcls = np.isin(self.kind, ["bridge", "t2t", "diag"])
        out["min_bridge_all"] = float(m_np[bcls].min())
        out["min_pinch_all"] = float(m_np[~bcls].min())
        out["n_pinch_all"] = out.get("n_pinch", 0) + out.get("n_lend", 0)
        out["n_hot_sites"] = out["n_bridge_all"] + out["n_pinch_all"]
        out["n_hot"] = out["n_hot_sites"] + topo_b + topo_o
        # secondary, stricter zone checker (reported, not the pass/fail criterion)
        out["zone_bridge"] = int(len(np.unique(label(zo))) - 1)
        out["zone_pinch"] = int(len(np.unique(label(zi))) - 1)
        out["sraf_print"] = extra
        out["zo"], out["zi"] = zo, zi
        return out

    # ---------------- ILT ---------------------------------------------------
    def run(self, rc: Recipe, theta0=None, verbose=False):
        n = self.g.n
        if theta0 is None:
            x0 = torch.clamp(self.T, 0.02, 0.98)
            theta0 = torch.log(x0 / (1 - x0))
        theta = theta0.clone().requires_grad_(True)
        opt = (torch.optim.Adam([theta], lr=rc.lr) if rc.optimizer == "adam"
               else torch.optim.SGD([theta], lr=rc.lr))
        a_site = rc.alpha if rc.alpha_site is None else rc.alpha_site
        corners = rc.corners_opt or default_corners(self.spec)
        biases = sorted({c[2] for c in corners})
        fids = sorted({c[1] for c in corners})
        sw = torch.ones(len(self.kind)) if rc.site_weights is None else torch.tensor(
            rc.site_weights, dtype=torch.float32)
        isb = torch.tensor(np.isin(self.kind, ["bridge", "t2t", "diag"]), dtype=torch.float32)
        isp = 1 - isb
        lat = torch.tensor(self.kind == "bridge")
        hist = []
        nb = len(rc.beta_sched)
        for it in range(rc.iters):
            beta = rc.beta_sched[min(it * nb // rc.iters, nb - 1)]
            rho = self.rho(theta, rc.R, rc.freeze, rc.x_frozen)
            g = grad_mag(rho, self.g.dx).detach()
            imgs = {}
            for b in biases:
                m = project(rho + b * g, beta)
                I = self.images(m, fids)
                for j, f in enumerate(fids):
                    imgs[(b, f)] = I[j]
            terms = {}
            Inom = imgs[(0.0, 0)] if (0.0, 0) in imgs else next(iter(imgs.values()))
            sig = lambda I, D: torch.sigmoid(rc.alpha * (D * I - 1))
            terms["fid"] = rc.w_fid * ((sig(Inom, 1.0) - self.T) ** 2).mean()
            if rc.w_out:
                terms["out"] = rc.w_out * sum((F.relu(sig(imgs[(c[2], c[1])], c[0]) - self.T) ** 2).mean()
                                              for c in corners if c[0] > 1)
            if rc.w_in:
                terms["in"] = rc.w_in * sum((F.relu(self.T - sig(imgs[(c[2], c[1])], c[0])) ** 2).mean()
                                            for c in corners if c[0] < 1)
            need_sites = rc.w_bridge or rc.w_pinch or rc.w_ils or rc.record_sites
            if need_sites:
                Q = torch.stack([measure(self.S.lengths(imgs[(c[2], c[1])], 1.0, c[0], a_site), self.S)
                                 for c in corners])                      # [C,sites]
                q = softmin(Q, rc.tau, 0)
                viol = self.lower + rc.margin - q
                if rc.hinge == "linear":
                    # unbounded margin reward: every site keeps being pushed
                    v = viol
                    if rc.cap is not None:
                        v = torch.clamp(v, max=rc.cap)
                    pen = v * sw
                else:
                    v = F.relu(viol) if rc.hinge else viol
                    if rc.cap is not None:
                        v = torch.clamp(v, max=rc.cap)
                    pen = v ** 2 * sw
                if rc.balance_sites is not None:
                    # replace each group's sum by its max (minimax balancing): the
                    # optimizer then only reduces the currently worst member
                    pen = pen.clone()
                    for grp in rc.balance_sites:
                        grp = list(grp)
                        mx = pen[grp].max()
                        pen[grp] = 0 * pen[grp]
                        pen[grp[0]] = mx
                if rc.w_bridge:
                    terms["bridge"] = rc.w_bridge * (pen * isb).sum() / max(1, isb.sum())
                if rc.w_pinch:
                    terms["pinch"] = rc.w_pinch * (pen * isp).sum() / max(1, isp.sum())
            if rc.w_zout:
                terms["zout"] = rc.w_zout * sum((sig(imgs[(c[2], c[1])], c[0]) * self.keep_out).pow(2).sum()
                                                for c in corners if c[0] >= 1) / self.keep_out.sum()
            if rc.w_zin:
                terms["zin"] = rc.w_zin * sum(((1 - sig(imgs[(c[2], c[1])], c[0])) * self.keep_in).pow(2).sum()
                                              for c in corners if c[0] <= 1) / self.keep_in.sum()
            if rc.w_sraf:
                terms["sraf"] = rc.w_sraf * sum((sig(imgs[(c[2], c[1])], c[0]) * self.far).pow(2).mean()
                                                for c in corners if c[0] > 1)
            if rc.w_ils:
                # log-contrast across each lateral cut: line centre vs space centre
                worst_c = None
                for c in corners:
                    smp = self.S.sample(imgs[(c[2], c[1])])[lat]
                    ns = torch.tensor(self.S.nsamp[lat.numpy()])
                    mid = smp[torch.arange(len(ns)), ns // 2]
                    ends = 0.5 * (smp[:, 0] + smp[torch.arange(len(ns)), ns - 1])
                    lc = torch.log(ends + 1e-3) - torch.log(mid + 1e-3)
                    worst_c = lc if worst_c is None else torch.minimum(worst_c, lc)
                terms["ils"] = -rc.w_ils * softmin(worst_c, 5.0, 0)
            loss = sum(terms.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if rc.callback is not None and (it % rc.cb_every == 0 or it == rc.iters - 1):
                rc.callback(it, theta.detach().clone())
            if it % rc.log_every == 0 or it == rc.iters - 1:
                rec = {"it": it, "loss": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}}
                if need_sites:
                    rec["q"] = q.detach().numpy().copy()
                hist.append(rec)
                if verbose:
                    print(it, {k: round(v, 5) for k, v in rec.items() if k not in ("q",)})
        rho = self.rho(theta.detach(), rc.R, rc.freeze, rc.x_frozen)
        return theta.detach(), rho, hist


def site_values(P, theta, rc, beta=None):
    """Soft worst-corner site values q (as in the cost) at the final beta."""
    beta = beta or rc.beta_sched[-1]
    corners = rc.corners_opt or default_corners(P.spec)
    rho = P.rho(theta, rc.R, rc.freeze, rc.x_frozen)
    g = grad_mag(rho, P.g.dx).detach()
    fids = sorted({c[1] for c in corners})
    imgs = {}
    for b in sorted({c[2] for c in corners}):
        I = P.images(project(rho + b * g, beta), fids)
        for j, f in enumerate(fids):
            imgs[(b, f)] = I[j]
    a_site = rc.alpha if rc.alpha_site is None else rc.alpha_site
    Q = torch.stack([measure(P.S.lengths(imgs[(c[2], c[1])], 1.0, c[0], a_site), P.S) for c in corners])
    return softmin(Q, rc.tau, 0)


def default_corners(sp: Spec):
    """Optimization corners: nominal + bridge-prone + pinch-prone."""
    hi, lo = max(sp.doses), min(sp.doses)
    bmax, bmin = max(sp.biases), min(sp.biases)
    c = [(1.0, 0, 0.0)]
    for f in sp.focus_idx:
        c.append((hi, f, bmax))
        c.append((lo, f, bmin))
    return c


def calibrate_threshold(im: Imager, g: Grid, w=37.0):
    x = (np.arange(g.n) + 0.5) * g.dx
    xc = x % g.period_x
    m = np.zeros((g.n, g.n), np.float32)
    m[:, np.abs(xc - g.period_x / 2) < w / 2] = 1
    # area-sampled edges
    sub = 8
    xs = ((np.arange(g.n * sub) + 0.5) * g.dx / sub) % g.period_x
    row = (np.abs(xs - g.period_x / 2) < w / 2).reshape(g.n, sub).mean(1)
    m[:] = row[None, :]
    I = im.image(torch.tensor(m), [0])[0, 0].numpy()
    # intensity at the target edge (interpolated)
    xe = g.period_x / 2 + w / 2
    return float(np.interp(xe, x[: int(g.period_x / g.dx) + 2], I[: int(g.period_x / g.dx) + 2]))
