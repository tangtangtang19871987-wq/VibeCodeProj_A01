"""Scalar Abbe imaging + threshold resist, differentiable (PyTorch).

Deliberately simple and fully specified so every number in the study can be
reproduced.  Units: nm.  Wafer-scale coordinates everywhere (mask is 1x).

Model
-----
* Partially coherent scalar Abbe imaging, immersion NA=1.35, lambda=193 nm.
  I(x) = sum_s w_s | IFFT( M(f) * P(f + f_s) ) |^2
* Defocus as an exact (non-paraxial) pupil phase in a medium of index n_imm.
* Resist: Gaussian blur of the aerial image (acid diffusion) + constant
  threshold.  A dose corner D scales the image: printed <=> D * I_blur > th.
* Mask: real transmission in [0, 1] (binary dark-field: bright = trench/line).

Limitations (stated in the report): scalar (no polarization), thin mask
(no M3D), constant-threshold resist, 2D only.  The study is about ILT recipe
mechanics on a fixed model, which is exactly the regime these limits allow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

torch.set_num_threads(4)
DTYPE = torch.float32
CDTYPE = torch.complex64


@dataclass
class Optics:
    wavelength: float = 193.0
    na: float = 1.35
    n_imm: float = 1.44
    # leaf-shaped X-dipole: sigma_x in [sx_in, 1], |sigma_y| <= sy_max
    sx_in: float = 0.90
    sy_max: float = 0.30
    src_step: float = 0.035          # source sampling step in sigma units
    blur_nm: float = 5.0             # resist acid-diffusion sigma
    focus_list: tuple = (0.0, -40.0, 40.0)
    # if set: keep only source points where the 0th AND one 1st order of this
    # pitch both pass the pupil (no background-only points)
    two_beam_pitch: float | None = None
    # attenuated PSM: intensity transmission of the dark area (phase pi); 0 = binary
    att_T: float = 0.0


@dataclass
class Grid:
    n: int = 256
    period_x: float = 74.5           # the metal pitch
    tracks: int = 8
    lx: float = field(init=False)
    dx: float = field(init=False)

    def __post_init__(self):
        self.lx = self.period_x * self.tracks
        self.dx = self.lx / self.n


def source_points(op: Optics):
    """Sample a symmetric leaf X-dipole on a square sigma grid."""
    pts = []
    s = np.arange(-1.0, 1.0 + 1e-9, op.src_step)
    for sx in s:
        for sy in s:
            if sx * sx + sy * sy > 1.0:
                continue
            if op.two_beam_pitch is not None:
                shift = op.wavelength / (op.two_beam_pitch * op.na)
                if (abs(sx) - shift) ** 2 + sy * sy <= 1.0:
                    pts.append((sx, sy))
            elif abs(sx) >= op.sx_in and abs(sy) <= op.sy_max:
                pts.append((sx, sy))
    pts = np.array(pts)
    w = np.ones(len(pts)) / len(pts)
    return pts, w


class Imager:
    """Precomputes shifted, defocused pupils for every (focus, source point)."""

    def __init__(self, grid: Grid, op: Optics | None = None, keep: float = 0.995):
        self.g = grid
        self.op = op or Optics()
        n, dx = grid.n, grid.dx
        f = np.fft.fftfreq(n, d=dx)
        fx, fy = np.meshgrid(f, f, indexing="xy")      # [y, x]
        self.fx, self.fy = fx, fy
        fc = self.op.na / self.op.wavelength
        self.src, self.w = source_points(self.op)
        pupils = []
        for z in self.op.focus_list:
            per_focus = []
            for sx, sy in self.src:
                gx, gy = fx + sx * fc, fy + sy * fc
                r2 = gx * gx + gy * gy
                inside = r2 <= fc * fc
                lam = self.op.wavelength
                n_i = self.op.n_imm
                kz = np.sqrt(np.maximum(n_i * n_i - lam * lam * r2, 0.0))
                phase = 2 * math.pi / lam * z * (kz - n_i)
                per_focus.append(np.where(inside, np.exp(1j * phase), 0))
            pupils.append(np.stack(per_focus))
        P = np.stack(pupils)                                   # [F,S,N,N]
        # SOCS: TCC = A^H A with A[s,f] = sqrt(w_s) P(f+f_s); kernels = right
        # singular vectors.  Exact up to the truncation (energy kept >= keep).
        band = np.any(np.abs(P) > 0, axis=(0, 1))
        idx = np.nonzero(band.ravel())[0]
        kers, eigs = [], []
        for fi in range(P.shape[0]):
            A = P[fi].reshape(P.shape[1], -1)[:, idx] * np.sqrt(self.w)[:, None]
            _, s, vh = np.linalg.svd(A, full_matrices=False)
            e = s ** 2
            k = int(np.searchsorted(np.cumsum(e) / e.sum(), keep) + 1)
            ker = np.zeros((k, n * n), complex)
            ker[:, idx] = vh[:k].conj()
            kers.append(ker.reshape(k, n, n))
            eigs.append(e[:k])
        kmax = max(len(e) for e in eigs)
        K = np.zeros((len(kers), kmax, n, n), complex)
        E = np.zeros((len(kers), kmax))
        for fi, (kk, ee) in enumerate(zip(kers, eigs)):
            K[fi, :len(ee)] = kk
            E[fi, :len(ee)] = ee
        self.n_kernels = [len(e) for e in eigs]
        self.kernels = torch.tensor(K, dtype=CDTYPE)           # [F,K,N,N]
        self.eig = torch.tensor(E, dtype=DTYPE)                # [F,K]
        b = self.op.blur_nm
        self.blur_tf = torch.tensor(
            np.exp(-2 * (math.pi * b) ** 2 * (fx * fx + fy * fy)), dtype=DTYPE)

    def aerial(self, mask: torch.Tensor, focus_idx=None) -> torch.Tensor:
        """mask [..., N, N] real -> aerial image [..., F, N, N]."""
        if self.op.att_T:
            t = -math.sqrt(self.op.att_T)                 # pi-phase background amplitude
            mask = t + (1 - t) * mask
        M = torch.fft.fft2(mask.to(CDTYPE))
        fsel = slice(None) if focus_idx is None else list(focus_idx)
        Kr, Ev = self.kernels[fsel], self.eig[fsel]
        E = torch.fft.ifft2(M[..., None, None, :, :] * Kr)
        I = (E.real ** 2 + E.imag ** 2)
        return torch.einsum("...fkyx,fk->...fyx", I, Ev)

    def resist(self, aerial: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(torch.fft.fft2(aerial.to(CDTYPE)) * self.blur_tf).real

    def image(self, mask, focus_idx=None):
        return self.resist(self.aerial(mask, focus_idx))


def clear_field_intensity(imager: Imager) -> float:
    one = torch.ones(imager.g.n, imager.g.n)
    return float(imager.aerial(one, [0])[0].mean())
