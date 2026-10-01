"""Recent first-order optimizers, implemented from their papers for ILT.

These three are the current state of the art for *stochastic, mini-batch
neural-network training* (AlgoPerf 2024 winners and their successors). ILT is
a different problem -- deterministic, full-batch, nonlinear least squares over
one image-shaped parameter -- so they are included to be measured against it,
not assumed to transfer. See ``gril.ilt.gauss_newton`` for the method that
exploits ILT's structure directly, and docs/findings.md F-OPT-01 for results.

* :class:`ScheduleFreeAdamW` -- Defazio et al., "The Road Less Scheduled"
  (NeurIPS 2024). Won the AlgoPerf 2024 self-tuning track. Replaces the LR
  schedule with interpolation between an SGD-style iterate ``z`` and a
  weighted running average ``x``; gradients are taken at ``y`` between them.
* :class:`Muon` -- Jordan et al. (2024). Momentum whose update is replaced by
  its nearest semi-orthogonal matrix (quintic Newton-Schulz), i.e. steepest
  descent under the spectral norm. Designed for 2-D weight matrices; the ILT
  parameter field is itself a 2-D matrix, which is the reason to test it.
* :class:`SOAP` -- Vyas et al., "SOAP: Improving and Stabilizing Shampoo
  using Adam" (2024). Runs Adam in the eigenbasis of Shampoo's Kronecker
  preconditioner (``L = E[G G^T]``, ``R = E[G^T G]``), refreshed every
  ``precondition_frequency`` steps.

All three accept ``(H, W)`` or ``(B, H, W)`` parameters; a leading batch
dimension is treated as independent 2-D problems, never as one big matrix.
"""

from __future__ import annotations

import math

import torch


def _as_3d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.dim() == 2 else t


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Schedule-Free AdamW (train-mode iterate is ``y``; call :meth:`eval` for ``x``).

    Mirrors the reference ``AdamWScheduleFree`` (facebookresearch/schedule_free):
    ``y`` is stored in ``p``; ``z`` lives in optimizer state; ``x`` is never
    stored explicitly -- it is recovered from ``y`` and ``z`` on :meth:`eval`.
    The deliverable of a run is ``x`` (the averaged iterate), so callers must
    call :meth:`eval` before reading the parameters out.
    """

    def __init__(self, params, lr: float = 0.0025, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, warmup_steps: int = 0, weight_lr_power: float = 2.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, weight_lr_power=weight_lr_power,
                        k=0, weight_sum=0.0, lr_max=-1.0, train_mode=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def eval(self) -> None:
        """Move ``p`` from ``y`` to the averaged iterate ``x``."""
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for p in group["params"]:
                z = self.state.get(p, {}).get("z")
                if z is not None:
                    p.lerp_(end=z, weight=1 - 1 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:
        """Move ``p`` back from ``x`` to ``y`` so stepping can resume."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1 = group["betas"][0]
            for p in group["params"]:
                z = self.state.get(p, {}).get("z")
                if z is not None:
                    p.lerp_(end=z, weight=1 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group["train_mode"]:
                raise RuntimeError("ScheduleFreeAdamW.step() called in eval mode; call .train() first")
            beta1, beta2 = group["betas"]
            k = group["k"]
            warmup = group["warmup_steps"]
            sched = (k + 1) / warmup if k < warmup else 1.0
            lr = group["lr"] * sched * math.sqrt(1 - beta2 ** (k + 1))
            group["lr_max"] = lr_max = max(lr, group["lr_max"])
            weight = lr_max ** group["weight_lr_power"]
            group["weight_sum"] += weight
            ckp1 = weight / group["weight_sum"] if group["weight_sum"] > 0 else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state["z"] = p.detach().clone()
                    state["exp_avg_sq"] = torch.zeros_like(p)
                z, v = state["z"], state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                g = p.grad / v.sqrt().add_(group["eps"])
                if group["weight_decay"] != 0:
                    g.add_(p, alpha=group["weight_decay"])
                p.lerp_(end=z, weight=ckp1)
                p.add_(g, alpha=lr * (beta1 * (1 - ckp1) - 1))
                z.sub_(g, alpha=lr)
            group["k"] = k + 1
        return loss


def newton_schulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate ``U V^T`` of ``g = U S V^T`` (Muon's quintic iteration), batched.

    The coefficients (3.4445, -4.7750, 2.0315) are the reference ones: they
    maximise the slope at zero, so small singular values are driven towards 1
    in few steps at the cost of not converging exactly (singular values land
    in roughly [0.7, 1.2], which the reference work shows does not hurt).
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = _as_3d(g).float()
    x = x / (x.flatten(1).norm(dim=1).view(-1, 1, 1) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        a_mat = x @ x.transpose(-2, -1)
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.transpose(-2, -1)
    return x.reshape(g.shape).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalisation of the update.

    The orthogonalised update has unit spectral norm, so its per-entry RMS is
    ~``1/sqrt(max(rows, cols))`` and shrinks with resolution. Following the
    Moonlight scaling (Liu et al. 2025), it is rescaled by
    ``rms_scale * sqrt(max(rows, cols))`` so the update RMS is ``rms_scale``
    and ``lr`` means the same thing as an Adam learning rate at any canvas size.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, rms_scale: float = 0.2):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, rms_scale=rms_scale))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() not in (2, 3):
                    raise ValueError(f"Muon needs (H,W) or (B,H,W) parameters, got shape {tuple(p.shape)}")
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(p))
                buf.mul_(mu).add_(p.grad)
                u = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                o = newton_schulz5(u, steps=group["ns_steps"])
                scale = group["rms_scale"] * math.sqrt(max(p.shape[-2], p.shape[-1]))
                p.add_(o, alpha=-group["lr"] * scale)
        return loss


class SOAP(torch.optim.Optimizer):
    """SOAP: Adam in the (slowly refreshed) eigenbasis of Shampoo's preconditioner.

    Follows the reference implementation (nikhilvyas/SOAP): the first step
    only initialises the preconditioner (no parameter update); the
    eigenbasis is initialised with ``eigh`` and thereafter refreshed every
    ``precondition_frequency`` steps with one power iteration + QR, with the
    second-moment estimate re-ordered to follow the new basis. First moments
    are kept in the original space and rotated on use; second moments live in
    the rotated space.
    """

    def __init__(self, params, lr: float = 3e-3, betas=(0.95, 0.95), shampoo_beta: float = -1.0,
                 eps: float = 1e-8, precondition_frequency: int = 10):
        super().__init__(params, dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                                      precondition_frequency=precondition_frequency))

    @staticmethod
    def _eig_basis(mat: torch.Tensor) -> torch.Tensor:
        _, q = torch.linalg.eigh(mat + 1e-30 * torch.eye(mat.shape[-1], dtype=mat.dtype))
        return q.flip(-1)  # descending eigenvalue order, as in the reference

    @staticmethod
    def _refresh(gg: torch.Tensor, q: torch.Tensor, exp_avg_sq: torch.Tensor, dim: int):
        est = torch.diagonal(q.transpose(-2, -1) @ gg @ q, dim1=-2, dim2=-1)
        order = torch.argsort(est, descending=True)
        exp_avg_sq = exp_avg_sq.index_select(dim, order)
        q = q[:, order]
        q, _ = torch.linalg.qr(gg @ q)
        return q, exp_avg_sq

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            sbeta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else beta2
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() not in (2, 3):
                    raise ValueError(f"SOAP needs (H,W) or (B,H,W) parameters, got shape {tuple(p.shape)}")
                grads = _as_3d(p.grad)
                data = _as_3d(p.data)
                state = self.state[p]
                if "slices" not in state:
                    state["step"] = 0
                    state["slices"] = []
                    for g in grads:
                        left = g @ g.T
                        right = g.T @ g
                        state["slices"].append(dict(
                            L=(1 - sbeta) * left, R=(1 - sbeta) * right,
                            QL=self._eig_basis(left), QR=self._eig_basis(right),
                            exp_avg=torch.zeros_like(g), exp_avg_sq=torch.zeros_like(g),
                        ))
                    continue  # reference behaviour: the first step only sets up
                state["step"] += 1
                t = state["step"]
                refresh = t % group["precondition_frequency"] == 0
                for b, g in enumerate(grads):
                    s = state["slices"][b]
                    ql, qr = s["QL"], s["QR"]
                    g_rot = ql.T @ g @ qr
                    s["exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
                    s["exp_avg_sq"].mul_(beta2).addcmul_(g_rot, g_rot, value=1 - beta2)
                    m_rot = ql.T @ s["exp_avg"] @ qr
                    norm = m_rot / (s["exp_avg_sq"].sqrt() + group["eps"])
                    update = ql @ norm @ qr.T
                    step_size = group["lr"] * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                    data[b].add_(update, alpha=-step_size)
                    s["L"].mul_(sbeta).add_(g @ g.T, alpha=1 - sbeta)
                    s["R"].mul_(sbeta).add_(g.T @ g, alpha=1 - sbeta)
                    if refresh:
                        s["QL"], s["exp_avg_sq"] = self._refresh(s["L"], ql, s["exp_avg_sq"], 0)
                        s["QR"], s["exp_avg_sq"] = self._refresh(s["R"], qr, s["exp_avg_sq"], 1)
        return loss
