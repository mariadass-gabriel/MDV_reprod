"""IVON and uCBOpt optimizers matching the paper's reference implementation."""

from __future__ import annotations
from contextlib import contextmanager
from math import pow
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


# IVON (exact copy of the paper's reference implementation)

def _welford_mean(avg: Optional[Tensor], newval: Tensor, count: int) -> Tensor:
    return newval if avg is None else avg + (newval - avg) / count


class IVON(Optimizer):
    hessian_approx_methods = ("price", "gradsq")

    def __init__(
        self,
        params,
        lr: float,
        ess: float,
        hess_init: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        weight_decay: float = 1e-4,
        mc_samples: int = 1,
        hess_approx: str = "price",
        clip_radius: float = float("inf"),
        debias: bool = True,
        rescale_lr: bool = True,
    ):
        defaults = dict(lr=lr, mc_samples=mc_samples, beta1=beta1, beta2=beta2,
                        weight_decay=weight_decay, hess_init=hess_init, ess=ess,
                        clip_radius=clip_radius)
        super().__init__(params, defaults)
        self.mc_samples = mc_samples
        self.hess_approx = hess_approx
        self.current_step = 0
        self.debias = debias
        self.rescale_lr = rescale_lr
        self._numel, self._device, self._dtype = self._get_param_configs()
        self._reset_samples()
        self._init_buffers()

    def _get_param_configs(self):
        all_params = [p for pg in self.param_groups for p in pg["params"] if p is not None]
        for pg in self.param_groups:
            pg["numel"] = sum(p.numel() for p in pg["params"] if p is not None)
        if not all_params:
            return 0, torch.device("cpu"), torch.get_default_dtype()
        device = next(iter({p.device for p in all_params}))
        dtype = next(iter({p.dtype for p in all_params}))
        total = sum(pg["numel"] for pg in self.param_groups)
        return total, device, dtype

    def _reset_samples(self):
        self.state["count"] = 0
        self.state["avg_grad"] = None
        self.state["avg_nxg"] = None
        self.state["avg_gsq"] = None

    def _init_buffers(self):
        for group in self.param_groups:
            group["momentum"] = torch.zeros(
                group["numel"], device=self._device, dtype=self._dtype)
            group["hess"] = torch.zeros(
                group["numel"], device=self._device, dtype=self._dtype
            ).add(group["hess_init"])

    @contextmanager
    def sampled_params(self, train: bool = False):
        param_avg, noise = self._sample_params()
        yield
        self._restore_param_average(train, param_avg, noise)

    def _sample_params(self) -> Tuple[Tensor, Tensor]:
        noise_samples, param_avgs = [], []
        offset = 0
        for group in self.param_groups:
            gnumel = group["numel"]
            noise_sample = (
                torch.randn(gnumel, device=self._device, dtype=self._dtype)
                / (group["ess"] * (group["hess"] + group["weight_decay"])).sqrt()
            )
            noise_samples.append(noise_sample)
            goffset = 0
            for p in group["params"]:
                if p is None:
                    continue
                p_avg = p.data.flatten()
                numel = p.numel()
                p.data = (p_avg + noise_sample[goffset:goffset + numel]).view(p.shape)
                param_avgs.append(p_avg)
                goffset += numel
                offset += numel
        return torch.cat(param_avgs), torch.cat(noise_samples)

    def _restore_param_average(self, train: bool, param_avg: Tensor, noise: Tensor):
        param_grads = []
        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue
                p_slice = slice(offset, offset + p.numel())
                p.data = param_avg[p_slice].view(p.shape)
                if train:
                    param_grads.append(
                        p.grad.flatten() if p.requires_grad else torch.zeros_like(p).flatten())
                offset += p.numel()
        if train:
            grad_sample = torch.cat(param_grads)
            count = self.state["count"] + 1
            self.state["count"] = count
            self.state["avg_grad"] = _welford_mean(self.state["avg_grad"], grad_sample, count)
            if self.hess_approx == "price":
                self.state["avg_nxg"] = _welford_mean(
                    self.state["avg_nxg"], noise * grad_sample, count)
            else:
                self.state["avg_gsq"] = _welford_mean(
                    self.state["avg_gsq"], grad_sample.square(), count)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        self._update()
        self._reset_samples()
        return loss

    def _update(self):
        self.current_step += 1
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["beta1"], group["beta2"]
            wd = group["weight_decay"]
            pg_slice = slice(offset, offset + group["numel"])

            param_avg = torch.cat([p.flatten() for p in group["params"] if p is not None])

            group["momentum"] = b1 * group["momentum"] + (1 - b1) * self.state["avg_grad"][pg_slice]

            hess = group["hess"]
            if self.hess_approx == "price":
                f = self.state["avg_nxg"][pg_slice] * (hess + wd) * group["ess"]
            else:
                f = self.state["avg_gsq"][pg_slice]
            group["hess"] = (b2 * hess + (1 - b2) * f
                             + 0.5 * (1 - b2) ** 2 * (hess - f).square() / (hess + wd))

            debias = 1.0 - pow(b1, float(self.current_step)) if self.debias else 1.0
            lr_eff = lr * (group["hess_init"] + wd) if self.rescale_lr else lr
            hess_new = group["hess"]
            step_dir = (group["momentum"] / debias + wd * param_avg) / (hess_new + wd)
            step_dir = torch.clamp(step_dir, -group["clip_radius"], group["clip_radius"])
            param_avg = param_avg - lr_eff * step_dir

            pg_offset = 0
            for p in group["params"]:
                if p is not None:
                    p.data = param_avg[pg_offset:pg_offset + p.numel()].view(p.shape)
                    pg_offset += p.numel()
            offset += group["numel"]


# uCBOpt (exact copy of the paper's reference implementation)

ClosureType = Callable[[], Tensor]


class uCBOpt(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 0.2,
        beta1: float = 0.9,
        beta2: float = 0.99999,
        hess_init: float = 0.5,
        weight_decay: float = 0.0,
        cand_curvature: float = 0.0,
        eps: float = 1e-8,
        rescale_lr: bool = False,
    ):
        assert cand_curvature <= weight_decay, "cand_curvature must be <= weight_decay"
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, hess_init=hess_init,
                        weight_decay=weight_decay, cand_curvature=cand_curvature,
                        eps=eps, rescale_lr=rescale_lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[ClosureType] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr_base = group["lr"]
            b1, b2 = group["beta1"], group["beta2"]
            hess_init = group["hess_init"]
            wd = group["weight_decay"]
            cand = group["cand_curvature"]
            eps = group["eps"]
            rescale_lr = group["rescale_lr"]
            lr_eff = lr_base * (hess_init + wd) if rescale_lr else lr_base

            params_with_grad, grads, m_list, v_list, step_counts = [], [], [], [], []

            for p in group["params"]:
                if p is None or p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.full_like(p, float(hess_init))
                state["step"] += 1
                params_with_grad.append(p)
                grads.append(p.grad)
                m_list.append(state["m"])
                v_list.append(state["v"])
                step_counts.append(state["step"])

            if not params_with_grad:
                continue

            h_list = torch._foreach_mul(grads, grads)
            torch._foreach_mul_(m_list, b1)
            torch._foreach_add_(m_list, grads, alpha=1.0 - b1)
            torch._foreach_mul_(v_list, b2)
            torch._foreach_add_(v_list, h_list, alpha=1.0 - b2)

            bias_corr_m = [1.0 - (b1 ** t) for t in step_counts]
            m_bar = torch._foreach_div(m_list, bias_corr_m)

            denom = torch._foreach_add(v_list, wd)
            torch._foreach_sub_(denom, cand)
            torch._foreach_clamp_min_(denom, eps)
            numer = torch._foreach_add(m_bar, params_with_grad, alpha=wd)
            step_dir = torch._foreach_div(numer, denom)
            torch._foreach_add_(params_with_grad, step_dir, alpha=-lr_eff)

        return loss


def build_optimizer(name: str, params, **kwargs):
    """Factory: 'adamw', 'ivon', 'ucbopt'."""
    name = name.lower()
    if name == "adamw":
        from torch.optim import AdamW
        return AdamW(
            params,
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-2),
            betas=(kwargs.get("beta1", 0.9), kwargs.get("beta2", 0.999)))
    elif name == "ivon":
        return IVON(
            params,
            lr=kwargs.get("lr", 0.2),
            ess=kwargs.get("ess", 50000),
            hess_init=kwargs.get("hess_init", 0.5),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.99999),
            weight_decay=kwargs.get("weight_decay", 2e-3),
            mc_samples=kwargs.get("mc_samples", 1),
            hess_approx=kwargs.get("hess_approx", "price"),
            rescale_lr=kwargs.get("rescale_lr", True))
    elif name == "ucbopt":
        return uCBOpt(
            params,
            lr=kwargs.get("lr", 1e-2),
            weight_decay=kwargs.get("weight_decay", 2e-3),
            hess_init=kwargs.get("hess_init", 0.05),
            cand_curvature=kwargs.get("cand_curvature", 8e-6),
            beta1=kwargs.get("beta1", 0.9),
            beta2=kwargs.get("beta2", 0.99999),
            rescale_lr=kwargs.get("rescale_lr", False))
    else:
        raise ValueError(f"Unknown optimizer: {name}")
