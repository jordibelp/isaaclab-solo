# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""LaProp optimizer and adaptive gradient clipping (DreamerV3 defaults).

LaProp (Ziyin et al., 2020) normalizes the gradient by its running second moment
*before* applying momentum, which DreamerV3 uses for stable, scale-invariant
updates.  Adaptive Gradient Clipping (AGC) rescales each parameter's gradient by
the ratio of its norm to the parameter norm, implemented with fused ``_foreach``
ops for speed.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import Optimizer


class LaProp(Optimizer):
    """LaProp: momentum on the second-moment-normalized gradient."""

    def __init__(self, params, lr: float = 4e-5, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-20,
                 weight_decay: float = 0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients.")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["exp_avg_lr_1"] = 0.0
                    state["exp_avg_lr_2"] = 0.0
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                state["exp_avg_lr_1"] = state["exp_avg_lr_1"] * beta1 + (1.0 - beta1) * lr
                state["exp_avg_lr_2"] = state["exp_avg_lr_2"] * beta2 + (1.0 - beta2)

                bias_correction1 = state["exp_avg_lr_1"] / lr if lr != 0.0 else 1.0
                step_size = 1.0 / bias_correction1 if bias_correction1 != 0.0 else 0.0
                bias_correction2 = state["exp_avg_lr_2"]

                denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                step_of_grad = grad / denom
                exp_avg.mul_(beta1).add_(step_of_grad, alpha=(1.0 - beta1) * lr)

                p.add_(exp_avg, alpha=-step_size)
                if weight_decay != 0.0:
                    p.add_(p, alpha=-weight_decay)
        return loss


@torch.no_grad()
def clip_grad_agc_(parameters: Iterable[torch.Tensor], clip: float, pmin: float) -> None:
    """Adaptive gradient clipping: scale g so ||g|| <= clip * max(||p||, pmin), per-tensor."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    params, grads = [], []
    for p in parameters:
        if p.grad is not None:
            params.append(p)
            grads.append(p.grad)
    if not grads:
        return

    # Group by (device, dtype) so the fused foreach ops are legal.
    from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

    grouped = _group_tensors_by_device_and_dtype([params, grads])
    for (_device, _dtype), ([dev_params, dev_grads], _indices) in grouped.items():
        pnorm = torch._foreach_norm(dev_params, 2)
        gnorm = torch._foreach_norm(dev_grads, 2)
        upper = torch._foreach_mul(torch._foreach_maximum(pnorm, pmin), clip)
        # scale = 1 / max(gnorm / upper, 1)  -> only shrinks gradients that exceed the bound.
        scale = torch._foreach_reciprocal(torch._foreach_maximum(torch._foreach_div(gnorm, upper), 1.0))
        torch._foreach_mul_(dev_grads, scale)
