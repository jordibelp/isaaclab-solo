# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-rank adapters for fine-tuning a pretrained network.

Fine-tuning every weight of a policy that already works is an easy way to lose it. A low-rank
adapter keeps the pretrained weights fixed and learns a small correction instead, so the
policy can only move inside a rank-limited subspace.

This mirrors the adapters in ``mujoco/train_lora.py``, which does the same thing for PPO in
JAX, so the two paths use the same ranks, gains and layer selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LAYER_CHOICES = ("all", "input_and_output", "input", "output")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank correction.

    The layer computes ``W x + b + (alpha / rank) * B A x``. ``A`` starts small and random and
    ``B`` starts at zero, so a freshly wrapped layer returns exactly what the layer it replaced
    returned. Fine-tuning starts from the pretrained policy rather than near it.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be at least 1, got {rank}.")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(0.01 * torch.randn(rank, base.in_features, **factory))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        correction = nn.functional.linear(nn.functional.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + self.scale * correction

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """The dense weight that reproduces this layer without an adapter."""
        return self.base.weight + self.scale * (self.lora_b @ self.lora_a)

    def extra_repr(self) -> str:
        return f"rank={self.lora_a.shape[0]}, scale={self.scale:g}"


def selected_layer_indices(count: int, mode: str) -> tuple[int, ...]:
    """Positions to adapt, using the same modes as ``mujoco/train_lora.py``."""
    if mode == "all":
        return tuple(range(count))
    if mode == "input":
        return (0,)
    if mode == "output":
        return (count - 1,)
    if mode == "input_and_output":
        return (0,) if count == 1 else (0, count - 1)
    raise ValueError(f"layers must be one of {LAYER_CHOICES}, got {mode!r}.")


def apply_lora(module: nn.Module, rank: int, alpha: float, layers: str = "all") -> int:
    """Freeze ``module`` and wrap the selected linear layers with adapters, in place.

    Layers are counted in registration order, which is forward order for the sequential MLPs
    used here. Returns how many were wrapped.
    """
    targets = [
        (parent, name, child)
        for parent in list(module.modules())
        for name, child in list(parent.named_children())
        if isinstance(child, nn.Linear)
    ]
    if not targets:
        raise ValueError("Found no linear layers to adapt.")
    selected = set(selected_layer_indices(len(targets), layers))

    # Freeze first: the adapters created below are the only parameters left trainable.
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    for index, (parent, name, child) in enumerate(targets):
        if index in selected:
            setattr(parent, name, LoRALinear(child, rank, alpha))
    return len(selected)


@torch.no_grad()
def merged_state_dict(module: nn.Module) -> dict:
    """State dict of ``module`` with every adapter folded into its dense weight.

    The keys match the module before ``apply_lora`` touched it, so a checkpoint written from
    this stays loadable by inference scripts and by later fine-tuning runs that know nothing
    about adapters.

    This reads the live module rather than a copy of it, because a model that has run a forward
    pass caches its output distribution and ``copy.deepcopy`` refuses the non-leaf tensors in it.
    """
    state = dict(module.state_dict())
    for prefix, adapter in module.named_modules():
        if not isinstance(adapter, LoRALinear):
            continue
        merged_weight = adapter.merged_weight()
        for suffix in ("base.weight", "lora_a", "lora_b"):
            del state[f"{prefix}.{suffix}"]
        state[f"{prefix}.weight"] = merged_weight
        bias = state.pop(f"{prefix}.base.bias", None)
        if bias is not None:
            state[f"{prefix}.bias"] = bias.clone()
    return state
