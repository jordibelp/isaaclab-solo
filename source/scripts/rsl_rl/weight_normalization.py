# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Weight normalization as used by XQC (arXiv:2509.25174, Sec. 4).

XQC does not use the Salimans & Kingma reparameterization ``w = g * v / ||v||``. After every optimizer
step it projects the weights back onto the unit sphere. This keeps the parameter norm, and therefore the
effective learning rate of layers followed by a normalization layer, constant. As in the official code
(``norm_dense_layer``), each hidden unit's incoming weights and bias are rescaled together to unit L2 norm.

The output layers stay free. XQC also projects them, but ours start at or near zero on purpose (small
action means, uniform categorical critic logits), and a zero row has no direction to project.
"""

from __future__ import annotations

import torch
from torch import nn


def hidden_linear_layers(network: nn.Module) -> list[nn.Linear]:
    """Every Linear layer of an MLP except its output layer."""
    return [module for module in network.modules() if isinstance(module, nn.Linear)][:-1]


@torch.no_grad()
def project_to_unit_sphere(layers: list[nn.Linear]) -> None:
    """Rescale each unit's incoming weights and bias together to unit L2 norm.

    Linear layers in front of a SAC BatchNorm critic's BatchNorm have no bias, as in XQC.
    Zero rows have no direction to project and stay zero instead of producing NaNs.
    """
    for layer in layers:
        if layer.bias is None:
            norm = layer.weight.norm(dim=1)
        else:
            norm = torch.cat((layer.weight, layer.bias.unsqueeze(1)), dim=1).norm(dim=1)
        norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        layer.weight.div_(norm.unsqueeze(1))
        if layer.bias is not None:
            layer.bias.div_(norm)


def attach(runner) -> dict[str, int]:
    """Project the actor and critic hidden layers now and after each of their optimizer steps.

    Returns the number of projected layers per optimizer.
    """
    alg = runner.alg
    policy = getattr(alg, "policy", None)
    if policy is not None:  # PPO: one optimizer for actor and critic
        networks = {"policy": (alg.optimizer, [policy.actor, policy.critic])}
    else:  # SAC
        networks = {
            "actor": (alg.actor_optimizer, [alg.actor.mlp]),
            "critic": (alg.critic_optimizer, [alg.critic.critic1, alg.critic.critic2]),
        }

    counts = {}
    for name, (optimizer, modules) in networks.items():
        # dict.fromkeys drops trunk layers that a shared actor-critic lists twice.
        layers = list(dict.fromkeys(layer for module in modules for layer in hidden_linear_layers(module)))
        project_to_unit_sphere(layers)
        optimizer.register_step_post_hook(lambda _opt, _args, _kwargs, layers=layers: project_to_unit_sphere(layers))
        counts[name] = len(layers)
    if policy is None:
        # The target critics start as copies of the online critics, so they must start projected too.
        alg.critic.init_target_networks()
    return counts
