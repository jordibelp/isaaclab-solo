# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchNorm in the PPO value network, the PPO side of ``agent.critic.batch_norm`` (XQC, arXiv:2509.25174).

RSL-RL's PPO policy has no normalization option, so BatchNorm is inserted into the built value network. PPO
keeps the policy in training mode during rollouts and updates, so the value network always uses batch
statistics. SAC builds its BatchNorm critic in ``rsl_rl_sac`` instead.
"""

from __future__ import annotations

from torch import nn


def insert(critic: nn.Sequential) -> list[nn.Parameter]:
    """Put BatchNorm between each hidden Linear layer and its activation, and return the new parameters.

    The Sequential is edited in place, so Linear state-dict keys and an already compiled forward stay valid.
    """
    if not isinstance(critic, nn.Sequential):
        raise ValueError("agent.critic.batch_norm needs a PPO value network built as a Sequential MLP.")
    slots = list(critic._modules.items())
    linear_positions = [index for index, (_, module) in enumerate(slots) if isinstance(module, nn.Linear)]
    if len(linear_positions) < 2 or linear_positions[-1] != len(slots) - 1:
        raise ValueError("agent.critic.batch_norm needs hidden Linear/activation pairs and a final Linear layer.")
    parameters = []
    for index in linear_positions[:-1]:
        linear = slots[index][1]
        name, activation = slots[index + 1]
        if isinstance(activation, (nn.Linear, nn.Sequential, nn.LayerNorm, nn.BatchNorm1d)):
            raise ValueError("agent.critic.batch_norm needs an activation directly after each hidden Linear layer.")
        # XQC's settings: running statistics decay by 0.99 per update, eps 1e-3.
        norm = nn.BatchNorm1d(linear.out_features, momentum=0.01, eps=1e-3, device=linear.weight.device)
        critic.add_module(name, nn.Sequential(norm, activation))
        parameters += norm.parameters()
    return parameters


def in_checkpoint(model_state_dict: dict) -> bool:
    """Whether a PPO checkpoint's value network was trained with :func:`insert`."""
    return any(key.startswith("critic.") and key.endswith(".running_mean") for key in model_state_dict)
