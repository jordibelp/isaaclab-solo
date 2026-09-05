# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


import torch

from isaaclab_tasks.direct.solo12_race.reward_utils import dense_reaction_force_reward


def _force(alpha: float, azimuth: float, magnitude: float = 100.0) -> list[float]:
    tangent = magnitude * math.sin(alpha)
    return [
        tangent * math.cos(azimuth),
        tangent * math.sin(azimuth),
        magnitude * math.cos(alpha),
    ]


def test_dense_reaction_force_reward_matches_angle_and_direction_definition():
    mu_static = torch.full((1, 4), 1.0)
    mu_dynamic = torch.full((1, 4), 0.5)
    alpha_static = math.atan(1.0)
    alpha_dynamic = math.atan(0.5)
    alpha_midpoint = 0.5 * (alpha_static + alpha_dynamic)

    forces = torch.tensor(
        [[
            _force(alpha_dynamic, 0.0),
            _force(alpha_midpoint, 0.0),
            _force(alpha_static, math.pi / 2),
            _force(alpha_static, math.pi),
        ]]
    )
    forward_axes = torch.tensor([[[1.0, 0.0, 0.0]] * 4])

    reward = dense_reaction_force_reward(forces, forward_axes, mu_static, mu_dynamic, contact_threshold=1.0)

    # Per-foot terms are 0, +0.5, 0, and -1 respectively.
    torch.testing.assert_close(reward, torch.tensor([-0.5]), atol=1.0e-6, rtol=0.0)


def test_dense_reaction_force_reward_clips_magnitude_and_ignores_invalid_contacts():
    mu_static = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mu_dynamic = torch.tensor([[0.5, 0.5, 1.0, 0.5]])
    forces = torch.tensor(
        [[
            _force(math.atan(2.0), 0.0),
            [0.0, 0.0, 0.5],
            _force(math.atan(1.0), 0.0),
            [10.0, 0.0, -10.0],
        ]]
    )
    forward_axes = torch.tensor([[[1.0, 0.0, 0.0]] * 4])

    reward = dense_reaction_force_reward(forces, forward_axes, mu_static, mu_dynamic, contact_threshold=1.0)

    # Only the first foot is valid: its angle is above alpha_static and therefore clips to 1.
    torch.testing.assert_close(reward, torch.tensor([1.0]), atol=1.0e-6, rtol=0.0)
