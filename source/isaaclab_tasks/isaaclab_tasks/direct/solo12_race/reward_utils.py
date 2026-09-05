# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def dense_reaction_force_reward(
    reaction_forces_w: torch.Tensor,
    foot_forward_axes_w: torch.Tensor,
    mu_static: torch.Tensor,
    mu_dynamic: torch.Tensor,
    contact_threshold: float,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Sum the friction-angle-weighted forward reaction-force alignment over the feet.

    The force inclination is ``alpha = atan2(||F_xy||, max(F_z, 0))``. Its magnitude score is
    ``clip((alpha - atan(mu_dynamic)) / (atan(mu_static) - atan(mu_dynamic)), 0, 1)``. The direction
    score is the cosine between ``F_xy`` and the horizontal projection of the foot-frame +x axis.
    """
    tangent_force_w = reaction_forces_w[..., :2]
    tangent_force_norm = torch.linalg.vector_norm(tangent_force_w, dim=-1)
    normal_force = torch.clamp(reaction_forces_w[..., 2], min=0.0)
    force_norm = torch.linalg.vector_norm(reaction_forces_w, dim=-1)

    alpha = torch.atan2(tangent_force_norm, normal_force)
    alpha_static = torch.atan(torch.clamp(mu_static, min=0.0))
    alpha_dynamic = torch.atan(torch.clamp(mu_dynamic, min=0.0))
    angle_range = alpha_static - alpha_dynamic
    magnitude_score = torch.clamp((alpha - alpha_dynamic) / torch.clamp(angle_range, min=eps), 0.0, 1.0)

    tangent_direction_w = tangent_force_w / torch.clamp(tangent_force_norm.unsqueeze(-1), min=eps)
    foot_forward_xy_w = foot_forward_axes_w[..., :2]
    foot_forward_xy_w = foot_forward_xy_w / torch.clamp(
        torch.linalg.vector_norm(foot_forward_xy_w, dim=-1, keepdim=True), min=eps
    )
    direction_cosine = torch.sum(tangent_direction_w * foot_forward_xy_w, dim=-1)

    valid_contact = (
        (force_norm > contact_threshold)
        & (normal_force > eps)
        & (tangent_force_norm > eps)
        & (angle_range > eps)
    )
    return torch.sum(torch.where(valid_contact, direction_cosine * magnitude_score, 0.0), dim=1)
