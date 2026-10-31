# SPDX-License-Identifier: BSD-3-Clause
"""Soft limits on raw policy outputs and pre-saturation controller demands.

Columns are squared relative thrust excess, squared relative torque excess,
and otherwise-clipped reference excess [action units^2]. Sum channels, not means.
Normalize both physical excesses by their positive upper command limit before
squaring. With the upper thrust cap disabled, retain its nominal scale only
for normalizing the lower-bound violation; do not add an upper-bound cost.
PPO INDI references remain unbounded SI values. Their actuator demands, not
the references themselves, are scored at the physics clock by the environment.
Each signed reward term is clipped to +/- cfg.rwd_limits_max_penalty per policy step.
"""
import torch


def relative_limit_cost(requested, lower, upper, scale):
    """Sum squared relative excesses, allowing an explicitly absent upper bound."""
    excess = (lower - requested).clamp_min(0)
    if upper is not None:
        excess = excess + (requested - upper).clamp_min(0)
    return (excess / scale).square().sum(-1)


def direct_thrust_request(actions, cfg, hover):
    """Raw direct-mode rotor demand [N], before action/thrust clipping or lag."""
    u = actions[:, :6]
    if cfg.sac_action_mapping:
        return hover * (1 + torch.where(u >= 0, (cfg.thrust_to_weight - 1) * u, u))
    return hover * (1 + cfg.rotor_action_scale * u)


def direct_thrust_limits(cfg, hover):
    """Bounds and normalization [N] used by the direct-mode soft-limit cost."""
    lower, upper = 0., hover * cfg.thrust_to_weight
    if not cfg.sac_action_mapping:
        if cfg.remove_thrust_clipping:
            upper = None
        else:
            span = hover * cfg.rotor_action_scale * cfg.action_clamp
            lower, upper = max(lower, hover - abs(span)), min(upper, hover + abs(span))
    return lower, upper, upper if upper is not None else hover * cfg.thrust_to_weight


def direct_joint_torque_request(actions, cfg):
    """Raw direct-mode joint demand [N*m]; PPO is physical, SAC normalized."""
    return actions[:, 6:8] * cfg.effort_lim_sim if cfg.sac_action_mapping else actions[:, 6:8]


def raw_action_limit_costs(actions, cfg, hover, *, legacy=False):
    costs = actions.new_zeros((actions.shape[0], 3))
    direct = legacy or cfg.control_mode in ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control")
    if direct and cfg.rwd_scale_thrust_limits:
        u = actions[:, :6]
        if legacy:
            upper = cfg.thrust_max if cfg.thrust_max > 0 else cfg.thrust_max_scale * hover
            lower = cfg.thrust_min
            if cfg.thrust_mode == "absolute":
                requested = lower + (u + 1) * 0.5 * (upper - lower)
            else:
                span = cfg.delta_thrust_scale * hover
                requested = hover + span * u
                # Include the earlier +/-1 policy clamp, not only the physical cap.
                lower, upper = max(lower, hover - span), min(upper, hover + span)
        else:
            requested = direct_thrust_request(actions, cfg, hover)
            lower, upper, _ = direct_thrust_limits(cfg, hover)
        thrust_limit = upper if upper is not None else hover * cfg.thrust_to_weight
        costs[:, 0] = relative_limit_cost(requested, lower, upper, thrust_limit)

    torque_output = (not cfg.des_joint_out if legacy else
                     cfg.control_mode == "thrust_and_torque_control" and not cfg.hold_arm)
    if torque_output and cfg.rwd_scale_torque_limits:
        if legacy:
            lower = actions.new_tensor(cfg.torque_min)
            upper = actions.new_tensor(cfg.torque_max)
            requested = lower + (actions[:, 6:8] + 1) * 0.5 * (upper - lower)
            requested[:, 1] += cfg.bias_torque_j2
            bias = actions.new_tensor([0., cfg.bias_torque_j2])
            lower = (lower + bias).clamp_min(-cfg.torque_lim)
            upper = (upper + bias).clamp_max(cfg.torque_lim)
        else:
            # Direct PPO actions already have N*m units; SAC actions remain normalized.
            requested = direct_joint_torque_request(actions, cfg)
            limit = cfg.effort_lim_sim
            lower, upper = -limit, limit
        costs[:, 1] = relative_limit_cost(requested, lower, upper, upper)

    if cfg.rwd_scale_action_limits:
        reference, limit = actions[:, :0], 1.
        if legacy and cfg.des_joint_out:
            reference = actions[:, 6:10]  # two atan2 vectors, clipped componentwise
        elif not legacy:
            if cfg.control_mode in ("indi", "wrench", "cbrt", "cbrt_indi"):
                if cfg.sac_action_mapping:
                    reference = actions[:, :-2] if cfg.hold_arm else actions  # normalized SAC references only
            elif not cfg.sac_action_mapping and not cfg.remove_thrust_clipping:
                limit = cfg.action_clamp
                if cfg.control_mode == "PI_rotors_PD_joint_pos_control":
                    reference = actions[:, :-2] if cfg.hold_arm else actions
                elif not torque_output and not cfg.hold_arm:
                    reference = actions[:, -2:]
        costs[:, 2] = (reference.abs() - limit).clamp_min(0).square().sum(-1)
    return costs


def clipped_limit_rewards(costs, cfg):
    """Signed soft-limit reward terms, each clipped to +/- cfg.rwd_limits_max_penalty."""
    bound = cfg.rwd_limits_max_penalty
    scales = {"thrust_limits": cfg.rwd_scale_thrust_limits, "torque_limits": cfg.rwd_scale_torque_limits,
              "action_limits": cfg.rwd_scale_action_limits}
    return {name: (scale * costs[:, i]).clamp(-bound, bound) for i, (name, scale) in enumerate(scales.items())}
