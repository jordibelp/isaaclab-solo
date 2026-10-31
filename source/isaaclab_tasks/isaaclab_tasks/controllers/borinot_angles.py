# SPDX-License-Identifier: BSD-3-Clause
"""Optional (cos, sin) policy outputs, expanded in place of scalar angles.

Pairs decode to radians, without scalar action scales/offsets. Scalar channels
retain their old contract. The zero vector means zero radians. This changes
only the policy representation, not the downstream joint PD/PID error law.
"""
import torch


def angle_action_slots(cfg):
    """Canonical scalar indices replaced by pairs, in policy-output order."""
    mode = cfg.control_mode
    slots = []
    if mode in ("indi", "wrench") and getattr(cfg, "predict_yaw_as_sincos", False):
        slots.append(6 if mode == "indi" else 3)
    if mode != "thrust_and_torque_control":
        start = 7 if mode in ("indi", "wrench") else (6 if mode == "thrust_and_PD_joint_pos_control" else 4)
        for index, joint in enumerate(("thigh", "calf")):
            if getattr(cfg, f"predict_{joint}_joint_pos_as_sincos", False):
                slots.append(start + index)
    return tuple(slots)


def angle_action_size(cfg):
    base = 9 if cfg.control_mode in ("indi", "wrench") else (
        8 if cfg.control_mode.startswith("thrust_and_") else 6)
    return base + len(angle_action_slots(cfg))


def angle_from_pair(pair):
    """Normalize robustly before atan2(sin, cos), including tiny/large vectors."""
    scale = pair.abs().amax(dim=-1, keepdim=True)
    unit = pair / scale.clamp_min(torch.finfo(pair.dtype).tiny)
    unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(pair.dtype).tiny)
    cosine = torch.where(scale[:, 0] > 0, unit[:, 0], 1.0)
    sine = torch.where(scale[:, 0] > 0, unit[:, 1], 0.0)
    return torch.atan2(sine, cosine)


def decode_angle_actions(actions, cfg):
    """Collapse pairs to the old controller layout; leave other samples intact."""
    slots = angle_action_slots(cfg)
    if not slots:
        return actions
    parts, start = [], 0
    for extra, slot in enumerate(slots):
        index = slot + extra
        parts.extend((actions[:, start:index], angle_from_pair(actions[:, index:index+2])[:, None]))
        start = index + 2
    parts.append(actions[:, start:])
    return torch.cat(parts, dim=-1)


def clamp_scalar_actions(actions, cfg, limit):
    """Clamp existing scalar channels only: component clipping distorts pair angles."""
    slots = angle_action_slots(cfg)
    if not slots:
        return actions.clamp(-limit, limit)
    parts, start = [], 0
    for extra, slot in enumerate(slots):
        index = slot + extra
        parts.extend((actions[:, start:index].clamp(-limit, limit), actions[:, index:index+2]))
        start = index + 2
    parts.append(actions[:, start:].clamp(-limit, limit))
    return torch.cat(parts, dim=-1)


def joint_angle_targets(decoded, cfg, scale, offset=None):
    """Apply old scalar mapping, replacing only enabled joints with decoded radians."""
    targets = decoded[:, -2:] * scale
    if offset is not None:
        targets = targets + offset
    for index, joint in enumerate(("thigh", "calf")):
        if getattr(cfg, f"predict_{joint}_joint_pos_as_sincos", False):
            targets[:, index] = decoded[:, -2+index]
    return targets
