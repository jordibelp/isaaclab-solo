# SPDX-License-Identifier: BSD-3-Clause
"""Episode statistics of raw direct thrust/torque requests, not applied actuation."""
import torch

from isaaclab_tasks.direct.borinot.action_limits import (
    direct_joint_torque_request, direct_thrust_limits, direct_thrust_request,
)


class DesiredActuationMetrics:
    """All policy samples enter per-environment buffers; only full episodes are logged."""

    def __init__(self, cfg, hover, num_envs, device):
        self.cfg, self.hover = cfg, hover
        lower, upper, scale = direct_thrust_limits(cfg, hover)
        self.lower = torch.tensor([lower] * 6 + [-cfg.effort_lim_sim] * 2, device=device)
        self.upper = torch.tensor([float("inf") if upper is None else upper] * 6
                                  + [cfg.effort_lim_sim] * 2, device=device)
        self.scale = torch.tensor([scale] * 6 + [cfg.effort_lim_sim] * 2, device=device)
        self.channels = [(f"Desired_Thrust/rotor_{i+1}", "n") for i in range(6)]
        self.channels += [(f"Desired_Torque/{name}", "nm") for name in cfg.joint_names]
        self.steps = torch.zeros(num_envs, device=device)
        self.sum = torch.zeros(num_envs, 8, device=device)
        self.square_sum = torch.zeros_like(self.sum)
        self.minimum = torch.full_like(self.sum, float("inf"))
        self.maximum = torch.full_like(self.sum, -float("inf"))
        self.outside = torch.zeros_like(self.sum)
        self.cost_sum = torch.zeros(num_envs, 2, device=device)

    def update(self, actions):
        requested = torch.cat((direct_thrust_request(actions, self.cfg, self.hover),
                               direct_joint_torque_request(actions, self.cfg)), dim=-1)
        excess = (self.lower - requested).clamp_min(0) + (requested - self.upper).clamp_min(0)
        self.steps += 1
        self.sum += requested
        self.square_sum += requested.square()
        torch.minimum(self.minimum, requested, out=self.minimum)
        torch.maximum(self.maximum, requested, out=self.maximum)
        self.outside += excess > 0
        cost = (excess / self.scale).square()
        self.cost_sum[:, 0] += cost[:, :6].sum(-1)
        self.cost_sum[:, 1] += cost[:, 6:].sum(-1)

    def packet(self, completed):
        """Mean of per-episode statistics, matching Episode_Reward's reset batches."""
        steps = self.steps[completed, None]
        stats = {
            "mean": (self.sum[completed] / steps).mean(0),
            "rms": (self.square_sum[completed] / steps).sqrt().mean(0),
            "min": self.minimum[completed].mean(0),
            "max": self.maximum[completed].mean(0),
            "outside_limits_fraction": (self.outside[completed] / steps).mean(0),
        }
        log = {}
        for index, (name, unit) in enumerate(self.channels):
            for stat, values in stats.items():
                suffix = stat if stat == "outside_limits_fraction" else f"{stat}_{unit}"
                log[f"{name}/{suffix}"] = values[index]
        for index, group in enumerate(("thrust", "torque")):
            cost = self.cost_sum[completed, index]
            log[f"Desired_Actuation/{group}_relative_excess_episode_sum"] = cost.mean()
            log[f"Desired_Actuation/{group}_relative_excess_mean_per_step"] = (cost / steps[:, 0]).mean()
        log["Desired_Actuation/thrust_lower_n"] = self.lower[0]
        log["Desired_Actuation/thrust_normalization_n"] = self.scale[0]
        log["Desired_Actuation/thrust_upper_limit_enabled"] = float(not (
            self.cfg.remove_thrust_clipping and not self.cfg.sac_action_mapping))
        if not (self.cfg.remove_thrust_clipping and not self.cfg.sac_action_mapping):
            log["Desired_Actuation/thrust_upper_n"] = self.upper[0]
        log["Desired_Actuation/torque_abs_limit_nm"] = self.scale[6]
        return log

    def reset(self, env_ids):
        self.steps[env_ids] = 0
        for buffer in (self.sum, self.square_sum, self.outside, self.cost_sum):
            buffer[env_ids] = 0
        self.minimum[env_ids] = float("inf")
        self.maximum[env_ids] = -float("inf")
