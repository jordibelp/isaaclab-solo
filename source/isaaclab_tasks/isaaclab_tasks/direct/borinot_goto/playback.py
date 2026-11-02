# SPDX-License-Identifier: BSD-3-Clause
"""Play-only fixed commands. Training always samples goals at resets."""
import math

from .borinot_goto_env import BorinotGoToEnv


class BorinotGoToPlayEnv(BorinotGoToEnv):
    def __init__(self, cfg, goal_offset=None, deadline_s=None, **kwargs):
        self._play_goal_offset = goal_offset
        self._play_deadline_s = deadline_s
        if (goal_offset is None) != (deadline_s is None):
            raise ValueError("Fixed playback needs both --goal-offset and --deadline_s.")
        if goal_offset is not None:
            if len(goal_offset) != 3 or not all(map(math.isfinite, goal_offset)):
                raise ValueError("goal-offset needs three finite metres.")
            if not math.isfinite(deadline_s) or deadline_s <= 0:
                raise ValueError("deadline_s must be finite and positive.")
            low, high = cfg.flight_height_range_m
            if not low < cfg.robot_home[2] + goal_offset[2] - cfg.reset_z_noise_m < high:
                raise ValueError("The fixed goal must lie inside the flight height bounds.")
            if not low < cfg.robot_home[2] + goal_offset[2] + cfg.reset_z_noise_m < high:
                raise ValueError("The fixed goal plus reset noise must lie inside the flight height bounds.")
            if any(abs(cfg.robot_home[i] + goal_offset[i]) + cfg.reset_xy_noise_m >= cfg.flight_xy_limit_m for i in (0, 1)):
                raise ValueError("The fixed goal must lie inside the XY flight bounds.")
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if self._play_goal_offset is not None:
            offset = self._goal_position_w.new_tensor(self._play_goal_offset)
            self._goal_position_w[env_ids] = self.robot.data.root_pos_w[env_ids] + offset
            self._goal_deadline_steps[env_ids] = max(1, math.ceil(self._play_deadline_s / self.step_dt))
            distance = offset.norm()
            self._initial_goal_distance[env_ids] = distance
            self._previous_goal_distance[env_ids] = distance
            self._minimum_goal_distance[env_ids] = distance
            self._first_arrival_s[env_ids] = 0.0 if distance <= self.cfg.goal_success_tolerance_m else -1.0


def checkpoint_goal_level(infos, fallback, override_keys):
    if "env.goal_curriculum_start_level" in override_keys:
        return fallback
    for key in ("best_model_curriculum_idx", "curriculum_global_idx"):
        if infos.get(key) is not None:
            return int(infos[key])
    return fallback
