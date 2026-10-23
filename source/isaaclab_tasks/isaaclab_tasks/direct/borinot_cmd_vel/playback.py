# SPDX-License-Identifier: BSD-3-Clause
"""Play-only command ownership. The registered training environment is unchanged."""
import math

import torch

from .borinot_cmd_vel_env import BorinotCmdVelEnv


def checkpoint_force_level(infos, configured_level, override_keys):
    """Use checkpoint stage metadata only when the caller did not select a stage."""
    if "env.curriculum_start_level" in override_keys or "env.skip_curriculum" in override_keys:
        return configured_level
    level = infos.get("best_model_curriculum_idx")
    if level is None:
        level = infos.get("curriculum_global_idx")
    return configured_level if level is None else int(level)


class BorinotCmdVelPlayEnv(BorinotCmdVelEnv):
    """Keep a live command across resampling, partial resets and auto resets."""

    def __init__(self, cfg, command=None, **kwargs):
        self.play_command = None
        super().__init__(cfg, **kwargs)
        if command is not None:
            self.set_play_command(command)

    def set_play_command(self, command):
        if len(command) != 4 or not all(math.isfinite(value) for value in command):
            raise ValueError("Command must contain four finite values: vx vy vz wz.")
        self.play_command = torch.tensor(command, device=self.device, dtype=self._commands.dtype)
        self._resample_commands(self.robot._ALL_INDICES)

    def _resample_commands(self, env_ids):
        if self.play_command is None:
            return super()._resample_commands(env_ids)
        self._commands[env_ids] = self.play_command
        self._command_steps_left[env_ids] = max(1, math.ceil(self.cfg.command_resampling_time_s / self.step_dt))
