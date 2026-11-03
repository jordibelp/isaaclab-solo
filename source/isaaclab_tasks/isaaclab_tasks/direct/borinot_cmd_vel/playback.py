# SPDX-License-Identifier: BSD-3-Clause
"""Play-only command ownership. The registered training environment is unchanged."""
import math
from pathlib import Path

import torch
import yaml

from .borinot_cmd_vel_env import BorinotCmdVelEnv
from .task_math import command_force_curriculum_stages, sample_steps, sample_vertical_pushes


def checkpoint_force_level(infos, configured_level, override_keys):
    """Use checkpoint stage metadata only when the caller did not select a stage."""
    if override_keys.intersection(("env.curriculum_start_level", "env.skip_curriculum",
                                   "env.command_lin_vel_lim_curriculum", "env.forces_applied_to_base_curriculum")):
        return configured_level
    level = infos.get("best_model_curriculum_idx")
    if level is None:
        level = infos.get("curriculum_global_idx")
    return configured_level if level is None else int(level)


def configure_play_curriculum(cfg, infos, checkpoint, override_keys, level=None):
    """Restore stage definitions, not training progression or other environment settings.

    Downloaded checkpoints may carry <stem>.env.yaml; checkpoints in their
    original run use params/env.yaml. Explicit Hydra fields always take priority.
    BaseLoader reads numeric sequences without constructing Python YAML objects.
    """
    checkpoint = Path(checkpoint)
    source = next((p for p in (checkpoint.with_suffix(".env.yaml"), checkpoint.parent / "params/env.yaml")
                   if p.is_file()), None)
    if source is not None:
        saved = yaml.load(source.read_text(), Loader=yaml.BaseLoader)
        for field in ("command_lin_vel_lim_curriculum", "forces_applied_to_base_curriculum",
                      "command_lin_vel_x_range", "command_lin_vel_y_range",
                      "command_lin_vel_z_range", "command_ang_vel_z_range"):
            if field in saved and f"env.{field}" not in override_keys:
                setattr(cfg, field, tuple(map(float, saved[field])))
    selected = (checkpoint_force_level(infos, cfg.curriculum_start_level, override_keys)
                if level is None else level)
    if level is not None:
        cfg.skip_curriculum = False
    if not cfg.forces_applied_to_base_curriculum:
        raise ValueError("forces_applied_to_base_curriculum must contain non-negative finite force limits.")
    stages = command_force_curriculum_stages(cfg.command_lin_vel_lim_curriculum,
                                             cfg.forces_applied_to_base_curriculum)
    if cfg.skip_curriculum:
        selected = len(stages) - 1
    if not 0 <= selected < len(stages):
        raise ValueError(
            f"Playback curriculum stage {selected} is outside 0..{len(stages) - 1}. "
            "The stage index alone does not contain the training schedule. Keep the run's params/env.yaml "
            "or a <checkpoint-stem>.env.yaml beside downloaded weights, pass the training "
            "env.command_lin_vel_lim_curriculum and env.forces_applied_to_base_curriculum, "
            "or select a valid fixed --curriculum-level.")
    cfg.curriculum_start_level = selected
    command_limit = stages[selected][0]
    if command_limit is not None:
        cfg.command_lin_vel_x_range = cfg.command_lin_vel_y_range = (-command_limit, command_limit)
    return source


class BorinotCmdVelPlayEnv(BorinotCmdVelEnv):
    """Keep a live command across resampling, partial resets and auto resets."""

    def __init__(self, cfg, command=None, **kwargs):
        self.play_command = None
        self.play_push_limit_n = None  # None delegates to the unchanged training scheduler.
        self.play_push_automatic = True
        self.play_push_point_xy = None
        self.play_push_request = None
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

    def configure_play_pushes(self, limit_n, automatic, point_xy=None):
        """Play-only settings. A changed limit affects the NEXT push, not an active one."""
        if not math.isfinite(limit_n) or limit_n < 0:
            raise ValueError("Push magnitude must be finite and non-negative.")
        if point_xy is not None:
            if len(point_xy) != 2 or not all(math.isfinite(v) for v in point_xy):
                raise ValueError("Push point must contain two finite body-link coordinates.")
            if math.hypot(*point_xy) > self.cfg.base_push_application_radius_m + 1e-9:
                raise ValueError("Push point must lie inside the training application disk.")
            point_xy = tuple(point_xy)
        self.play_push_limit_n = float(limit_n)
        self.play_push_automatic = bool(automatic)
        self.play_push_point_xy = point_xy

    def request_play_push(self, mode="sample"):
        if mode not in ("sample", "up", "down"):
            raise ValueError("Push mode must be sample, up, or down.")
        self.play_push_request = mode

    def stop_play_pushes(self):
        self.play_push_automatic = False
        self.play_push_request = None
        self._base_push_steps_left.zero_()
        self._base_push_force_z_w.zero_()

    def _update_pushes(self):
        if self.play_push_limit_n is None and self.play_push_automatic and self.play_push_request is None:
            return super()._update_pushes()
        # Same sampler, policy-clock durations, start-to-start timing and physical
        # force/CoM-moment path as training. Only scheduling and selection differ.
        if self.play_push_request is not None:
            ids = self.robot._ALL_INDICES
        elif self.play_push_automatic:
            ids = (self._base_push_steps_until_next <= 0).nonzero().flatten()
        else:
            ids = self.robot._ALL_INDICES[:0]
        if len(ids):
            limits = (self._force_limits[self._episode_stage[ids]] if self.play_push_limit_n is None else
                      self._base_push_force_z_w.new_full((len(ids),), self.play_push_limit_n))
            force, points = sample_vertical_pushes(limits, self.cfg.base_push_application_radius_m,
                                                    self.cfg.base_push_application_half_height_m)
            if self.play_push_request in ("up", "down"):
                force = limits * (1 if self.play_push_request == "up" else -1)
                points[:, 2] = torch.where(force >= 0, -self.cfg.base_push_application_half_height_m,
                                           self.cfg.base_push_application_half_height_m)
            if self.play_push_point_xy is not None:
                points[:, :2] = points.new_tensor(self.play_push_point_xy)
            self._base_push_force_z_w[ids], self._base_push_points_b[ids] = force, points
            self._base_push_steps_left[ids] = sample_steps(len(ids), self.cfg.base_push_duration_range_s,
                                                           self.step_dt, self.device)
            self._base_push_steps_until_next[ids] = sample_steps(len(ids), self.cfg.base_push_interval_range_s,
                                                                self.step_dt, self.device)
        self.play_push_request = None
        self._base_push_force_z_w[self._base_push_steps_left <= 0] = 0.
        self._base_push_steps_left.sub_(1).clamp_min_(0)
        self._base_push_steps_until_next.sub_(1)
