# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import DelayBuffer

from ..solo12.solo12_env import Solo12Env
from .solo12_backflip_env_cfg import Solo12BackflipEnvCfg


# A landing up to this angle short of a whole turn still counts as a completed backflip.
_BACKFLIP_COUNT_TOLERANCE = math.pi / 4


def _backward_pitch(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """Nose-up angle of the base in its x-z plane: 0 upright, pi/2 nose straight up, pi upside down."""
    return torch.atan2(-projected_gravity_b[:, 0], -projected_gravity_b[:, 2])


def _completed_backflips(rotation: torch.Tensor) -> torch.Tensor:
    """Count whole backward turns in a net rotation angle."""
    return torch.floor((rotation + _BACKFLIP_COUNT_TOLERANCE) / (2.0 * math.pi)).clamp_min(0.0)


class Solo12BackflipEnv(Solo12Env):
    cfg: Solo12BackflipEnvCfg

    def __init__(self, cfg: Solo12BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        num_phases = len(cfg.backflip_curriculum_advance_thresholds) + 1
        self._backflip_phase = num_phases if cfg.skip_curriculum else int(cfg.backflip_curriculum_start_phase)
        super().__init__(cfg, render_mode, **kwargs)

        self._episode_sums["backflip_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        # Net backward rotation of the base since reset, unwrapped from the gravity direction (rad).
        self._backflip_rotation = torch.zeros(self.num_envs, device=self.device)
        self._backflip_pitch = torch.zeros(self.num_envs, device=self.device)
        self._backflip_pitch_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._airborne_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._window_backflips = 0.0
        self._window_episodes = 0
        self._last_window_mean_backflips: float | None = None
        if self.cfg.backflip_curriculum:
            self._validate_backflip_curriculum()
            max_delay = max(int(delay_range[1]) for delay_range in self.cfg.actuation_delay_range_curriculum)
            self._action_delay_buffer = DelayBuffer(max_delay, self.num_envs, device=self.device)
            self._set_backflip_phase(self._backflip_phase)

    def _validate_backflip_curriculum(self):
        num_phases = len(self.cfg.backflip_curriculum_advance_thresholds) + 1
        for name in (
            "include_events_randomization_curriculum",
            "actuation_delay_range_curriculum",
            "forces_applied_to_base_curriculum_by_phase",
            "base_push_force_z_range_curriculum",
        ):
            if len(getattr(self.cfg, name)) != num_phases:
                raise ValueError(
                    f"{name} needs one entry per phase ({num_phases}), got {len(getattr(self.cfg, name))}."
                )
        if not 1 <= self._backflip_phase <= num_phases:
            raise ValueError(f"backflip_curriculum_start_phase must be in [1, {num_phases}], got {self._backflip_phase}.")
        events = [bool(value) for value in self.cfg.include_events_randomization_curriculum]
        if any(active and not later for active, later in zip(events, events[1:])):
            raise ValueError("include_events_randomization_curriculum cannot switch randomization off again.")

    def _set_backflip_phase(self, phase: int):
        idx = phase - 1
        self._backflip_phase = phase
        self.cfg.actuation_delay_range = tuple(int(value) for value in self.cfg.actuation_delay_range_curriculum[idx])
        force = float(self.cfg.forces_applied_to_base_curriculum_by_phase[idx])
        self.cfg.base_push_force_xy_range = (-force, force)
        self.cfg.base_push_force_z_range = tuple(
            float(value) for value in self.cfg.base_push_force_z_range_curriculum[idx]
        )
        if self.cfg.include_events_randomization_curriculum[idx]:
            self._activate_curriculum_event_randomization()
        self._window_backflips = 0.0
        self._window_episodes = 0

    def _update_backflip_curriculum(self, backflips: torch.Tensor):
        """Advance once a window of num_envs finished episodes averages enough completed backflips."""
        if (
            not self.cfg.backflip_curriculum
            or self._backflip_phase > len(self.cfg.backflip_curriculum_advance_thresholds)
            or len(backflips) == 0
        ):
            return
        self._window_backflips += backflips.sum().item()
        self._window_episodes += len(backflips)
        if self._window_episodes < self.num_envs:
            return
        self._last_window_mean_backflips = self._window_backflips / self._window_episodes
        self._window_backflips = 0.0
        self._window_episodes = 0
        if self._last_window_mean_backflips >= self.cfg.backflip_curriculum_advance_thresholds[self._backflip_phase - 1]:
            self._set_backflip_phase(self._backflip_phase + 1)

    def get_curriculum_global_idx(self) -> int | None:
        return self._backflip_phase - 1 if self.cfg.backflip_curriculum else None

    def get_curriculum_max_global_idx(self) -> int | None:
        return len(self.cfg.backflip_curriculum_advance_thresholds) if self.cfg.backflip_curriculum else None

    def _reward_scales(self) -> dict[str, float]:
        return {**super()._reward_scales(), "backflip_ang_vel": self.cfg.backflip_ang_vel_reward_scale}

    def _get_rewards(self) -> torch.Tensor:
        self._update_backflip_rotation()
        return super()._get_rewards()

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        backward_ang_vel = -self._robot.data.root_ang_vel_b[:, 1]
        clip = self.cfg.backflip_ang_vel_clip
        if clip > 0.0:
            backward_ang_vel = backward_ang_vel.clamp(-clip, clip)
        terms = super()._reward_terms()
        terms["backflip_ang_vel"] = backward_ang_vel * self.cfg.backflip_ang_vel_reward_scale * self.step_dt
        return terms

    def _update_backflip_rotation(self):
        """Accumulate the net backward rotation from the change of the gravity direction in the base frame."""
        gravity_b = self._robot.data.projected_gravity_b
        pitch = _backward_pitch(gravity_b)
        # The angle is undefined while the lateral axis is near vertical (robot on its side); skip those steps.
        valid = torch.linalg.vector_norm(gravity_b[:, [0, 2]], dim=1) > 0.5
        delta = math_utils.wrap_to_pi(pitch - self._backflip_pitch)
        self._backflip_rotation += torch.where(valid & self._backflip_pitch_valid, delta, 0.0)
        self._backflip_pitch = pitch
        self._backflip_pitch_valid = valid

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Only episodes that started on the ground measure flips the policy made itself.
        finished = env_ids[(self.episode_length_buf[env_ids] > 0) & ~self._airborne_start[env_ids]]
        rotation = self._backflip_rotation[finished]
        backflips = _completed_backflips(rotation)
        self._update_backflip_curriculum(backflips)

        super()._reset_idx(env_ids)

        log = self.extras["log"]
        if len(finished) > 0:
            log["Episode/backflips"] = backflips.mean().item()
            log["Episode/backward_rotation_rad"] = rotation.mean().item()
        log["Curriculum/backflip_phase"] = self._backflip_phase
        log["Curriculum/base_push_force_z_abs"] = max(abs(value) for value in self.cfg.base_push_force_z_range)
        if self._last_window_mean_backflips is not None:
            log["Curriculum/backflip_window_mean_backflips"] = self._last_window_mean_backflips

        self._backflip_rotation[env_ids] = 0.0
        self._backflip_pitch_valid[env_ids] = False
        self._reset_airborne(env_ids)

    def _reset_airborne(self, env_ids: torch.Tensor):
        """Move a random subset of the reset envs mid-air, already rotating backwards."""
        self._airborne_start[env_ids] = False
        if self.cfg.backflip_airborne_reset_prob <= 0.0:
            return
        env_ids = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.backflip_airborne_reset_prob]
        if len(env_ids) == 0:
            return
        self._airborne_start[env_ids] = True
        count = len(env_ids)

        # Keep the position and heading of the regular reset, then lift and pitch the base nose-up.
        root_pose = self._robot.data.root_link_pose_w[env_ids].clone()
        root_pose[:, 2] = self._terrain.env_origins[env_ids, 2] + torch.empty(count, device=self.device).uniform_(
            *self.cfg.backflip_airborne_reset_height_range
        )
        rotation = torch.empty(count, device=self.device).uniform_(*self.cfg.backflip_airborne_reset_rotation_range)
        lateral_axis = torch.tensor((0.0, 1.0, 0.0), device=self.device).expand(count, -1)
        root_pose[:, 3:7] = math_utils.quat_mul(
            root_pose[:, 3:7], math_utils.quat_from_angle_axis(-rotation, lateral_axis)
        )

        ang_vel_b = torch.zeros(count, 3, device=self.device)
        ang_vel_b[:, 1] = -torch.empty(count, device=self.device).uniform_(*self.cfg.backflip_airborne_reset_ang_vel_range)
        root_velocity = torch.zeros(count, 6, device=self.device)
        root_velocity[:, 3:6] = math_utils.quat_apply(root_pose[:, 3:7], ang_vel_b)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids)
