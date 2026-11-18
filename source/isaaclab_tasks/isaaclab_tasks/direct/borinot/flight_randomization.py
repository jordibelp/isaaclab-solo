# SPDX-License-Identifier: BSD-3-Clause
"""Borinot rotor plant and atmospheric disturbances. No simulator dependencies."""
import math

import torch


def validate_flight_randomization(cfg):
    def bounds(name, values, allow_zero=False):
        if (not isinstance(values, (list, tuple)) or len(values) != 2
                or not all(math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1]
                or (not allow_zero and values[0] == 0)):
            raise ValueError(f"{name} must be ordered, finite and {'non-negative' if allow_zero else 'positive'}.")

    if cfg.motor_time_constant_range_s is not None:
        bounds("motor_time_constant_range_s", cfg.motor_time_constant_range_s, allow_zero=True)
    for name in ("thrust_shared_gain_delta", "thrust_rotor_gain_delta", "km_gain_delta",
                 "wind_steady_max_m_s", "wind_gust_max_m_s", "wind_drag_n_per_m_s_squared", "external_moment_max_nm"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if cfg.thrust_shared_gain_delta + cfg.thrust_rotor_gain_delta >= 1 or cfg.km_gain_delta >= 1:
        raise ValueError("Rotor coefficient gains must stay strictly positive.")
    if not isinstance(cfg.wind_enabled, bool):
        raise ValueError("wind_enabled must be boolean.")
    bounds("gust_interval_range_s", cfg.gust_interval_range_s)
    bounds("gust_duration_range_s", cfg.gust_duration_range_s)
    if cfg.gust_duration_range_s[1] > cfg.gust_interval_range_s[0]:
        raise ValueError("Maximum gust duration must not exceed the minimum start-to-start interval.")


def make_flight_randomization(env, reset_thrust, nominal_tau):
    cfg = env.cfg.flight_randomization
    rotor = None
    if cfg.motor_time_constant_range_s is not None or any(
        (cfg.thrust_shared_gain_delta, cfg.thrust_rotor_gain_delta, cfg.km_gain_delta)
    ):
        rotor = RotorPlant(env.cfg, env.num_envs, env.physics_dt, env.device, reset_thrust, nominal_tau)
    wind = WindDisturbances(cfg, env.num_envs, env.step_dt, env.device) if cfg.wind_enabled else None
    return rotor, wind


class RotorPlant:
    """Convert nominal thrust commands to actual speed, force, and aerodynamic moment."""

    def __init__(self, task, n, dt, device, reset_thrust, nominal_tau):
        cfg = task.flight_randomization
        self.dt, self.nominal_kf = dt, task.kf
        self.spin = torch.tensor(task.rotor_spin, device=device)
        self.thrust_lag = getattr(task, "control_mode", None) == "PI_rotors_PD_joint_pos_control"

        def symmetric(width, columns):
            return torch.empty(n, columns, device=device).uniform_(-width, width) if width else 0.0

        bounds = cfg.motor_time_constant_range_s
        if bounds is None:
            self.alpha = -math.expm1(-dt / nominal_tau) if nominal_tau else 1.0
        else:
            tau = torch.empty(n, len(self.spin), device=device).uniform_(*bounds)
            # At tau=0 this evaluates to one (instant response).
            self.alpha = -torch.expm1(-dt / tau)
        gain = 1 + symmetric(cfg.thrust_shared_gain_delta, 1) + symmetric(cfg.thrust_rotor_gain_delta, len(self.spin))
        self.kf = torch.as_tensor(task.kf * gain, device=device).expand(n, len(self.spin))
        self.km = task.km * (1 + symmetric(cfg.km_gain_delta, 1))
        self.reset_speed = math.sqrt(reset_thrust / task.kf)
        self.speed = torch.full((n, len(self.spin)), self.reset_speed, device=device)

    def reset(self, ids):
        self.speed[ids] = self.reset_speed
        return self.kf[ids] * self.reset_speed**2

    def step(self, command_thrust, transient_effectiveness=None):
        # Normalize using the controller's nominal calibration, not the sampled kf.
        command = command_thrust / self.nominal_kf
        if self.thrust_lag:
            speed_sq = self.speed.square()
            next_speed = (speed_sq + self.alpha * (command - speed_sq)).clamp_min(0).sqrt()
        else:
            next_speed = self.speed + self.alpha * (command.sqrt() - self.speed)
        transient = None
        if transient_effectiveness is not None:
            transient = ((next_speed - self.speed) / self.dt) @ transient_effectiveness.T
        self.speed.copy_(next_speed)
        thrust = self.kf * self.speed.square()
        yaw = (self.km * self.speed.square() * self.spin).sum(-1)
        return thrust, yaw, transient

    def power(self):
        return (self.km * self.speed**3).sum(-1)


class WindDisturbances:
    """Episode-constant wind plus bounded gust/torque pulses, independent of vertical pushes."""

    def __init__(self, cfg, n, step_dt, device):
        self.cfg, self.device = cfg, device
        self.interval = tuple(max(1, math.ceil(t / step_dt)) for t in cfg.gust_interval_range_s)
        self.duration = tuple(max(1, math.ceil(t / step_dt)) for t in cfg.gust_duration_range_s)
        self.steady = torch.zeros(n, 3, device=device)
        self.gust = torch.zeros_like(self.steady)
        self.moment = torch.zeros_like(self.steady)
        self.until_next = torch.zeros(n, dtype=torch.long, device=device)
        self.left = torch.zeros_like(self.until_next)
        self.total = torch.ones_like(self.until_next)
        self.weight = torch.zeros(n, 1, device=device)

    def _horizontal(self, n, limit):
        angle, radius = torch.rand(2, n, device=self.device)
        angle *= 2 * math.pi
        radius = limit * radius.sqrt()
        return torch.stack((radius * angle.cos(), radius * angle.sin(), torch.zeros_like(radius)), dim=-1)

    def reset(self, ids):
        self.steady[ids] = self._horizontal(len(ids), self.cfg.wind_steady_max_m_s)
        self.gust[ids] = self.moment[ids] = self.weight[ids] = 0.0
        self.left[ids] = 0
        self.until_next[ids] = (torch.rand(len(ids), device=self.device) * self.interval[1]).long()

    def update(self):
        if not (self.cfg.wind_gust_max_m_s or self.cfg.external_moment_max_nm):
            return
        starts = (self.until_next <= 0).nonzero().flatten()
        n = len(starts)
        if n:
            self.gust[starts] = self._horizontal(n, self.cfg.wind_gust_max_m_s)
            # Uniform direction and bounded magnitude; no preferred world axis.
            direction = torch.nn.functional.normalize(torch.randn(n, 3, device=self.device), dim=-1)
            self.moment[starts] = direction * (torch.rand(n, 1, device=self.device) * self.cfg.external_moment_max_nm)
            self.total[starts] = self.left[starts] = torch.randint(
                self.duration[0], self.duration[1] + 1, (n,), device=self.device)
            self.until_next[starts] = torch.randint(self.interval[0], self.interval[1] + 1, (n,), device=self.device)
        # Sample a half-sine at tick centers; one-tick pulses still reach full strength.
        phase = (self.total - self.left + 0.5) / self.total
        self.weight[:, 0] = torch.where(self.left > 0, (math.pi * phase).sin(), 0.0)
        self.left.sub_(1).clamp_min_(0)
        self.until_next.sub_(1)

    def wrench_world(self, velocity_w):
        air_relative = self.steady + self.weight * self.gust - velocity_w
        force = self.cfg.wind_drag_n_per_m_s_squared * air_relative.norm(dim=-1, keepdim=True) * air_relative
        return force, self.weight * self.moment
