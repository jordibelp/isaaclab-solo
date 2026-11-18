# SPDX-License-Identifier: BSD-3-Clause
"""Policy-command latency shared by the Borinot tasks. Pure PyTorch; no simulator imports."""
import math

import torch


def _checked_seconds(seconds, name):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite non-negative number of seconds, got {seconds!r}.")
    return seconds


def delay_steps(seconds, physics_hz, name="actuator_delay"):
    """Whole physics steps for a delay in seconds. Off-grid values fail instead of rounding."""
    seconds = _checked_seconds(seconds, name)
    steps = round(seconds * physics_hz)
    if abs(steps - seconds * physics_hz) > 1e-6:
        raise ValueError(f"{name}={seconds:g} s is not a whole number of {1000 / physics_hz:g} ms physics steps.")
    return steps


def _delay_plan(cfg, physics_hz):
    """Return queue bounds and, if enabled, the requested continuous range in seconds."""
    mode = getattr(cfg, "actuator_delay_sampling", "ticks")
    if mode not in ("ticks", "seconds"):
        raise ValueError("actuator_delay_sampling must be 'ticks' or 'seconds'.")
    fixed = delay_steps(cfg.actuator_delay, physics_hz)
    bounds = cfg.actuator_delay_range
    if bounds is None or (bounds == "auto" and fixed):
        return fixed, fixed, None
    if bounds == "auto":
        if mode == "ticks":
            return 0, math.floor(0.020 * physics_hz + 1e-9), None
        bounds = (0.0, 0.020)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("actuator_delay_range must be auto, None or (min, max) seconds.")
    if fixed:
        raise ValueError("Set either actuator_delay or actuator_delay_range, not both.")
    name = "actuator_delay_range"
    low, high = (_checked_seconds(value, name) for value in bounds)
    if low > high:
        raise ValueError(f"{name} must be ordered (min, max).")
    if mode == "seconds":
        # Avoid a spurious extra tick when an exact grid endpoint has floating-point error.
        return math.ceil(low * physics_hz - 1e-9), math.ceil(high * physics_hz - 1e-9), (low, high)
    return delay_steps(low, physics_hz, name), delay_steps(high, physics_hz, name), None


def actuator_delay_bounds(cfg):
    """Effective (min, max) policy-command delay in physics steps."""
    hz = cfg.physics_hz if hasattr(cfg, "physics_hz") else 1 / cfg.sim.dt
    low, high, _ = _delay_plan(cfg, hz)
    return low, high


class ActuatorDelay:
    """Delay each policy command by whole physics steps, before every controller and motor model.

    INDI, the PI rate loop and the joint PD/PID still read current measurements; only the command
    they receive is old. Every robot reset keeps actuator_delay, or draws a uniform delay from
    actuator_delay_range, and refills that environment's queue with zero actions. For the modern
    tasks this means hover, home heading, and zero joint target or torque. The legacy task keeps
    its own zero-action mapping (including its configured thrust mode and torque bias).
    """

    def __init__(self, cfg, num_envs, action_dim, device, dtype=torch.float32):
        self.physics_dt = 1 / cfg.physics_hz if hasattr(cfg, "physics_hz") else cfg.sim.dt
        self._hz = cfg.physics_hz if hasattr(cfg, "physics_hz") else 1 / cfg.sim.dt
        self.low, self.high, self._seconds_range = _delay_plan(cfg, self._hz)
        self.steps = torch.full((num_envs,), self.low, dtype=torch.long, device=device)
        # A zero delay allocates nothing and returns the policy command itself.
        self._queue = torch.zeros(self.high + 1, num_envs, action_dim, dtype=dtype, device=device) if self.high else None
        self._cursor = 0  # queue slot of the newest command
        self._env_ids = torch.arange(num_envs, device=device)

    @property
    def seconds(self):
        return self.steps * self.physics_dt

    def reset(self, env_ids):
        if self.high > self.low:
            if self._seconds_range is None:
                self.steps[env_ids] = torch.randint(self.low, self.high + 1, (len(env_ids),), device=self.steps.device)
            else:
                low, high = self._seconds_range
                seconds = low + torch.rand(len(env_ids), dtype=torch.float64, device=self.steps.device) * (high - low)
                self.steps[env_ids] = (seconds * self._hz).ceil().long().clamp_(self.low, self.high)
        if self._queue is not None:
            self._queue[:, env_ids] = 0.0

    def __call__(self, actions):
        """Queue this physics step's command; return the one queued `steps` physics steps earlier."""
        if self._queue is None:
            return actions
        self._cursor = (self._cursor + 1) % len(self._queue)
        self._queue[self._cursor] = actions
        return self._queue[(self._cursor - self.steps) % len(self._queue), self._env_ids]
