# SPDX-License-Identifier: BSD-3-Clause
"""Policy-command latency shared by the Borinot tasks. Pure PyTorch; no simulator imports."""
import math

import torch


def delay_steps(seconds, physics_hz, name="actuator_delay"):
    """Whole physics steps for a delay in seconds. Off-grid values fail instead of rounding."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite non-negative number of seconds, got {seconds!r}.")
    steps = round(seconds * physics_hz)
    if abs(steps - seconds * physics_hz) > 1e-6:
        raise ValueError(f"{name}={seconds:g} s is not a whole number of {1000 / physics_hz:g} ms physics steps.")
    return steps


def actuator_delay_bounds(cfg):
    """(min, max) policy-command delay in physics steps, from actuator_delay or actuator_delay_range."""
    physics_hz = cfg.physics_hz if hasattr(cfg, "physics_hz") else 1 / cfg.sim.dt
    fixed = delay_steps(cfg.actuator_delay, physics_hz)
    if cfg.actuator_delay_range is None:
        return fixed, fixed
    if not isinstance(cfg.actuator_delay_range, (list, tuple)) or len(cfg.actuator_delay_range) != 2:
        raise ValueError("actuator_delay_range must be None or (min, max) seconds.")
    if fixed:
        raise ValueError("Set either actuator_delay or actuator_delay_range, not both.")
    low, high = (delay_steps(value, physics_hz, "actuator_delay_range") for value in cfg.actuator_delay_range)
    if low > high:
        raise ValueError("actuator_delay_range must be ordered (min, max).")
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
        self.low, self.high = actuator_delay_bounds(cfg)
        self.physics_dt = 1 / cfg.physics_hz if hasattr(cfg, "physics_hz") else cfg.sim.dt
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
            self.steps[env_ids] = torch.randint(self.low, self.high + 1, (len(env_ids),), device=self.steps.device)
        if self._queue is not None:
            self._queue[:, env_ids] = 0.0

    def __call__(self, actions):
        """Queue this physics step's command; return the one queued `steps` physics steps earlier."""
        if self._queue is None:
            return actions
        self._cursor = (self._cursor + 1) % len(self._queue)
        self._queue[self._cursor] = actions
        return self._queue[(self._cursor - self.steps) % len(self._queue), self._env_ids]
