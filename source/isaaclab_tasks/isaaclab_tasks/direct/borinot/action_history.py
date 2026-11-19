# SPDX-License-Identifier: BSD-3-Clause
"""Policy-rate memory of commands issued before actuator delay and motor lag."""
import torch


def extra_action_history_dim(cfg):
    steps = cfg.include_prev_action_steps
    if type(steps) is not int or steps < 0:
        raise ValueError("include_prev_action_steps must be a non-negative integer.")
    return max(0, steps - 1) * cfg.action_space


class OlderPolicyActions:
    """Ring buffer for the older commands; the newest is already in env.actions."""

    def __init__(self, latest, steps):
        self._buffer = latest.new_zeros(latest.shape[0], steps - 1, latest.shape[1])
        self._head = 0

    def push(self, previous):
        self._head = (self._head - 1) % self._buffer.shape[1]
        self._buffer[:, self._head].copy_(previous)

    def reset(self, env_ids):
        self._buffer[env_ids] = 0

    def parts(self):
        """Flat, newest-to-oldest views, split at ring wraparound if needed."""
        parts = [self._buffer[:, self._head:].flatten(1)]
        if self._head:
            parts.append(self._buffer[:, :self._head].flatten(1))
        return parts

    def append(self, obs, *, critic=False):
        parts = self.parts()
        policy = torch.cat([obs["policy"], *parts], dim=-1)
        value = obs["critic"]
        if critic:
            value = policy if value is obs["policy"] else torch.cat([value, *parts], dim=-1)
        return {"policy": policy, "critic": value}
