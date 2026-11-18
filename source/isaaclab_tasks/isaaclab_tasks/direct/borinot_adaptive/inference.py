# SPDX-License-Identifier: BSD-3-Clause
"""Hold the adaptive latent across actor ticks during playback."""

import torch


def encoder_period(policy_hz, encoder_hz):
    """Return the number of actor ticks per encoder evaluation (zero means same rate)."""
    if encoder_hz == 0:
        return 1
    ticks = policy_hz / encoder_hz
    if ticks < 1 or not ticks.is_integer():
        raise ValueError("encoder_hz must be zero or a positive divisor of policy_hz.")
    return int(ticks)


class HeldLatentInference:
    def __init__(self, model, period):
        self.model = model
        self.period = period
        self.remaining = None
        self.latent = None

    def reset(self, dones):
        if self.remaining is not None:
            self.remaining.masked_fill_(dones.reshape(-1).bool(), 0)

    def __call__(self, obs):
        model = self.model
        actor_obs = model.actor_obs_normalizer(model.get_actor_obs(obs))
        if self.remaining is None:
            self.remaining = torch.zeros(actor_obs.shape[0], device=actor_obs.device, dtype=torch.long)
        due = self.remaining.eq(0).nonzero(as_tuple=True)[0]
        if due.numel():
            latent = model._actor_input(actor_obs[due])[:, :-model.command_dim]
            if self.latent is None:
                self.latent = latent.clone()
            else:
                self.latent[due] = latent
            self.remaining[due] = self.period
        self.remaining.sub_(1)
        return model.actor(torch.cat((self.latent, actor_obs[:, -model.command_dim:]), dim=-1))
