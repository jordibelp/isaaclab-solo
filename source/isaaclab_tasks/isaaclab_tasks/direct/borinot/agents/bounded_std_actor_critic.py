# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL ActorCritic whose action-noise std stays inside fixed bounds."""
import math

import torch
from rsl_rl.modules import ActorCritic


class BoundedStdActorCritic(ActorCritic):
    """Same networks and checkpoint keys as ActorCritic, with std limited to [min_action_std, max_action_std].

    The std parameter is projected back into range before every distribution update. This is not a clamp
    inside the autograd graph, so gradients still reach the parameter at the bound and the std can shrink again.
    """

    def __init__(self, *args, min_action_std: float, max_action_std: float, **kwargs):
        super().__init__(*args, **kwargs)
        if self.state_dependent_std:
            raise ValueError("BoundedStdActorCritic only bounds the state-independent std parameter.")
        self.min_action_std, self.max_action_std = min_action_std, max_action_std

    @torch.no_grad()
    def sanitize_action_std_(self) -> None:
        """Project the std parameter into range; train.py also calls this right after a resume."""
        if self.noise_std_type == "log":
            self.log_std.clamp_(math.log(self.min_action_std), math.log(self.max_action_std))
        else:
            self.std.clamp_(self.min_action_std, self.max_action_std)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        self.sanitize_action_std_()
        super()._update_distribution(obs)
