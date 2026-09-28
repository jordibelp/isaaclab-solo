# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Solo12 PPO with gradient diagnostics for the installed RSL-RL update loop."""

import torch
from rsl_rl.algorithms import PPO


class Solo12PPO(PPO):
    """PPO with optional compiled forwards and pre-clipping gradient diagnostics."""

    def __init__(self, *args, torch_compile: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if torch_compile:
            if self.policy.is_recurrent:
                raise ValueError("Solo12 PPO torch_compile does not support recurrent policies.")
            self.policy.actor.forward = torch.compile(self.policy.actor.forward)
            self.policy.critic.forward = torch.compile(self.policy.critic.forward)
            print("[INFO]: Enabled torch.compile for Solo12 PPO actor and critic forwards.")

    def update(self) -> dict[str, float]:
        actor_params = [param for name, param in self.policy.named_parameters() if not name.startswith("critic")]
        critic_params = [param for name, param in self.policy.named_parameters() if name.startswith("critic")]
        actor_norms, critic_norms, clipped_actor_norms, clipped_critic_norms = [], [], [], []
        max_grad_norm = self.max_grad_norm

        def record_norms_and_clip(_optimizer, _args, _kwargs):
            actor_gradients = [p.grad for p in actor_params if p.grad is not None]
            critic_gradients = [p.grad for p in critic_params if p.grad is not None]
            actor_norms.append(torch.nn.utils.get_total_norm(actor_gradients))
            critic_norms.append(torch.nn.utils.get_total_norm(critic_gradients))
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            # The combined clip scales both groups by one factor, so measure each group again.
            clipped_actor_norms.append(torch.nn.utils.get_total_norm(actor_gradients))
            clipped_critic_norms.append(torch.nn.utils.get_total_norm(critic_gradients))

        # Upstream PPO clips the whole policy immediately before optimizer.step(). Let that
        # call be a no-op, then measure both groups and apply the same combined clip here.
        hook = self.optimizer.register_step_pre_hook(record_norms_and_clip)
        self.max_grad_norm = float("inf")
        try:
            losses = super().update()
        finally:
            self.max_grad_norm = max_grad_norm
            hook.remove()

        if actor_norms:
            names = ("grad_norm_actor", "grad_norm_critic", "clipped_grad_norm_actor", "clipped_grad_norm_critic")
            norms = (actor_norms, critic_norms, clipped_actor_norms, clipped_critic_norms)
            losses.update(zip(names, torch.stack([torch.stack(values).mean() for values in norms]).tolist()))
        return losses
