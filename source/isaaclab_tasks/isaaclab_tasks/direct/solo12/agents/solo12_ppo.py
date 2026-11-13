# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Solo12 PPO with gradient diagnostics for the installed RSL-RL update loop."""

import torch
from rsl_rl.algorithms import PPO


class Solo12PPO(PPO):
    """PPO with optional compiled forwards and pre-clipping gradient diagnostics."""

    def __init__(self, *args, torch_compile: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_warmup_remaining = 0
        self.critic_warmup_completed = 0
        if torch_compile:
            if self.policy.is_recurrent:
                raise ValueError("Solo12 PPO torch_compile does not support recurrent policies.")
            self.policy.actor.forward = torch.compile(self.policy.actor.forward)
            self.policy.critic.forward = torch.compile(self.policy.critic.forward)
            print("[INFO]: Enabled torch.compile for Solo12 PPO actor and critic forwards.")

    def start_critic_warmup(self, iterations: int, *, completed: int = 0) -> None:
        """Freeze the sampled student policy while fitting its privileged critic."""
        if type(iterations) is not int or iterations < 0 or type(completed) is not int or completed < 0:
            raise ValueError("Critic warm-up iteration counts must be non-negative integers.")
        if not hasattr(self.policy, "actor_history_encoder") or self.policy.is_recurrent:
            raise ValueError("Critic warm-up requires the base-IMU feedforward TCN student.")
        if self.is_multi_gpu:
            raise ValueError("Critic warm-up currently supports single-process training only.")
        self.critic_warmup_remaining = iterations
        self.critic_warmup_completed = completed
        self.policy.critic_warmup_active = iterations > 0

    def compute_returns(self, obs) -> None:
        if not self.critic_warmup_remaining:
            super().compute_returns(obs)
            return
        # Use observed rewards through the rollout, not intermediate inherited value
        # predictions. This remains bootstrapped at rollout boundaries/timeouts.
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(last_values, self.gamma, 1.0, normalize_advantage=False)

    def _update_critic_warmup(self) -> dict[str, float]:
        critic_params = [p for name, p in self.policy.named_parameters() if name.startswith("critic")]
        actor_params = [
            (p, p.requires_grad) for name, p in self.policy.named_parameters() if not name.startswith("critic")
        ]
        for param, _ in actor_params:
            param.requires_grad_(False)
        losses, grad_norms = [], []
        try:
            for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
                obs_batch, returns_batch = batch[0], batch[4]
                value_loss = (self.policy.evaluate(obs_batch) - returns_batch).square().mean()
                # No policy/entropy/symmetry loss, KL schedule, or value clipping here.
                self.optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                grad_norms.append(torch.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm).detach())
                self.optimizer.step()
                losses.append(value_loss.detach())
        finally:
            for param, requires_grad in actor_params:
                param.requires_grad_(requires_grad)
        self.storage.clear()
        self.critic_warmup_remaining -= 1
        self.critic_warmup_completed += 1
        self.policy.critic_warmup_active = self.critic_warmup_remaining > 0
        return {
            "value_function": torch.stack(losses).mean().item(),
            "grad_norm_critic": torch.stack(grad_norms).mean().item(),
            "critic_warmup_active": 1.0,
            "critic_warmup_remaining_iterations": float(self.critic_warmup_remaining),
            "critic_warmup_completed_iterations": float(self.critic_warmup_completed),
        }

    def update(self) -> dict[str, float]:
        if self.critic_warmup_remaining:
            return self._update_critic_warmup()
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
        if self.critic_warmup_completed:
            losses.update(
                critic_warmup_active=0.0,
                critic_warmup_remaining_iterations=0.0,
                critic_warmup_completed_iterations=float(self.critic_warmup_completed),
            )
        return losses
