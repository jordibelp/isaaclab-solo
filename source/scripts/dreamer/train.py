# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a small DreamerV3-style world-model agent on Isaac Lab tasks."""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train a DreamerV3-style agent.")
parser.add_argument("--task", type=str, default="Solo12-simple-dreamerV3", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="dreamer_cfg_entry_point", help="Name of the Dreamer config entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Training seed. Use -1 for a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of collect/train iterations.")
parser.add_argument("--run-name", type=str, default=None, help="Override the configured run name.")
parser.add_argument("--logger", type=str, default=None, choices=["wandb", "tensorboard", "none"], help="Logger backend.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional Dreamer checkpoint to resume.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from gymnasium.spaces import flatdim  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import borinotIsaacLab.tasks  # noqa: F401, E402


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_set(cfg: Any, name: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _sanitize_run_name(name: str) -> str:
    return re.sub(r"[\\/]+", "-", name).strip()


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, *, layer_norm: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def unimix_logits(logits: torch.Tensor, unimix: float = 0.01) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    return torch.log(probs.clamp_min(1e-8))


def straight_through_categorical(logits: torch.Tensor) -> torch.Tensor:
    logits = unimix_logits(logits)
    probs = torch.softmax(logits, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    indices = torch.multinomial(flat_probs, 1).squeeze(-1)
    one_hot = F.one_hot(indices, probs.shape[-1]).to(dtype=probs.dtype)
    one_hot = one_hot.reshape_as(probs)
    return one_hot + probs - probs.detach()


def categorical_kl(lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
    lhs_logits = unimix_logits(lhs_logits)
    rhs_logits = unimix_logits(rhs_logits)
    lhs_probs = torch.softmax(lhs_logits, dim=-1)
    kl = lhs_probs * (lhs_logits - rhs_logits)
    return kl.sum(dim=(-1, -2))


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor


class RSSMWorldModel(nn.Module):
    def __init__(self, obs_dim: int, command_dim: int, action_dim: int, cfg: Any):
        super().__init__()
        self.obs_dim = obs_dim
        self.command_dim = command_dim
        self.action_dim = action_dim
        self.deter_dim = int(_cfg_get(cfg, "deter_dim"))
        self.stoch_dim = int(_cfg_get(cfg, "stoch_dim"))
        self.discrete_dim = int(_cfg_get(cfg, "discrete_dim"))
        self.stoch_flat_dim = self.stoch_dim * self.discrete_dim
        hidden = int(_cfg_get(cfg, "model_hidden_dim"))

        encoder_hidden = list(_cfg_get(cfg, "encoder_hidden_dims"))
        self.encoder = mlp(obs_dim, encoder_hidden, hidden)
        self.gru = nn.GRUCell(self.stoch_flat_dim + action_dim, self.deter_dim)
        self.prior = mlp(self.deter_dim, [hidden], self.stoch_flat_dim)
        self.posterior = mlp(self.deter_dim + hidden, [hidden], self.stoch_flat_dim)
        feature_dim = self.feature_dim
        self.decoder = mlp(feature_dim, [hidden, hidden], obs_dim)
        self.reward = mlp(feature_dim, [hidden, hidden], 1)
        self.continue_head = mlp(feature_dim, [hidden, hidden], 1)

    @property
    def feature_dim(self) -> int:
        return self.deter_dim + self.stoch_flat_dim + self.command_dim

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        stoch[..., 0] = 1.0
        return RSSMState(deter=deter, stoch=stoch)

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(symlog(obs))

    def _posterior_from_embed(self, deter: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.posterior(torch.cat((deter, embed), dim=-1))
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.discrete_dim)
        stoch = straight_through_categorical(logits)
        return logits, stoch

    def initial_from_obs(self, obs: torch.Tensor) -> tuple[RSSMState, torch.Tensor]:
        state = self.initial(obs.shape[0], obs.device)
        embed = self._embed(obs)
        post_logits, stoch = self._posterior_from_embed(state.deter, embed)
        return RSSMState(state.deter, stoch), post_logits

    def feature(self, state: RSSMState, command: torch.Tensor) -> torch.Tensor:
        stoch = state.stoch.reshape(*state.stoch.shape[:-2], self.stoch_flat_dim)
        return torch.cat((state.deter, stoch, command), dim=-1)

    def observe_next(
        self, state: RSSMState, action: torch.Tensor, next_obs: torch.Tensor
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        prev_stoch = state.stoch.reshape(state.stoch.shape[0], self.stoch_flat_dim)
        deter = self.gru(torch.cat((prev_stoch, action), dim=-1), state.deter)
        prior_logits = self.prior(deter).reshape(-1, self.stoch_dim, self.discrete_dim)
        embed = self._embed(next_obs)
        post_logits, stoch = self._posterior_from_embed(deter, embed)
        return RSSMState(deter=deter, stoch=stoch), prior_logits, post_logits

    def imagine_next(self, state: RSSMState, action: torch.Tensor) -> tuple[RSSMState, torch.Tensor]:
        prev_stoch = state.stoch.reshape(state.stoch.shape[0], self.stoch_flat_dim)
        deter = self.gru(torch.cat((prev_stoch, action), dim=-1), state.deter)
        prior_logits = self.prior(deter).reshape(-1, self.stoch_dim, self.discrete_dim)
        stoch = straight_through_categorical(prior_logits)
        return RSSMState(deter=deter, stoch=stoch), prior_logits

    def reset_where(self, state: RSSMState, obs: torch.Tensor, reset: torch.Tensor) -> RSSMState:
        if not torch.any(reset):
            return state
        reset_state, _ = self.initial_from_obs(obs)
        mask = reset.reshape(-1, 1)
        stoch_mask = reset.reshape(-1, 1, 1)
        return RSSMState(
            deter=torch.where(mask, reset_state.deter, state.deter),
            stoch=torch.where(stoch_mask, reset_state.stoch, state.stoch),
        )


class TanhNormalActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = mlp(input_dim, hidden_dims, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(features).chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(features)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log1p(-action.pow(2) + 1e-6)
        entropy = normal.entropy()
        return action, log_prob.sum(dim=-1), entropy.sum(dim=-1)

    def mode(self, features: torch.Tensor) -> torch.Tensor:
        mean, _ = self(features)
        return torch.tanh(mean)


class DreamerAgent(nn.Module):
    def __init__(self, obs_dim: int, command_dim: int, action_dim: int, cfg: Any, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.world = RSSMWorldModel(obs_dim, command_dim, action_dim, cfg).to(device)
        self.actor = TanhNormalActor(self.world.feature_dim, action_dim, list(_cfg_get(cfg, "actor_hidden_dims"))).to(device)
        self.critic = mlp(self.world.feature_dim, list(_cfg_get(cfg, "critic_hidden_dims")), 1).to(device)
        self.target_critic = mlp(self.world.feature_dim, list(_cfg_get(cfg, "critic_hidden_dims")), 1).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=float(_cfg_get(cfg, "model_lr")))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(_cfg_get(cfg, "actor_lr")))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(_cfg_get(cfg, "critic_lr")))

    @torch.no_grad()
    def act(self, state: RSSMState, command: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.world.feature(state, command)
        if deterministic:
            return self.actor.mode(features)
        action, _, _ = self.actor.sample(features)
        return action

    @torch.no_grad()
    def observe_next(self, state: RSSMState, action: torch.Tensor, obs: torch.Tensor, done: torch.Tensor) -> RSSMState:
        next_state, _, _ = self.world.observe_next(state, action, obs)
        return self.world.reset_where(next_state, obs, done)

    def train_on_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        model_metrics, starts = self._train_world_model(batch)
        actor_metrics = self._train_actor_critic(starts)
        metrics = {}
        metrics.update(model_metrics)
        metrics.update(actor_metrics)
        return metrics

    def _train_world_model(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, float], tuple[RSSMState, torch.Tensor]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        next_obs = batch["next_obs"]
        next_commands = batch["next_commands"]

        state, _ = self.world.initial_from_obs(obs[:, 0])
        states_deter = []
        states_stoch = []
        model_loss = torch.zeros((), device=self.device)
        obs_loss_sum = torch.zeros((), device=self.device)
        reward_loss_sum = torch.zeros((), device=self.device)
        continue_loss_sum = torch.zeros((), device=self.device)
        kl_dyn_sum = torch.zeros((), device=self.device)
        kl_rep_sum = torch.zeros((), device=self.device)
        valid_count = torch.zeros((), device=self.device)

        free_nats = float(_cfg_get(self.cfg, "free_nats"))
        for t in range(actions.shape[1]):
            next_state, prior_logits, post_logits = self.world.observe_next(state, actions[:, t], next_obs[:, t])
            features = self.world.feature(next_state, next_commands[:, t])
            valid = (1.0 - dones[:, t].float()).unsqueeze(-1)

            obs_pred = self.world.decoder(features)
            reward_pred = self.world.reward(features).squeeze(-1)
            continue_logits = self.world.continue_head(features).squeeze(-1)
            continue_target = 1.0 - dones[:, t].float()

            obs_loss = F.mse_loss(obs_pred, symlog(next_obs[:, t]), reduction="none").mean(dim=-1, keepdim=True)
            reward_loss = F.mse_loss(reward_pred, symlog(rewards[:, t]), reduction="none")
            continue_loss = F.binary_cross_entropy_with_logits(continue_logits, continue_target, reduction="none")
            kl_dyn = torch.clamp(categorical_kl(post_logits.detach(), prior_logits), min=free_nats)
            kl_rep = torch.clamp(categorical_kl(post_logits, prior_logits.detach()), min=free_nats)

            step_loss = (
                float(_cfg_get(self.cfg, "obs_loss_scale")) * (obs_loss * valid).mean()
                + float(_cfg_get(self.cfg, "reward_loss_scale")) * reward_loss.mean()
                + float(_cfg_get(self.cfg, "continue_loss_scale")) * continue_loss.mean()
                + float(_cfg_get(self.cfg, "kl_dyn_scale")) * (kl_dyn * valid.squeeze(-1)).mean()
                + float(_cfg_get(self.cfg, "kl_rep_scale")) * (kl_rep * valid.squeeze(-1)).mean()
            )
            model_loss = model_loss + step_loss
            obs_loss_sum = obs_loss_sum + (obs_loss * valid).sum()
            reward_loss_sum = reward_loss_sum + reward_loss.sum()
            continue_loss_sum = continue_loss_sum + continue_loss.sum()
            kl_dyn_sum = kl_dyn_sum + (kl_dyn * valid.squeeze(-1)).sum()
            kl_rep_sum = kl_rep_sum + (kl_rep * valid.squeeze(-1)).sum()
            valid_count = valid_count + valid.sum().clamp_min(1.0)

            states_deter.append(next_state.deter)
            states_stoch.append(next_state.stoch)
            state = self.world.reset_where(next_state, next_obs[:, t], dones[:, t].bool())

        model_loss = model_loss / actions.shape[1]
        self.model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), float(_cfg_get(self.cfg, "grad_clip")))
        self.model_opt.step()

        start_state = RSSMState(
            deter=torch.stack(states_deter, dim=1).detach().reshape(-1, self.world.deter_dim),
            stoch=torch.stack(states_stoch, dim=1).detach().reshape(-1, self.world.stoch_dim, self.world.discrete_dim),
        )
        start_commands = next_commands.detach().reshape(-1, next_commands.shape[-1])
        denom = valid_count.clamp_min(1.0)
        metrics = {
            "loss/model": float(model_loss.detach().cpu()),
            "loss/obs": float((obs_loss_sum / denom).detach().cpu()),
            "loss/reward": float((reward_loss_sum / rewards.numel()).detach().cpu()),
            "loss/continue": float((continue_loss_sum / rewards.numel()).detach().cpu()),
            "loss/kl_dyn": float((kl_dyn_sum / denom).detach().cpu()),
            "loss/kl_rep": float((kl_rep_sum / denom).detach().cpu()),
        }
        return metrics, (start_state, start_commands)

    def _train_actor_critic(self, starts: tuple[RSSMState, torch.Tensor]) -> dict[str, float]:
        start_state, command = starts
        state = RSSMState(start_state.deter.detach(), start_state.stoch.detach())
        command = command.detach()
        horizon = int(_cfg_get(self.cfg, "imag_horizon"))
        discount = float(_cfg_get(self.cfg, "discount"))
        lambda_ = float(_cfg_get(self.cfg, "lambda_"))

        features = []
        log_probs = []
        entropies = []
        rewards = []
        continues = []

        for _ in range(horizon):
            feat = self.world.feature(state, command).detach()
            action, log_prob, entropy = self.actor.sample(feat)
            with torch.no_grad():
                next_state, _ = self.world.imagine_next(state, action.detach())
                next_feat = self.world.feature(next_state, command)
                reward = symexp(self.world.reward(next_feat).squeeze(-1))
                cont = torch.sigmoid(self.world.continue_head(next_feat).squeeze(-1))
            features.append(feat)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            continues.append(cont)
            state = RSSMState(next_state.deter.detach(), next_state.stoch.detach())

        feats = torch.stack(features, dim=0)
        log_probs_t = torch.stack(log_probs, dim=0)
        entropies_t = torch.stack(entropies, dim=0)
        rewards_t = torch.stack(rewards, dim=0)
        discounts_t = discount * torch.stack(continues, dim=0)

        with torch.no_grad():
            target_values = self.target_critic(feats.reshape(-1, feats.shape[-1])).reshape(horizon, -1)
            bootstrap = self.target_critic(self.world.feature(state, command).detach()).squeeze(-1)
            returns = lambda_returns(rewards_t, discounts_t, target_values, bootstrap, lambda_)
            weights = torch.cumprod(
                torch.cat((torch.ones_like(discounts_t[:1]), discounts_t[:-1]), dim=0), dim=0
            ).detach()

        critic_values_for_adv = self.critic(feats.reshape(-1, feats.shape[-1])).reshape(horizon, -1).detach()
        advantages = returns - critic_values_for_adv
        actor_loss = -(
            weights
            * (
                log_probs_t * advantages.detach()
                + float(_cfg_get(self.cfg, "actor_entropy_scale")) * entropies_t
            )
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), float(_cfg_get(self.cfg, "grad_clip")))
        self.actor_opt.step()

        critic_values = self.critic(feats.detach().reshape(-1, feats.shape[-1])).reshape(horizon, -1)
        critic_loss = (weights * F.mse_loss(critic_values, returns.detach(), reduction="none")).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), float(_cfg_get(self.cfg, "grad_clip")))
        self.critic_opt.step()
        self._update_target_critic()

        return {
            "loss/actor": float(actor_loss.detach().cpu()),
            "loss/critic": float(critic_loss.detach().cpu()),
            "imag/reward": float(rewards_t.mean().detach().cpu()),
            "imag/continue": float((discounts_t / discount).mean().detach().cpu()),
            "imag/return": float(returns.mean().detach().cpu()),
            "policy/entropy": float(entropies_t.mean().detach().cpu()),
        }

    def _update_target_critic(self):
        tau = float(_cfg_get(self.cfg, "slow_critic_tau"))
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save(self, path: Path, iteration: int, total_steps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "iteration": iteration,
                "total_steps": total_steps,
                "cfg": _cfg_to_dict(self.cfg),
                "world": self.world.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "model_opt": self.model_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> tuple[int, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.world.load_state_dict(checkpoint["world"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint.get("target_critic", checkpoint["critic"]))
        self.model_opt.load_state_dict(checkpoint["model_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
        return int(checkpoint.get("iteration", 0)), int(checkpoint.get("total_steps", 0))


def lambda_returns(
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    next_value = bootstrap
    returns = []
    for t in reversed(range(rewards.shape[0])):
        next_values = bootstrap if t == rewards.shape[0] - 1 else values[t + 1]
        next_value = rewards[t] + discounts[t] * ((1.0 - lambda_) * next_values + lambda_ * next_value)
        returns.append(next_value)
    return torch.stack(list(reversed(returns)), dim=0)


class SequenceReplayBuffer:
    def __init__(self, capacity: int, num_envs: int, obs_dim: int, command_dim: int, action_dim: int):
        self.num_envs = num_envs
        self.capacity_steps = max(2, int(capacity) // num_envs)
        self.obs = torch.empty(self.capacity_steps, num_envs, obs_dim, dtype=torch.float32)
        self.commands = torch.empty(self.capacity_steps, num_envs, command_dim, dtype=torch.float32)
        self.actions = torch.empty(self.capacity_steps, num_envs, action_dim, dtype=torch.float32)
        self.rewards = torch.empty(self.capacity_steps, num_envs, dtype=torch.float32)
        self.dones = torch.empty(self.capacity_steps, num_envs, dtype=torch.bool)
        self.next_obs = torch.empty(self.capacity_steps, num_envs, obs_dim, dtype=torch.float32)
        self.next_commands = torch.empty(self.capacity_steps, num_envs, command_dim, dtype=torch.float32)
        self.pos = 0
        self.filled = 0
        self.total = 0

    def add(
        self,
        obs: torch.Tensor,
        command: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        next_command: torch.Tensor,
    ):
        idx = self.pos
        self.obs[idx].copy_(obs.detach().cpu())
        self.commands[idx].copy_(command.detach().cpu())
        self.actions[idx].copy_(action.detach().cpu())
        self.rewards[idx].copy_(reward.detach().cpu())
        self.dones[idx].copy_(done.detach().cpu())
        self.next_obs[idx].copy_(next_obs.detach().cpu())
        self.next_commands[idx].copy_(next_command.detach().cpu())
        self.pos = (self.pos + 1) % self.capacity_steps
        self.filled = min(self.filled + 1, self.capacity_steps)
        self.total += self.num_envs

    def can_sample(self, batch_length: int, min_steps: int) -> bool:
        return self.total >= min_steps and self.filled >= batch_length

    def _physical_indices(self, logical: torch.Tensor) -> torch.Tensor:
        if self.filled < self.capacity_steps:
            return logical
        return (self.pos + logical) % self.capacity_steps

    def sample(self, batch_size: int, batch_length: int, device: torch.device) -> dict[str, torch.Tensor]:
        max_start = self.filled - batch_length
        starts = torch.randint(0, max_start + 1, (batch_size,))
        env_ids = torch.randint(0, self.num_envs, (batch_size,))
        offsets = torch.arange(batch_length)
        logical = starts[:, None] + offsets[None, :]
        phys = self._physical_indices(logical)

        def gather(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[phys, env_ids[:, None]].to(device=device, non_blocking=True)

        return {
            "obs": gather(self.obs),
            "commands": gather(self.commands),
            "actions": gather(self.actions),
            "rewards": gather(self.rewards),
            "dones": gather(self.dones),
            "next_obs": gather(self.next_obs),
            "next_commands": gather(self.next_commands),
        }


class ScalarLogger:
    def __init__(self, log_dir: Path, cfg: Any):
        self.writer = None
        self.wandb = None
        logger_name = str(_cfg_get(cfg, "logger", "none")).lower()
        if logger_name == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(log_dir))
        elif logger_name == "wandb":
            try:
                import wandb

                self.wandb = wandb
                self.wandb.init(
                    project=str(_cfg_get(cfg, "wandb_project")),
                    entity=_cfg_get(cfg, "wandb_entity"),
                    name=log_dir.name,
                    config=_cfg_to_dict(cfg),
                    sync_tensorboard=False,
                )
            except Exception as exc:
                print(f"[WARN] Could not initialize W&B ({exc}); continuing with stdout logging.", flush=True)

    def log(self, metrics: dict[str, float], step: int):
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.wandb is not None:
            self.wandb.finish()


def _obs_command(obs_dict: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obs = obs_dict["policy"].to(device)
    command = obs_dict.get("command")
    if command is None:
        command = torch.zeros(obs.shape[0], 0, device=device)
    else:
        command = command.to(device)
    return obs, command


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Any):
    if _cfg_get(agent_cfg, "seed") is None:
        agent_seed = 42
    else:
        agent_seed = int(_cfg_get(agent_cfg, "seed"))
    seed = args_cli.seed if args_cli.seed is not None else agent_seed
    if seed == -1:
        seed = random.randint(0, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    env_cfg.seed = seed

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = int(_cfg_get(agent_cfg, "num_envs", env_cfg.scene.num_envs))
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    else:
        env_cfg.sim.device = str(_cfg_get(agent_cfg, "device", env_cfg.sim.device))
    if args_cli.max_iterations is not None:
        _cfg_set(agent_cfg, "max_iterations", args_cli.max_iterations)
    if args_cli.run_name is not None:
        _cfg_set(agent_cfg, "run_name", args_cli.run_name)
    if args_cli.logger is not None:
        _cfg_set(agent_cfg, "logger", args_cli.logger)

    if getattr(env_cfg, "policy_model", None) != "simple_dreamer_v3":
        raise ValueError("Dreamer trainer expects a task with env.policy_model='simple_dreamer_v3'.")

    log_root = Path("logs") / "dreamer" / str(_cfg_get(agent_cfg, "experiment_name"))
    log_root = log_root.resolve()
    run_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = _sanitize_run_name(f"{run_prefix}_{_cfg_get(agent_cfg, 'run_name')}")
    log_dir = log_root / run_name
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
    print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)

    env_cfg.log_dir = str(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        device = torch.device(env.unwrapped.device)
        obs_dict, _ = env.reset()
        obs, command = _obs_command(obs_dict, device)
        obs_dim = obs.shape[-1]
        command_dim = command.shape[-1]
        action_dim = flatdim(env.unwrapped.single_action_space)

        agent = DreamerAgent(obs_dim, command_dim, action_dim, agent_cfg, device)
        start_iteration = 0
        total_steps = 0
        if args_cli.checkpoint:
            start_iteration, total_steps = agent.load(args_cli.checkpoint)
            print(f"[INFO] Resumed Dreamer checkpoint from {args_cli.checkpoint}", flush=True)

        replay = SequenceReplayBuffer(
            int(_cfg_get(agent_cfg, "replay_size")),
            env.unwrapped.num_envs,
            obs_dim,
            command_dim,
            action_dim,
        )
        state, _ = agent.world.initial_from_obs(obs)
        episode_returns = torch.zeros(env.unwrapped.num_envs, device=device)
        recent_returns: list[float] = []
        logger = ScalarLogger(log_dir, agent_cfg)
        start_time = time.time()

        max_iterations = int(_cfg_get(agent_cfg, "max_iterations"))
        steps_per_env = int(_cfg_get(agent_cfg, "steps_per_env"))
        batch_size = int(_cfg_get(agent_cfg, "batch_size"))
        batch_length = int(_cfg_get(agent_cfg, "batch_length"))
        prefill_steps = int(_cfg_get(agent_cfg, "prefill_steps"))
        train_steps_per_iteration = int(_cfg_get(agent_cfg, "train_steps_per_iteration"))
        log_interval = int(_cfg_get(agent_cfg, "log_interval"))
        save_interval = int(_cfg_get(agent_cfg, "save_interval"))

        for iteration in range(start_iteration + 1, max_iterations + 1):
            for _ in range(steps_per_env):
                if replay.total < prefill_steps:
                    action = torch.empty(obs.shape[0], action_dim, device=device).uniform_(-1.0, 1.0)
                else:
                    action = agent.act(state, command)

                next_obs_dict, reward, terminated, truncated, extras = env.step(action)
                done = terminated | truncated
                next_obs, next_command = _obs_command(next_obs_dict, device)
                replay.add(obs, command, action, reward, done, next_obs, next_command)

                episode_returns += reward
                if torch.any(done):
                    recent_returns.extend(episode_returns[done].detach().cpu().tolist())
                    recent_returns = recent_returns[-100:]
                    episode_returns[done] = 0.0

                state = agent.observe_next(state, action, next_obs, done)
                obs, command = next_obs, next_command
                total_steps += env.unwrapped.num_envs

            train_metrics: dict[str, float] = {}
            if replay.can_sample(batch_length, prefill_steps):
                for _ in range(train_steps_per_iteration):
                    batch = replay.sample(batch_size, batch_length, device)
                    train_metrics = agent.train_on_batch(batch)

            if iteration % log_interval == 0 or iteration == 1:
                elapsed = max(time.time() - start_time, 1e-6)
                metrics = {
                    "train/iteration": float(iteration),
                    "train/env_steps": float(total_steps),
                    "train/fps": float(total_steps / elapsed),
                    "replay/steps": float(replay.total),
                    "episode/return_mean_100": float(sum(recent_returns) / len(recent_returns))
                    if recent_returns
                    else 0.0,
                }
                metrics.update(train_metrics)
                env_logs = extras.get("log", {}) if isinstance(extras, dict) else {}
                for key, value in env_logs.items():
                    if isinstance(value, (int, float)):
                        metrics[f"env/{key}"] = float(value)
                logger.log(metrics, total_steps)
                print(
                    f"[INFO] iter={iteration} steps={total_steps} "
                    f"return100={metrics['episode/return_mean_100']:.3f} "
                    f"model_loss={metrics.get('loss/model', 0.0):.4f} "
                    f"actor_loss={metrics.get('loss/actor', 0.0):.4f}",
                    flush=True,
                )

            if iteration % save_interval == 0:
                agent.save(log_dir / "checkpoints" / f"model_{iteration}.pt", iteration, total_steps)

        agent.save(log_dir / "checkpoints" / "last.pt", max_iterations, total_steps)
        logger.close()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
