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

_THIS_DIR = Path(__file__).resolve().parent
_RSL_RL_SCRIPT_DIR = _THIS_DIR.parent / "rsl_rl"
if str(_RSL_RL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))

from isaaclab.app import AppLauncher


_HYDRA_OVERRIDE_HELP = """Hydra override examples:
  agent.num_batches_trained_per_iteration=32
  agent.batch_size=1024
  agent.hidden_vector_deter_dims=128
  agent.stoch_dim=32 agent.num_bins_encoding=64
  agent.model_lr=5e-5 agent.actor_entropy_scale=1e-4
  'agent.actor_hidden_dims=[512,256,128]'
  'agent.run_name="[Cluster]-dreamer-bins64"'

Use the agent.<field>=<value> form for fields in dreamer_v3_cfg.py.
Use env.<field>=<value> for environment config fields, matching normal IsaacLab commands.
"""

parser = argparse.ArgumentParser(
    description="Train a DreamerV3-style agent.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=_HYDRA_OVERRIDE_HELP,
)
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
parser.add_argument(
    "--use-cbp",
    "--cbp-enable",
    "--cbp_enable",
    dest="use_cbp",
    action="store_true",
    default=False,
    help="Enable Continual Backpropagation neuron replacement for Dreamer MLPs.",
)
parser.add_argument(
    "--cbp-replacement-rate",
    "--cbp_replacement_rate",
    type=float,
    default=1.0e-4,
    help="CBP replacement rate per mature hidden unit and optimizer step.",
)
parser.add_argument(
    "--cbp-maturity-threshold",
    "--cbp_maturity_threshold",
    type=int,
    default=10_000,
    help="Number of optimizer steps before a hidden unit is eligible for CBP replacement.",
)
parser.add_argument(
    "--cbp-decay-rate",
    "--cbp_decay_rate",
    type=float,
    default=0.99,
    help="Exponential decay rate for CBP feature-utility estimates.",
)
parser.add_argument(
    "--cbp-util-type",
    "--cbp_util_type",
    type=str,
    choices=(
        "contribution",
        "zero_contribution",
        "adaptable_contribution",
        "weight",
        "adaptation",
        "feature_by_input",
        "random",
    ),
    default="contribution",
    help="CBP utility score used to choose mature hidden units for replacement.",
)
parser.add_argument(
    "--cbp-init",
    "--cbp_init",
    type=str,
    choices=("default", "xavier", "lecun", "kaiming"),
    default="kaiming",
    help="Initialization bound used when CBP resets incoming weights.",
)
parser.add_argument(
    "--cbp-accumulate",
    "--cbp_accumulate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Accumulate fractional CBP replacements across optimizer steps.",
)
AppLauncher.add_app_launcher_args(parser)
_RAW_CLI_ARGS = tuple(sys.argv[1:])
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

from continual_backprop import build_continual_backprop_manager, cbp_specs_for_sequential_mlp  # noqa: E402

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


def _cli_option_was_provided(*option_names: str) -> bool:
    for arg in _RAW_CLI_ARGS:
        for option_name in option_names:
            if arg == option_name or arg.startswith(f"{option_name}="):
                return True
    return False


def _cfg_or_cli(
    cfg: Any,
    cfg_name: str,
    parsed_args: argparse.Namespace,
    arg_name: str,
    *option_names: str,
) -> Any:
    if _cli_option_was_provided(*option_names):
        return getattr(parsed_args, arg_name)
    return _cfg_get(cfg, cfg_name, getattr(parsed_args, arg_name))


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _jsonify(obj: Any, max_str: int = 4000) -> Any:
    """Convert nested config objects to values that W&B can store in run config."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str) and len(obj) > max_str:
            return obj[:max_str] + "...(truncated)"
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x, max_str=max_str) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v, max_str=max_str) for k, v in obj.items()}
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _jsonify(obj.to_dict(), max_str=max_str)
        except Exception:
            pass

    value = repr(obj)
    if len(value) > max_str:
        value = value[:max_str] + "...(truncated)"
    return value


def _wandb_config_payload(agent_cfg: Any, env_cfg: Any, args_cli: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    agent_dict = _jsonify(agent_cfg)
    payload: dict[str, Any] = {
        "log_dir": str(log_dir),
        "wandb_run_id": os.environ.get("WANDB_RUN_ID"),
        "cli": _jsonify(vars(args_cli)),
        "agent_cfg": agent_dict,
        "env_cfg": _jsonify(env_cfg),
    }
    if isinstance(agent_dict, dict):
        # Keep the old top-level agent fields so existing W&B panels/sweeps keep working.
        payload.update(agent_dict)
    return payload


def _generate_wandb_run_id_if_needed(agent_cfg: Any) -> str | None:
    if str(_cfg_get(agent_cfg, "logger", "none")).lower() != "wandb":
        return None

    try:
        import wandb
    except Exception as exc:
        print(f"[WARN] Could not pre-generate W&B run id ({exc}); log folder will not include it.", flush=True)
        return None

    wandb_run_id = wandb.util.generate_id()
    os.environ["WANDB_RUN_ID"] = wandb_run_id
    os.environ["WANDB_RESUME"] = "allow"
    return wandb_run_id


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


def _world_model_cbp_specs(world: "RSSMWorldModel") -> list[Any]:
    specs = []
    for name in ("encoder", "prior", "posterior", "decoder", "continue_head"):
        specs.extend(cbp_specs_for_sequential_mlp(f"world.{name}", getattr(world, name)))
    specs.extend(cbp_specs_for_sequential_mlp("world.reward.net", world.reward.net))
    return specs


def _build_cbp_manager(name: str, specs: list[Any], parsed_args: argparse.Namespace):
    if not specs:
        print(f"[WARN] CBP requested, but no eligible hidden groups were found for {name}; skipping.", flush=True)
        return None
    return build_continual_backprop_manager(
        specs,
        replacement_rate=float(parsed_args.cbp_replacement_rate),
        maturity_threshold=int(parsed_args.cbp_maturity_threshold),
        decay_rate=float(parsed_args.cbp_decay_rate),
        util_type=str(parsed_args.cbp_util_type),
        init=str(parsed_args.cbp_init),
        accumulate=bool(parsed_args.cbp_accumulate),
    )


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


@dataclass
class ImaginationStarts:
    state: RSSMState
    command: torch.Tensor


class PercentileReturnNormalizer(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool,
        rate: float,
        limit: float,
        percentile_low: float,
        percentile_high: float,
        device: torch.device,
    ):
        super().__init__()
        if percentile_low >= percentile_high:
            raise ValueError("return_norm_percentile_low must be smaller than return_norm_percentile_high.")
        self.enabled = enabled
        self.rate = rate
        self.limit = limit
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), device=device))
        self.register_buffer("high", torch.zeros((), device=device))

    @torch.no_grad()
    def scale(self, returns: torch.Tensor, *, update: bool) -> torch.Tensor:
        if not self.enabled:
            return returns.new_tensor(1.0)
        flat = returns.detach().reshape(-1).float()
        if flat.numel() == 0:
            return (self.high - self.low).clamp_min(self.limit).to(device=returns.device, dtype=returns.dtype)
        if update:
            low = torch.quantile(flat, self.percentile_low / 100.0).to(device=self.low.device, dtype=self.low.dtype)
            high = torch.quantile(flat, self.percentile_high / 100.0).to(device=self.high.device, dtype=self.high.dtype)
            self.low.mul_(1.0 - self.rate).add_(low, alpha=self.rate)
            self.high.mul_(1.0 - self.rate).add_(high, alpha=self.rate)
        return (self.high - self.low).clamp_min(self.limit).to(device=returns.device, dtype=returns.dtype)


class SymexpTwoHotHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        *,
        num_bins: int,
        symlog_range: float,
        layer_norm: bool = True,
    ):
        super().__init__()
        if num_bins < 3:
            raise ValueError("reward_value_num_bins must be at least 3.")
        self.num_bins = int(num_bins)
        self.symlog_range = float(symlog_range)
        self.net = mlp(input_dim, hidden_dims, self.num_bins, layer_norm=layer_norm)
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.register_buffer("bins", self._make_bins(self.num_bins, self.symlog_range))

    @staticmethod
    def _make_bins(num_bins: int, symlog_range: float) -> torch.Tensor:
        if num_bins % 2 == 1:
            half = torch.linspace(-symlog_range, 0.0, (num_bins - 1) // 2 + 1, dtype=torch.float32)
            half = symexp(half)
            return torch.cat((half, -half[:-1].flip(0)), dim=0)
        half = torch.linspace(-symlog_range, 0.0, num_bins // 2, dtype=torch.float32)
        half = symexp(half)
        return torch.cat((half, -half.flip(0)), dim=0)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def pred_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        bins = self.bins.to(device=probs.device, dtype=probs.dtype)
        n = bins.shape[0]
        if n % 2 == 1:
            mid = (n - 1) // 2
            left = (probs[..., :mid] * bins[:mid]).flip(-1)
            center = (probs[..., mid : mid + 1] * bins[mid : mid + 1]).sum(dim=-1)
            right = probs[..., mid + 1 :] * bins[mid + 1 :]
            return center + (left + right).sum(dim=-1)
        left = (probs[..., : n // 2] * bins[: n // 2]).flip(-1)
        right = probs[..., n // 2 :] * bins[n // 2 :]
        return (left + right).sum(dim=-1)

    def pred(self, features: torch.Tensor) -> torch.Tensor:
        return self.pred_from_logits(self.logits(features))

    def loss_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(device=logits.device, dtype=logits.dtype).contiguous()
        bins = self.bins.to(device=logits.device, dtype=logits.dtype)
        above = torch.searchsorted(bins, target.detach(), right=True)
        below = (above - 1).clamp(0, self.num_bins - 1)
        above = above.clamp(0, self.num_bins - 1)
        equal = below == above
        dist_to_below = torch.where(equal, torch.ones_like(target), (bins[below] - target).abs())
        dist_to_above = torch.where(equal, torch.ones_like(target), (bins[above] - target).abs())
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        twohot = (
            F.one_hot(below, self.num_bins).to(dtype=logits.dtype) * weight_below.unsqueeze(-1)
            + F.one_hot(above, self.num_bins).to(dtype=logits.dtype) * weight_above.unsqueeze(-1)
        )
        return -(twohot * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def loss(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_from_logits(self.logits(features), target)


class RSSMWorldModel(nn.Module):
    def __init__(self, obs_dim: int, command_dim: int, action_dim: int, cfg: Any):
        super().__init__()
        self.obs_dim = obs_dim
        self.command_dim = command_dim
        self.action_dim = action_dim
        self.hidden_vector_deter_dims = int(_cfg_get(cfg, "hidden_vector_deter_dims"))
        self.stoch_dim = int(_cfg_get(cfg, "stoch_dim"))
        self.num_bins_encoding = int(_cfg_get(cfg, "num_bins_encoding"))
        self.stoch_flat_dim = self.stoch_dim * self.num_bins_encoding
        hidden = int(_cfg_get(cfg, "model_hidden_dim"))

        encoder_hidden = list(_cfg_get(cfg, "encoder_hidden_dims"))
        self.encoder = mlp(obs_dim, encoder_hidden, hidden)
        self.gru = nn.GRUCell(self.stoch_flat_dim + action_dim, self.hidden_vector_deter_dims)
        self.prior = mlp(self.hidden_vector_deter_dims, [hidden], self.stoch_flat_dim)
        self.posterior = mlp(self.hidden_vector_deter_dims + hidden, [hidden], self.stoch_flat_dim)
        feature_dim = self.feature_dim
        self.decoder = mlp(feature_dim, [hidden, hidden], obs_dim)
        self.reward = SymexpTwoHotHead(
            feature_dim,
            [hidden, hidden],
            num_bins=int(_cfg_get(cfg, "reward_value_num_bins", 255)),
            symlog_range=float(_cfg_get(cfg, "reward_value_symlog_range", 20.0)),
        )
        self.continue_head = mlp(feature_dim, [hidden, hidden], 1)

    @property
    def feature_dim(self) -> int:
        return self.hidden_vector_deter_dims + self.stoch_flat_dim + self.command_dim

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.hidden_vector_deter_dims, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, self.num_bins_encoding, device=device)
        stoch[..., 0] = 1.0
        return RSSMState(deter=deter, stoch=stoch)

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(symlog(obs))

    def _posterior_from_embed(self, deter: torch.Tensor, embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.posterior(torch.cat((deter, embed), dim=-1))
        logits = logits.reshape(*logits.shape[:-1], self.stoch_dim, self.num_bins_encoding)
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
    ) -> tuple[RSSMState, RSSMState, torch.Tensor, torch.Tensor]:
        prev_stoch = state.stoch.reshape(state.stoch.shape[0], self.stoch_flat_dim)
        deter = self.gru(torch.cat((prev_stoch, action), dim=-1), state.deter)
        prior_logits = self.prior(deter).reshape(-1, self.stoch_dim, self.num_bins_encoding)
        prior_stoch = straight_through_categorical(prior_logits)
        prior_state = RSSMState(deter=deter, stoch=prior_stoch)
        embed = self._embed(next_obs)
        post_logits, stoch = self._posterior_from_embed(deter, embed)
        return RSSMState(deter=deter, stoch=stoch), prior_state, prior_logits, post_logits

    def imagine_next(self, state: RSSMState, action: torch.Tensor) -> tuple[RSSMState, torch.Tensor]:
        prev_stoch = state.stoch.reshape(state.stoch.shape[0], self.stoch_flat_dim)
        deter = self.gru(torch.cat((prev_stoch, action), dim=-1), state.deter)
        prior_logits = self.prior(deter).reshape(-1, self.stoch_dim, self.num_bins_encoding)
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
    def __init__(
        self,
        obs_dim: int,
        command_dim: int,
        action_dim: int,
        cfg: Any,
        device: torch.device,
        cbp_args: argparse.Namespace | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.cbp_managers = {}
        self.world = RSSMWorldModel(obs_dim, command_dim, action_dim, cfg).to(device)
        self.actor = TanhNormalActor(self.world.feature_dim, action_dim, list(_cfg_get(cfg, "actor_hidden_dims"))).to(device)
        self.critic = SymexpTwoHotHead(
            self.world.feature_dim,
            list(_cfg_get(cfg, "critic_hidden_dims")),
            num_bins=int(_cfg_get(cfg, "reward_value_num_bins", 255)),
            symlog_range=float(_cfg_get(cfg, "reward_value_symlog_range", 20.0)),
        ).to(device)
        self.target_critic = SymexpTwoHotHead(
            self.world.feature_dim,
            list(_cfg_get(cfg, "critic_hidden_dims")),
            num_bins=int(_cfg_get(cfg, "reward_value_num_bins", 255)),
            symlog_range=float(_cfg_get(cfg, "reward_value_symlog_range", 20.0)),
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.return_normalizer = PercentileReturnNormalizer(
            enabled=bool(_cfg_get(cfg, "normalize_actor_returns", True)),
            rate=float(_cfg_get(cfg, "return_norm_rate", 0.01)),
            limit=float(_cfg_get(cfg, "return_norm_limit", 1.0)),
            percentile_low=float(_cfg_get(cfg, "return_norm_percentile_low", 5.0)),
            percentile_high=float(_cfg_get(cfg, "return_norm_percentile_high", 95.0)),
            device=device,
        )
        self.model_opt = torch.optim.Adam(self.world.parameters(), lr=float(_cfg_get(cfg, "model_lr")))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(_cfg_get(cfg, "actor_lr")))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(_cfg_get(cfg, "critic_lr")))
        if cbp_args is not None and bool(getattr(cbp_args, "use_cbp", False)):
            self._attach_continual_backprop(cbp_args)

    def _attach_continual_backprop(self, parsed_args: argparse.Namespace) -> None:
        managers = {
            "world": _build_cbp_manager("world", _world_model_cbp_specs(self.world), parsed_args),
            "actor": _build_cbp_manager("actor", cbp_specs_for_sequential_mlp("actor.net", self.actor.net), parsed_args),
            "critic": _build_cbp_manager("critic", cbp_specs_for_sequential_mlp("critic.net", self.critic.net), parsed_args),
        }
        self.cbp_managers = {name: manager for name, manager in managers.items() if manager is not None}
        if not self.cbp_managers:
            raise ValueError("--use-cbp was provided, but no eligible Dreamer MLP hidden groups were found.")
        for name, manager in self.cbp_managers.items():
            group_names = ", ".join(group.name for group in manager.groups)
            print(
                "[INFO] Continual Backpropagation enabled for "
                f"{name} (replacement_rate={manager.replacement_rate:g}, "
                f"maturity_threshold={manager.maturity_threshold}, decay_rate={manager.decay_rate:g}, "
                f"util_type={manager.util_type}, init={manager.init}, accumulate={manager.accumulate}).",
                flush=True,
            )
            print(f"[INFO] CBP {name} feature groups: {group_names}", flush=True)

    def _after_optimizer_step(self, manager_name: str, optimizer: torch.optim.Optimizer) -> None:
        manager = self.cbp_managers.get(manager_name)
        if manager is not None:
            manager.after_optimizer_step(optimizer)

    def _cbp_metrics(self) -> dict[str, float]:
        metrics = {}
        for manager_name, manager in self.cbp_managers.items():
            prefix = f"CBP/{manager_name}"
            last_total = sum(manager.last_replacements.values())
            total = sum(manager.total_replacements_by_group.values())
            metrics[f"{prefix}/optimizer_steps"] = float(manager.optimizer_steps)
            metrics[f"{prefix}/replacements_last_update"] = float(last_total)
            metrics[f"{prefix}/replacements_total"] = float(total)
            for group_name, group_total in manager.total_replacements_by_group.items():
                metrics[f"{prefix}/replacements_total/{group_name}"] = float(group_total)
        return metrics

    def _cbp_state_dict(self) -> dict[str, Any]:
        return {name: manager.state_dict() for name, manager in self.cbp_managers.items()}

    def _cbp_summary(self) -> dict[str, Any]:
        return {name: manager.summary() for name, manager in self.cbp_managers.items()}

    def _restore_cbp_state(self, checkpoint: dict[str, Any], checkpoint_path: str) -> None:
        if not self.cbp_managers:
            return
        state = checkpoint.get("continual_backprop_state_dict")
        if isinstance(state, dict):
            for name, manager in self.cbp_managers.items():
                manager_state = state.get(name)
                if isinstance(manager_state, dict):
                    report = manager.load_state_dict(manager_state)
                    print(
                        "[INFO] Restored Dreamer CBP state "
                        f"for {name} from {checkpoint_path} "
                        f"(optimizer_steps={report['optimizer_steps']}, "
                        f"groups={report['groups_loaded']}/{report['groups_total']}, "
                        f"age_tensors={report['age_tensors_loaded']}/{report['groups_total']} exact, "
                        f"fallback={report['age_tensors_from_optimizer_steps']}).",
                        flush=True,
                    )
                else:
                    print(f"[INFO] No CBP state for Dreamer {name}; starting that manager from scratch.", flush=True)
            return

        summary = checkpoint.get("continual_backprop", {})
        if not isinstance(summary, dict):
            print("[INFO] Checkpoint has no Dreamer CBP state; starting CBP ages/utilities from scratch.", flush=True)
            return
        for name, manager in self.cbp_managers.items():
            component_summary = summary.get(name, {})
            optimizer_steps = (
                component_summary.get("optimizer_steps", None) if isinstance(component_summary, dict) else None
            )
            if optimizer_steps is None:
                print(f"[INFO] No CBP optimizer-step summary for Dreamer {name}; starting from scratch.", flush=True)
                continue
            report = manager.initialize_ages_from_optimizer_steps(int(optimizer_steps))
            print(
                "[INFO] Checkpoint has only a Dreamer CBP summary, not exact per-neuron state; "
                f"initialized {name} ages from optimizer_steps={report['optimizer_steps']} for "
                f"{report['age_tensors_from_optimizer_steps']}/{report['groups_total']} groups.",
                flush=True,
            )

    @torch.no_grad()
    def act(self, state: RSSMState, command: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        features = self.world.feature(state, command)
        if deterministic:
            return self.actor.mode(features)
        action, _, _ = self.actor.sample(features)
        return action

    @torch.no_grad()
    def observe_next(self, state: RSSMState, action: torch.Tensor, obs: torch.Tensor, done: torch.Tensor) -> RSSMState:
        next_state, _, _, _ = self.world.observe_next(state, action, obs)
        return self.world.reset_where(next_state, obs, done)

    def train_on_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        model_metrics, starts = self._train_world_model(batch)
        if starts.state.deter.shape[0] > 0:
            actor_metrics = self._train_actor_critic(starts)
        else:
            actor_metrics = {"train/actor_skipped_no_valid_starts": 1.0}
        metrics = {}
        metrics.update(model_metrics)
        metrics.update(actor_metrics)
        metrics.update(self._cbp_metrics())
        return metrics

    def _train_world_model(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, float], ImaginationStarts]:
        obs = batch["obs"]
        commands = batch["commands"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminals = batch["terminals"] if "terminals" in batch else batch["dones"]
        truncations = batch["truncations"] if "truncations" in batch else torch.zeros_like(terminals)
        dones = terminals | truncations
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
            next_state, prior_state, prior_logits, post_logits = self.world.observe_next(
                state, actions[:, t], next_obs[:, t]
            )
            command_for_loss = torch.where(dones[:, t].unsqueeze(-1), commands[:, t], next_commands[:, t])
            features = self.world.feature(next_state, command_for_loss)
            prior_features = self.world.feature(prior_state, command_for_loss)
            valid = (1.0 - dones[:, t].float()).unsqueeze(-1)

            obs_pred = self.world.decoder(features)
            reward_features = torch.where(dones[:, t].unsqueeze(-1), prior_features, features)
            reward_loss = self.world.reward.loss(reward_features, rewards[:, t])
            continue_logits = self.world.continue_head(prior_features).squeeze(-1)
            continue_target = 1.0 - terminals[:, t].float()

            obs_loss = F.mse_loss(obs_pred, symlog(next_obs[:, t]), reduction="none").mean(dim=-1, keepdim=True)
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
        self._after_optimizer_step("world", self.model_opt)

        states_deter_t = torch.stack(states_deter, dim=1).detach()
        states_stoch_t = torch.stack(states_stoch, dim=1).detach()
        imag_last = int(_cfg_get(self.cfg, "imag_last", 0))
        start_count_per_sequence = min(imag_last or actions.shape[1], actions.shape[1])
        start_deter = states_deter_t[:, -start_count_per_sequence:]
        start_stoch = states_stoch_t[:, -start_count_per_sequence:]
        start_commands = next_commands[:, -start_count_per_sequence:].detach()
        start_boundaries = dones[:, -start_count_per_sequence:]
        if bool(_cfg_get(self.cfg, "filter_done_imagination_starts", True)):
            start_mask = ~start_boundaries
        else:
            start_mask = torch.ones_like(start_boundaries, dtype=torch.bool)

        total_start_candidates = start_mask.numel()
        valid_start_candidates = int(start_mask.sum().detach().cpu())
        if valid_start_candidates > 0:
            start_state = RSSMState(
                deter=start_deter[start_mask].reshape(-1, self.world.hidden_vector_deter_dims),
                stoch=start_stoch[start_mask].reshape(-1, self.world.stoch_dim, self.world.num_bins_encoding),
            )
            start_commands = start_commands[start_mask].reshape(-1, start_commands.shape[-1])
        else:
            start_state = RSSMState(
                deter=start_deter.reshape(-1, self.world.hidden_vector_deter_dims)[:0],
                stoch=start_stoch.reshape(-1, self.world.stoch_dim, self.world.num_bins_encoding)[:0],
            )
            start_commands = start_commands.reshape(-1, start_commands.shape[-1])[:0]
        denom = valid_count.clamp_min(1.0)
        metrics = {
            "loss/model": float(model_loss.detach().cpu()),
            "loss/obs": float((obs_loss_sum / denom).detach().cpu()),
            "loss/reward": float((reward_loss_sum / rewards.numel()).detach().cpu()),
            "loss/continue": float((continue_loss_sum / rewards.numel()).detach().cpu()),
            "loss/kl_dyn": float((kl_dyn_sum / denom).detach().cpu()),
            "loss/kl_rep": float((kl_rep_sum / denom).detach().cpu()),
            "imag/start_candidates": float(total_start_candidates),
            "imag/start_count": float(valid_start_candidates),
            "imag/start_fraction": float(valid_start_candidates / max(total_start_candidates, 1)),
            "imag/last": float(start_count_per_sequence),
        }
        return metrics, ImaginationStarts(state=start_state, command=start_commands)

    def _train_actor_critic(self, starts: ImaginationStarts) -> dict[str, float]:
        start_state = starts.state
        command = starts.command
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
                reward = self.world.reward.pred(next_feat)
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
            target_values = self.target_critic.pred(feats.reshape(-1, feats.shape[-1])).reshape(horizon, -1)
            bootstrap = self.target_critic.pred(self.world.feature(state, command).detach())
            returns = lambda_returns(rewards_t, discounts_t, target_values, bootstrap, lambda_)
            weights = torch.cumprod(
                torch.cat((torch.ones_like(discounts_t[:1]), discounts_t[:-1]), dim=0), dim=0
            ).detach()
            return_scale = self.return_normalizer.scale(returns, update=True)

        critic_values_for_adv = self.critic.pred(feats.reshape(-1, feats.shape[-1])).reshape(horizon, -1).detach()
        raw_advantages = returns - critic_values_for_adv
        advantages = raw_advantages / return_scale
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
        self._after_optimizer_step("actor", self.actor_opt)

        critic_loss_per_step = self.critic.loss(
            feats.detach().reshape(-1, feats.shape[-1]), returns.detach().reshape(-1)
        ).reshape(horizon, -1)
        critic_loss = (weights * critic_loss_per_step).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), float(_cfg_get(self.cfg, "grad_clip")))
        self.critic_opt.step()
        self._after_optimizer_step("critic", self.critic_opt)
        self._update_target_critic()

        return {
            "loss/actor": float(actor_loss.detach().cpu()),
            "loss/critic": float(critic_loss.detach().cpu()),
            "imag/reward": float(rewards_t.mean().detach().cpu()),
            "imag/continue": float((discounts_t / discount).mean().detach().cpu()),
            "imag/return": float(returns.mean().detach().cpu()),
            "imag/return_norm_scale": float(return_scale.detach().cpu()),
            "imag/advantage": float(raw_advantages.mean().detach().cpu()),
            "imag/advantage_normed": float(advantages.mean().detach().cpu()),
            "policy/entropy": float(entropies_t.mean().detach().cpu()),
        }

    def _update_target_critic(self):
        tau = float(_cfg_get(self.cfg, "slow_critic_tau"))
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save(self, path: Path, iteration: int, total_steps: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
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
            "return_normalizer": self.return_normalizer.state_dict(),
        }
        if self.cbp_managers:
            payload["continual_backprop"] = self._cbp_summary()
            payload["continual_backprop_state_dict"] = self._cbp_state_dict()
        torch.save(payload, path)

    def _load_component_state(self, component: nn.Module, name: str, state: dict[str, Any], checkpoint_path: str) -> None:
        try:
            component.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load Dreamer {name} weights from {checkpoint_path}. "
                "If this checkpoint predates the symexp-two-hot reward/value heads, "
                "start a fresh Dreamer run or resume from a two-hot checkpoint."
            ) from exc

    def load(self, path: str) -> tuple[int, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._load_component_state(self.world, "world model", checkpoint["world"], path)
        self._load_component_state(self.actor, "actor", checkpoint["actor"], path)
        self._load_component_state(self.critic, "critic", checkpoint["critic"], path)
        self._load_component_state(
            self.target_critic, "target critic", checkpoint.get("target_critic", checkpoint["critic"]), path
        )
        self.model_opt.load_state_dict(checkpoint["model_opt"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt"])
        if "return_normalizer" in checkpoint:
            self.return_normalizer.load_state_dict(checkpoint["return_normalizer"])
        else:
            print("[INFO] Checkpoint has no return normalizer state; starting it from scratch.", flush=True)
        self._restore_cbp_state(checkpoint, path)
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
        self.terminals = torch.empty(self.capacity_steps, num_envs, dtype=torch.bool)
        self.truncations = torch.empty(self.capacity_steps, num_envs, dtype=torch.bool)
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
        terminal: torch.Tensor,
        truncation: torch.Tensor,
        next_obs: torch.Tensor,
        next_command: torch.Tensor,
    ):
        idx = self.pos
        self.obs[idx].copy_(obs.detach().cpu())
        self.commands[idx].copy_(command.detach().cpu())
        self.actions[idx].copy_(action.detach().cpu())
        self.rewards[idx].copy_(reward.detach().cpu())
        self.terminals[idx].copy_(terminal.detach().cpu())
        self.truncations[idx].copy_(truncation.detach().cpu())
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

        terminals = gather(self.terminals)
        truncations = gather(self.truncations)
        return {
            "obs": gather(self.obs),
            "commands": gather(self.commands),
            "actions": gather(self.actions),
            "rewards": gather(self.rewards),
            "terminals": terminals,
            "truncations": truncations,
            "dones": terminals | truncations,
            "next_obs": gather(self.next_obs),
            "next_commands": gather(self.next_commands),
        }


class ScalarLogger:
    def __init__(self, log_dir: Path, cfg: Any, env_cfg: Any, args_cli: argparse.Namespace):
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
                    id=os.environ.get("WANDB_RUN_ID"),
                    resume=os.environ.get("WANDB_RESUME", "allow"),
                    config=_wandb_config_payload(cfg, env_cfg, args_cli, log_dir),
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
    _cfg_set(agent_cfg, "seed", seed)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = int(_cfg_get(agent_cfg, "num_envs", env_cfg.scene.num_envs))
    _cfg_set(agent_cfg, "num_envs", int(env_cfg.scene.num_envs))
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    else:
        env_cfg.sim.device = str(_cfg_get(agent_cfg, "device", env_cfg.sim.device))
    _cfg_set(agent_cfg, "device", str(env_cfg.sim.device))
    if args_cli.max_iterations is not None:
        _cfg_set(agent_cfg, "max_iterations", args_cli.max_iterations)
    if args_cli.run_name is not None:
        _cfg_set(agent_cfg, "run_name", args_cli.run_name)
    if args_cli.logger is not None:
        _cfg_set(agent_cfg, "logger", args_cli.logger)
    cbp_enabled = bool(args_cli.use_cbp or _cfg_get(agent_cfg, "use_cbp", False))
    cbp_replacement_rate = float(
        _cfg_or_cli(agent_cfg, "cbp_replacement_rate", args_cli, "cbp_replacement_rate", "--cbp-replacement-rate", "--cbp_replacement_rate")
    )
    cbp_maturity_threshold = int(
        _cfg_or_cli(agent_cfg, "cbp_maturity_threshold", args_cli, "cbp_maturity_threshold", "--cbp-maturity-threshold", "--cbp_maturity_threshold")
    )
    cbp_decay_rate = float(
        _cfg_or_cli(agent_cfg, "cbp_decay_rate", args_cli, "cbp_decay_rate", "--cbp-decay-rate", "--cbp_decay_rate")
    )
    cbp_util_type = str(
        _cfg_or_cli(agent_cfg, "cbp_util_type", args_cli, "cbp_util_type", "--cbp-util-type", "--cbp_util_type")
    )
    cbp_init = str(_cfg_or_cli(agent_cfg, "cbp_init", args_cli, "cbp_init", "--cbp-init", "--cbp_init"))
    cbp_accumulate = bool(
        _cfg_or_cli(
            agent_cfg,
            "cbp_accumulate",
            args_cli,
            "cbp_accumulate",
            "--cbp-accumulate",
            "--no-cbp-accumulate",
            "--cbp_accumulate",
            "--no-cbp_accumulate",
        )
    )
    args_cli.use_cbp = cbp_enabled
    args_cli.cbp_replacement_rate = cbp_replacement_rate
    args_cli.cbp_maturity_threshold = cbp_maturity_threshold
    args_cli.cbp_decay_rate = cbp_decay_rate
    args_cli.cbp_util_type = cbp_util_type
    args_cli.cbp_init = cbp_init
    args_cli.cbp_accumulate = cbp_accumulate
    _cfg_set(agent_cfg, "use_cbp", cbp_enabled)
    _cfg_set(agent_cfg, "cbp_replacement_rate", cbp_replacement_rate)
    _cfg_set(agent_cfg, "cbp_maturity_threshold", cbp_maturity_threshold)
    _cfg_set(agent_cfg, "cbp_decay_rate", cbp_decay_rate)
    _cfg_set(agent_cfg, "cbp_util_type", cbp_util_type)
    _cfg_set(agent_cfg, "cbp_init", cbp_init)
    _cfg_set(agent_cfg, "cbp_accumulate", cbp_accumulate)

    if getattr(env_cfg, "policy_model", None) != "simple_dreamer_v3":
        raise ValueError("Dreamer trainer expects a task with env.policy_model='simple_dreamer_v3'.")

    log_root = Path("logs") / "dreamer" / str(_cfg_get(agent_cfg, "experiment_name"))
    log_root = log_root.resolve()
    run_prefix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = _sanitize_run_name(f"{run_prefix}_{_cfg_get(agent_cfg, 'run_name')}")
    wandb_run_id = _generate_wandb_run_id_if_needed(agent_cfg)
    if wandb_run_id is not None:
        run_name = _sanitize_run_name(f"{run_name}_{wandb_run_id}")
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

        agent = DreamerAgent(
            obs_dim,
            command_dim,
            action_dim,
            agent_cfg,
            device,
            cbp_args=args_cli if args_cli.use_cbp else None,
        )
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
        episode_rewards = torch.zeros(env.unwrapped.num_envs, device=device)
        recent_episodic_rewards: list[float] = []
        episodic_reward_window = int(env.unwrapped.num_envs)
        logger = ScalarLogger(log_dir, agent_cfg, env_cfg, args_cli)
        start_time = time.time()

        max_iterations = int(_cfg_get(agent_cfg, "max_iterations"))
        steps_per_env = int(_cfg_get(agent_cfg, "steps_per_env"))
        batch_size = int(_cfg_get(agent_cfg, "batch_size"))
        batch_length = int(_cfg_get(agent_cfg, "batch_length"))
        prefill_steps = int(_cfg_get(agent_cfg, "prefill_steps"))
        num_batches_trained_per_iteration = int(_cfg_get(agent_cfg, "num_batches_trained_per_iteration"))
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
                replay.add(obs, command, action, reward, terminated, truncated, next_obs, next_command)

                episode_rewards += reward
                if torch.any(done):
                    recent_episodic_rewards.extend(episode_rewards[done].detach().cpu().tolist())
                    recent_episodic_rewards = recent_episodic_rewards[-episodic_reward_window:]
                    episode_rewards[done] = 0.0

                state = agent.observe_next(state, action, next_obs, done)
                obs, command = next_obs, next_command
                total_steps += env.unwrapped.num_envs

            train_metrics: dict[str, float] = {}
            if replay.can_sample(batch_length, prefill_steps):
                for _ in range(num_batches_trained_per_iteration):
                    batch = replay.sample(batch_size, batch_length, device)
                    train_metrics = agent.train_on_batch(batch)

            if iteration % log_interval == 0 or iteration == 1:
                elapsed = max(time.time() - start_time, 1e-6)
                metrics = {
                    "train/iteration": float(iteration),
                    "train/env_steps": float(total_steps),
                    "train/fps": float(total_steps / elapsed),
                    "replay/steps": float(replay.total),
                    "episode/episodic_reward": float(sum(recent_episodic_rewards) / len(recent_episodic_rewards))
                    if recent_episodic_rewards
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
                    f"episodic_reward={metrics['episode/episodic_reward']:.3f} "
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
