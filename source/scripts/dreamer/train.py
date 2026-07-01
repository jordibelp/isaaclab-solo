# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a high-performance DreamerV3 world-model agent on Isaac Lab tasks.

The learning core lives in :mod:`dreamer_core` (block-diagonal GRU RSSM, symexp
two-hot heads, LaProp + adaptive gradient clipping, a single fused backward with
frozen-network imagination, AMP, ``torch.compile`` and a GPU-resident replay
buffer with latent caching).  This script keeps the Isaac Lab integration:
environment stepping, command handling, W&B/TensorBoard logging, checkpointing
and optional Continual-Backprop.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
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
  agent.hidden_vector_deter_dims=512
  agent.stoch_dim=32 agent.num_bins_encoding=32
  agent.model_lr=1e-4 agent.actor_entropy_scale=3e-4
  agent.use_compile=false agent.use_amp=true
  'agent.actor_hidden_dims=[512,256,128]'
  'agent.run_name="[Local]-dreamer-hp"'

Use the agent.<field>=<value> form for fields in dreamer_v3_cfg.py.
Use env.<field>=<value> for environment config fields, matching normal IsaacLab commands.
"""

parser = argparse.ArgumentParser(
    description="Train a high-performance DreamerV3 agent.",
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
from gymnasium.spaces import flatdim  # noqa: E402

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

from continual_backprop import build_continual_backprop_manager, cbp_specs_for_sequential_mlp  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import borinotIsaacLab.tasks  # noqa: F401, E402

from dreamer_core import DreamerAgent, SequenceReplayBuffer  # noqa: E402
from dreamer_core.agent import DreamerConfig  # noqa: E402
from dreamer_core.rssm import RSSMState  # noqa: E402


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_set(cfg: Any, name: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def _cfg_delete(cfg: Any, name: str) -> None:
    if isinstance(cfg, dict):
        cfg.pop(name, None)
    elif hasattr(cfg, name):
        delattr(cfg, name)


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


def _replay_ratio_metrics(agent_cfg: Any) -> dict[str, float]:
    num_envs = int(_cfg_get(agent_cfg, "num_envs"))
    steps_per_env = int(_cfg_get(agent_cfg, "steps_per_env"))
    num_batches = int(_cfg_get(agent_cfg, "num_batches_trained_per_iteration"))
    batch_size = int(_cfg_get(agent_cfg, "batch_size"))
    batch_length = int(_cfg_get(agent_cfg, "batch_length"))

    batch_timesteps = batch_size * batch_length
    env_timesteps = num_envs * steps_per_env
    replay_ratio = (num_batches * batch_timesteps) / env_timesteps
    return {
        "replay_ratio": float(replay_ratio),
        "num_gradients_per_policy_step": float(replay_ratio / batch_timesteps),
    }


def _wandb_config_payload(agent_cfg: Any, env_cfg: Any, args_cli: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    replay_ratio_metrics = _replay_ratio_metrics(agent_cfg)
    agent_dict = _jsonify(agent_cfg)
    if isinstance(agent_dict, dict):
        agent_dict.update(replay_ratio_metrics)
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
    payload.update(replay_ratio_metrics)
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


def _largest_divisor_at_most(value: int, cap: int) -> int:
    """Largest divisor of ``value`` that is <= ``cap`` (used to make deter block-safe)."""
    for candidate in range(min(cap, value), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def build_dreamer_config(agent_cfg: Any, obs_dim: int, command_dim: int, action_dim: int) -> DreamerConfig:
    """Map the Isaac Lab agent config (dreamer_v3_cfg.py) onto a DreamerConfig."""
    deter = int(_cfg_get(agent_cfg, "hidden_vector_deter_dims"))
    blocks = int(_cfg_get(agent_cfg, "blocks", 8))
    if deter % blocks != 0:
        adjusted = _largest_divisor_at_most(deter, blocks)
        print(f"[WARN] deter={deter} not divisible by blocks={blocks}; using blocks={adjusted}.", flush=True)
        blocks = adjusted
    model_hidden = int(_cfg_get(agent_cfg, "model_hidden_dim"))
    head_hidden = list(_cfg_get(agent_cfg, "head_hidden_dims", [model_hidden, model_hidden]))
    return DreamerConfig(
        obs_dim=obs_dim,
        command_dim=command_dim,
        action_dim=action_dim,
        device=str(_cfg_get(agent_cfg, "device", "cuda:0")),
        deter=deter,
        stoch=int(_cfg_get(agent_cfg, "stoch_dim")),
        discrete=int(_cfg_get(agent_cfg, "num_bins_encoding")),
        model_hidden=model_hidden,
        blocks=blocks,
        obs_layers=int(_cfg_get(agent_cfg, "obs_layers", 1)),
        img_layers=int(_cfg_get(agent_cfg, "img_layers", 2)),
        dyn_layers=int(_cfg_get(agent_cfg, "dyn_layers", 1)),
        unimix=float(_cfg_get(agent_cfg, "unimix", 0.01)),
        encoder_hidden_dims=list(_cfg_get(agent_cfg, "encoder_hidden_dims")),
        head_hidden_dims=head_hidden,
        actor_hidden_dims=list(_cfg_get(agent_cfg, "actor_hidden_dims")),
        critic_hidden_dims=list(_cfg_get(agent_cfg, "critic_hidden_dims")),
        num_bins=int(_cfg_get(agent_cfg, "reward_value_num_bins", 255)),
        symlog_range=float(_cfg_get(agent_cfg, "reward_value_symlog_range", 20.0)),
        actor_min_std=float(_cfg_get(agent_cfg, "actor_min_std", 0.1)),
        actor_max_std=float(_cfg_get(agent_cfg, "actor_max_std", 1.0)),
        kl_free=float(_cfg_get(agent_cfg, "free_nats", 1.0)),
        kl_dyn_scale=float(_cfg_get(agent_cfg, "kl_dyn_scale", 0.5)),
        kl_rep_scale=float(_cfg_get(agent_cfg, "kl_rep_scale", 0.1)),
        decoder_scale=float(_cfg_get(agent_cfg, "obs_loss_scale", 1.0)),
        reward_scale=float(_cfg_get(agent_cfg, "reward_loss_scale", 1.0)),
        cont_scale=float(_cfg_get(agent_cfg, "continue_loss_scale", 1.0)),
        policy_scale=float(_cfg_get(agent_cfg, "policy_loss_scale", 1.0)),
        value_scale=float(_cfg_get(agent_cfg, "value_loss_scale", 1.0)),
        repval_scale=float(_cfg_get(agent_cfg, "repval_loss_scale", 0.3)),
        act_entropy_scale=float(_cfg_get(agent_cfg, "actor_entropy_scale", 3e-4)),
        slowreg=float(_cfg_get(agent_cfg, "slowreg", 1.0)),
        imag_horizon=int(_cfg_get(agent_cfg, "imag_horizon", 15)),
        discount=float(_cfg_get(agent_cfg, "discount", 0.99)),
        lambda_=float(_cfg_get(agent_cfg, "lambda_", 0.95)),
        normalize_actor_returns=bool(_cfg_get(agent_cfg, "normalize_actor_returns", True)),
        return_norm_rate=float(_cfg_get(agent_cfg, "return_norm_rate", 0.01)),
        return_norm_limit=float(_cfg_get(agent_cfg, "return_norm_limit", 1.0)),
        return_norm_pct_low=float(_cfg_get(agent_cfg, "return_norm_percentile_low", 5.0)),
        return_norm_pct_high=float(_cfg_get(agent_cfg, "return_norm_percentile_high", 95.0)),
        slow_critic_tau=float(_cfg_get(agent_cfg, "slow_critic_tau", 0.02)),
        model_lr=float(_cfg_get(agent_cfg, "model_lr")),
        actor_lr=float(_cfg_get(agent_cfg, "actor_lr")),
        critic_lr=float(_cfg_get(agent_cfg, "critic_lr")),
        laprop_beta1=float(_cfg_get(agent_cfg, "laprop_beta1", 0.9)),
        laprop_beta2=float(_cfg_get(agent_cfg, "laprop_beta2", 0.999)),
        laprop_eps=float(_cfg_get(agent_cfg, "laprop_eps", 1e-20)),
        agc_clip=float(_cfg_get(agent_cfg, "agc_clip", 0.3)),
        agc_pmin=float(_cfg_get(agent_cfg, "agc_pmin", 1e-3)),
        warmup_steps=int(_cfg_get(agent_cfg, "warmup_steps", 1000)),
        use_amp=bool(_cfg_get(agent_cfg, "use_amp", True)),
        use_compile=bool(_cfg_get(agent_cfg, "use_compile", True)),
    )


def _attach_dreamer_cbp(agent: DreamerAgent, parsed_args: argparse.Namespace) -> None:
    """Wire optional Continual-Backprop managers onto the agent's MLP groups."""
    managers: dict[str, object] = {}
    for group_name, mlps in agent.cbp_target_mlps().items():
        specs: list[Any] = []
        for i, net in enumerate(mlps):
            specs.extend(cbp_specs_for_sequential_mlp(f"{group_name}.mlp{i}", net))
        if not specs:
            continue
        managers[group_name] = build_continual_backprop_manager(
            specs,
            replacement_rate=float(parsed_args.cbp_replacement_rate),
            maturity_threshold=int(parsed_args.cbp_maturity_threshold),
            decay_rate=float(parsed_args.cbp_decay_rate),
            util_type=str(parsed_args.cbp_util_type),
            init=str(parsed_args.cbp_init),
            accumulate=bool(parsed_args.cbp_accumulate),
        )
    if not managers:
        raise ValueError("--use-cbp was provided, but no eligible Dreamer MLP hidden groups were found.")
    agent.attach_cbp(managers)
    for name, manager in managers.items():
        print(
            f"[INFO] Continual Backpropagation enabled for {name} "
            f"(replacement_rate={manager.replacement_rate:g}, maturity_threshold={manager.maturity_threshold}, "
            f"decay_rate={manager.decay_rate:g}, util_type={manager.util_type}, init={manager.init}).",
            flush=True,
        )


def _cbp_metrics(agent: DreamerAgent) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, manager in agent.cbp_managers.items():
        prefix = f"CBP/{name}"
        metrics[f"{prefix}/optimizer_steps"] = float(manager.optimizer_steps)
        metrics[f"{prefix}/replacements_last_update"] = float(sum(manager.last_replacements.values()))
        metrics[f"{prefix}/replacements_total"] = float(sum(manager.total_replacements_by_group.values()))
    return metrics


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
                self.wandb = None
                print(f"[WARN] Could not initialize W&B ({exc}); continuing with stdout logging.", flush=True)
            if self.wandb is not None:
                try:
                    self.wandb.define_metric("num_env_interactions")
                    self.wandb.define_metric("num_optimization_steps")
                    self.wandb.define_metric("*", step_metric="num_env_interactions")
                except Exception as exc:
                    print(f"[WARN] Could not define W&B step metrics ({exc}); continuing.", flush=True)

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
    # ---- seeds / device / iteration count -------------------------------
    agent_seed = 42 if _cfg_get(agent_cfg, "seed") is None else int(_cfg_get(agent_cfg, "seed"))
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

    # ---- CBP config resolution (CLI overrides config) -------------------
    cbp_enabled = bool(args_cli.use_cbp or _cfg_get(agent_cfg, "use_cbp", False))
    args_cli.use_cbp = cbp_enabled
    args_cli.cbp_replacement_rate = float(
        _cfg_or_cli(agent_cfg, "cbp_replacement_rate", args_cli, "cbp_replacement_rate", "--cbp-replacement-rate", "--cbp_replacement_rate")
    )
    args_cli.cbp_maturity_threshold = int(
        _cfg_or_cli(agent_cfg, "cbp_maturity_threshold", args_cli, "cbp_maturity_threshold", "--cbp-maturity-threshold", "--cbp_maturity_threshold")
    )
    args_cli.cbp_decay_rate = float(
        _cfg_or_cli(agent_cfg, "cbp_decay_rate", args_cli, "cbp_decay_rate", "--cbp-decay-rate", "--cbp_decay_rate")
    )
    args_cli.cbp_util_type = str(_cfg_or_cli(agent_cfg, "cbp_util_type", args_cli, "cbp_util_type", "--cbp-util-type", "--cbp_util_type"))
    args_cli.cbp_init = str(_cfg_or_cli(agent_cfg, "cbp_init", args_cli, "cbp_init", "--cbp-init", "--cbp_init"))
    args_cli.cbp_accumulate = bool(
        _cfg_or_cli(agent_cfg, "cbp_accumulate", args_cli, "cbp_accumulate", "--cbp-accumulate", "--no-cbp-accumulate", "--cbp_accumulate", "--no-cbp_accumulate")
    )
    for key in ("use_cbp", "cbp_replacement_rate", "cbp_maturity_threshold", "cbp_decay_rate", "cbp_util_type", "cbp_init", "cbp_accumulate"):
        _cfg_set(agent_cfg, key, getattr(args_cli, key))

    if getattr(env_cfg, "policy_model", None) != "simple_dreamer_v3":
        raise ValueError("Dreamer trainer expects a task with env.policy_model='simple_dreamer_v3'.")
    command_outside_observation = bool(_cfg_get(agent_cfg, "command_outside_observation", False))
    _cfg_set(env_cfg, "_dreamer_command_outside_observation", command_outside_observation)
    if hasattr(env_cfg, "refresh_observation_dimensions"):
        env_cfg.refresh_observation_dimensions()
    _cfg_delete(env_cfg, "_dreamer_command_outside_observation")

    # ---- logging directory ----------------------------------------------
    log_root = (Path("logs") / "dreamer" / str(_cfg_get(agent_cfg, "experiment_name"))).resolve()
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
    _cfg_set(env_cfg, "_dreamer_command_outside_observation", command_outside_observation)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        device = torch.device(env.unwrapped.device)
        torch.set_float32_matmul_precision("high")
        obs_dict, _ = env.reset()
        obs, command = _obs_command(obs_dict, device)
        num_envs = env.unwrapped.num_envs
        obs_dim = obs.shape[-1]
        command_dim = command.shape[-1]
        action_dim = flatdim(env.unwrapped.single_action_space)

        cfg = build_dreamer_config(agent_cfg, obs_dim, command_dim, action_dim)
        agent = DreamerAgent(cfg)
        print(f"[INFO] DreamerV3 agent: {sum(p.numel() for p in agent.parameters() if p.requires_grad):,} trainable params "
              f"(obs={obs_dim}, command={command_dim}, action={action_dim}, deter={cfg.deter}, "
              f"stoch={cfg.stoch}x{cfg.discrete}, amp={cfg.use_amp}, compile={cfg.use_compile}).", flush=True)
        if cbp_enabled:
            _attach_dreamer_cbp(agent, args_cli)

        start_iteration = 0
        total_steps = 0
        if args_cli.checkpoint:
            ckpt = agent.load(args_cli.checkpoint)
            start_iteration = int(ckpt.get("iteration", 0))
            total_steps = int(ckpt.get("total_steps", 0))
            print(f"[INFO] Resumed Dreamer checkpoint from {args_cli.checkpoint}", flush=True)

        # ---- hyperparameters -------------------------------------------
        max_iterations = int(_cfg_get(agent_cfg, "max_iterations"))
        steps_per_env = int(_cfg_get(agent_cfg, "steps_per_env"))
        batch_size = int(_cfg_get(agent_cfg, "batch_size"))
        batch_length = int(_cfg_get(agent_cfg, "batch_length"))
        seq_len = batch_length + 1  # +1 context step for the RSSM warm start
        prefill_steps = int(_cfg_get(agent_cfg, "prefill_steps"))
        num_batches = int(_cfg_get(agent_cfg, "num_batches_trained_per_iteration"))
        log_interval = int(_cfg_get(agent_cfg, "log_interval"))
        save_interval = int(_cfg_get(agent_cfg, "save_interval"))
        save_best_checkpoint = bool(_cfg_get(agent_cfg, "save_best_checkpoint", True))
        recent_fraction = float(_cfg_get(agent_cfg, "replay_recent_fraction", 0.0))
        if bool(_cfg_get(agent_cfg, "use_uniform_replay_buffer_with_online_queue", False)) and recent_fraction == 0.0:
            recent_fraction = 0.1  # honour the "mix in recent data" intent of the old online queue
        replay_ratio_metrics = _replay_ratio_metrics(agent_cfg)

        replay = SequenceReplayBuffer(
            capacity=int(_cfg_get(agent_cfg, "replay_size")),
            num_envs=num_envs,
            obs_dim=obs_dim,
            command_dim=command_dim,
            action_dim=action_dim,
            stoch=cfg.stoch,
            discrete=cfg.discrete,
            deter=cfg.deter,
            device=device,
            storage_device=torch.device(str(_cfg_get(agent_cfg, "replay_storage_device", device))),
            recent_fraction=recent_fraction,
        )

        # ---- collection state ------------------------------------------
        state = agent.initial_state(num_envs)
        prev_action = torch.zeros(num_envs, action_dim, device=device)
        is_first = torch.ones(num_envs, dtype=torch.bool, device=device)
        episode_rewards = torch.zeros(num_envs, device=device)
        recent_episodic_rewards: list[float] = []
        best_episodic_reward = float("-inf")
        best_iteration = 0
        last_metrics: dict[str, float] = {}
        logger = ScalarLogger(log_dir, agent_cfg, env_cfg, args_cli)
        start_time = time.time()

        for iteration in range(start_iteration + 1, max_iterations + 1):
            # ---- collect --------------------------------------------------
            for _ in range(steps_per_env):
                action_policy, state = agent.act(state, prev_action, command, is_first, obs, eval=False)
                if replay.total < prefill_steps:
                    action = torch.empty(num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
                else:
                    action = action_policy

                next_obs_dict, reward, terminated, truncated, extras = env.step(action)
                done = terminated | truncated
                next_obs, next_command = _obs_command(next_obs_dict, device)

                replay.add(
                    obs=obs, command=command, action=action, reward=reward.float(),
                    is_first=is_first, is_terminal=terminated, is_last=done,
                    stoch=state.stoch, deter=state.deter,
                )

                episode_rewards += reward
                if torch.any(done):
                    recent_episodic_rewards.extend(episode_rewards[done].detach().cpu().tolist())
                    recent_episodic_rewards = recent_episodic_rewards[-num_envs:]
                    episode_rewards[done] = 0.0

                obs, command = next_obs, next_command
                prev_action = action
                is_first = done
                total_steps += num_envs

            # ---- train ----------------------------------------------------
            if replay.can_sample(seq_len, prefill_steps):
                for _ in range(num_batches):
                    batch, writeback = replay.sample(batch_size, seq_len)
                    metrics = agent.update(batch, writeback, replay)
                last_metrics = agent.metrics_to_float(metrics)
                last_metrics.update(_cbp_metrics(agent))

            # ---- best-checkpoint bookkeeping -----------------------------
            episodic_reward_metric = (
                float(sum(recent_episodic_rewards) / len(recent_episodic_rewards)) if recent_episodic_rewards else None
            )
            if save_best_checkpoint and episodic_reward_metric is not None and episodic_reward_metric > best_episodic_reward:
                best_episodic_reward = episodic_reward_metric
                best_iteration = iteration
                agent.save(
                    log_dir / "checkpoints" / "best_model.pt",
                    extra={
                        "iteration": iteration, "total_steps": total_steps,
                        "cfg": _cfg_to_dict(agent_cfg),
                        "best_model_metric": "episode/episodic_reward",
                        "best_model_value": best_episodic_reward,
                    },
                )
                print(f"[INFO] Saved new Dreamer best checkpoint episode/episodic_reward={best_episodic_reward:.3f} "
                      f"at iteration={iteration} steps={total_steps}", flush=True)

            # ---- logging --------------------------------------------------
            if iteration % log_interval == 0 or iteration == 1:
                elapsed = max(time.time() - start_time, 1e-6)
                metrics = {
                    "num_env_interactions": float(total_steps),
                    "num_optimization_steps": float(agent.num_optimization_steps),
                    "train/iteration": float(iteration),
                    "train/env_steps": float(total_steps),
                    "train/fps": float(total_steps / elapsed),
                    "replay/steps": float(replay.total),
                    "replay/recent_fraction": float(recent_fraction),
                    "train/replay_ratio": replay_ratio_metrics["replay_ratio"],
                    "train/num_gradients_per_policy_step": replay_ratio_metrics["num_gradients_per_policy_step"],
                    "episode/episodic_reward": episodic_reward_metric if episodic_reward_metric is not None else 0.0,
                    "checkpoint/best_episodic_reward": best_episodic_reward if best_iteration else 0.0,
                    "checkpoint/best_iteration": float(best_iteration),
                }
                metrics.update(last_metrics)
                env_logs = extras.get("log", {}) if isinstance(extras, dict) else {}
                for key, value in env_logs.items():
                    if isinstance(value, (int, float)):
                        metrics[f"env/{key}"] = float(value)
                logger.log(metrics, total_steps)
                print(f"[INFO] iter={iteration} steps={total_steps} opt_steps={agent.num_optimization_steps} "
                      f"episodic_reward={metrics['episode/episodic_reward']:.3f} "
                      f"total_loss={last_metrics.get('total_loss', 0.0):.4f} "
                      f"fps={metrics['train/fps']:.0f}", flush=True)

            if iteration % save_interval == 0:
                agent.save(log_dir / "checkpoints" / f"model_{iteration}.pt",
                           extra={"iteration": iteration, "total_steps": total_steps, "cfg": _cfg_to_dict(agent_cfg)})

        agent.save(log_dir / "checkpoints" / "last.pt",
                   extra={"iteration": max_iterations, "total_steps": total_steps, "cfg": _cfg_to_dict(agent_cfg)})
        logger.close()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
