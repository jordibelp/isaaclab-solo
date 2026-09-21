#!/usr/bin/env python3
"""Interaction-budgeted MJX SAC fine-tuning for Solo12 (arXiv:2602.20220).

This uses the same RSL-RL-SAC actor, critic, replay buffer, checkpoints, symmetry
augmentation, timeout handling, and n-step targets as Isaac training. MJX only
replaces the vectorized simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import statistics
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# Share the GPU with PyTorch instead of letting JAX reserve almost all VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tensordict import TensorDict

import train_lora as mjx_env

_ROOT = Path(__file__).resolve().parents[1]
for _path in (
    _ROOT / "source" / "scripts" / "skrl",
    _ROOT / "source" / "rsl_rl_sac_vendor",
    _ROOT / "source",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import solo12_symmetry
from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.modules import LAYER_CHOICES, apply_lora, merged_state_dict
from rsl_rl_sac.runners import OffPolicyRunner
from rsl_rl_sac.storage import MixedReplayBuffer, ReplayBuffer
from rsl_rl_sac.utils import resolve_optimizer


def reproducible_command(argv=None):
    """Return a shell-safe command that recreates this training invocation."""
    arguments = sys.argv[1:] if argv is None else argv
    return shlex.join(["./isaaclab.sh", "-p", "mujoco/train_sac.py", *arguments])


def _bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="solo12-two-feet")
    p.add_argument("--checkpoint", default=None, help="RSL-RL-SAC checkpoint to fine-tune/resume from.")
    p.add_argument("--resume", action="store_true", help="Restore optimizers and iteration as well as networks.")
    p.add_argument("--reset-optimizers", action="store_true",
                   help="Start fresh Adam moments instead of transferring compatible checkpoint optimizers.")
    p.add_argument("--freeze-alpha", action="store_true", help="Keep the SAC entropy temperature fixed.")
    p.add_argument(
        "--source-env-cfg",
        default=None,
        metavar="PATH|auto",
        help="Copy the environment settings of the Isaac run that produced the checkpoint, from its"
        " params/env.yaml. 'auto' reads <checkpoint dir>/params/env.yaml. Explicit env.* overrides still win."
        " Always attach the value with '=', because a detached value would swallow the next env.* override.",
    )
    p.add_argument(
        "--curriculum-stage",
        type=int,
        default=None,
        help="Curriculum stage to copy with --source-env-cfg (default: the last stage).",
    )
    p.add_argument("--run-name", default="[mujoco] Solo12 SAC")
    p.add_argument("--num_envs", "--num-envs", type=int, default=1)
    p.add_argument(
        "--max-env-interactions",
        type=int,
        default=50_000,
        help="Total environment interactions to collect, including warm-up.",
    )
    p.add_argument(
        "--rollout-steps",
        type=int,
        default=1000,
        help=(
            "Interactions per environment collected before each update phase. Collection"
            " runs across episode ends, so an early termination no longer shortens the"
            " update budget."
        ),
    )
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument(
        "--episode-log-window",
        "--episode_log_window",
        type=int,
        default=5,
        help="Episodes averaged by Train/mean_reward and Train/mean_episode_length. Isaac uses 100,"
        " which lags a whole fine-tuning budget at one environment.",
    )
    warmup = p.add_mutually_exclusive_group()
    warmup.add_argument(
        "--num-transitions-before-weight-updates",
        "--num_transitions_before_weight_updates",
        "--transitions-before-updates",
        dest="transitions_before_updates",
        type=int,
        default=5000,
        help=(
            "Transitions collected with the loaded policy before the first gradient update."
            " Default 5000. Updates start at the first rollout boundary"
            " at or beyond this count; retained offline data does not count."
        ),
    )
    warmup.add_argument(
        "--start-training", type=int, default=None,
        help="Deprecated: warm-up in full rollout lengths. Prefer --num-transitions-before-weight-updates.",
    )
    p.add_argument("--replay-buffer-size", type=int, default=500_000)
    p.add_argument(
        "--updates-per-iteration",
        type=int,
        default=None,
        help="Fixed gradient updates per episode/rollout (K), instead of --utd.",
    )
    p.add_argument(
        "--utd",
        type=float,
        default=None,
        help=(
            "Critic update-to-data ratio (eta = K / transitions per iteration). Default 1.25,"
            " which is 1250 updates on the 1000-step Solo12 episode and matches the Go1"
            " setting of arXiv:2602.20220."
        ),
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--tau", type=float, default=0.003)
    p.add_argument(
        "--offline-replay-buffer",
        default=None,
        help=(
            "replay_buffer.pt written by a pretraining run. Its transitions are retained and"
            " mixed into every mini-batch to stabilize early fine-tuning (arXiv:2602.20220)."
        ),
    )
    p.add_argument(
        "--offline-fraction",
        type=float,
        default=None,
        help="Share of each mini-batch taken from the retained data at the start. Default 0.5.",
    )
    p.add_argument(
        "--offline-fraction-final",
        type=float,
        default=None,
        help="Share reached at the end of the anneal. Default 0.0, so training ends on MJX data only.",
    )
    p.add_argument(
        "--offline-anneal-iterations",
        type=int,
        default=None,
        help=(
            "Update-bearing episodes/rollouts over which the offline share is annealed,"
            " excluding warm-up. Default: half of the interaction-budget iterations."
        ),
    )
    p.add_argument(
        "--rank",
        type=int,
        default=0,
        help=(
            "Shared actor/critic LoRA rank. 0 (default) trains all weights; positive ranks"
            " train only adapters. Override with --actor-rank or --critic-rank."
        ),
    )
    p.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help="Shared adapter gain. Defaults to each network's own rank (scale alpha/rank = 1).",
    )
    p.add_argument("--lora-layers", choices=LAYER_CHOICES, default="all")
    for network in ("actor", "critic"):
        p.add_argument(
            f"--{network}-rank", type=int, default=None,
            help=f"Override --rank for the {network}: 0 = full tuning, positive = LoRA.",
        )
        p.add_argument(
            f"--freeze-{network}", action="store_true",
            help=f"Freeze the {network}, including its normalizer; overrides shared --rank.",
        )
        p.add_argument(f"--{network}-lora-alpha", type=float, default=None)
        p.add_argument(f"--{network}-lora-layers", choices=LAYER_CHOICES, default=None)
    p.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Snapshot the MJX replay buffer so a later run can retain it in turn.",
    )
    p.add_argument("--save-replay-buffer-every", type=int, default=500)
    p.add_argument(
        "--actor-update-every",
        type=int,
        default=20,
        help=(
            "Critic updates per actor update (M in arXiv:2602.20220). The paper shows that"
            " updating the actor every critic step destabilizes transfer on every platform"
            " it tested; 20 is its recommendation. Use 1 for synchronous updates."
        ),
    )
    p.add_argument("--actor-learning-rate", type=float, default=1.0e-5)
    p.add_argument("--critic-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--alpha-learning-rate", type=float, default=2.0e-5)
    p.add_argument("--initial-alpha", type=float, default=None,
                   help="Override checkpoint entropy temperature; without a checkpoint use 0.001.")
    p.add_argument("--target-entropy-scale", type=float, default=0.167)
    p.add_argument("--initial-std", type=float, default=0.15)
    p.add_argument("--actor-activation", default=None, help="Override checkpoint-sidecar activation (fallback: swish).")
    p.add_argument(
        "--critic-activation", default=None, help="Override checkpoint-sidecar activation (fallback: swish)."
    )
    p.add_argument(
        "--critic-loss", choices=("mse", "two_hot", "hl_gauss"), default=None,
        help="Override critic label loss; detached categorical checkpoints default to two_hot.",
    )
    p.add_argument("--hl-gauss-sigma-ratio", type=float, default=None)
    p.add_argument("--symmetry-mode", choices=("none", "augmentation", "loss", "both"), default="augmentation")
    p.add_argument("--symmetry-loss-coeff", type=float, default=0.1)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("logs/mujoco/sac"))
    p.add_argument("--wandb-project", default="solo12-two-feet-lora")
    p.add_argument("--wandb-entity", default="jordibelp")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--headless", action="store_true", help="Accepted for CLI compatibility; MJX is headless.")
    return p


def _torch_from_jax(value: jax.Array, device: torch.device) -> torch.Tensor:
    value = jnp.asarray(value)
    tensor = torch.utils.dlpack.from_dlpack(value)
    return tensor.to(device=device)


def _jax_from_torch(value: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(value.detach().contiguous())


class MjxSolo12VecEnv:
    """RSL-RL VecEnv-compatible adapter around the existing parallel MJX task."""

    def __init__(self, env_cfg: dict, num_envs: int, seed: int, torch_device: str):
        self.num_envs = int(num_envs)
        self.num_actions = 12
        self.device = torch.device(torch_device)
        self.cfg = SimpleNamespace(**env_cfg, is_finite_horizon=False, action_scale=float(mjx_env.ACTION_SCALE))
        self.max_episode_length = max(1, round(self.cfg.episode_length_s / mjx_env.STEP_DT))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.unwrapped = self
        model_info = mjx_env.build_model(
            Path(__file__).with_name("solo12.xml"), float(env_cfg["kp"]), float(env_cfg["kd"]), env_cfg
        )
        self.action_lower_magnitude, self.action_upper_magnitude = self._action_bounds(model_info.model)
        self._reset_fn, self._step_fn = mjx_env.make_training_functions(model_info, env_cfg, self.num_envs)
        self._key = jax.random.PRNGKey(seed)
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._reset_fn(reset_key)

    def _action_bounds(self, model=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Both simulators drive ``target = SAFE_Q + ACTION_SCALE * action``, so the action
        # centre is SAFE_Q, not q=0. Measure each joint range from that centre.
        import mujoco

        if model is None:
            model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("solo12.xml")))
        ranges = []
        for name in mjx_env.JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ranges.append(model.jnt_range[jid].copy())
        ranges = np.asarray(ranges)
        centre = np.asarray(mjx_env.SAFE_Q, dtype=np.float64)
        lower = torch.as_tensor(
            (centre - ranges[:, 0]) / mjx_env.ACTION_SCALE, dtype=torch.float32, device=self.device
        )
        upper = torch.as_tensor(
            (ranges[:, 1] - centre) / mjx_env.ACTION_SCALE, dtype=torch.float32, device=self.device
        )
        if torch.any(lower <= 0) or torch.any(upper <= 0):
            raise ValueError("MJX SAC requires the SAFE_Q action centre to lie inside every joint range.")
        return lower, upper

    def get_observations(self) -> TensorDict:
        obs = _torch_from_jax(self._state.obs, self.device)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._reset_fn(reset_key)
        self.episode_length_buf.zero_()
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor):
        self._key, step_key = jax.random.split(self._key)
        result = self._step_fn(self._state, _jax_from_torch(actions), step_key)
        (
            self._state,
            rewards,
            reward_terms,
            dones,
            terminated,
            termination_causes,
            final_obs,
            episode_returns,
            episode_steps,
            _episode_reward_sums,
        ) = result
        obs = self.get_observations()
        rewards_t = _torch_from_jax(rewards, self.device)
        dones_t = _torch_from_jax(dones, self.device).bool()
        terminated_t = _torch_from_jax(terminated, self.device).bool()
        timeout_t = dones_t & ~terminated_t
        self.episode_length_buf = _torch_from_jax(self._state.episode_steps, self.device).long()
        final_obs_t = _torch_from_jax(final_obs, self.device)
        terms_t = _torch_from_jax(reward_terms, self.device)
        log = {f"RewardsPerStep/{name}": terms_t[:, i].mean()
               for i, name in enumerate(mjx_env.REWARD_TERM_NAMES)}
        log["RewardsPerStep/total"] = rewards_t.mean()
        causes_t = _torch_from_jax(termination_causes, self.device).float()
        for i, name in enumerate(("base_contact", "front_contact", "diverged")):
            log[f"Terminations/{name}_per_step"] = causes_t[:, i].mean()
        log["Terminations/timeout_per_step"] = timeout_t.float().mean()
        if dones_t.any():
            log["Episodes/return"] = _torch_from_jax(episode_returns, self.device)[dones_t]
            log["Episodes/length_steps"] = _torch_from_jax(episode_steps, self.device)[dones_t].float()
        extras = {
            "time_outs": timeout_t,
            "time_outs_obs": TensorDict(
                {"policy": final_obs_t}, batch_size=[self.num_envs], device=self.device
            ),
            "log": log,
        }
        return obs, rewards_t, dones_t, extras

    def close(self):
        pass


def _mjx_action_scaling(env, device: str):
    unwrapped = getattr(env, "unwrapped", env)
    return (
        unwrapped.action_upper_magnitude.to(device),
        unwrapped.action_lower_magnitude.to(device),
    )


def _report_checkpoint_action_scaling(actor, env) -> None:
    """Report the transferred action map; a loaded actor keeps its own tanh scaling.

    ``action = action_range * tanh(latent) + action_bias`` is part of the trained policy,
    not a property of the simulator. Both simulators apply the same
    ``target = SAFE_Q + ACTION_SCALE * action``, so rescaling a transferred actor to the raw
    XML joint ranges multiplies every joint target it emits and destroys the policy.
    """
    upper, lower = _mjx_action_scaling(env, str(actor.action_bias.device))
    to_degrees = mjx_env.ACTION_SCALE * 180.0 / math.pi
    span = (2.0 * actor.action_range * to_degrees)
    allowed = (upper + lower) * to_degrees
    print(
        f"[INFO] Action scaling kept from the checkpoint: joint target spans"
        f" {span.min():.0f}-{span.max():.0f} deg around SAFE_Q"
        f" (this simulator permits up to {allowed.min():.0f}-{allowed.max():.0f} deg)."
    )


def _apply_finetuning(runner, args) -> None:
    """Configure each network after transfer (or before dense optimizer resume)."""
    settings = _finetuning_config(args)
    algorithm = runner.alg
    for name, config in settings.items():
        model = getattr(algorithm, name)
        adapted = 0
        if config["mode"] == "frozen":
            model.requires_grad_(False)
            model.eval()
        elif config["mode"] == "lora":
            # Apply the layer selection independently to each online Q-network.
            # Targets stay dense, frozen, and retain the loaded checkpoint's lag.
            networks = (model,) if name == "actor" else (model.critic1, model.critic2)
            for network in networks:
                adapted += apply_lora(network, config["rank"], config["alpha"], config["layers"])
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = (
            resolve_optimizer("adam")(parameters, lr=getattr(args, f"{name}_learning_rate"))
            if parameters else None
        )
        setattr(algorithm, f"{name}_parameters", parameters)
        setattr(algorithm, f"{name}_optimizer", optimizer)
        trainable = sum(p.numel() for p in parameters)
        print(
            f"[INFO] {name.capitalize()} tuning: {config}; {adapted} adapted layers; "
            f"{trainable:,} trainable parameters (critic targets always excluded)."
        )

    original_save = algorithm.save

    def save_with_merged_networks() -> dict:
        payload = original_save()
        payload["actor_state_dict"] = merged_state_dict(algorithm.actor)
        payload["critic_state_dict"] = merged_state_dict(algorithm.critic)
        payload["mujoco_finetuning"] = settings
        if any(config["mode"] == "lora" for config in settings.values()):
            payload["mujoco_lora"] = settings
        return payload

    algorithm.save = save_with_merged_networks


def _finetuning_config(args) -> dict:
    """Resolve shared defaults, per-network overrides, and explicit freeze flags."""
    if args.rank < 0:
        raise ValueError(f"--rank must be non-negative, got {args.rank}.")
    settings = {}
    for name in ("actor", "critic"):
        override = getattr(args, f"{name}_rank")
        rank = args.rank if override is None else override
        frozen = getattr(args, f"freeze_{name}")
        alpha = getattr(args, f"{name}_lora_alpha")
        layers = getattr(args, f"{name}_lora_layers")
        if rank < 0:
            raise ValueError(f"--{name}-rank must be non-negative.")
        if frozen and override not in (None, 0):
            raise ValueError(f"--freeze-{name} conflicts with a positive --{name}-rank.")
        mode = "frozen" if frozen else ("lora" if rank > 0 else "full")
        if mode != "lora" and (alpha is not None or layers is not None):
            raise ValueError(f"--{name}-lora-* requires a positive rank on an unfrozen {name}.")
        alpha = args.lora_alpha if alpha is None else alpha
        alpha = float(rank) if alpha is None else alpha
        if mode == "lora" and (not math.isfinite(alpha) or alpha <= 0):
            raise ValueError(f"{name} LoRA alpha must be finite and positive.")
        settings[name] = {
            "mode": mode, "rank": rank if mode == "lora" else 0,
            "alpha": alpha if mode == "lora" else None,
            "layers": (layers or args.lora_layers) if mode == "lora" else None,
        }
    has_lora = any(config["mode"] == "lora" for config in settings.values())
    if not has_lora and (args.lora_alpha is not None or args.lora_layers != "all"):
        raise ValueError("Shared --lora-* settings require a positive rank on an unfrozen network.")
    if has_lora and args.resume:
        raise ValueError(
            "--resume cannot restore LoRA optimizers from merged networks. "
            "Start a new run with --checkpoint instead."
        )
    return settings


def _restore_finetuning_state(algorithm, payload: dict, args) -> None:
    """Transfer temperature and compatible Adam moments without resuming run counters.

    Dense optimizers cannot be mapped to LoRA factors. Target networks and normalizers
    have already been loaded with the model state; leave their checkpoint lag intact.
    """
    if args.initial_alpha is None:
        saved_log_alpha = payload.get("log_alpha")
        saved_alpha = payload.get("alpha")
        if saved_log_alpha is not None:
            algorithm.log_alpha.data.copy_(saved_log_alpha.to(algorithm.device))
        elif saved_alpha is not None:
            algorithm.log_alpha.data.fill_(math.log(saved_alpha))
        algorithm.alpha = algorithm.log_alpha.exp().item()
    restored = []
    if not args.reset_optimizers:
        settings = _finetuning_config(args)
        for name in ("actor", "critic", "alpha"):
            optimizer = getattr(algorithm, f"{name}_optimizer")
            state = payload.get(f"{name}_optimizer_state_dict")
            if optimizer is None or state is None:
                continue
            if name == "alpha":
                if args.initial_alpha is not None:
                    continue
            elif settings[name]["mode"] != "full" or "mujoco_lora" in payload:
                continue
            optimizer.load_state_dict(state)
            # Adam state includes the OLD LR. Keep the requested fine-tuning LR.
            for group in optimizer.param_groups:
                group["lr"] = getattr(args, f"{name}_learning_rate")
            restored.append(name)
    print(f"[INFO] Transfer: alpha={algorithm.log_alpha.exp().item():.8g}; restored Adam moments: {restored}.")


def _interaction_schedule(args, env_cfg: dict) -> dict:
    """Resolve the interaction budget and the fixed update period.

    The budget is counted in environment interactions rather than episodes, so an early
    termination costs the run one short episode instead of a whole iteration.
    """
    if args.utd is not None and args.updates_per_iteration is not None:
        raise ValueError("--utd and --updates-per-iteration set the same quantity; pass only one.")
    for name in (
        "num_envs", "batch_size", "n_steps", "actor_update_every", "rollout_steps",
        "max_env_interactions", "save_interval", "log_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1.")
    duration = float(env_cfg["episode_length_s"])
    if not math.isfinite(duration) or duration < mjx_env.STEP_DT:
        raise ValueError("env.episode_length_s must be finite and at least one control step (0.02 s).")
    if args.replay_buffer_size // args.num_envs < args.n_steps:
        raise ValueError("--replay-buffer-size must hold at least --n-steps per environment.")
    transitions = args.num_envs * args.rollout_steps
    utd = 1.25 if args.utd is None else args.utd
    if not math.isfinite(utd) or utd <= 0:
        raise ValueError("--utd must be finite and positive.")
    if args.updates_per_iteration is not None and args.updates_per_iteration < 1:
        raise ValueError("--updates-per-iteration must be at least 1.")
    warmup = args.transitions_before_updates
    if args.start_training is not None:
        if args.start_training < 0:
            raise ValueError("--start-training must be non-negative.")
        warmup = args.start_training * transitions
    if warmup < 0:
        raise ValueError("--num-transitions-before-weight-updates must be non-negative.")
    updates = args.updates_per_iteration
    return {
        "max_env_interactions": args.max_env_interactions,
        "max_iterations": math.ceil(args.max_env_interactions / transitions),
        "rollout_steps": args.rollout_steps,
        "episode_steps": round(duration / mjx_env.STEP_DT),
        "updates_per_iteration": updates if updates is not None else math.floor(utd * transitions),
        "fixed_updates": updates,
        "utd": utd if updates is None else updates / transitions,
        "transitions_before_updates": warmup,
        "transitions_per_iteration": transitions,
    }


def _report_schedule(schedule: dict, args) -> None:
    """Print the resolved budget, update period and warm-up."""
    print(
        f"[INFO] Budget: {schedule['max_env_interactions']:,} env interactions ="
        f" {schedule['max_iterations']:,} iterations of {args.num_envs} env(s)"
        f" x {schedule['rollout_steps']:,} steps."
    )
    print(
        f"[INFO] Update every {schedule['transitions_per_iteration']:,} interactions:"
        f" {schedule['updates_per_iteration']:,} gradient updates (UTD {schedule['utd']:g},"
        f" batch {args.batch_size}, actor every {args.actor_update_every} critic updates)."
    )
    print(
        f"[INFO] Warm start: first update once {schedule['transitions_before_updates']:,}"
        f" new transitions exist. Episodes end at most every {schedule['episode_steps']:,}"
        f" steps and do not interrupt collection."
    )
    if args.start_training is not None:
        print("[WARN] --start-training is deprecated; use --num-transitions-before-weight-updates.")


def _validate_offline_arguments(args) -> None:
    """Reject a mixing schedule that has no retained data to apply it to."""
    mixing_args = (args.offline_fraction, args.offline_fraction_final, args.offline_anneal_iterations)
    if args.offline_replay_buffer is None and any(value is not None for value in mixing_args):
        raise ValueError(
            "--offline-fraction, --offline-fraction-final and --offline-anneal-iterations only take"
            " effect together with --offline-replay-buffer."
        )


def _resolved_offline_replay_config(args, schedule: dict) -> dict:
    """Return the effective retained-replay settings recorded for this run."""
    if args.offline_replay_buffer is None:
        return {
            "offline_replay_buffer": None,
            "offline_fraction": None,
            "offline_fraction_final": None,
            "offline_anneal_iterations": None,
        }
    return {
        "offline_replay_buffer": str(Path(args.offline_replay_buffer).expanduser().resolve()),
        "offline_fraction": 0.5 if args.offline_fraction is None else args.offline_fraction,
        "offline_fraction_final": 0.0 if args.offline_fraction_final is None else args.offline_fraction_final,
        "offline_anneal_iterations": (
            max(schedule["max_iterations"] // 2, 1)
            if args.offline_anneal_iterations is None
            else args.offline_anneal_iterations
        ),
    }


def _install_retained_replay(runner, args, schedule: dict) -> None:
    """Load pretraining transitions and mix them into every mini-batch.

    This is the retained-replay step of arXiv:2602.20220. The recorded transitions anchor the
    critic while the policy meets MJX dynamics, and their share is annealed to
    ``--offline-fraction-final`` so the final policy is fitted on MJX data.
    """
    offline_config = _resolved_offline_replay_config(args, schedule)
    path = offline_config["offline_replay_buffer"]
    initial = offline_config["offline_fraction"]
    final = offline_config["offline_fraction_final"]
    anneal = offline_config["offline_anneal_iterations"]
    offline = ReplayBuffer.load_snapshot(path, args.device, n_steps=args.n_steps, gamma=args.gamma)

    runner.alg.replay_buffer = MixedReplayBuffer(
        runner.alg.replay_buffer,
        offline,
        initial_offline_fraction=initial,
        final_offline_fraction=final,
        anneal_iterations=anneal,
    )
    print(
        f"[INFO] Retained replay: {offline.num_envs * offline.num_transitions:,} transitions from {path}; "
        f"offline share {initial:g} -> {final:g} over {anneal} iterations."
    )


def _install_best_weights_hook(runner, source_checkpoint: str | None) -> None:
    """Overwrite ``best_weights.pt`` whenever the rolling episode return improves.

    ``Train/mean_reward`` is the full return averaged over ``episode_log_window``
    completed episodes. It is less noisy than the current log batch's
    ``Episodes/return`` and, unlike a per-step reward, includes survival duration.
    """
    original_log = runner.logger.log
    best_mean_reward = float("-inf")

    def _log_with_best_weights(*log_args, **log_kwargs):
        nonlocal best_mean_reward
        original_log(*log_args, **log_kwargs)

        logger = runner.logger
        if logger.log_dir is None or logger.writer is None or not logger.rewbuffer:
            return
        mean_reward = float(statistics.mean(logger.rewbuffer))
        if not math.isfinite(mean_reward) or mean_reward <= best_mean_reward:
            return
        best_mean_reward = mean_reward
        iteration = log_kwargs.get("it", log_args[0] if log_args else None)
        infos = {
            "best_model_metric": "Train/mean_reward",
            "best_model_value": mean_reward,
            "best_model_iteration": iteration,
            "best_model_total_timesteps": logger.tot_timesteps,
            "best_model_total_time": logger.tot_time,
            "source_checkpoint": source_checkpoint,
        }
        path = os.path.join(logger.log_dir, "best_weights.pt")
        runner.save(path, infos=infos)
        print(
            f"[INFO] Saved new best weights to {path} "
            f"(iteration={iteration}, Train/mean_reward={mean_reward:.4f}).",
            flush=True,
        )

    runner.logger.log = _log_with_best_weights


def _physical_gpu_id(local_index: int) -> str:
    """Translate a process-local CUDA index into the GPU number the node uses.

    Slurm usually renumbers ``CUDA_VISIBLE_DEVICES`` from 0 inside the job, so the Slurm
    variables are the reliable source for the node-level GPU number reported by nvidia-smi.
    """
    for env_var in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
        visible = [entry.strip() for entry in os.environ.get(env_var, "").split(",") if entry.strip()]
        if len(visible) > local_index:
            return visible[local_index]
    return str(local_index)


def _runtime_placement(device: str) -> dict:
    """Describe where this run executes: process, Slurm job, host and physical GPU.

    Several runs share one cluster node, so these fields are what lets a W&B run be tied back
    to the exact process and card it used. Same keys as source/scripts/rsl_rl/train.py.
    """
    placement = {
        "pid": os.getpid(),
        "hostname": os.environ.get("SLURMD_NODENAME") or platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        local_index = torch.cuda.current_device() if torch_device.index is None else torch_device.index
        properties = torch.cuda.get_device_properties(local_index)
        placement["gpu_id"] = _physical_gpu_id(local_index)
        placement["gpu_name"] = properties.name
        # Unique per physical card, so it still identifies the GPU when index numbering is ambiguous.
        placement["gpu_uuid"] = str(getattr(properties, "uuid", ""))
    return placement


def _runner_config(args, schedule: dict) -> dict:
    algorithm = {
        "class_name": "SAC",
        "replay_buffer_size": args.replay_buffer_size,
        "num_learning_epochs": 1,
        "num_mini_batches": schedule["updates_per_iteration"],
        "mini_batch_size": args.batch_size,
        "actor_learning_rate": args.actor_learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        "alpha_learning_rate": args.alpha_learning_rate,
        "actor_optimizer": "adam",
        "critic_optimizer": "adam",
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": 0.001 if args.initial_alpha is None else args.initial_alpha,
        "auto_alpha": not args.freeze_alpha,
        "target_entropy_scale": args.target_entropy_scale,
        "max_grad_norm": 1.0,
        "policy_frequency": args.actor_update_every,
        "n_steps": args.n_steps,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    }
    if args.symmetry_mode != "none":
        algorithm["symmetry_cfg"] = {
            "use_data_augmentation": args.symmetry_mode in ("augmentation", "both"),
            "use_mirror_loss": args.symmetry_mode in ("loss", "both"),
            "mirror_loss_coeff": args.symmetry_loss_coeff,
            "data_augmentation_func": solo12_symmetry.compute_symmetric_observations_actions,
        }
    return {
        "class_name": "OffPolicyRunner",
        "finetuning": _finetuning_config(args),
        "seed": args.seed,
        "device": args.device,
        "num_steps_per_env": schedule["rollout_steps"],
        "max_env_interactions": schedule["max_env_interactions"],
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "episode_log_window": args.episode_log_window,
        "update_schedule": schedule,
        "offline_replay": _resolved_offline_replay_config(args, schedule),
        "save_replay_buffer": args.save_replay_buffer,
        "save_replay_buffer_every": args.save_replay_buffer_every,
        "experiment_name": "solo12_mujoco_sac",
        "run_name": args.run_name,
        "logger": "tensorboard" if args.no_wandb else "wandb",
        "wandb_project": args.wandb_project,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "actor": {
            "class_name": "SACActorModel",
            "hidden_dims": [512, 256, 128],
            "activation": "swish",
            "obs_normalization": True,
            "layer_norm": False,
            "init_noise_std": args.initial_std,
            "log_std_min": -20.0,
            "log_std_max": 2.0,
        },
        "critic": {
            "class_name": "SACCriticModel",
            "hidden_dims": [512, 256, 128],
            "activation": "swish",
            "obs_normalization": True,
            "layer_norm": False,
        },
        "algorithm": algorithm,
    }


def _configure_checkpoint_models(cfg, args) -> None:
    """Match checkpoint tensor shapes and recover non-tensor settings from its sidecar."""
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if args.initial_alpha is None:
            if payload.get("log_alpha") is not None:
                cfg["algorithm"]["alpha"] = payload["log_alpha"].exp().item()
            elif payload.get("alpha") is not None:
                cfg["algorithm"]["alpha"] = payload["alpha"]
        saved_cfg = {}
        if (path.parent / "run_config.json").is_file():
            saved_cfg = json.loads((path.parent / "run_config.json").read_text())["agent"]
        elif (path.parent / "params" / "agent.yaml").is_file():
            import yaml

            saved_cfg = yaml.safe_load((path.parent / "params" / "agent.yaml").read_text())
        else:
            print(
                "[INFO] No checkpoint config sidecar: assuming swish activations and two_hot "
                "labels for categorical critics. Override --actor-activation, --critic-activation, "
                "or --critic-loss if needed."
            )
        for name, prefix in (("actor", "mlp."), ("critic", "critic1.")):
            state = payload[f"{name}_state_dict"]
            weights = sorted(
                ((int(k.split(".")[1]), v) for k, v in state.items()
                 if k.startswith(prefix) and k.endswith(".weight")),
                key=lambda item: item[0],
            )
            linear = [v for _, v in weights if v.ndim == 2]
            cfg[name].update(
                hidden_dims=[v.shape[0] for v in linear[:-1]],
                layer_norm=any(v.ndim == 1 for _, v in weights),
                obs_normalization=any(k.startswith("obs_normalizer.") for k in state),
            )
            # Shapes cannot reveal activation functions or the categorical label scheme.
            for key in ("activation", "log_std_min", "log_std_max", "distributional_loss", "hl_gauss_sigma_ratio"):
                if key in saved_cfg.get(name, {}):
                    cfg[name][key] = saved_cfg[name][key]
        cfg["actor"]["state_dependent_std"] = "log_std" not in payload["actor_state_dict"]
        support = payload["critic_state_dict"].get("value_support")
        if support is not None:
            if cfg["critic"].get("distributional_loss", "mse") == "mse":
                cfg["critic"]["distributional_loss"] = "two_hot"
            cfg["critic"]["distributional_num_bins"] = support.numel()
            cfg["critic"]["distributional_symlog_limit"] = support[-1].log1p().item()
        else:
            cfg["critic"]["distributional_loss"] = "mse"
        if args.critic_loss is not None and (args.critic_loss == "mse") != (support is None):
            raise ValueError("--critic-loss must match the checkpoint's scalar/categorical head type.")
    for name in ("actor", "critic"):
        if getattr(args, f"{name}_activation") is not None:
            cfg[name]["activation"] = getattr(args, f"{name}_activation")
    if args.critic_loss is not None:
        cfg["critic"]["distributional_loss"] = args.critic_loss
    if args.hl_gauss_sigma_ratio is not None:
        cfg["critic"]["hl_gauss_sigma_ratio"] = args.hl_gauss_sigma_ratio


def _source_env_cfg(args) -> dict | None:
    """Start from the Isaac run that trained the checkpoint instead of the MJX defaults."""
    if args.source_env_cfg is None:
        if args.curriculum_stage is not None:
            raise ValueError("--curriculum-stage only applies together with --source-env-cfg.")
        return None
    if args.source_env_cfg == "auto":
        if args.checkpoint is None:
            raise ValueError("--source-env-cfg needs a path when there is no --checkpoint to locate it from.")
        path = Path(args.checkpoint).expanduser().resolve().parent / "params" / "env.yaml"
    else:
        path = Path(args.source_env_cfg).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No Isaac environment config at {path}.")
    overrides, notes = mjx_env.source_env_overrides(path, args.curriculum_stage)
    stage = "last" if args.curriculum_stage is None else args.curriculum_stage
    print(f"[INFO] Environment settings copied from {path} at curriculum stage {stage}:")
    for name, value in sorted(overrides.items()):
        marker = " " if value == mjx_env.DEFAULT_ENV[name] else "*"
        print(f"  {marker} {name} = {value!r}")
    for note in notes:
        print(f"[WARN] {note}")
    return {**mjx_env.DEFAULT_ENV, **overrides}


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if args.task != "solo12-two-feet":
        raise ValueError("mujoco/train_sac.py currently supports --task=solo12-two-feet only.")
    env_cfg, unsupported = mjx_env.parse_env_overrides(unknown, _source_env_cfg(args))
    if unsupported:
        raise ValueError("Unsupported arguments/overrides: " + " ".join(unsupported))
    if any(abs(x) > 1e-9 for x in env_cfg["forces_applied_to_base_curriculum"]):
        raise ValueError("MJX SAC currently supports zero external pushes only.")
    if any(abs(x) > 1e-9 for x in env_cfg["base_push_force_z_range"]):
        raise ValueError("MJX SAC currently supports zero external pushes only.")
    if env_cfg["include_events_randomization"]:
        raise ValueError("MJX startup property randomization is not implemented.")
    _validate_offline_arguments(args)
    if args.initial_alpha is not None and (not math.isfinite(args.initial_alpha) or args.initial_alpha <= 0):
        raise ValueError("--initial-alpha must be finite and positive.")
    if args.resume and (args.reset_optimizers or args.initial_alpha is not None or args.freeze_alpha):
        raise ValueError("Temperature/optimizer overrides require fine-tuning without --resume.")
    _finetuning_config(args)
    schedule = _interaction_schedule(args, env_cfg)
    _report_schedule(schedule, args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = _runner_config(args, schedule)
    _configure_checkpoint_models(cfg, args)
    command = reproducible_command()
    cfg["command"] = command
    cfg["run_placement"] = _runtime_placement(args.device)
    print(f"[INFO] Run placement: {cfg['run_placement']}")

    # The released constructor calls SAC._compute_action_scaling explicitly.
    SAC._compute_action_scaling = staticmethod(_mjx_action_scaling)
    env = MjxSolo12VecEnv(env_cfg, args.num_envs, args.seed, args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = None
    if not args.no_wandb:
        import wandb

        run_id = wandb.util.generate_id()
        os.environ["WANDB_RUN_ID"] = run_id
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["BORINOT_WANDB_NAME"] = args.run_name
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
    suffix = f"_{run_id}" if run_id else ""
    log_dir = args.output_dir / f"{timestamp}_{args.run_name.replace('/', '_')}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "run_config.json").write_text(
        json.dumps({"command": command, "args": vars(args), "env": env_cfg, "agent": cfg}, indent=2, default=str)
        + "\n"
    )

    runner = OffPolicyRunner(env, cfg, log_dir=str(log_dir), device=args.device)
    if args.resume:
        _apply_finetuning(runner, args)
    if args.checkpoint:
        checkpoint = str(Path(args.checkpoint).expanduser().resolve())
        if args.resume:
            runner.load(checkpoint)
        else:
            # Load networks first, then adapt/freeze layers and restore compatible
            # optimizer state. Fine-tuning starts new iteration and interaction counters.
            runner.load(
                checkpoint,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            )
        _report_checkpoint_action_scaling(runner.alg.actor, env)
        print(f"[INFO] Loaded SAC checkpoint: {checkpoint} (exact resume={args.resume})")
    if not args.resume:
        _apply_finetuning(runner, args)
        if args.checkpoint:
            _restore_finetuning_state(
                runner.alg, torch.load(checkpoint, map_location=args.device, weights_only=False), args
            )
    if args.offline_replay_buffer:
        _install_retained_replay(runner, args, schedule)
    source_checkpoint = str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None
    _install_best_weights_hook(runner, source_checkpoint)
    print(
        f"[INFO] Best-weight tracking: {log_dir / 'best_weights.pt'} from Train/mean_reward "
        f"(last {args.episode_log_window} completed episodes)."
    )
    runner.learn(num_learning_iterations=schedule["max_iterations"], init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
