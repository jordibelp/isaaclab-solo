#!/usr/bin/env python3
"""Parallel MJX SAC fine-tuning for Solo12 (arXiv:2605.24975).

This uses the same RSL-RL-SAC actor, critic, replay buffer, checkpoints, symmetry
augmentation, timeout handling, and n-step targets as Isaac training. MJX only
replaces the vectorized simulator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
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
    p.add_argument("--run-name", default="[mujoco] Solo12 SAC")
    p.add_argument("--num_envs", "--num-envs", type=int, default=256)
    p.add_argument("--max-iterations", type=int, default=1500)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument(
        "--start-training",
        type=int,
        default=1,
        help=(
            "Iterations of pure data collection with the loaded policy before the first update."
            " This is the warm start of arXiv:2602.20220; one iteration already collects"
            " num_envs * rollout_steps transitions."
        ),
    )
    p.add_argument("--replay-buffer-size", type=int, default=int(5.0e6))
    p.add_argument("--updates-per-iteration", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
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
        help="Iterations over which the offline share is annealed. Default: half of --max-iterations.",
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
    p.add_argument("--initial-alpha", type=float, default=0.001)
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
        self.cfg = SimpleNamespace(
            is_finite_horizon=False,
            episode_length_s=float(env_cfg["episode_length_s"]),
            action_scale=float(mjx_env.ACTION_SCALE),
        )
        self.max_episode_length = max(1, round(self.cfg.episode_length_s / mjx_env.STEP_DT))
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.unwrapped = self
        self.action_lower_magnitude, self.action_upper_magnitude = self._action_bounds()
        model_info = mjx_env.build_model(
            Path(__file__).with_name("solo12.xml"), float(env_cfg["kp"]), float(env_cfg["kd"])
        )
        self._reset_fn, self._step_fn = mjx_env.make_training_functions(model_info, env_cfg, self.num_envs)
        self._key = jax.random.PRNGKey(seed)
        self._key, reset_key = jax.random.split(self._key)
        self._state = self._reset_fn(reset_key)

    def _action_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The MJX XML uses the same Solo12 soft joint ranges and q=0 action centre
        # as the direct Isaac environment. Read them instead of hard-coding values.
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("solo12.xml")))
        ranges = []
        for name in mjx_env.JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ranges.append(model.jnt_range[jid].copy())
        ranges = np.asarray(ranges)
        lower = torch.as_tensor(-ranges[:, 0] / mjx_env.ACTION_SCALE, dtype=torch.float32, device=self.device)
        upper = torch.as_tensor(ranges[:, 1] / mjx_env.ACTION_SCALE, dtype=torch.float32, device=self.device)
        if torch.any(lower <= 0) or torch.any(upper <= 0):
            raise ValueError("MJX SAC requires q=0 to lie inside every joint range.")
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
            _reward_terms,
            dones,
            terminated,
            _termination_causes,
            final_obs,
            _episode_returns,
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
        extras = {
            "time_outs": timeout_t,
            "time_outs_obs": TensorDict(
                {"policy": final_obs_t}, batch_size=[self.num_envs], device=self.device
            ),
            "log": {},
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


def _apply_environment_action_scaling(actor, env) -> None:
    """Keep target-simulator bounds instead of checkpoint-serialized source bounds."""

    upper, lower = _mjx_action_scaling(env, str(actor.action_bias.device))
    lower_signed = -lower
    with torch.no_grad():
        actor.action_bias.copy_(0.5 * (upper + lower_signed))
        actor.action_range.copy_(0.5 * (upper - lower_signed))
        actor.log_action_range.copy_(torch.log(actor.action_range).sum())


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


def _validate_offline_arguments(args) -> None:
    """Reject a mixing schedule that has no retained data to apply it to."""
    mixing_args = (args.offline_fraction, args.offline_fraction_final, args.offline_anneal_iterations)
    if args.offline_replay_buffer is None and any(value is not None for value in mixing_args):
        raise ValueError(
            "--offline-fraction, --offline-fraction-final and --offline-anneal-iterations only take"
            " effect together with --offline-replay-buffer."
        )


def _install_retained_replay(runner, args) -> None:
    """Load pretraining transitions and mix them into every mini-batch.

    This is the retained-replay step of arXiv:2602.20220. The recorded transitions anchor the
    critic while the policy meets MJX dynamics, and their share is annealed to
    ``--offline-fraction-final`` so the final policy is fitted on MJX data.
    """
    path = str(Path(args.offline_replay_buffer).expanduser().resolve())
    offline = ReplayBuffer.load_snapshot(path, args.device, n_steps=args.n_steps, gamma=args.gamma)

    initial = 0.5 if args.offline_fraction is None else args.offline_fraction
    final = 0.0 if args.offline_fraction_final is None else args.offline_fraction_final
    anneal = (
        max(args.max_iterations // 2, 1)
        if args.offline_anneal_iterations is None
        else args.offline_anneal_iterations
    )

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


def _runner_config(args) -> dict:
    algorithm = {
        "class_name": "SAC",
        "replay_buffer_size": args.replay_buffer_size,
        "num_learning_epochs": 1,
        "num_mini_batches": args.updates_per_iteration,
        "mini_batch_size": args.batch_size,
        "actor_learning_rate": args.actor_learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        "alpha_learning_rate": args.alpha_learning_rate,
        "actor_optimizer": "adam",
        "critic_optimizer": "adam",
        "gamma": args.gamma,
        "tau": args.tau,
        "alpha": args.initial_alpha,
        "auto_alpha": True,
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
        "num_steps_per_env": args.rollout_steps,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "start_training": args.start_training,
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


def main() -> None:
    args, unknown = build_parser().parse_known_args()
    if args.task != "solo12-two-feet":
        raise ValueError("mujoco/train_sac.py currently supports --task=solo12-two-feet only.")
    env_cfg, unsupported = mjx_env.parse_env_overrides(unknown)
    if unsupported:
        raise ValueError("Unsupported arguments/overrides: " + " ".join(unsupported))
    if any(abs(x) > 1e-9 for x in env_cfg["forces_applied_to_base_curriculum"]):
        raise ValueError("MJX SAC currently supports zero external pushes only.")
    if any(abs(x) > 1e-9 for x in env_cfg["base_push_force_z_range"]):
        raise ValueError("MJX SAC currently supports zero external pushes only.")
    if env_cfg["include_events_randomization"]:
        raise ValueError("MJX startup property randomization is not implemented.")
    _validate_offline_arguments(args)
    _finetuning_config(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = _runner_config(args)
    _configure_checkpoint_models(cfg, args)
    command = reproducible_command()
    cfg["command"] = command

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
            # Sim-to-MJX fine-tuning deliberately starts fresh optimizers, alpha,
            # iteration count, and replay data while transferring actor and critics.
            runner.load(
                checkpoint,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            )
        _apply_environment_action_scaling(runner.alg.actor, env)
        print(f"[INFO] Loaded SAC checkpoint: {checkpoint} (exact resume={args.resume})")
    if not args.resume:
        _apply_finetuning(runner, args)
    if args.offline_replay_buffer:
        _install_retained_replay(runner, args)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
