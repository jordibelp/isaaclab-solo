#!/usr/bin/env python3
"""Parallel MJX SAC fine-tuning for Solo12 (arXiv:2605.24975).

This uses the same RSL-RL-SAC actor, critic, replay buffer, checkpoints, symmetry
augmentation, timeout handling, and n-step targets as Isaac training. MJX only
replaces the vectorized simulator.
"""

from __future__ import annotations

import argparse
import json
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
from rsl_rl_sac.runners import OffPolicyRunner


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
    p.add_argument("--start-training", type=int, default=1)
    p.add_argument("--replay-buffer-size", type=int, default=int(5.0e6))
    p.add_argument("--updates-per-iteration", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--n-steps", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--tau", type=float, default=0.003)
    p.add_argument("--actor-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--critic-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--alpha-learning-rate", type=float, default=2.0e-5)
    p.add_argument("--initial-alpha", type=float, default=0.001)
    p.add_argument("--target-entropy-scale", type=float, default=0.167)
    p.add_argument("--initial-std", type=float, default=0.15)
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
        "policy_frequency": 1,
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
        "seed": args.seed,
        "device": args.device,
        "num_steps_per_env": args.rollout_steps,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "start_training": args.start_training,
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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = _runner_config(args)
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
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
