#!/usr/bin/env python3
"""Fine-tune Solo12 RSL-RL actor/critic networks with LoRA or full PPO in MJX.

The target dynamics are the same MuJoCo model used by ``play_direct_mujoco.py``.
MJX batches the physics over ``--num_envs`` on a GPU (or CPU fallback).  The
By default, pre-trained dense weights and biases remain frozen while PPO updates
low-rank adapters in both actor and critic plus the actor exploration log
standard deviation. Rank zero either fully fine-tunes the dense networks or,
when the base remains frozen, runs a no-learning baseline.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import mujoco
import numpy as np
import torch

# Avoid JAX preallocating essentially all VRAM before PyTorch reads the checkpoint.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from mujoco import mjx

from lora_ppo_cfg import DEFAULT_AGENT_CFG


PHYSICS_DT = 1.0 / 200.0
DECIMATION = 4
STEP_DT = PHYSICS_DT * DECIMATION
ACTION_SCALE = 0.25
EFFORT_LIMIT = 2.65
OBS_NORM_EPS = 1.0e-2
SAFE_Q = np.array((0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8), np.float32)
JOINT_NAMES = tuple(
    f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")
)
LAYER_CHOICES = ("all", "input_and_output", "input", "output")
LR_PERM = jnp.array((3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8))
LR_SIGN = jnp.array((-1.0, 1.0, 1.0) * 4)
# exp() of the PPO ratio overflows to inf well before this; clamping keeps a diverged
# update finite so the NaN guard can reject it instead of poisoning every parameter.
LOG_RATIO_LIMIT = 10.0

REWARD_TERM_NAMES = (
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "action_rate_l2",
    "dof_torques_l2",
    "three_or_more_feet_contact",
    "two_feet_above_height",
    "undesired_contacts",
    "foot_contact",
)
REWARD_SCALE_KEYS = (
    "track_lin_vel_xy_reward_scale",
    "track_ang_vel_z_reward_scale",
    "action_rate_reward_scale",
    "joint_torque_reward_scale",
    "three_or_more_feet_contact_penalty_reward_scale",
    "two_feet_above_height_reward_scale",
    "undesired_contact_reward_scale",
    "foot_contact_reward_scale",
)


DEFAULT_ENV = {
    "kp": 9.0,
    "kd": 0.2,
    "episode_length_s": 20.0,
    "command_lin_vel_x_range": (-0.5, 0.5),
    "command_lin_vel_y_range": (-0.3, 0.3),
    "command_ang_vel_z_range": (-0.5, 0.5),
    "command_resampling_time_s": 10.0,
    "standing_env_prob": 0.02,
    "opposite_direction_cmd_prob": 0.05,
    "reset_x_pos": 0.5,
    "reset_y_pos": 0.5,
    "reset_base_lin_vel_range": (-0.3, 0.3),
    "reset_base_ang_vel_range": (-0.1, 0.1),
    "joint_pos_noise_range": (-0.07, 0.07),
    "actuation_delay_range": (0, 3),
    "enable_observation_corruption": True,
    "base_lin_vel_noise": (-0.1, 0.1),
    "base_ang_vel_noise": (-0.2, 0.2),
    "projected_gravity_noise": (-0.05, 0.05),
    "joint_pos_noise": (-0.01, 0.01),
    "joint_vel_noise": (-1.5, 1.5),
    "tracking_std": 0.5,
    "track_lin_vel_xy_reward_scale": 1.6,
    "track_ang_vel_z_reward_scale": 0.5,
    "action_rate_reward_scale": -0.05,
    "joint_torque_reward_scale": -0.5e-3,
    "two_feet_above_height_reward_scale": 1.3,
    "two_feet_above_height_threshold": 0.45,
    "two_feet_above_height_alpha": 25.0,
    "three_or_more_feet_contact_penalty_reward_scale": -100.0,
    "undesired_contact_reward_scale": -2.25,
    "foot_contact_reward_scale": -1.0e-3,
    "front_back_asymetry": True,
    "rear_feet_in_contact_for_twofeet": False,
    "finish_on_front_feet_contact": False,
    "finish_on_front_feet_contact_after": 1.3,
    "include_events_randomization": False,
    "forces_applied_to_base_curriculum": (0.0,),
    "base_push_force_z_range": (0.0, 0.0),
}


def _literal(raw: str):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        lowered = raw.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return raw


def _boolean(raw: str | bool) -> bool:
    if isinstance(raw, bool):
        return raw
    value = raw.lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {raw!r}")


def parse_env_overrides(tokens: list[str]) -> tuple[dict, list[str]]:
    cfg = copy.deepcopy(DEFAULT_ENV)
    unsupported = []
    aliases = {"max_velx_range_curriculum": None, "curriculum_two_feet": None, "initial_position": None,
               "tricky_terrain": None, "track_base_height_reward_scale": None, "enabled_self_collisions": None,
               "base_filtered_pairs": None, "extra_mass_on_front_feet": None}
    for token in tokens:
        key, sep, raw = token.partition("=")
        if not sep or not key.startswith("env."):
            unsupported.append(token)
            continue
        name = key[4:]
        value = _literal(raw)
        if name in cfg:
            if isinstance(cfg[name], tuple):
                value = tuple(value)
            cfg[name] = value
        elif name in aliases:
            # Accepted compatibility settings: flat terrain/model XML already encode them.
            continue
        else:
            unsupported.append(token)
    return cfg, unsupported


def build_parser() -> argparse.ArgumentParser:
    agent = DEFAULT_AGENT_CFG
    policy = agent.policy
    algorithm = agent.algorithm
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="solo12-two-feet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--trainable_layers", "--trainable-layers", choices=LAYER_CHOICES, default=policy.trainable_layers)
    p.add_argument("--rank", type=int, default=policy.rank, help="LoRA rank; 0 disables adapters for a full/frozen baseline.")
    p.add_argument(
        "--frozen-base-weights",
        type=_boolean,
        nargs="?",
        const=True,
        default=policy.frozen_base_weights,
        help="Freeze pretrained actor/critic weights and biases (default: true). Use false with --rank=0 for full fine-tuning.",
    )
    p.add_argument("--lora-alpha", type=float, default=policy.lora_alpha, help="Adapter gain; the applied scale is alpha/rank.")
    p.add_argument(
        "--constrain-delta-lora-degrees",
        type=float,
        default=policy.constrain_delta_lora_degrees,
        help="Per-joint bound in degrees on the LoRA-induced position-target correction; 0 disables it.",
    )
    p.add_argument(
        "--clipping-includes-exploratory-noisy",
        type=_boolean,
        nargs="?",
        const=True,
        default=policy.clipping_includes_exploratory_noisy,
        help="Include Gaussian exploration in the correction clipped around the frozen policy mean.",
    )
    p.add_argument("--num_envs", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=agent.max_iterations)
    p.add_argument("--rollout-steps", type=int, default=agent.num_steps_per_env)
    p.add_argument("--learning-rate", type=float, default=algorithm.learning_rate,
                   help="Initial LR. With --lr-schedule=adaptive this is only a starting point.")
    p.add_argument("--lr-schedule", choices=("adaptive", "fixed"), default=algorithm.schedule,
                   help="'adaptive' matches the RSL-RL KL-targeting schedule used for the frozen policy.")
    p.add_argument("--desired-kl", type=float, default=algorithm.desired_kl)
    p.add_argument("--lr-range", type=float, nargs=2, default=algorithm.learning_rate_range)
    p.add_argument("--ppo-epochs", type=int, default=algorithm.num_learning_epochs)
    p.add_argument("--num-minibatches", type=int, default=algorithm.num_mini_batches)
    p.add_argument("--gamma", type=float, default=algorithm.gamma)
    p.add_argument("--gae-lambda", type=float, default=algorithm.lam)
    p.add_argument("--clip-param", type=float, default=algorithm.clip_param)
    p.add_argument("--value-loss-coef", type=float, default=algorithm.value_loss_coef)
    p.add_argument("--entropy-coef", type=float, default=algorithm.entropy_coef, help="Matches the Isaac solo12 PPO config.")
    p.add_argument("--max-grad-norm", type=float, default=algorithm.max_grad_norm)
    p.add_argument("--log-std-range", type=float, nargs=2, default=policy.log_std_range,
                   help="Hard bounds on the exploration log_std; the entropy bonus otherwise drifts it up without limit.")
    p.add_argument("--seed", type=int, default=agent.seed)
    p.add_argument("--save-interval", type=int, default=agent.save_interval)
    p.add_argument("--run-name", default="mujoco LoRA")
    p.add_argument("--symmetry-mode", choices=("none", "augmentation"), default="augmentation")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("logs/mujoco/lora_ppo"))
    p.add_argument("--wandb-project", default="solo12-two-feet-lora")
    p.add_argument("--wandb-entity")
    p.add_argument(
        "--metrics-smoothing-window",
        type=int,
        default=100,
        help="Number of PPO iterations pooled into Smoothed* metrics; 1 reproduces the unsmoothed values.",
    )
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Build everything and run one rollout/update only.")
    return p


AGENT_OVERRIDE_TO_DEST = {
    "agent.seed": "seed",
    "agent.num_steps_per_env": "rollout_steps",
    "agent.max_iterations": "max_iterations",
    "agent.save_interval": "save_interval",
    "agent.policy.trainable_layers": "trainable_layers",
    "agent.policy.rank": "rank",
    "agent.policy.frozen_base_weights": "frozen_base_weights",
    "agent.policy.lora_alpha": "lora_alpha",
    "agent.policy.constrain_delta_lora_degrees": "constrain_delta_lora_degrees",
    "agent.policy.clipping_includes_exploratory_noisy": "clipping_includes_exploratory_noisy",
    "agent.policy.log_std_range": "log_std_range",
    "agent.algorithm.learning_rate": "learning_rate",
    "agent.algorithm.schedule": "lr_schedule",
    "agent.algorithm.desired_kl": "desired_kl",
    "agent.algorithm.learning_rate_range": "lr_range",
    "agent.algorithm.num_learning_epochs": "ppo_epochs",
    "agent.algorithm.num_mini_batches": "num_minibatches",
    "agent.algorithm.gamma": "gamma",
    "agent.algorithm.lam": "gae_lambda",
    "agent.algorithm.clip_param": "clip_param",
    "agent.algorithm.value_loss_coef": "value_loss_coef",
    "agent.algorithm.entropy_coef": "entropy_coef",
    "agent.algorithm.max_grad_norm": "max_grad_norm",
}


def apply_agent_overrides(args: argparse.Namespace, tokens: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Apply Hydra-style ``agent.*=value`` tokens after ordinary argparse flags."""
    remaining = []
    for token in tokens:
        key, sep, raw = token.partition("=")
        if not sep or not key.startswith("agent."):
            remaining.append(token)
            continue
        if key not in AGENT_OVERRIDE_TO_DEST:
            raise ValueError(f"Unsupported agent override: {token}")
        dest = AGENT_OVERRIDE_TO_DEST[key]
        value = _literal(raw)
        current = getattr(args, dest)
        if isinstance(current, tuple):
            value = tuple(value)
        elif isinstance(current, bool):
            value = bool(value)
        elif current is not None:
            value = type(current)(value)
        setattr(args, dest, value)
    if args.trainable_layers not in LAYER_CHOICES:
        raise ValueError(f"agent.policy.trainable_layers must be one of {LAYER_CHOICES}.")
    if args.lr_schedule not in ("adaptive", "fixed"):
        raise ValueError("agent.algorithm.schedule must be 'adaptive' or 'fixed'.")
    return args, remaining


def resolved_agent_config(args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "num_steps_per_env": args.rollout_steps,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "policy": {
            "trainable_layers": args.trainable_layers,
            "rank": args.rank,
            "frozen_base_weights": args.frozen_base_weights,
            "lora_alpha": args.lora_alpha,
            "constrain_delta_lora_degrees": args.constrain_delta_lora_degrees,
            "clipping_includes_exploratory_noisy": args.clipping_includes_exploratory_noisy,
            "log_std_range": tuple(args.log_std_range),
        },
        "algorithm": {
            "learning_rate": args.learning_rate,
            "schedule": args.lr_schedule,
            "desired_kl": args.desired_kl,
            "learning_rate_range": tuple(args.lr_range),
            "num_learning_epochs": args.ppo_epochs,
            "num_mini_batches": args.num_minibatches,
            "gamma": args.gamma,
            "lam": args.gae_lambda,
            "clip_param": args.clip_param,
            "value_loss_coef": args.value_loss_coef,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
        },
    }


@dataclass(frozen=True)
class ModelInfo:
    model: mujoco.MjModel
    mjx_model: object
    joint_qpos: np.ndarray
    joint_dof: np.ndarray
    actuator_ids: np.ndarray
    base_id: int
    ground_geom_id: int
    foot_body_ids: np.ndarray
    foot_geom_ids: np.ndarray
    front_thigh_geom_ids: np.ndarray
    thigh_geom_ids: np.ndarray
    base_geom_ids: np.ndarray


def build_model(xml_path: Path, kp: float, kd: float) -> ModelInfo:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    joint_qpos = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    joint_dof = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in JOINT_NAMES])
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd
    model.actuator_forcerange[actuator_ids, :] = (-EFFORT_LIMIT, EFFORT_LIMIT)
    name_id = lambda typ, name: mujoco.mj_name2id(model, typ, name)
    ground = name_id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
    foot_geom_ids = np.array([name_id(mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot_geom") for leg in ("FL", "FR", "RL", "RR")])
    # MJX 3.3 does not implement cylinder-box collision. Keep cylindrical feet against the plane,
    # while excluding only cylinder/self-box pairs; all other authored self-collisions remain.
    model.geom_contype[ground], model.geom_conaffinity[ground] = 1, 2
    model.geom_contype[foot_geom_ids], model.geom_conaffinity[foot_geom_ids] = 2, 0
    return ModelInfo(
        model=model,
        mjx_model=mjx.put_model(model),
        joint_qpos=joint_qpos,
        joint_dof=joint_dof,
        actuator_ids=actuator_ids,
        base_id=name_id(mujoco.mjtObj.mjOBJ_BODY, "base"),
        ground_geom_id=ground,
        foot_body_ids=np.array([name_id(mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot") for leg in ("FL", "FR", "RL", "RR")]),
        foot_geom_ids=foot_geom_ids,
        front_thigh_geom_ids=np.array([name_id(mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FL_thigh_top", "FL_knee", "FR_thigh_top", "FR_knee")]),
        thigh_geom_ids=np.array([name_id(mujoco.mjtObj.mjOBJ_GEOM, n) for leg in ("FL", "FR", "RL", "RR") for n in (f"{leg}_thigh_top", f"{leg}_knee")]),
        base_geom_ids=np.array([i for i in range(model.ngeom) if model.geom_bodyid[i] == name_id(mujoco.mjtObj.mjOBJ_BODY, "base") and model.geom_contype[i] != 0]),
    )


def selected_layer_indices(count: int, mode: str) -> tuple[int, ...]:
    if mode == "all": return tuple(range(count))
    if mode == "input": return (0,)
    if mode == "output": return (count - 1,)
    return (0, count - 1)


def load_frozen_networks(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    def layers(prefix):
        keys = sorted((k for k in state if k.startswith(prefix + ".") and k.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
        return tuple((jnp.asarray(state[k].numpy()), jnp.asarray(state[k.replace(".weight", ".bias")].numpy())) for k in keys), keys
    actor, actor_keys = layers("actor")
    critic, critic_keys = layers("critic")
    norms = {
        "actor_mean": jnp.asarray(state["actor_obs_normalizer._mean"].numpy().reshape(-1)),
        "actor_std": jnp.asarray(state["actor_obs_normalizer._std"].numpy().reshape(-1)),
        "critic_mean": jnp.asarray(state["critic_obs_normalizer._mean"].numpy().reshape(-1)),
        "critic_std": jnp.asarray(state["critic_obs_normalizer._std"].numpy().reshape(-1)),
    }
    return payload, actor, critic, actor_keys, critic_keys, norms, jnp.asarray(state["log_std"].numpy())


def init_trainable(key, actor, critic, rank: int, selected: tuple[int, ...], log_std, frozen_base_weights: bool = True):
    keys = jax.random.split(key, 2 * (len(actor) + len(critic)))
    def trainable_layers(layers, offset):
        result = []
        for i, (weight, bias) in enumerate(layers):
            if not frozen_base_weights:
                result.append({"weight": weight, "bias": bias})
            elif rank > 0 and i in selected:
                a = 0.01 * jax.random.normal(keys[offset + 2*i], (rank, weight.shape[1]))
                b = jnp.zeros((weight.shape[0], rank), weight.dtype)
                result.append({"a": a, "b": b})
            elif rank > 0:
                result.append({
                    "a": jnp.zeros((0, weight.shape[1]), weight.dtype),
                    "b": jnp.zeros((weight.shape[0], 0), weight.dtype),
                })
            else:
                result.append({})
        return tuple(result)
    return {"actor": trainable_layers(actor, 0), "critic": trainable_layers(critic, 2*len(actor)), "log_std": log_std}


def dense_lora(x, frozen, adapter, scale):
    weight, bias = frozen
    if "weight" in adapter:
        return x @ adapter["weight"].T + adapter["bias"]
    residual = (x @ adapter["a"].T) @ adapter["b"].T if "a" in adapter else 0.0
    return x @ weight.T + bias + scale * residual


def network(x, frozen, adapters, scale):
    for i, layer in enumerate(frozen):
        if adapters:
            x = dense_lora(x, layer, adapters[i], scale)
        else:
            weight, bias = layer
            x = x @ weight.T + bias
        if i != len(frozen) - 1:
            x = jax.nn.elu(x)
    return x


def actor_mean(params, obs, frozen, norms, scale, max_delta_radians=0.0, clipping_includes_noise=False):
    x = (obs - norms["actor_mean"]) / (norms["actor_std"] + OBS_NORM_EPS)
    adapted = network(x, frozen, params["actor"], scale)
    if max_delta_radians <= 0.0 or clipping_includes_noise:
        return adapted
    frozen_mean = network(x, frozen, (), 0.0)
    max_delta_action = max_delta_radians / ACTION_SCALE
    return frozen_mean + jnp.clip(adapted - frozen_mean, -max_delta_action, max_delta_action)


def clip_exploratory_action(action, obs, frozen, norms, max_delta_radians):
    """Clip the combined LoRA and exploration correction around the frozen mean."""
    if max_delta_radians <= 0.0:
        return action
    x = (obs - norms["actor_mean"]) / (norms["actor_std"] + OBS_NORM_EPS)
    frozen_mean = network(x, frozen, (), 0.0)
    max_delta_action = max_delta_radians / ACTION_SCALE
    return frozen_mean + jnp.clip(action - frozen_mean, -max_delta_action, max_delta_action)


def critic_value(params, obs, frozen, norms, scale):
    x = (obs - norms["critic_mean"]) / (norms["critic_std"] + OBS_NORM_EPS)
    return network(x, frozen, params["critic"], scale)[..., 0]


def gaussian_log_prob(action, mean, log_std):
    return -0.5 * jnp.sum(((action - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + math.log(2 * math.pi), axis=-1)


def quat_matrix(q):
    w, x, y, z = jnp.moveaxis(q, -1, 0)
    return jnp.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1] + (3, 3))


class EnvState(NamedTuple):
    data: object
    obs: jax.Array
    command: jax.Array
    command_steps: jax.Array
    action: jax.Array
    previous_action: jax.Array
    target_history: jax.Array
    delay: jax.Array
    episode_steps: jax.Array
    episode_return: jax.Array
    episode_reward_sums: jax.Array


class Transition(NamedTuple):
    obs: jax.Array; action: jax.Array; log_prob: jax.Array; reward: jax.Array
    done: jax.Array; value: jax.Array; mean: jax.Array


def make_training_functions(info: ModelInfo, cfg: dict, num_envs: int):
    model = info.mjx_model
    qids, dids = jnp.asarray(info.joint_qpos), jnp.asarray(info.joint_dof)
    foot_bodies, base_geoms = jnp.asarray(info.foot_body_ids), jnp.asarray(info.base_geom_ids)
    ground_geom = jnp.asarray(info.ground_geom_id)
    safe_q = jnp.asarray(SAFE_Q)
    cpu_data = mujoco.MjData(info.model)
    template = mjx.put_data(info.model, cpu_data)
    forward_batch = jax.vmap(mjx.forward, in_axes=(None, 0))
    step_batch = jax.vmap(mjx.step, in_axes=(None, 0))
    command_interval = max(1, round(float(cfg["command_resampling_time_s"]) / STEP_DT))
    max_episode_steps = max(1, round(float(cfg["episode_length_s"]) / STEP_DT))

    def sample_commands(key, previous):
        kx, ky, kz, ks, ko = jax.random.split(key, 5)
        ranges = (cfg["command_lin_vel_x_range"], cfg["command_lin_vel_y_range"], cfg["command_ang_vel_z_range"])
        cmd = jnp.stack(tuple(jax.random.uniform(k, (num_envs,), minval=r[0], maxval=r[1]) for k, r in zip((kx,ky,kz), ranges)), -1)
        flip = (jax.random.uniform(ko, cmd.shape) < cfg["opposite_direction_cmd_prob"]) & (jnp.abs(previous) > 1e-6)
        cmd = jnp.where(flip, -previous, cmd)
        return jnp.where((jax.random.uniform(ks, (num_envs,)) < cfg["standing_env_prob"])[:, None], 0.0, cmd)

    # One membership column per reward group, so every contact query is a gather into this
    # table rather than a separate full pass over the constraint force vector.
    group_names = tuple(f"foot_{i}" for i in range(4)) + tuple(f"thigh_{i}" for i in range(4)) + ("front_thigh",)
    membership = np.zeros((info.model.ngeom, len(group_names)), bool)
    for i, geom in enumerate(info.foot_geom_ids):
        membership[geom, i] = True
    for i, group in enumerate(info.thigh_geom_ids.reshape(4, 2)):
        membership[group, 4 + i] = True
    membership[info.front_thigh_geom_ids, 8] = True
    membership = jnp.asarray(membership)
    base_membership = jnp.zeros((info.model.ngeom,), bool).at[base_geoms].set(True)

    def contact_terms(data):
        """Per-contact geom pair and |constraint force|, evaluated once per physics step."""
        impl = getattr(data, "_impl", data)  # MJX >=3.3 moved these fields behind ``_impl``.
        contact, efc_force = impl.contact, impl.efc_force
        address = jnp.broadcast_to(
            contact.efc_address, efc_force.shape[:-1] + (contact.efc_address.shape[-1],)
        )
        force = jnp.take_along_axis(efc_force, jnp.maximum(address, 0), axis=-1)
        return contact.geom, jnp.where(address >= 0, jnp.abs(force), 0.0)

    def group_forces(terms):
        """Contact force per reward group, shape (..., len(group_names))."""
        pairs, force = terms
        hit = membership[pairs[..., 0]] | membership[pairs[..., 1]]
        return jnp.sum(force[..., None] * hit, axis=-2)

    def force_against_ground(terms, geom_membership):
        pairs, force = terms
        touches = geom_membership[pairs[..., 0]] | geom_membership[pairs[..., 1]]
        ground = (pairs[..., 0] == ground_geom) | (pairs[..., 1] == ground_geom)
        return jnp.sum(jnp.where(touches & ground, force, 0.0), axis=-1)

    def kinematics(data):
        rot = quat_matrix(data.qpos[..., 3:7])
        omega_w = jnp.einsum("...ij,...j->...i", rot, data.qvel[..., 3:6])
        offset = data.xipos[..., info.base_id, :] - data.xpos[..., info.base_id, :]
        lin_w = data.qvel[..., :3] + jnp.cross(omega_w, offset)
        lin_b = jnp.einsum("...ji,...j->...i", rot, lin_w)
        ang_b = data.qvel[..., 3:6]
        gravity_b = jnp.einsum("...ji,j->...i", rot, jnp.array((0.0, 0.0, -1.0)))
        lateral = rot[..., :2, 1]
        lateral = lateral / jnp.maximum(jnp.linalg.norm(lateral, axis=-1, keepdims=True), 1e-8)
        forward = jnp.stack((lateral[..., 1], -lateral[..., 0]), -1)
        tracked = jnp.stack((jnp.sum(lin_w[..., :2]*forward, -1), jnp.sum(lin_w[..., :2]*lateral, -1)), -1)
        return lin_b, ang_b, gravity_b, lin_w, omega_w, tracked

    def observation(data, command, action, key, corrupt=True):
        lin_b, ang_b, gravity_b, _, _, _ = kinematics(data)
        q, qd = data.qpos[..., qids], data.qvel[..., dids]
        if corrupt and cfg["enable_observation_corruption"]:
            keys = jax.random.split(key, 5)
            def noise(k, shape, bounds): return jax.random.uniform(k, shape, minval=bounds[0], maxval=bounds[1])
            lin_b += noise(keys[0], lin_b.shape, cfg["base_lin_vel_noise"])
            ang_b += noise(keys[1], ang_b.shape, cfg["base_ang_vel_noise"])
            gravity_b += noise(keys[2], gravity_b.shape, cfg["projected_gravity_noise"])
            q += noise(keys[3], q.shape, cfg["joint_pos_noise"])
            qd += noise(keys[4], qd.shape, cfg["joint_vel_noise"])
        return jnp.concatenate((lin_b, ang_b, gravity_b, command, q-safe_q, qd, action), -1)

    def reset(key, old: EnvState | None = None, mask=None):
        keys = jax.random.split(key, 8)
        if mask is None: mask = jnp.ones((num_envs,), bool)
        qpos = jnp.broadcast_to(jnp.asarray(template.qpos), (num_envs, template.qpos.shape[0]))
        qvel = jnp.zeros((num_envs, template.qvel.shape[0]))
        qpos = qpos.at[:, 0].set(jax.random.uniform(keys[0], (num_envs,), minval=-cfg["reset_x_pos"], maxval=cfg["reset_x_pos"]))
        qpos = qpos.at[:, 1].set(jax.random.uniform(keys[1], (num_envs,), minval=-cfg["reset_y_pos"], maxval=cfg["reset_y_pos"]))
        qpos = qpos.at[:, 2].set(0.35)
        qpos = qpos.at[:, 3:7].set(jnp.array((0.0, 0.0, 0.0, 1.0)))
        qnoise = jax.random.uniform(keys[2], (num_envs, 12), minval=cfg["joint_pos_noise_range"][0], maxval=cfg["joint_pos_noise_range"][1])
        qpos = qpos.at[:, qids].set(safe_q + qnoise)
        qvel = qvel.at[:, :3].set(jax.random.uniform(keys[3], (num_envs,3), minval=cfg["reset_base_lin_vel_range"][0], maxval=cfg["reset_base_lin_vel_range"][1]))
        qvel = qvel.at[:, 3:6].set(jax.random.uniform(keys[4], (num_envs,3), minval=cfg["reset_base_ang_vel_range"][0], maxval=cfg["reset_base_ang_vel_range"][1]))
        base_data = jax.vmap(lambda q, v: template.replace(qpos=q, qvel=v, ctrl=safe_q))(qpos, qvel)
        new_data = forward_batch(model, base_data)
        commands = sample_commands(keys[5], jnp.zeros((num_envs,3)))
        delay = jax.random.randint(keys[6], (num_envs,), cfg["actuation_delay_range"][0], cfg["actuation_delay_range"][1]+1)
        zeros_a = jnp.zeros((num_envs,12)); history = jnp.broadcast_to(safe_q, (4,num_envs,12))
        if old is not None:
            def choose(n, o):
                shape = (num_envs,) + (1,)*(n.ndim-1)
                return jnp.where(mask.reshape(shape), n, o)
            new_data = jax.tree_util.tree_map(choose, new_data, old.data)
            commands = jnp.where(mask[:,None], commands, old.command)
            delay = jnp.where(mask, delay, old.delay)
            zeros_a = jnp.where(mask[:,None], zeros_a, old.action)
            history = jnp.where(mask[None,:,None], history, old.target_history)
        # ``reset`` also runs as the post-step merge, so every field must fall back to the old
        # value where ``mask`` is false -- including the command countdown, which otherwise
        # gets rearmed every step and disables in-episode command resampling entirely.
        countdown = jnp.full((num_envs,), command_interval)
        if old is not None:
            countdown = jnp.where(mask, countdown, old.command_steps)
        return EnvState(new_data, observation(new_data,commands,zeros_a,keys[7]), commands,
                        countdown, zeros_a, zeros_a, history, delay,
                        jnp.zeros((num_envs,),jnp.int32) if old is None else jnp.where(mask,0,old.episode_steps),
                        jnp.zeros((num_envs,)) if old is None else jnp.where(mask,0.0,old.episode_return),
                        jnp.zeros((num_envs,len(REWARD_TERM_NAMES))) if old is None else
                        jnp.where(mask[:,None],0.0,old.episode_reward_sums))

    def env_step(state: EnvState, action, key):
        kcmd, kobs, kreset = jax.random.split(key, 3)
        resample = state.command_steps <= 0
        sampled = sample_commands(kcmd, state.command)
        command = jnp.where(resample[:,None], sampled, state.command)
        command_steps = jnp.where(resample, command_interval, state.command_steps) - 1
        target = safe_q + ACTION_SCALE * action
        history = state.target_history
        data = state.data
        def physics(carry, _):
            data, hist = carry
            hist = jnp.concatenate((target[None],hist[:-1]),0)
            ctrl = jnp.take_along_axis(jnp.moveaxis(hist,0,1), state.delay[:,None,None], axis=1)[:,0]
            data = data.replace(ctrl=ctrl)
            return (step_batch(model,data),hist), None
        (data,history),_ = jax.lax.scan(physics,(data,history),None,length=DECIMATION)
        lin_b, ang_b, gravity_b, _, omega_w, tracked = kinematics(data)
        lin_err = jnp.sum((command[:,:2]-tracked)**2,-1)
        yaw_err = (command[:,2]-omega_w[:,2])**2
        track_lin = cfg["track_lin_vel_xy_reward_scale"]*jnp.exp(-lin_err/cfg["tracking_std"]**2)*STEP_DT
        track_ang = cfg["track_ang_vel_z_reward_scale"]*jnp.exp(-yaw_err/cfg["tracking_std"]**2)*STEP_DT
        action_rate = cfg["action_rate_reward_scale"]*jnp.sum((action-state.action)**2,-1)*STEP_DT
        dof_torques = cfg["joint_torque_reward_scale"]*jnp.sum(data.qfrc_actuator[:,dids]**2,-1)*STEP_DT
        terms = contact_terms(data)
        forces = group_forces(terms)
        foot_forces, thigh_forces, front_thigh_force = forces[:, :4], forces[:, 4:8], forces[:, 8]
        feet_contact = foot_forces > 1.0
        front_contact = jnp.any(feet_contact[:,:2],-1) | (front_thigh_force > 1.0)
        forbidden = front_contact if cfg["front_back_asymetry"] else jnp.sum(feet_contact,-1)>=3
        forbidden_contact = cfg["three_or_more_feet_contact_penalty_reward_scale"]*forbidden*STEP_DT
        foot_h = data.xpos[:,foot_bodies,2]
        front_h = jnp.mean(foot_h[:,:2],-1)
        kernel = jnp.where(front_h>=cfg["two_feet_above_height_threshold"],1.0,jnp.exp(-cfg["two_feet_above_height_alpha"]*(cfg["two_feet_above_height_threshold"]-front_h)**2))
        lift = kernel*(~jnp.any(feet_contact[:,:2],-1))
        if cfg["rear_feet_in_contact_for_twofeet"]: lift *= jnp.all(feet_contact[:,2:],-1)
        two_feet_height = cfg["two_feet_above_height_reward_scale"]*lift*STEP_DT
        undesired_contacts = cfg["undesired_contact_reward_scale"]*jnp.sum(thigh_forces > 0.6,axis=-1)*STEP_DT
        foot_excess = jnp.maximum(foot_forces-10.0,0.0)
        foot_contact = cfg["foot_contact_reward_scale"]*jnp.sum(foot_excess**2,axis=-1)*STEP_DT
        reward_terms = jnp.stack((track_lin,track_ang,action_rate,dof_torques,forbidden_contact,
                                  two_feet_height,undesired_contacts,foot_contact),axis=-1)
        reward = jnp.sum(reward_terms,axis=-1)
        base_hit = force_against_ground(terms,base_membership) > 1.0
        steps = state.episode_steps+1
        timeout = steps>=max_episode_steps
        diverged = ~jnp.isfinite(data.qpos).all(-1)
        terminate_front = cfg["finish_on_front_feet_contact"] & forbidden & (steps*STEP_DT>=cfg["finish_on_front_feet_contact_after"])
        terminated = base_hit | terminate_front | diverged
        done = terminated | timeout
        reward_terms = jnp.where(diverged[:,None],0.0,reward_terms)
        reward = jnp.sum(reward_terms,axis=-1)  # a diverged rollout must not poison the batch
        returns = state.episode_return+reward
        reward_sums = state.episode_reward_sums+reward_terms
        # Observation of the pre-reset state, so a timeout can be bootstrapped from the value
        # the episode actually ended at rather than from the fresh reset observation.
        final_obs = observation(data,command,action,kobs)
        finished = EnvState(data,final_obs,command,command_steps,action,state.action,history,state.delay,steps,returns,reward_sums)
        next_state = reset(kreset,finished,done)
        termination_causes = jnp.stack((base_hit,terminate_front,diverged),axis=-1)
        return next_state,reward,reward_terms,done,terminated,termination_causes,final_obs,returns,steps,reward_sums

    return jax.jit(reset), jax.jit(env_step)


def mirror_lr(obs, action):
    obs = obs.copy()
    obs = obs.at[...,0:3].multiply(jnp.array((1.,-1.,1.)))
    obs = obs.at[...,3:6].multiply(jnp.array((-1.,1.,-1.)))
    obs = obs.at[...,6:9].multiply(jnp.array((1.,-1.,1.)))
    obs = obs.at[...,9:12].multiply(jnp.array((1.,-1.,-1.)))
    for start in (12,24,36): obs = obs.at[...,start:start+12].set(obs[...,start:start+12][...,LR_PERM]*LR_SIGN)
    return obs, action[...,LR_PERM]*LR_SIGN


def zeros_like_tree(tree): return jax.tree_util.tree_map(jnp.zeros_like,tree)


def adam_update(params, grads, state, lr, max_norm):
    """Adam with a hard non-finite guard: a diverged gradient is dropped, not applied.

    Without this a single inf/NaN gradient permanently poisons the parameters and the run
    keeps burning GPU time while logging NaN forever.
    """
    step,m,v=state
    norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(grads)))
    ok=jnp.isfinite(norm)
    clip=jnp.where(ok,jnp.minimum(1.0,max_norm/(norm+1e-8)),0.0)
    grads=jax.tree_util.tree_map(lambda g:jnp.nan_to_num(g,nan=0.0,posinf=0.0,neginf=0.0)*clip,grads)
    step=step+ok.astype(step.dtype)
    m=jax.tree_util.tree_map(lambda a,g:jnp.where(ok,.9*a+.1*g,a),m,grads)
    v=jax.tree_util.tree_map(lambda a,g:jnp.where(ok,.999*a+.001*g*g,a),v,grads)
    mh=jax.tree_util.tree_map(lambda x:x/(1-.9**step),m);vh=jax.tree_util.tree_map(lambda x:x/(1-.999**step),v)
    params=jax.tree_util.tree_map(lambda p,a,b:jnp.where(ok,p-lr*a/(jnp.sqrt(b)+1e-8),p),params,mh,vh)
    return params,(step,m,v),norm,ok


def gaussian_kl(mean_old, log_std_old, mean_new, log_std_new):
    """Analytic KL(old || new) per sample, as used by the RSL-RL adaptive LR schedule."""
    var_old, var_new = jnp.exp(2*log_std_old), jnp.exp(2*log_std_new)
    return jnp.sum(log_std_new-log_std_old+(var_old+(mean_old-mean_new)**2)/(2*var_new)-0.5,axis=-1)


def augment_left_right_batch(obs, actions, old_logp, old_means, old_log_std, advantages, returns):
    """Add exact left-right transformed behavior samples to a PPO batch.

    The mirrored action was sampled by transforming the original action, so its behavior
    distribution is the transformed original Gaussian. Evaluating it under the policy at the
    mirrored observation instead would assign a likelihood from a distribution that did not
    generate the transition and would make the PPO ratio artificially start at one.
    """
    mirrored_obs, mirrored_actions = mirror_lr(obs, actions)
    _, mirrored_old_means = mirror_lr(obs, old_means)
    mirrored_old_log_std = old_log_std[..., LR_PERM]
    return (
        jnp.concatenate((obs, mirrored_obs)),
        jnp.concatenate((actions, mirrored_actions)),
        jnp.concatenate((old_logp, old_logp)),
        jnp.concatenate((old_means, mirrored_old_means)),
        jnp.concatenate((old_log_std, mirrored_old_log_std)),
        jnp.concatenate((advantages, advantages)),
        jnp.concatenate((returns, returns)),
    )


def reward_metrics(reward_terms, completed_reward_sums, completed_episodes: int, cfg: dict) -> dict[str, float]:
    """Build RSL-compatible scaled and scale-independent reward diagnostics."""
    terms = np.asarray(reward_terms)
    completed = np.asarray(completed_reward_sums)
    log = {}
    for index, (name, scale_key) in enumerate(zip(REWARD_TERM_NAMES, REWARD_SCALE_KEYS)):
        scale = float(cfg[scale_key])
        mean_contribution = float(np.mean(terms[..., index]))
        log[f"RewardsPerStep/{name}"] = mean_contribution
        if scale != 0.0:
            log[f"PerStepRewardRatio/{name}"] = mean_contribution / (scale * STEP_DT)
        if completed_episodes:
            # Match IsaacLab's Episode_Reward convention: magnitude per configured maximum
            # episode second. Keep this key for side-by-side RSL/MJX charts.
            log[f"Episode_Reward/{name}"] = abs(float(completed[index]) / completed_episodes) / float(
                cfg["episode_length_s"]
            )
    log["RewardsPerStep/cmd_tracking"] = (
        log["RewardsPerStep/track_lin_vel_xy_exp"] + log["RewardsPerStep/track_ang_vel_z_exp"]
    )
    log["RewardsPerStep/total"] = float(np.mean(np.sum(terms, axis=-1)))
    if completed_episodes:
        log["Episode_Reward/cmd_tracking"] = (
            log["Episode_Reward/track_lin_vel_xy_exp"] + log["Episode_Reward/track_ang_vel_z_exp"]
        )
    return log


class MetricSmoother:
    """Compute rolling metrics from raw rollout and completed-episode statistics."""

    def __init__(self, window: int, cfg: dict):
        if window < 1:
            raise ValueError("--metrics-smoothing-window must be at least 1")
        self.entries = deque(maxlen=window)
        self.cfg = cfg

    def update(
        self,
        rollout_metrics: dict[str, float],
        episode_return_sum: float,
        episode_steps_sum: float,
        completed_episodes: int,
        completed_reward_sums,
    ) -> dict[str, float]:
        continuous = {
            key: float(value)
            for key, value in rollout_metrics.items()
            if key == "reward/mean_step"
            or key.startswith("RewardsPerStep/")
            or key.startswith("PerStepRewardRatio/")
        }
        self.entries.append(
            (
                continuous,
                float(episode_return_sum),
                float(episode_steps_sum),
                int(completed_episodes),
                np.asarray(completed_reward_sums, dtype=np.float64),
            )
        )

        result = {
            "Smoothed/window_iterations": len(self.entries),
            "SmoothedEpisode/completed_count": sum(entry[3] for entry in self.entries),
        }
        for key in continuous:
            values = [entry[0][key] for entry in self.entries]
            prefix, name = key.split("/", 1)
            smoothed_prefix = "SmoothedReward" if prefix == "reward" else f"Smoothed{prefix}"
            result[f"{smoothed_prefix}/{name}"] = float(np.mean(values))

        episode_count = int(result["SmoothedEpisode/completed_count"])
        if episode_count:
            return_sum = sum(entry[1] for entry in self.entries)
            steps_sum = sum(entry[2] for entry in self.entries)
            reward_sums = np.sum([entry[4] for entry in self.entries], axis=0)
            result.update(
                {
                    "SmoothedEpisode/length_steps": steps_sum / episode_count,
                    "SmoothedEpisode/length_seconds": steps_sum / episode_count * STEP_DT,
                    "SmoothedEpisode_Reward/total": return_sum / episode_count,
                }
            )
            for index, name in enumerate(REWARD_TERM_NAMES):
                result[f"SmoothedEpisode_Reward/{name}"] = (
                    abs(float(reward_sums[index]) / episode_count) / float(self.cfg["episode_length_s"])
                )
            result["SmoothedEpisode_Reward/cmd_tracking"] = (
                result["SmoothedEpisode_Reward/track_lin_vel_xy_exp"]
                + result["SmoothedEpisode_Reward/track_ang_vel_z_exp"]
            )
        return result


def save_checkpoints(output: Path, iteration: int, original_payload, params, actor, critic, actor_keys, critic_keys, scale, config):
    output.mkdir(parents=True,exist_ok=True)
    adapter={"iteration":iteration,"config":config,"actor":[],"critic":[],"log_std":np.asarray(params["log_std"])}
    merged=copy.deepcopy(original_payload);state=merged["model_state_dict"]
    for name,layers,keys in (("actor",actor,actor_keys),("critic",critic,critic_keys)):
        for frozen,ad,key in zip(layers,params[name],keys):
            bias_key=key.replace(".weight", ".bias")
            if "weight" in ad:
                state[key]=torch.as_tensor(np.asarray(ad["weight"]),dtype=state[key].dtype)
                state[bias_key]=torch.as_tensor(np.asarray(ad["bias"]),dtype=state[bias_key].dtype)
                adapter[name].append({"weight":np.asarray(ad["weight"]),"bias":np.asarray(ad["bias"]),"weight_key":key})
            elif "a" in ad:
                delta=scale*np.asarray(ad["b"]@ad["a"]);state[key]=torch.as_tensor(np.asarray(frozen[0])+delta,dtype=state[key].dtype)
                adapter[name].append({"a":np.asarray(ad["a"]),"b":np.asarray(ad["b"]),"weight_key":key})
            else:
                adapter[name].append({"weight_key":key})
    state["log_std"]=torch.as_tensor(np.asarray(params["log_std"]).copy(),dtype=state["log_std"].dtype)
    merged["iter"]=iteration
    if merged.get("infos") is None:
        merged["infos"] = {}
    merged["infos"]["mujoco_lora"]={"iteration":iteration,**config}
    torch.save(adapter,output/f"adapter_{iteration}.pt");torch.save(merged,output/f"model_{iteration}.pt")
    torch.save(adapter,output/"adapter_latest.pt");torch.save(merged,output/"model_latest.pt")


def run_directory_name(timestamp, run_name, wandb_run_id=None):
    name = f"{timestamp}_{run_name.replace('/', '_')}"
    return f"{name}_{wandb_run_id}" if wandb_run_id else name


def main():
    args,unknown=build_parser().parse_known_args()
    args,unknown=apply_agent_overrides(args,unknown)
    if args.task!="solo12-two-feet": raise ValueError("Only --task=solo12-two-feet is supported.")
    if args.rank<0 or args.num_envs<1: raise ValueError("--rank must be non-negative and --num_envs must be positive.")
    if args.rank>0 and not args.frozen_base_weights:
        raise ValueError("Unfreezing base weights is supported only with --rank=0 (full fine-tuning).")
    if args.metrics_smoothing_window < 1: raise ValueError("--metrics-smoothing-window must be at least 1.")
    if not math.isfinite(args.constrain_delta_lora_degrees) or args.constrain_delta_lora_degrees < 0.0:
        raise ValueError("--constrain-delta-lora-degrees must be finite and non-negative (0 disables it).")
    cfg,unsupported=parse_env_overrides(unknown)
    if unsupported: raise ValueError("Unsupported arguments/overrides: "+" ".join(unsupported))
    if any(abs(x)>1e-9 for x in cfg["forces_applied_to_base_curriculum"]) or any(abs(x)>1e-9 for x in cfg["base_push_force_z_range"]):
        raise ValueError("MJX LoRA training currently supports zero external pushes only.")
    if cfg["include_events_randomization"]: raise ValueError("MJX startup property randomization is not implemented; use env.include_events_randomization=False.")
    if args.device!="auto": jax.config.update("jax_platform_name",args.device)
    checkpoint=Path(args.checkpoint).expanduser().resolve();xml=Path(__file__).with_name("solo12.xml")
    payload,actor,critic,akeys,ckeys,norms,log_std=load_frozen_networks(checkpoint)
    if len(actor)!=len(critic): raise ValueError("Actor and critic layer counts differ.")
    selected=selected_layer_indices(len(actor),args.trainable_layers) if args.rank > 0 else ()
    scale=args.lora_alpha/args.rank if args.rank > 0 else 0.0
    max_delta_lora_radians=math.radians(args.constrain_delta_lora_degrees)
    key=jax.random.PRNGKey(args.seed);key,kinit,kenv=jax.random.split(key,3)
    params=init_trainable(kinit,actor,critic,args.rank,selected,log_std,args.frozen_base_weights)
    model_info=build_model(xml,float(cfg["kp"]),float(cfg["kd"]));reset_fn,step_fn=make_training_functions(model_info,cfg,args.num_envs)
    state=reset_fn(kenv)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    config={**vars(args),"checkpoint":str(checkpoint),"selected_layer_indices":selected,"lora_scale":scale,"agent":resolved_agent_config(args),"env":cfg,"jax_devices":[str(x) for x in jax.devices()],"paper":"arXiv:2603.17092"}
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()}
    wandb_run=None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=args.run_name,config=config)
        except Exception as exc: print(f"[WARN] W&B disabled: {exc}",flush=True)
    wandb_run_id=wandb_run.id if wandb_run else None
    output=args.output_dir/run_directory_name(timestamp,args.run_name,wandb_run_id);output.mkdir(parents=True,exist_ok=False)
    config.update({"output_dir":str(output),"wandb_run_id":wandb_run_id})
    (output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    if wandb_run: wandb_run.config.update({"output_dir":str(output),"wandb_run_id":wandb_run_id})
    log_std_min,log_std_max=map(float,args.log_std_range);lr_min,lr_max=map(float,args.lr_range)
    optimizer=(jnp.array(0),zeros_like_tree(params),zeros_like_tree(params),jnp.asarray(args.learning_rate))

    @jax.jit
    def collect(params,state,key):
        def body(carry,_):
            state,key=carry;key,ka,ks=jax.random.split(key,3)
            mean=actor_mean(params,state.obs,actor,norms,scale,max_delta_lora_radians,args.clipping_includes_exploratory_noisy)
            action=mean+jnp.exp(params["log_std"])*jax.random.normal(ka,mean.shape)
            executed_action=(clip_exploratory_action(action,state.obs,actor,norms,max_delta_lora_radians)
                             if args.clipping_includes_exploratory_noisy else action)
            logp=gaussian_log_prob(action,mean,params["log_std"]);value=critic_value(params,state.obs,critic,norms,scale)
            (next_state,reward,reward_terms,done,terminated,termination_causes,final_obs,
             ep_return,ep_steps,ep_reward_sums)=step_fn(state,executed_action,ks)
            # Bootstrap timeouts (RSL-RL's ``bootstrap_on_time_outs``) so a truncated episode is
            # not scored as a failure; genuine terminations keep the hard cut.
            timeout=done&~terminated
            reward=reward+args.gamma*critic_value(params,final_obs,critic,norms,scale)*timeout
            stats=(done*ep_return,done*ep_steps,done,terminated,termination_causes,
                   reward_terms,done[:,None]*ep_reward_sums)
            return (next_state,key),(Transition(state.obs,action,logp,reward,done,value,mean),stats)
        return jax.lax.scan(body,(state,key),None,length=args.rollout_steps)

    def update_batch(params,opt,batch):
        obs,action,old_logp,old_mean,old_log_std,adv,returns=batch
        def loss(p):
            mean=actor_mean(p,obs,actor,norms,scale,max_delta_lora_radians,args.clipping_includes_exploratory_noisy);logp=gaussian_log_prob(action,mean,p["log_std"])
            # Clamping before exp keeps a diverged step finite so the NaN guard can reject it.
            ratio=jnp.exp(jnp.clip(logp-old_logp,-LOG_RATIO_LIMIT,LOG_RATIO_LIMIT))
            policy=-jnp.mean(jnp.minimum(ratio*adv,jnp.clip(ratio,1-args.clip_param,1+args.clip_param)*adv))
            value=critic_value(p,obs,critic,norms,scale);vloss=.5*jnp.mean((returns-value)**2)
            entropy=jnp.sum(p["log_std"]+.5*math.log(2*math.pi*math.e))
            total=policy+args.value_loss_coef*vloss-args.entropy_coef*entropy
            kl=jnp.mean(gaussian_kl(old_mean,old_log_std,mean,p["log_std"]))
            return total,(policy,vloss,entropy,jnp.mean(jnp.abs(ratio-1)>args.clip_param),kl)
        (total,metrics),grads=jax.value_and_grad(loss,has_aux=True)(params)
        if args.rank == 0 and args.frozen_base_weights:
            grads=zeros_like_tree(grads)
        step,m,v,lr=opt
        kl=metrics[-1]
        if args.lr_schedule=="adaptive":
            # RSL-RL schedule: shrink the step when the policy moves further than desired_kl.
            lr=jnp.where(kl>2.0*args.desired_kl,lr/1.5,jnp.where((kl<0.5*args.desired_kl)&(kl>0.0),lr*1.5,lr))
            lr=jnp.clip(lr,lr_min,lr_max)
        params,(step,m,v),gn,ok=adam_update(params,grads,(step,m,v),lr,args.max_grad_norm)
        params={**params,"log_std":jnp.clip(params["log_std"],log_std_min,log_std_max)}
        return params,(step,m,v,lr),(total,*metrics,gn,lr,ok.astype(jnp.float32))

    @jax.jit
    def ppo_update(params,opt,batch,key):
        """All epochs and minibatches in one dispatch; the Python loop cost dominated at low --num_envs."""
        size=batch[0].shape[0];mb=max(1,size//args.num_minibatches);usable=mb*args.num_minibatches
        def epoch(carry,ekey):
            perm=jax.random.permutation(ekey,size)[:usable].reshape(args.num_minibatches,mb)
            def minibatch(carry,idx):
                params,opt=carry
                params,opt,metrics=update_batch(params,opt,tuple(x[idx] for x in batch))
                return (params,opt),metrics
            return jax.lax.scan(minibatch,carry,perm)
        (params,opt),metrics=jax.lax.scan(epoch,(params,opt),jax.random.split(key,args.ppo_epochs))
        return params,opt,jax.tree_util.tree_map(lambda x:jnp.mean(x),metrics)

    @jax.jit
    def build_batch(params,traj,last_obs):
        bootstrap=critic_value(params,last_obs,critic,norms,scale)
        def gae_step(carry,x):
            adv,next_v=carry;reward,done,value=x
            delta=reward+args.gamma*(1-done)*next_v-value
            adv=delta+args.gamma*args.gae_lambda*(1-done)*adv
            return (adv,value),adv
        _,adv_rev=jax.lax.scan(gae_step,(jnp.zeros_like(bootstrap),bootstrap),(traj.reward[::-1],traj.done[::-1],traj.value[::-1]))
        adv=adv_rev[::-1];returns=adv+traj.value
        flat=lambda x:x.reshape((-1,)+x.shape[2:])
        obs,actions,old_logp,means=flat(traj.obs),flat(traj.action),flat(traj.log_prob),flat(traj.mean)
        adv,returns=flat(adv),flat(returns)
        log_std=jnp.broadcast_to(params["log_std"],means.shape)
        if args.symmetry_mode=="augmentation":
            obs,actions,old_logp,means,log_std,adv,returns=augment_left_right_batch(
                obs,actions,old_logp,means,log_std,adv,returns
            )
        adv=(adv-jnp.mean(adv))/(jnp.std(adv)+1e-8)
        return obs,actions,old_logp,means,log_std,adv,returns

    print(f"[INFO] MJX devices={jax.devices()} envs={args.num_envs} rank={args.rank} layers={selected} trainable={sum(x.size for x in jax.tree_util.tree_leaves(params)):,}",flush=True)
    samples=args.num_envs*args.rollout_steps
    if samples<256:
        print(f"[WARN] {samples} samples/iteration ({args.num_envs} envs x {args.rollout_steps} steps) is a very small PPO batch; "
              f"expect noisy advantages. Raise --num_envs or --rollout-steps if training is unstable.",flush=True)
    history=[];metrics_file=(output/"metrics.jsonl").open("a");rejected=0
    metric_smoother = MetricSmoother(args.metrics_smoothing_window, cfg)
    iterations=1 if args.dry_run else args.max_iterations
    for iteration in range(1,iterations+1):
        started=time.perf_counter();key,kroll,kperm=jax.random.split(key,3)
        (state,_),(traj,stats)=collect(params,state,kroll)
        batch=build_batch(params,traj,state.obs)
        params,optimizer,m=ppo_update(params,optimizer,batch,kperm)
        m=[float(x) for x in m];elapsed=time.perf_counter()-started
        ep_return,ep_steps,dones,terminations=(np.asarray(jnp.sum(x)) for x in stats[:4])
        termination_causes=np.asarray(jnp.sum(stats[4],axis=(0,1)))
        reward_terms=np.asarray(stats[5]);completed_reward_sums=np.asarray(jnp.sum(stats[6],axis=(0,1)))
        dones=int(dones);terminations=int(terminations);timeouts=dones-terminations
        rollout_envs=samples
        log={"iteration":iteration,"reward/mean_step":float(jnp.mean(traj.reward)),
             "episodes":dones,
             "Episode/count":dones,"Episode/termination_count":terminations,"Episode/timeout_count":timeouts,
             "Episode/completion_rate_per_env_step":dones/rollout_envs,
             "Episode/termination_rate_per_env_step":terminations/rollout_envs,
             "Episode/timeout_rate_per_env_step":timeouts/rollout_envs,
             "Episode_Termination/base_contact":int(termination_causes[0]),
             "Episode_Termination/forbidden_feet_contact":int(termination_causes[1]),
             "Episode_Termination/diverged":int(termination_causes[2]),
             "Episode_Termination/time_out":timeouts,
             "terminations":terminations,"timeouts":timeouts,
             "loss/total":m[0],"loss/policy":m[1],"loss/value":m[2],"policy/entropy":m[3],
             "policy/clip_fraction":m[4],"policy/kl":m[5],"grad_norm":m[6],"learning_rate":m[7],
             "updates_applied_frac":m[8],"policy/action_std":float(jnp.mean(jnp.exp(params["log_std"]))),
             "throughput_steps_s":samples/elapsed,"wall_time_s":elapsed}
        log |= reward_metrics(reward_terms,completed_reward_sums,dones,cfg)
        if dones:
            log |= {"episode/length_s":float(ep_steps/dones)*STEP_DT,
                    "Episode_Reward/total":float(ep_return/dones),
                    "Episode/length_steps":float(ep_steps/dones),"Episode/length_seconds":float(ep_steps/dones)*STEP_DT}
        log |= metric_smoother.update(log, ep_return, ep_steps, dones, completed_reward_sums)
        history.append({k:float(v) if isinstance(v,(np.floating,float)) else int(v) for k,v in log.items()})
        print("[INFO] "+" ".join(f"{k}={v:.5g}" if isinstance(v,float) else f"{k}={v}" for k,v in log.items()),flush=True)
        if wandb_run: wandb_run.log(log,step=iteration)
        metrics_file.write(json.dumps(history[-1])+"\n");metrics_file.flush()
        # The NaN guard leaves the parameters untouched when a gradient blows up, so a transient
        # spike is recoverable; only a sustained total rejection means the run is unsalvageable.
        rejected = rejected+1 if m[8]<1.0 else 0
        if m[8]<1.0:
            print(f"[WARN] {(1-m[8])*100:.0f}% of updates rejected as non-finite at iteration {iteration}.",flush=True)
        if rejected>=5:
            print("[ERROR] Gradients non-finite for 5 consecutive iterations; aborting with the last "
                  "good parameters. Lower --learning-rate or raise --num_envs/--rollout-steps.",flush=True)
            save_checkpoints(output,iteration,payload,params,actor,critic,akeys,ckeys,scale,config)
            break
        if iteration%args.save_interval==0 or iteration==iterations: save_checkpoints(output,iteration,payload,params,actor,critic,akeys,ckeys,scale,config)
    metrics_file.close()
    if wandb_run:
        import wandb
        artifact=wandb.Artifact(f"mujoco-lora-{timestamp}",type="model");artifact.add_dir(str(output));wandb_run.log_artifact(artifact);wandb_run.finish()
    print(f"[RESULT] LoRA training artifacts: {output}",flush=True)


if __name__=="__main__": main()
