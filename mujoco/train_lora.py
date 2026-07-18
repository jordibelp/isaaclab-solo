#!/usr/bin/env python3
"""Fine-tune frozen Solo12 RSL-RL actor/critic networks with LoRA PPO in MJX.

The target dynamics are the same MuJoCo model used by ``play_direct_mujoco.py``.
MJX batches the physics over ``--num_envs`` on a GPU (or CPU fallback).  The
pre-trained dense weights and biases remain frozen; PPO updates low-rank
adapters in both actor and critic plus the actor exploration log standard
deviation.  No recovery policy or safety filter is used.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import time
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="solo12-two-feet")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--trainable_layers", "--trainable-layers", choices=LAYER_CHOICES, default="all")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--lora-alpha", type=float, default=1.0)
    p.add_argument("--num_envs", "--num-envs", type=int, default=512)
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--rollout-steps", type=int, default=24)
    p.add_argument("--learning-rate", type=float, default=1.0e-2, help="Paper LoRA default: 1e-2.")
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-param", type=float, default=0.2)
    p.add_argument("--value-loss-coef", type=float, default=1.0)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--run-name", default="mujoco LoRA")
    p.add_argument("--symmetry-mode", choices=("none", "augmentation"), default="augmentation")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("logs/mujoco/lora_ppo"))
    p.add_argument("--wandb-project", default="solo12-two-feet-lora")
    p.add_argument("--wandb-entity")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Build everything and run one rollout/update only.")
    return p


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


def init_trainable(key, actor, critic, rank: int, selected: tuple[int, ...], log_std):
    keys = jax.random.split(key, 2 * (len(actor) + len(critic)))
    def adapters(layers, offset):
        result = []
        for i, (weight, _) in enumerate(layers):
            if i in selected:
                a = 0.01 * jax.random.normal(keys[offset + 2*i], (rank, weight.shape[1]))
                b = jnp.zeros((weight.shape[0], rank), weight.dtype)
            else:
                a = jnp.zeros((0, weight.shape[1]), weight.dtype)
                b = jnp.zeros((weight.shape[0], 0), weight.dtype)
            result.append({"a": a, "b": b})
        return tuple(result)
    return {"actor": adapters(actor, 0), "critic": adapters(critic, 2*len(actor)), "log_std": log_std}


def dense_lora(x, frozen, adapter, scale):
    weight, bias = frozen
    residual = (x @ adapter["a"].T) @ adapter["b"].T if adapter["a"].shape[0] else 0.0
    return x @ weight.T + bias + scale * residual


def network(x, frozen, adapters, scale):
    for i, layer in enumerate(frozen):
        x = dense_lora(x, layer, adapters[i], scale)
        if i != len(frozen) - 1:
            x = jax.nn.elu(x)
    return x


def actor_mean(params, obs, frozen, norms, scale):
    x = (obs - norms["actor_mean"]) / (norms["actor_std"] + OBS_NORM_EPS)
    return network(x, frozen, params["actor"], scale)


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


class Transition(NamedTuple):
    obs: jax.Array; action: jax.Array; log_prob: jax.Array; reward: jax.Array
    done: jax.Array; value: jax.Array


def make_training_functions(info: ModelInfo, cfg: dict, num_envs: int):
    model = info.mjx_model
    qids, dids, aids = map(jnp.asarray, (info.joint_qpos, info.joint_dof, info.actuator_ids))
    foot_bodies, foot_geoms = jnp.asarray(info.foot_body_ids), jnp.asarray(info.foot_geom_ids)
    front_thigh, thigh_geoms, base_geoms = map(jnp.asarray, (info.front_thigh_geom_ids, info.thigh_geom_ids, info.base_geom_ids))
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

    def contact_force(data, geom_ids):
        pairs = data.contact.geom
        address = jnp.broadcast_to(
            data.contact.efc_address,
            data.efc_force.shape[:-1] + (data.contact.efc_address.shape[-1],),
        )
        belongs = jnp.any((pairs[..., 0, None] == geom_ids) | (pairs[..., 1, None] == geom_ids), axis=-1)
        force = jnp.take_along_axis(data.efc_force, jnp.maximum(address, 0), axis=-1)
        return jnp.sum(jnp.where((address >= 0) & belongs, jnp.abs(force), 0.0), axis=-1)

    def contact_force_between(data, first_ids, second_id):
        pairs = data.contact.geom
        address = jnp.broadcast_to(
            data.contact.efc_address,
            data.efc_force.shape[:-1] + (data.contact.efc_address.shape[-1],),
        )
        first = jnp.any((pairs[..., 0, None] == first_ids) | (pairs[..., 1, None] == first_ids), axis=-1)
        second = (pairs[..., 0] == second_id) | (pairs[..., 1] == second_id)
        force = jnp.take_along_axis(data.efc_force, jnp.maximum(address, 0), axis=-1)
        return jnp.sum(jnp.where((address >= 0) & first & second, jnp.abs(force), 0.0), axis=-1)

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
        state = EnvState(new_data, jnp.empty((num_envs,48)), commands, jnp.full((num_envs,),command_interval), zeros_a, zeros_a, history, delay,
                         jnp.zeros((num_envs,),jnp.int32) if old is None else jnp.where(mask,0,old.episode_steps),
                         jnp.zeros((num_envs,)) if old is None else jnp.where(mask,0.0,old.episode_return))
        return state._replace(obs=observation(state.data,state.command,state.action,keys[7]))

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
        reward = (cfg["track_lin_vel_xy_reward_scale"]*jnp.exp(-lin_err/cfg["tracking_std"]**2)
                  +cfg["track_ang_vel_z_reward_scale"]*jnp.exp(-yaw_err/cfg["tracking_std"]**2))*STEP_DT
        reward += cfg["action_rate_reward_scale"]*jnp.sum((action-state.action)**2,-1)*STEP_DT
        reward += cfg["joint_torque_reward_scale"]*jnp.sum(data.qfrc_actuator[:,dids]**2,-1)*STEP_DT
        foot_forces = jnp.stack([contact_force(data, jnp.array((g,))) for g in foot_geoms], axis=-1)
        feet_contact = foot_forces > 1.0
        front_contact = jnp.any(feet_contact[:,:2],-1) | (contact_force(data,front_thigh) > 1.0)
        forbidden = front_contact if cfg["front_back_asymetry"] else jnp.sum(feet_contact,-1)>=3
        reward += cfg["three_or_more_feet_contact_penalty_reward_scale"]*forbidden*STEP_DT
        foot_h = data.xpos[:,foot_bodies,2]
        front_h = jnp.mean(foot_h[:,:2],-1)
        kernel = jnp.where(front_h>=cfg["two_feet_above_height_threshold"],1.0,jnp.exp(-cfg["two_feet_above_height_alpha"]*(cfg["two_feet_above_height_threshold"]-front_h)**2))
        lift = kernel*(~jnp.any(feet_contact[:,:2],-1))
        if cfg["rear_feet_in_contact_for_twofeet"]: lift *= jnp.all(feet_contact[:,2:],-1)
        reward += cfg["two_feet_above_height_reward_scale"]*lift*STEP_DT
        thigh_groups = thigh_geoms.reshape((4, 2))
        thigh_contacts = jnp.stack([contact_force(data, group) > 0.6 for group in thigh_groups], axis=-1)
        reward += cfg["undesired_contact_reward_scale"]*jnp.sum(thigh_contacts,axis=-1)*STEP_DT
        foot_excess = jnp.maximum(foot_forces-10.0,0.0)
        reward += cfg["foot_contact_reward_scale"]*jnp.sum(foot_excess**2,axis=-1)*STEP_DT
        base_hit = contact_force_between(data,base_geoms,ground_geom) > 1.0
        steps = state.episode_steps+1
        timeout = steps>=max_episode_steps
        terminate_front = cfg["finish_on_front_feet_contact"] & forbidden & (steps*STEP_DT>=cfg["finish_on_front_feet_contact_after"])
        done = base_hit | timeout | terminate_front | ~jnp.isfinite(data.qpos).all(-1)
        returns = state.episode_return+reward
        next_state = EnvState(data,jnp.empty_like(state.obs),command,command_steps,action,state.action,history,state.delay,steps,returns)
        next_state = next_state._replace(obs=observation(data,command,action,kobs))
        next_state = reset(kreset,next_state,done)
        return next_state,reward,done,returns

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
    step,m,v=state; step+=1
    norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(grads)))
    grads=jax.tree_util.tree_map(lambda g:g*jnp.minimum(1.0,max_norm/(norm+1e-8)),grads)
    m=jax.tree_util.tree_map(lambda a,g:.9*a+.1*g,m,grads);v=jax.tree_util.tree_map(lambda a,g:.999*a+.001*g*g,v,grads)
    mh=jax.tree_util.tree_map(lambda x:x/(1-.9**step),m);vh=jax.tree_util.tree_map(lambda x:x/(1-.999**step),v)
    params=jax.tree_util.tree_map(lambda p,a,b:p-lr*a/(jnp.sqrt(b)+1e-8),params,mh,vh)
    return params,(step,m,v),norm


def save_checkpoints(output: Path, iteration: int, original_payload, params, actor, critic, actor_keys, critic_keys, scale, config):
    output.mkdir(parents=True,exist_ok=True)
    adapter={"iteration":iteration,"config":config,"actor":[],"critic":[],"log_std":np.asarray(params["log_std"])}
    merged=copy.deepcopy(original_payload);state=merged["model_state_dict"]
    for name,layers,keys in (("actor",actor,actor_keys),("critic",critic,critic_keys)):
        for frozen,ad,key in zip(layers,params[name],keys):
            delta=scale*np.asarray(ad["b"]@ad["a"]);state[key]=torch.as_tensor(np.asarray(frozen[0])+delta,dtype=state[key].dtype)
            adapter[name].append({"a":np.asarray(ad["a"]),"b":np.asarray(ad["b"]),"weight_key":key})
    state["log_std"]=torch.as_tensor(np.asarray(params["log_std"]).copy(),dtype=state["log_std"].dtype)
    merged["iter"]=iteration
    if merged.get("infos") is None:
        merged["infos"] = {}
    merged["infos"]["mujoco_lora"]={"iteration":iteration,**config}
    torch.save(adapter,output/f"adapter_{iteration}.pt");torch.save(merged,output/f"model_{iteration}.pt")
    torch.save(adapter,output/"adapter_latest.pt");torch.save(merged,output/"model_latest.pt")


def main():
    args,unknown=build_parser().parse_known_args()
    if args.task!="solo12-two-feet": raise ValueError("Only --task=solo12-two-feet is supported.")
    if args.rank<1 or args.num_envs<1: raise ValueError("--rank and --num_envs must be positive.")
    cfg,unsupported=parse_env_overrides(unknown)
    if unsupported: raise ValueError("Unsupported arguments/overrides: "+" ".join(unsupported))
    if any(abs(x)>1e-9 for x in cfg["forces_applied_to_base_curriculum"]) or any(abs(x)>1e-9 for x in cfg["base_push_force_z_range"]):
        raise ValueError("MJX LoRA training currently supports zero external pushes only.")
    if cfg["include_events_randomization"]: raise ValueError("MJX startup property randomization is not implemented; use env.include_events_randomization=False.")
    if args.device!="auto": jax.config.update("jax_platform_name",args.device)
    checkpoint=Path(args.checkpoint).expanduser().resolve();xml=Path(__file__).with_name("solo12.xml")
    payload,actor,critic,akeys,ckeys,norms,log_std=load_frozen_networks(checkpoint)
    if len(actor)!=len(critic): raise ValueError("Actor and critic layer counts differ.")
    selected=selected_layer_indices(len(actor),args.trainable_layers);scale=args.lora_alpha/args.rank
    key=jax.random.PRNGKey(args.seed);key,kinit,kenv=jax.random.split(key,3)
    params=init_trainable(kinit,actor,critic,args.rank,selected,log_std)
    model_info=build_model(xml,float(cfg["kp"]),float(cfg["kd"]));reset_fn,step_fn=make_training_functions(model_info,cfg,args.num_envs)
    state=reset_fn(kenv)
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S");run_slug=args.run_name.replace("/","_")
    output=args.output_dir/f"{timestamp}_{run_slug}";output.mkdir(parents=True,exist_ok=False)
    config={**vars(args),"checkpoint":str(checkpoint),"output_dir":str(output),"selected_layer_indices":selected,"lora_scale":scale,"env":cfg,"jax_devices":[str(x) for x in jax.devices()],"paper":"arXiv:2603.17092"}
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in config.items()};(output/"run_config.json").write_text(json.dumps(config,indent=2)+"\n")
    wandb_run=None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=args.run_name,config=config)
        except Exception as exc: print(f"[WARN] W&B disabled: {exc}",flush=True)
    optimizer=(jnp.array(0),zeros_like_tree(params),zeros_like_tree(params))

    @jax.jit
    def collect(params,state,key):
        def body(carry,_):
            state,key=carry;key,ka,ks=jax.random.split(key,3)
            mean=actor_mean(params,state.obs,actor,norms,scale);action=mean+jnp.exp(params["log_std"])*jax.random.normal(ka,mean.shape)
            logp=gaussian_log_prob(action,mean,params["log_std"]);value=critic_value(params,state.obs,critic,norms,scale)
            next_state,reward,done,_=step_fn(state,action,ks)
            return (next_state,key),Transition(state.obs,action,logp,reward,done,value)
        return jax.lax.scan(body,(state,key),None,length=args.rollout_steps)

    @jax.jit
    def update_batch(params,opt,batch):
        obs,action,old_logp,adv,returns=batch
        def loss(p):
            mean=actor_mean(p,obs,actor,norms,scale);logp=gaussian_log_prob(action,mean,p["log_std"])
            ratio=jnp.exp(logp-old_logp);policy=-jnp.mean(jnp.minimum(ratio*adv,jnp.clip(ratio,1-args.clip_param,1+args.clip_param)*adv))
            value=critic_value(p,obs,critic,norms,scale);vloss=.5*jnp.mean((returns-value)**2)
            entropy=jnp.sum(p["log_std"]+.5*math.log(2*math.pi*math.e))
            total=policy+args.value_loss_coef*vloss-args.entropy_coef*entropy
            return total,(policy,vloss,entropy,jnp.mean(jnp.abs(ratio-1)>args.clip_param))
        (loss,metrics),grads=jax.value_and_grad(loss,has_aux=True)(params)
        params,opt,gn=adam_update(params,grads,opt,args.learning_rate,args.max_grad_norm)
        return params,opt,(loss,*metrics,gn)

    print(f"[INFO] MJX devices={jax.devices()} envs={args.num_envs} rank={args.rank} layers={selected} trainable={sum(x.size for x in jax.tree_util.tree_leaves(params)):,}",flush=True)
    history=[]
    iterations=1 if args.dry_run else args.max_iterations
    for iteration in range(1,iterations+1):
        started=time.perf_counter();key,kroll,kperm=jax.random.split(key,3);(state,key2),traj=collect(params,state,kroll);key=key2
        bootstrap=critic_value(params,state.obs,critic,norms,scale)
        def gae_step(carry,x):
            adv,next_v=carry;reward,done,value=x;delta=reward+args.gamma*(1-done)*next_v-value;adv=delta+args.gamma*args.gae_lambda*(1-done)*adv;return (adv,value),adv
        _,adv_rev=jax.lax.scan(gae_step,(jnp.zeros_like(bootstrap),bootstrap),(traj.reward[::-1],traj.done[::-1],traj.value[::-1]))
        adv=adv_rev[::-1];returns=adv+traj.value
        obs,actions,old_logp=traj.obs.reshape((-1,48)),traj.action.reshape((-1,12)),traj.log_prob.reshape(-1);adv=adv.reshape(-1);returns=returns.reshape(-1)
        if args.symmetry_mode=="augmentation":
            mo,ma=mirror_lr(obs,actions);ml=gaussian_log_prob(ma,actor_mean(params,mo,actor,norms,scale),params["log_std"])
            obs=jnp.concatenate((obs,mo));actions=jnp.concatenate((actions,ma));old_logp=jnp.concatenate((old_logp,ml));adv=jnp.concatenate((adv,adv));returns=jnp.concatenate((returns,returns))
        adv=(adv-jnp.mean(adv))/(jnp.std(adv)+1e-8);size=obs.shape[0];mb=max(1,size//args.num_minibatches);metrics=[]
        for epoch in range(args.ppo_epochs):
            kperm,sub=jax.random.split(kperm);perm=jax.random.permutation(sub,size)
            for start in range(0,size-mb+1,mb):
                idx=perm[start:start+mb];params,optimizer,m=update_batch(params,optimizer,(obs[idx],actions[idx],old_logp[idx],adv[idx],returns[idx]));metrics.append(m)
        m=np.asarray(jnp.mean(jnp.stack([jnp.stack(values) for values in metrics]),0));elapsed=time.perf_counter()-started
        log={"iteration":iteration,"reward/mean_step":float(jnp.mean(traj.reward)),"resets":int(jnp.sum(traj.done)),"loss/total":m[0],"loss/policy":m[1],"loss/value":m[2],"policy/entropy":m[3],"policy/clip_fraction":m[4],"grad_norm":m[5],"throughput_steps_s":args.num_envs*args.rollout_steps/elapsed,"wall_time_s":elapsed}
        history.append({k:float(v) if isinstance(v,(np.floating,float)) else int(v) for k,v in log.items()});print("[INFO] "+" ".join(f"{k}={v:.5g}" if isinstance(v,float) else f"{k}={v}" for k,v in log.items()),flush=True)
        if wandb_run: wandb_run.log(log,step=iteration)
        if iteration%args.save_interval==0 or iteration==iterations: save_checkpoints(output,iteration,payload,params,actor,critic,akeys,ckeys,scale,config)
        (output/"metrics.jsonl").write_text("".join(json.dumps(x)+"\n" for x in history))
    if wandb_run:
        import wandb
        artifact=wandb.Artifact(f"mujoco-lora-{timestamp}",type="model");artifact.add_dir(str(output));wandb_run.log_artifact(artifact);wandb_run.finish()
    print(f"[RESULT] LoRA training artifacts: {output}",flush=True)


if __name__=="__main__": main()
