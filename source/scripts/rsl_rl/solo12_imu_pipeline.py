# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint contracts for the Solo12 base-IMU teacher / DAgger / PPO pipeline.

Importable without Isaac Sim. A contract preserves signal meanings, not just tensor sizes.
It intentionally does not copy rewards or curriculum progress between training stages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch


CONTRACT_FILE = "solo12_imu_contract.json"
ENV_FIELDS = (
    "include_foot_height_obs", "include_lin_acc_steps_in_imu_obs", "teacher_include_base_force",
    "teacher_foot_force_obs", "teacher_latent_activation", "apply_tanh_to_latent", "action_obs_is_last_executed", "base_imu_history_action_is_last_executed",
    "imu_ekf_processed_inputs", "use_rotMat_on_imu_encoder", "decimation",
)
POLICY_FIELDS = (
    "actor_hidden_dims", "critic_hidden_dims", "activation", "actor_obs_normalization",
    "critic_obs_normalization", "noise_std_type", "teacher_encoder_hidden_dims", "teacher_latent_dim", "teacher_latent_activation", "apply_tanh_to_latent",
    "priv_critic_includes_perception_encoding", "priv_critic_detached_perception_encoding",
)
PERCEPTION_DEFAULTS = {"priv_critic_includes_perception_encoding": False,
                       "priv_critic_detached_perception_encoding": True}


def observation_contract(env_cfg) -> dict:
    return {**{name: getattr(env_cfg, name) for name in ENV_FIELDS}, "physics_dt": float(env_cfg.sim.dt)}


def apply_observation_contract(env_cfg, contract: dict) -> None:
    for name in ENV_FIELDS:
        if name in contract:
            setattr(env_cfg, name, contract[name])
    if "physics_dt" in contract:
        env_cfg.sim.dt = float(contract["physics_dt"])
        if hasattr(env_cfg, "physics_hz"):
            # Legacy contracts store dt/decimation, not the newer authoritative Hz fields.
            env_cfg.physics_hz = 1.0 / env_cfg.sim.dt
            env_cfg.policy_hz = env_cfg.physics_hz / env_cfg.decimation
    env_cfg.refresh_runtime_dependent_config()


def adapter_latent_tanh(checkpoint: dict) -> bool:
    """Old adapters always used tanh; new metadata must agree across both saved contracts."""
    contract = checkpoint.get("observation_contract") or {}
    value = checkpoint.get("layout", {}).get("apply_tanh_to_latent", contract.get("apply_tanh_to_latent", True))
    if "apply_tanh_to_latent" in contract and value != contract["apply_tanh_to_latent"]:
        raise ValueError("DAgger layout and observation contract disagree on apply_tanh_to_latent.")
    return value


def restore_latent_activation_from_checkpoint(env_cfg, checkpoint_path: str) -> None:
    """Restore activation and critic topology for standalone PPO resume/playback."""
    path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "adapter_state_dict" in checkpoint:
        saved = {**(checkpoint.get("observation_contract") or {}),
                 "apply_tanh_to_latent": adapter_latent_tanh(checkpoint)}
        critic_contract = checkpoint.get("teacher_policy", {})
    else:
        saved = (checkpoint.get("infos") or {}).get("solo12_imu_latent")
        critic_contract = (checkpoint.get("infos") or {}).get("solo12_imu_critic")
        if saved is None:
            sidecar = path.parent / "params" / CONTRACT_FILE
            if not sidecar.is_file():
                sidecar = path.parent.parent / "params" / CONTRACT_FILE
            sidecar_data = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
            saved = sidecar_data.get("observation_contract", {})
            if critic_contract is None:
                critic_contract = sidecar_data.get("policy", {})
    for name in ("apply_tanh_to_latent", "teacher_latent_activation"):
        if name in saved:
            setattr(env_cfg, name, saved[name])
    for name, default in PERCEPTION_DEFAULTS.items():
        value = (critic_contract or {}).get(name, default)
        if getattr(env_cfg, name, default) != value:
            print(f"[INFO]: Restoring checkpoint {name}={value}; full PPO resume/playback preserves saved topology.")
        setattr(env_cfg, name, value)
    env_cfg.refresh_runtime_dependent_config()


def attach_latent_contract_to_runner(runner, env_cfg) -> None:
    """Embed activation, critic topology and warm-up progress in every standalone PPO .pt."""
    original_save = runner.save

    def save(path, infos=None):
        infos = {**(infos or {}), "solo12_imu_latent": {
            "apply_tanh_to_latent": env_cfg.apply_tanh_to_latent,
            "teacher_latent_activation": env_cfg.teacher_latent_activation,
        }}
        infos["solo12_imu_critic"] = {name: getattr(env_cfg, name, default)
                                       for name, default in PERCEPTION_DEFAULTS.items()}
        alg = getattr(runner, "alg", None)
        if hasattr(alg, "critic_warmup_remaining") and hasattr(alg.policy, "actor_history_encoder"):
            infos["solo12_critic_warmup"] = {"remaining": alg.critic_warmup_remaining,
                                            "completed": alg.critic_warmup_completed}
        return original_save(path, infos=infos)

    runner.save = save


def save_training_contract(log_dir: str, env_cfg, policy_cfg) -> None:
    if not str(getattr(env_cfg, "policy_model", "")).startswith("base_imu_"):
        return
    path = Path(log_dir) / "params" / CONTRACT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "observation_contract": observation_contract(env_cfg),
        "policy": {name: getattr(policy_cfg, name) for name in POLICY_FIELDS if hasattr(policy_cfg, name)},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def checkpoint_model_state(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError(f"No model state in teacher checkpoint: {path}")


def mlp_shape(state: dict, prefix: str) -> tuple[int, list[int], int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    layers = sorted(
        (int(match.group(1)), value) for key, value in state.items()
        if (match := pattern.match(key)) is not None and torch.is_tensor(value) and value.ndim == 2
    )
    if not layers:
        raise ValueError(f"Teacher checkpoint has no {prefix} MLP weights.")
    return layers[0][1].shape[1], [value.shape[0] for _, value in layers[:-1]], layers[-1][1].shape[0]


def configure_teacher_for_dagger(env_cfg, policy_cfg, checkpoint_path: str) -> dict:
    """Restore the teacher's input semantics and architecture before building the DAgger env."""
    if env_cfg.policy_model != "base_imu_student_dagger":
        raise ValueError("DAgger requires a base_imu_student_dagger task.")
    path = Path(checkpoint_path)
    sidecar = path.parent / "params" / CONTRACT_FILE
    if not sidecar.is_file():
        sidecar = path.parent.parent / "params" / CONTRACT_FILE
    saved = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    state = checkpoint_model_state(checkpoint_path)
    encoder_in, encoder_hidden, latent_dim = mlp_shape(state, "teacher_encoder")
    critic_in, _, _ = mlp_shape(state, "critic")
    if critic_in not in {encoder_in + 3, encoder_in + 3 + latent_dim}:
        raise ValueError("Teacher critic input does not match its privileged/perception contract.")
    includes_encoding = critic_in == encoder_in + 3 + latent_dim
    if saved:
        apply_observation_contract(env_cfg, saved["observation_contract"])
        for name, value in saved.get("policy", {}).items():
            if name in POLICY_FIELDS and hasattr(policy_cfg, name):
                setattr(policy_cfg, name, value)
    else:
        restore_latent_activation_from_checkpoint(env_cfg, checkpoint_path)
    env_cfg.teacher_encoder_hidden_dims = encoder_hidden
    env_cfg.teacher_latent_dim = latent_dim
    env_cfg.base_imu_tcn_latent_dim = latent_dim
    env_cfg.priv_critic_includes_perception_encoding = includes_encoding
    env_cfg.priv_critic_detached_perception_encoding = saved.get("policy", {}).get(
        "priv_critic_detached_perception_encoding", getattr(env_cfg, "priv_critic_detached_perception_encoding", True))
    env_cfg.refresh_runtime_dependent_config()
    if env_cfg.teacher_encoder_obs_dim != encoder_in:
        raise ValueError(
            f"Teacher input contract mismatch: env={env_cfg.teacher_encoder_obs_dim}, checkpoint={encoder_in}. "
            "Use the teacher's observation overrides (legacy checkpoints have no contract sidecar)."
        )
    policy_cfg.teacher_encoder_obs_dim = encoder_in
    policy_cfg.teacher_encoder_hidden_dims = encoder_hidden
    policy_cfg.teacher_latent_dim = latent_dim
    for name in PERCEPTION_DEFAULTS:
        setattr(policy_cfg, name, getattr(env_cfg, name))
    for prefix in ("actor", "critic"):
        setattr(policy_cfg, f"{prefix}_hidden_dims", mlp_shape(state, prefix)[1])
        setattr(policy_cfg, f"{prefix}_obs_normalization", any(k.startswith(f"{prefix}_obs_normalizer.") for k in state))
    policy_cfg.noise_std_type = "log" if "log_std" in state else "scalar"
    return state


def adapter_student_actor_state(checkpoint: dict) -> dict | None:
    """Never silently discard a fine-tuned actor or accept metadata without its weights."""
    state = checkpoint.get("student_actor_state_dict")
    if state is not None and not isinstance(state, dict):
        raise ValueError("DAgger checkpoint has an invalid student_actor_state_dict.")
    has_actor = state is not None
    declared = (checkpoint.get("objective") or {}).get("finetune_student_actor", has_actor)
    if bool(declared) != has_actor:
        raise ValueError("DAgger checkpoint has inconsistent actor-fine-tuning metadata and student_actor_state_dict.")
    return state


def adapter_metadata(env_cfg, policy_cfg, teacher_checkpoint: str, *, action_loss_coef: float,
                     finetune_student_actor: bool = False) -> dict:
    return {
        "layout": {
            "kind": "base_imu", "history_len": int(env_cfg.base_imu_history_length),
            "history_dim": int(env_cfg.base_imu_history_sample_dim), "channels": int(env_cfg.base_imu_tcn_channels),
            "kernel_size": int(env_cfg.base_imu_tcn_kernel_size), "activation": str(env_cfg.base_imu_tcn_activation),
            "apply_tanh_to_latent": bool(env_cfg.apply_tanh_to_latent),
        },
        "observation_contract": observation_contract(env_cfg),
        "teacher_policy": {name: getattr(policy_cfg, name) for name in POLICY_FIELDS if hasattr(policy_cfg, name)},
        "teacher_checkpoint": str(teacher_checkpoint),
        "objective": {"supervise_action": action_loss_coef != 0., "action_loss_coef": action_loss_coef,
                      "finetune_student_actor": finetune_student_actor},
    }


def configure_base_imu_student(env_cfg, policy_cfg, checkpoint: dict, teacher_checkpoint: str) -> None:
    if env_cfg.policy_model != "base_imu_student_rl":
        raise ValueError("Base-IMU adapter initialization requires a base_imu_student_rl task.")
    dims = checkpoint["dims"]
    layout = checkpoint.get("layout", {})
    if layout.get("kind", "base_imu") != "base_imu":
        raise ValueError("The selected student requires a base_imu adapter, not a race adapter.")
    contract = checkpoint.get("observation_contract")
    if contract is not None:
        apply_observation_contract(env_cfg, contract)
    else:
        for name in ("imu_ekf_processed_inputs", "use_rotMat_on_imu_encoder"):
            if name in checkpoint.get("imu_layout", {}):
                setattr(env_cfg, name, checkpoint["imu_layout"][name])
    # Old adapters used a hard-coded tanh. New contracts preserve this parameter-free choice explicitly.
    env_cfg.apply_tanh_to_latent = adapter_latent_tanh(checkpoint)
    state = checkpoint_model_state(teacher_checkpoint)
    encoder_in, _, latent_dim = mlp_shape(state, "teacher_encoder")
    if contract is None and encoder_in in (45, 49):
        env_cfg.include_foot_height_obs = encoder_in == 49
    history_len = int(dims["history_len"])
    if history_len < 1 or (history_len - 1) % env_cfg.decimation:
        raise ValueError(f"Endpoint-inclusive IMU history_len={history_len} is incompatible with decimation={env_cfg.decimation}.")
    env_cfg.base_imu_history_policy_steps = (history_len - 1) // env_cfg.decimation
    adapter_state = checkpoint["adapter_state_dict"]
    conv = adapter_state["conv_stack.0.conv.weight"]
    env_cfg.base_imu_tcn_channels = int(layout.get("channels", conv.shape[0]))
    env_cfg.base_imu_tcn_kernel_size = int(layout.get("kernel_size", conv.shape[2]))
    env_cfg.base_imu_tcn_activation = str(layout.get("activation", "elu"))
    env_cfg.base_imu_tcn_latent_dim = latent_dim
    env_cfg.feed_history_encoding_to_critic = False
    env_cfg.refresh_runtime_dependent_config()
    if (env_cfg.teacher_encoder_obs_dim != encoder_in or env_cfg.teacher_critic_obs_dim != int(dims["teacher_obs_dim"])
            or env_cfg.base_imu_history_sample_dim != int(dims["history_sample_dim"])
            or latent_dim != int(dims["latent_dim"])):
        raise ValueError("DAgger/teacher observation contract does not match the selected student task.")
    critic_in = mlp_shape(state, "critic")[0]
    if critic_in not in {env_cfg.teacher_critic_obs_dim, env_cfg.teacher_critic_obs_dim + latent_dim}:
        raise ValueError("Teacher critic input does not match the privileged/perception contract.")
    if critic_in != env_cfg.teacher_critic_obs_dim and not getattr(
        env_cfg, "priv_critic_includes_perception_encoding", False
    ):
        raise ValueError(
            "Teacher critic uses a perception encoding; enable env.priv_critic_includes_perception_encoding=True."
        )
    for name, value in checkpoint.get("teacher_policy", {}).items():
        if name in POLICY_FIELDS and hasattr(policy_cfg, name):
            setattr(policy_cfg, name, value)
    for name, value in {
        "history_len": history_len, "history_sample_dim": env_cfg.base_imu_history_sample_dim,
        "teacher_critic_obs_dim": env_cfg.teacher_critic_obs_dim, "tcn_channels": env_cfg.base_imu_tcn_channels,
        "tcn_kernel_size": env_cfg.base_imu_tcn_kernel_size, "tcn_activation": env_cfg.base_imu_tcn_activation,
        "tcn_latent_dim": latent_dim, "feed_history_encoding_to_critic": False,
        "apply_tanh_to_latent": env_cfg.apply_tanh_to_latent,
        "actor_hidden_dims": mlp_shape(state, "actor")[1], "critic_hidden_dims": mlp_shape(state, "critic")[1],
        "actor_obs_normalization": isinstance(checkpoint.get("history_normalizer_state_dict"), dict)
                                   or any(k.startswith("actor_obs_normalizer.") for k in state),
        "critic_obs_normalization": any(k.startswith("critic_obs_normalizer.") for k in state),
        "noise_std_type": "log" if "log_std" in state else "scalar",
        **{name: getattr(env_cfg, name, default) for name, default in PERCEPTION_DEFAULTS.items()},
    }.items():
        setattr(policy_cfg, name, value)
    print(f"[INFO]: Configured base-IMU student: history={history_len}x{env_cfg.base_imu_history_sample_dim}, "
          f"privileged critic={env_cfg.teacher_critic_obs_dim}, TCN={env_cfg.base_imu_tcn_channels}ch/"
          f"k{env_cfg.base_imu_tcn_kernel_size}, latent={latent_dim}, "
          f"apply_tanh_to_latent={env_cfg.apply_tanh_to_latent}.", flush=True)


def expand_privileged_critic_weight(source: torch.Tensor, target: torch.Tensor,
                                    privileged_obs_dim: int, latent_dim: int, command_dim: int) -> torch.Tensor:
    """Add zero-weight encoding columns, preserving a direct privileged critic's function."""
    if source.shape != (target.shape[0], privileged_obs_dim) or target.shape[1] != privileged_obs_dim + latent_dim:
        raise ValueError("Cannot expand critic weight: expected a direct privileged critic with the same hidden width.")
    expanded = torch.zeros_like(target)
    features = privileged_obs_dim - command_dim
    expanded[:, :features] = source[:, :features].to(expanded.device)
    expanded[:, -command_dim:] = source[:, -command_dim:].to(expanded.device)
    return expanded


def configure_critic_warmup(runner, iterations: int, *, from_adapter: bool, checkpoint_infos=None) -> None:
    """Start once at a handoff, or continue (never restart) checkpointed warm-up progress."""
    if type(iterations) is not int or iterations < 0:
        raise ValueError("agent.critic_warmup_iterations must be a non-negative integer.")
    saved = (checkpoint_infos or {}).get("solo12_critic_warmup")
    if from_adapter:
        if iterations:
            runner.alg.start_critic_warmup(iterations)
    elif saved is not None:
        runner.alg.start_critic_warmup(saved["remaining"], completed=saved["completed"])
    elif iterations:
        raise ValueError("Critic warm-up requires a DAgger adapter handoff; PPO resume needs saved warm-up progress.")
    if getattr(runner.alg, "critic_warmup_remaining", 0):
        print(f"[INFO]: Critic-only warm-up: {runner.alg.critic_warmup_remaining} rollout/update iterations remaining; "
              "actor, perception encoder, action noise and actor normalization stay fixed.", flush=True)
