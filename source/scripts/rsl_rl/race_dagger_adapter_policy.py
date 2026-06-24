# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for playing/evaluating Solo12 race env-param DAgger adapters.

A phase-2 DAgger checkpoint only contains the history->latent adapter. To act in
an environment we also need the frozen phase-1 ParamsConditionedEnc teacher: the
adapter predicts z_hat from history, then z_hat is fed through the teacher actor
head together with the current race observation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict

from isaaclab_tasks.direct.solo12_race.agents.env_params_conditioned_encoder_actor import (
    EnvParamsConditionedEncoderActor,
)
from isaaclab_tasks.direct.solo12_race.agents.imu_tcn_actor_critic import FootImuTcnEncoder


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return dict(cfg)
    return dict(vars(cfg))


def _checkpoint_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint state_dict type: {type(state_dict)!r}")
    return state_dict


def _infer_mlp_hidden_dims(state_dict: dict[str, torch.Tensor], prefix: str) -> list[int] | None:
    """Infer hidden layer widths from an MLP state dict prefix.

    RSL-RL MLP modules are saved as e.g. actor.0.weight, actor.2.weight, ..., where the last Linear is the
    output layer. The hidden architecture is therefore every Linear out_features except the final one.
    """

    layers: list[tuple[int, int]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.weight$")
    for key, value in state_dict.items():
        match = pattern.match(str(key))
        if match is None or not isinstance(value, torch.Tensor) or value.ndim < 2:
            continue
        layers.append((int(match.group(1)), int(value.shape[0])))
    layers.sort(key=lambda item: item[0])
    if len(layers) < 2:
        return None
    return [out_features for _, out_features in layers[:-1]]


def _apply_checkpoint_architecture(policy_kwargs: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> None:
    """Make the teacher policy config match the checkpoint head widths before loading weights.

    The EnvParamsConditionedEncoderActor loader can adapt observation-layout changes, but it cannot recover an omitted
    hidden layer. If we instantiate the teacher with the task default [256, 128, 64] and then load a checkpoint trained
    with [256, 256, 128, 64], the permissive loader has to skip most of the actor/critic head. For evaluation that would
    leave a partly-random teacher, so infer the saved MLP widths from the checkpoint first.
    """

    for prefix, cfg_key in (("actor", "actor_hidden_dims"), ("critic", "critic_hidden_dims")):
        hidden_dims = _infer_mlp_hidden_dims(state_dict, prefix)
        if not hidden_dims:
            continue
        configured = list(policy_kwargs.get(cfg_key, []))
        if configured != hidden_dims:
            print(
                f"[INFO] Inferred checkpoint {cfg_key}={hidden_dims} "
                f"(config had {configured}).",
                flush=True,
            )
            policy_kwargs[cfg_key] = hidden_dims


def apply_checkpoint_architecture_to_policy_cfg(policy_cfg: Any, checkpoint_path: str) -> None:
    """Mutate an RSL-RL policy config so actor/critic MLP widths match a checkpoint.

    This keeps evaluation/play commands honest when older training runs used Hydra overrides such as
    ``actor_hidden_dims=[256,256,128,64]`` but the current task config defaults have drifted. The permissive custom
    loaders can warm-start observation-layout changes, but skipped hidden-layer tensors would leave random network
    pieces during eval. Infer the saved architecture before constructing the runner/policy.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _checkpoint_model_state_dict(checkpoint)
    policy_kwargs = _cfg_to_dict(policy_cfg)
    _apply_checkpoint_architecture(policy_kwargs, state_dict)
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        if key in policy_kwargs and hasattr(policy_cfg, key):
            setattr(policy_cfg, key, list(policy_kwargs[key]))


def load_dagger_adapter_checkpoint(path: str) -> dict[str, Any] | None:
    """Return a DAgger adapter checkpoint dict, or None for a normal RSL-RL checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "adapter_state_dict" in checkpoint and "layout" in checkpoint:
        return checkpoint
    return None


def is_dagger_adapter_checkpoint(path: str) -> bool:
    try:
        return load_dagger_adapter_checkpoint(path) is not None
    except Exception:
        return False


def resolve_teacher_checkpoint(
    *,
    adapter_checkpoint: dict[str, Any],
    adapter_checkpoint_path: str,
    override_path: str | None = None,
) -> str:
    """Resolve the frozen phase-1 teacher checkpoint for a DAgger adapter.

    Cluster-trained adapter checkpoints can store a remote teacher path such as
    /home/jbeltran/.../best_model.pt. If that path is unavailable locally, search
    nearby flattened checkpoint exports for the run id (e.g. f50n1qmb).
    """

    candidate_paths: list[str] = []
    if override_path:
        candidate_paths.append(override_path)
    saved_path = str(adapter_checkpoint.get("teacher_checkpoint") or "")
    if saved_path:
        candidate_paths.append(saved_path)

    for candidate in candidate_paths:
        expanded = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isfile(expanded):
            return expanded

    adapter_path = Path(adapter_checkpoint_path).expanduser().resolve()
    search_roots = [
        adapter_path.parent,
        adapter_path.parent.parent,
        Path.cwd() / "logs" / "skrl" / "checkpoints",
        Path.cwd() / "logs" / "rsl_rl",
        Path("/home/jordibelp/IsaacLab/logs/skrl/checkpoints"),
        Path("/home/jordibelp/IsaacLab/logs/rsl_rl"),
    ]
    tokens = list(dict.fromkeys(re.findall(r"[0-9a-z]{8}", saved_path)))
    # Prefer the last id in the original training run name. For f50n1qmb-style
    # exports this is usually the actual teacher run id, while earlier ids can be
    # sweep/experiment ids embedded in the run name.
    for token in reversed(tokens):
        matches: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            try:
                matches.extend(root.glob(f"**/*{token}*best_model*.pt"))
                matches.extend(root.glob(f"**/*{token}*.pt"))
            except Exception:
                continue
        matches = sorted({m.resolve() for m in matches if m.is_file()})
        if matches:
            return str(matches[0])

    hint = override_path or saved_path or "<missing>"
    raise FileNotFoundError(
        "Could not resolve the phase-1 teacher checkpoint for this DAgger adapter. "
        f"Tried override/saved path: {hint}. Pass --dagger-teacher-checkpoint /path/to/best_model.pt."
    )


def configure_env_cfg_for_dagger_adapter(env_cfg: Any, adapter_checkpoint: dict[str, Any]) -> None:
    """Mutate a Solo12 race env cfg so observations match the adapter checkpoint."""

    layout = dict(adapter_checkpoint.get("layout") or {})
    dims = dict(adapter_checkpoint.get("dims") or {})
    kind = str(layout.get("kind") or "")
    if kind not in {"joint_state", "joint_state_imu"}:
        raise ValueError(f"Unsupported DAgger adapter layout kind: {kind!r}")

    # The saved DAgger dims expect current race obs + privileged GT env params + history.
    if int(dims.get("env_params_dim", 0)) > 0:
        if hasattr(env_cfg, "include_forces_to_gt_obs"):
            env_cfg.include_forces_to_gt_obs = True
        if hasattr(env_cfg, "include_mu_coefs_to_gt_obs"):
            env_cfg.include_mu_coefs_to_gt_obs = True

    if hasattr(env_cfg, "include_joint_state_history_obs"):
        env_cfg.include_joint_state_history_obs = True
    if hasattr(env_cfg, "include_foot_imu_obs"):
        env_cfg.include_foot_imu_obs = kind == "joint_state_imu"
    if hasattr(env_cfg, "policy_model"):
        env_cfg.policy_model = (
            "env_params_dagger_joint_state_imu_tcn" if kind == "joint_state_imu" else "env_params_dagger_joint_state_tcn"
        )

    history_len = int(layout.get("history_len") or 0)
    decimation = int(getattr(env_cfg, "decimation", 1))
    if history_len > 0:
        if history_len % decimation != 0:
            raise ValueError(
                f"DAgger adapter history_len={history_len} is not divisible by env decimation={decimation}; "
                "cannot express it as a Solo12 race history-policy-step window."
            )
        history_policy_steps = history_len // decimation
        for attr in (
            "foot_imu_history_policy_steps",
            "joint_state_history_policy_steps",
            "joint_imu_history_policy_steps",
        ):
            if hasattr(env_cfg, attr):
                setattr(env_cfg, attr, history_policy_steps)

    # Recompute observation_space/history lengths after mutating config flags.
    post_init = getattr(env_cfg, "__post_init__", None)
    if callable(post_init):
        post_init()

    expected_obs_dim = int(dims.get("history_start", 0)) + int(dims.get("history_flat_dim", layout.get("flat_dim", 0)))
    actual_obs_dim = int(getattr(env_cfg, "observation_space", expected_obs_dim))
    if expected_obs_dim and actual_obs_dim != expected_obs_dim:
        raise ValueError(
            f"DAgger adapter expects obs dim {expected_obs_dim}, but configured task has obs dim {actual_obs_dim}."
        )


def _load_teacher(
    *,
    checkpoint_path: str,
    policy_cfg: Any,
    num_actions: int,
    device: torch.device,
    teacher_shared_networks: bool = False,
) -> EnvParamsConditionedEncoderActor:
    policy_kwargs = _cfg_to_dict(policy_cfg)
    policy_kwargs.pop("class_name", None)
    if teacher_shared_networks:
        if "shared_networks" not in policy_kwargs:
            raise ValueError("--dagger-teacher-shared-networks was set, but this policy config has no shared_networks field.")
        policy_kwargs["shared_networks"] = True

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _checkpoint_model_state_dict(checkpoint)
    _apply_checkpoint_architecture(policy_kwargs, state_dict)

    current_obs_dim = int(policy_kwargs.get("current_obs_dim", 57))
    env_params_dim = int(policy_kwargs.get("env_params_dim", 16))
    teacher_obs_dim = current_obs_dim + env_params_dim
    dummy_obs = TensorDict({"policy": torch.zeros((1, teacher_obs_dim), device=device)}, batch_size=[1])
    teacher = EnvParamsConditionedEncoderActor(
        obs=dummy_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=num_actions,
        **policy_kwargs,
    ).to(device)

    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _actor_mean_and_std(
    teacher: EnvParamsConditionedEncoderActor, actor_input: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = teacher.actor(actor_input)
    if teacher.state_dependent_std:
        if teacher.noise_std_type == "scalar":
            mean, std = torch.unbind(output, dim=-2)
        elif teacher.noise_std_type == "log":
            mean, log_std = torch.unbind(output, dim=-2)
            std = torch.exp(log_std)
        else:
            raise ValueError(f"Unsupported teacher noise_std_type: {teacher.noise_std_type}")
        return mean, std

    if teacher.noise_std_type == "scalar":
        std = teacher.std.expand_as(output)
    elif teacher.noise_std_type == "log":
        std = torch.exp(teacher.log_std).expand_as(output)
    else:
        raise ValueError(f"Unsupported teacher noise_std_type: {teacher.noise_std_type}")
    return output, std


class DaggerLatentPolicy(nn.Module):
    """Callable policy for phase-2 DAgger adapter inference."""

    is_dagger_adapter_policy = True

    def __init__(
        self,
        *,
        teacher: EnvParamsConditionedEncoderActor,
        adapter: nn.Module,
        history_normalizer: EmpiricalNormalization | None,
        dims: dict[str, int],
        stochastic_actions: bool = False,
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.adapter = adapter
        self.history_normalizer = history_normalizer
        self.dims = {k: int(v) for k, v in dims.items()}
        self.stochastic_actions = bool(stochastic_actions)

    @property
    def actor_critic(self):
        # Existing play/eval loops expect policy.actor_critic.reset(dones).
        # Keep this as a property instead of a Module attribute to avoid
        # recursively registering self as a child module.
        return self

    def reset(self, dones: torch.Tensor | None = None) -> None:
        return None

    def forward(self, obs: Any) -> torch.Tensor:
        policy_obs = obs["policy"] if isinstance(obs, (dict, TensorDict)) else obs
        teacher_obs_dim = int(self.dims["teacher_obs_dim"])
        history_start = int(self.dims["history_start"])
        history_flat_dim = int(self.dims["history_flat_dim"])

        teacher_obs_raw = policy_obs[:, :teacher_obs_dim]
        history_raw = policy_obs[:, history_start : history_start + history_flat_dim]
        if history_raw.shape[-1] != history_flat_dim:
            raise RuntimeError(f"Expected history dim {history_flat_dim}, got {history_raw.shape[-1]} from observations.")

        if self.history_normalizer is not None:
            adapter_input = self.history_normalizer(history_raw)
        else:
            adapter_input = history_raw
        z_hat = self.adapter(adapter_input)

        teacher_obs = self.teacher.actor_obs_normalizer(teacher_obs_raw)
        current_obs = teacher_obs[:, : self.teacher.current_obs_dim]
        actor_input = torch.cat((current_obs, z_hat), dim=-1)
        mean, std = _actor_mean_and_std(self.teacher, actor_input)
        if self.stochastic_actions:
            return mean + torch.randn_like(mean) * std
        return mean


def load_dagger_latent_policy(
    *,
    adapter_checkpoint_path: str,
    adapter_checkpoint: dict[str, Any],
    teacher_checkpoint_path: str | None,
    policy_cfg: Any,
    num_actions: int,
    device: torch.device,
    teacher_shared_networks: bool = False,
    stochastic_actions: bool = False,
) -> tuple[DaggerLatentPolicy, str]:
    """Load a DAgger adapter policy and return (policy, resolved_teacher_checkpoint)."""

    layout = dict(adapter_checkpoint["layout"])
    dims = {k: int(v) for k, v in dict(adapter_checkpoint["dims"]).items()}
    teacher_path = resolve_teacher_checkpoint(
        adapter_checkpoint=adapter_checkpoint,
        adapter_checkpoint_path=adapter_checkpoint_path,
        override_path=teacher_checkpoint_path,
    )
    teacher = _load_teacher(
        checkpoint_path=teacher_path,
        policy_cfg=policy_cfg,
        num_actions=num_actions,
        device=device,
        teacher_shared_networks=teacher_shared_networks,
    )

    adapter = FootImuTcnEncoder(
        history_len=int(layout["history_len"]),
        imu_dim=int(layout["history_dim"]),
        channels=int(layout["channels"]),
        latent_dim=int(dims["latent_dim"]),
        kernel_size=int(layout["kernel_size"]),
        activation=str(layout["activation"]),
    ).to(device)
    adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    for param in adapter.parameters():
        param.requires_grad_(False)

    history_normalizer = None
    normalizer_state = adapter_checkpoint.get("history_normalizer_state_dict")
    if normalizer_state is not None:
        history_normalizer = EmpiricalNormalization(int(dims["history_flat_dim"])).to(device)
        history_normalizer.load_state_dict(normalizer_state, strict=True)
        history_normalizer.eval()
        for param in history_normalizer.parameters():
            param.requires_grad_(False)

    policy = DaggerLatentPolicy(
        teacher=teacher,
        adapter=adapter,
        history_normalizer=history_normalizer,
        dims=dims,
        stochastic_actions=stochastic_actions,
    ).to(device)
    policy.eval()
    return policy, teacher_path
