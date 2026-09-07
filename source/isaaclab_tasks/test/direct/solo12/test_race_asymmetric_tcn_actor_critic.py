# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict


def _load_agent_module(module_basename: str):
    """Load a focused race agent module without importing Isaac Sim task registration."""

    agents_dir = (
        Path(__file__).resolve().parents[3]
        / "isaaclab_tasks"
        / "direct"
        / "solo12_race"
        / "agents"
    )
    package_name = "_solo12_race_asymmetric_agents_test"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(agents_dir)]
        sys.modules[package_name] = package

    module_name = f"{package_name}.{module_basename}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, agents_dir / f"{module_basename}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


ActorCriticFootImuTcn = _load_agent_module("imu_tcn_actor_critic").ActorCriticFootImuTcn
FootImuTcnEncoder = _load_agent_module("imu_tcn_actor_critic").FootImuTcnEncoder
EnvParamsConditionedEncoderActor = _load_agent_module(
    "env_params_conditioned_encoder_actor"
).EnvParamsConditionedEncoderActor


TRAIN_PATH = Path(__file__).resolve().parents[4] / "scripts" / "rsl_rl" / "train.py"


def _load_train_definitions(*names):
    """Load handoff helpers without launching the train.py Isaac application."""

    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    future_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {"Path": Path, "re": __import__("re"), "torch": torch}
    exec(compile(ast.Module(body=future_imports + definitions, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


CURRENT_OBS_DIM = 5
HISTORY_LEN = 8
HISTORY_DIM = 3
HISTORY_FLAT_DIM = HISTORY_LEN * HISTORY_DIM
TCN_LATENT_DIM = 3
ENV_PARAMS_DIM = 4
ENV_PARAMS_LATENT_DIM = 2


def _observations(batch_size: int = 4) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(batch_size, CURRENT_OBS_DIM + HISTORY_FLAT_DIM),
            "critic": torch.randn(batch_size, CURRENT_OBS_DIM + ENV_PARAMS_DIM),
        },
        batch_size=[batch_size],
    )


def _make_asymmetric_model(*, normalize_observations: bool = False) -> ActorCriticFootImuTcn:
    return ActorCriticFootImuTcn(
        obs=_observations(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=normalize_observations,
        critic_obs_normalization=normalize_observations,
        actor_hidden_dims=[7],
        critic_hidden_dims=[7],
        current_obs_dim=CURRENT_OBS_DIM,
        history_len=HISTORY_LEN,
        history_dim=HISTORY_DIM,
        tcn_channels=4,
        tcn_latent_dim=TCN_LATENT_DIM,
        tcn_kernel_size=3,
        asymmetric_actor_critic=True,
        env_params_dim=ENV_PARAMS_DIM,
        env_params_encoder_hidden_dims=[6],
        env_params_latent_dim=ENV_PARAMS_LATENT_DIM,
    )


def test_asymmetric_tcn_routes_history_only_to_actor_and_privileged_params_only_to_critic():
    model = _make_asymmetric_model()
    observations = _observations()

    assert hasattr(model, "actor_imu_encoder")
    assert hasattr(model, "critic_env_params_encoder")
    assert not hasattr(model, "critic_imu_encoder")
    assert model.act_inference(observations).shape == (4, 2)
    assert model.evaluate(observations).shape == (4, 1)

    changed_critic = observations.clone()
    changed_critic["critic"] += 100.0
    torch.testing.assert_close(model.act_inference(observations), model.act_inference(changed_critic))

    changed_policy = observations.clone()
    changed_policy["policy"] -= 100.0
    torch.testing.assert_close(model.evaluate(observations), model.evaluate(changed_policy))

    model.zero_grad()
    model.evaluate(observations).sum().backward()
    assert any(parameter.grad is not None for parameter in model.critic_env_params_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.actor_imu_encoder.parameters())


def test_asymmetric_critic_state_is_shape_compatible_with_complete_teacher_critic():
    student = _make_asymmetric_model(normalize_observations=True)
    teacher_obs = TensorDict(
        {
            "policy": torch.zeros(4, CURRENT_OBS_DIM + ENV_PARAMS_DIM),
            "critic": torch.zeros(4, CURRENT_OBS_DIM + ENV_PARAMS_DIM),
        },
        batch_size=[4],
    )
    teacher = EnvParamsConditionedEncoderActor(
        obs=teacher_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[7],
        critic_hidden_dims=[7],
        current_obs_dim=CURRENT_OBS_DIM,
        env_params_dim=ENV_PARAMS_DIM,
        env_params_encoder_hidden_dims=[6],
        env_params_latent_dim=ENV_PARAMS_LATENT_DIM,
    )

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    reusable_prefixes = ("critic_env_params_encoder.", "critic.", "critic_obs_normalizer.")
    reusable_keys = [key for key in teacher_state if key.startswith(reusable_prefixes)]

    assert reusable_keys
    for key in reusable_keys:
        assert key in student_state
        assert student_state[key].shape == teacher_state[key].shape

    for key in reusable_keys:
        student_state[key] = teacher_state[key].clone()
    assert student.load_state_dict(student_state, strict=True) is True
    critic_observations = TensorDict(
        {
            "policy": torch.randn(4, CURRENT_OBS_DIM + HISTORY_FLAT_DIM),
            "critic": torch.randn(4, CURRENT_OBS_DIM + ENV_PARAMS_DIM),
        },
        batch_size=[4],
    )
    torch.testing.assert_close(student.evaluate(critic_observations), teacher.evaluate(critic_observations))


def test_asymmetric_tcn_validates_critic_observation_width_and_disallows_shared_networks():
    observations = _observations()
    observations["critic"] = torch.zeros(4, CURRENT_OBS_DIM + ENV_PARAMS_DIM + 1)

    with pytest.raises(ValueError, match="Critic observation dim"):
        ActorCriticFootImuTcn(
            obs=observations,
            obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=2,
            current_obs_dim=CURRENT_OBS_DIM,
            history_len=HISTORY_LEN,
            history_dim=HISTORY_DIM,
            tcn_channels=4,
            tcn_latent_dim=TCN_LATENT_DIM,
            tcn_kernel_size=3,
            asymmetric_actor_critic=True,
            env_params_dim=ENV_PARAMS_DIM,
            env_params_latent_dim=ENV_PARAMS_LATENT_DIM,
        )

    with pytest.raises(ValueError, match="shared_networks=True"):
        ActorCriticFootImuTcn(
            obs=_observations(),
            obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=2,
            current_obs_dim=CURRENT_OBS_DIM,
            history_len=HISTORY_LEN,
            history_dim=HISTORY_DIM,
            tcn_channels=4,
            tcn_latent_dim=TCN_LATENT_DIM,
            tcn_kernel_size=3,
            shared_networks=True,
            asymmetric_actor_critic=True,
            env_params_dim=ENV_PARAMS_DIM,
            env_params_latent_dim=ENV_PARAMS_LATENT_DIM,
        )


def test_symmetric_tcn_remains_the_default_compatibility_path():
    observations = TensorDict(
        {"policy": torch.randn(4, CURRENT_OBS_DIM + HISTORY_FLAT_DIM)},
        batch_size=[4],
    )
    model = ActorCriticFootImuTcn(
        obs=observations,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=2,
        actor_hidden_dims=[7],
        critic_hidden_dims=[7],
        current_obs_dim=CURRENT_OBS_DIM,
        history_len=HISTORY_LEN,
        history_dim=HISTORY_DIM,
        tcn_channels=4,
        tcn_latent_dim=TCN_LATENT_DIM,
        tcn_kernel_size=3,
    )

    assert model.asymmetric_actor_critic is False
    assert hasattr(model, "critic_imu_encoder")
    assert not hasattr(model, "critic_env_params_encoder")
    assert model.act_inference(observations).shape == (4, 2)
    assert model.evaluate(observations).shape == (4, 1)


class _CombinedHistoryCfg:
    decimation = 2
    remove_c_close_vectors_from_observation = False
    base_observation_dim = 63
    foot_imu_history_policy_steps = 5
    joint_state_history_policy_steps = 5
    joint_imu_history_policy_steps = 5
    foot_imu_history_length = 10
    joint_state_history_length = 10
    joint_imu_history_length = 10
    asymmetric_actor_critic = True

    def __post_init__(self):
        self.base_observation_dim = 57 if self.remove_c_close_vectors_from_observation else 63
        self.foot_imu_history_length = self.decimation * self.foot_imu_history_policy_steps
        self.joint_state_history_length = self.decimation * self.joint_state_history_policy_steps
        if self.foot_imu_history_length != self.joint_state_history_length:
            raise ValueError("combined histories must match")
        self.joint_imu_history_length = self.foot_imu_history_length
        self.observation_space = self.base_observation_dim + 48 * self.joint_imu_history_length
        self.state_space = self.base_observation_dim + 16


def _teacher_state_for_config_test() -> dict[str, torch.Tensor]:
    state = {
        "actor.0.weight": torch.zeros(7, 65),
        "actor.0.bias": torch.zeros(7),
        "actor.2.weight": torch.zeros(2, 7),
        "actor.2.bias": torch.zeros(2),
        "critic.0.weight": torch.zeros(7, 65),
        "critic.0.bias": torch.zeros(7),
        "critic.2.weight": torch.zeros(1, 7),
        "critic.2.bias": torch.zeros(1),
        "critic_env_params_encoder.0.weight": torch.zeros(6, 16),
        "critic_env_params_encoder.0.bias": torch.zeros(6),
        "critic_env_params_encoder.2.weight": torch.zeros(8, 6),
        "critic_env_params_encoder.2.bias": torch.zeros(8),
        "log_std": torch.zeros(2),
    }
    for prefix in ("actor_obs_normalizer", "critic_obs_normalizer"):
        state[f"{prefix}._mean"] = torch.zeros(1, 73)
        state[f"{prefix}._var"] = torch.ones(1, 73)
        state[f"{prefix}._std"] = torch.ones(1, 73)
        state[f"{prefix}.count"] = torch.tensor(1.0)
    return state


def test_dagger_metadata_restores_legacy_current_obs_and_all_combined_history_windows():
    (
        resolve_path,
        checkpoint_state,
        infer_hidden,
        infer_io,
        configure_policy,
    ) = _load_train_definitions(
        "_resolve_existing_model_path",
        "_checkpoint_model_state_dict",
        "_infer_mlp_hidden_dims",
        "_infer_mlp_input_output_dims",
        "_configure_student_policy_from_dagger_adapter",
    )
    del resolve_path, checkpoint_state, infer_hidden, infer_io

    policy_cfg = SimpleNamespace(
        history_name="joint-state + foot-IMU",
        history_len=10,
        history_dim=48,
        imu_history_len=10,
        imu_dim=48,
        tcn_channels=32,
        tcn_latent_dim=8,
        tcn_kernel_size=5,
        tcn_activation="relu",
        current_obs_dim=63,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        asymmetric_actor_critic=True,
        env_params_dim=16,
        env_params_encoder_hidden_dims=[64, 32],
        env_params_latent_dim=8,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        noise_std_type="scalar",
    )
    env_cfg = _CombinedHistoryCfg()

    with tempfile.TemporaryDirectory() as tmp_dir:
        teacher_path = Path(tmp_dir) / "teacher.pt"
        adapter_path = Path(tmp_dir) / "adapter.pt"
        torch.save({"model_state_dict": _teacher_state_for_config_test()}, teacher_path)
        torch.save(
            {
                "teacher_checkpoint": str(teacher_path),
                "adapter_state_dict": {"weight": torch.zeros(1)},
                "history_normalizer_state_dict": {"_mean": torch.zeros(1, 576)},
                "layout": {
                    "kind": "joint_state_imu",
                    "history_len": 12,
                    "history_dim": 48,
                    "channels": 4,
                    "kernel_size": 3,
                    "activation": "elu",
                },
                "dims": {"current_obs_dim": 57, "env_params_dim": 16, "latent_dim": 8},
            },
            adapter_path,
        )

        configure_policy(env_cfg, policy_cfg, str(adapter_path))

    assert env_cfg.remove_c_close_vectors_from_observation is True
    assert env_cfg.base_observation_dim == 57
    assert env_cfg.foot_imu_history_policy_steps == 6
    assert env_cfg.joint_state_history_policy_steps == 6
    assert env_cfg.joint_imu_history_policy_steps == 6
    assert env_cfg.joint_imu_history_length == 12
    assert policy_cfg.current_obs_dim == 57
    assert policy_cfg.history_len == 12
    assert policy_cfg.actor_hidden_dims == [7]
    assert policy_cfg.critic_hidden_dims == [7]
    assert policy_cfg.actor_obs_normalization is True
    assert policy_cfg.critic_obs_normalization is True
    assert policy_cfg.noise_std_type == "log"


def _make_exact_handoff_fixture(tmp_dir: str):
    current_dim = 5
    env_params_dim = 4
    latent_dim = 2
    history_len = 8
    history_dim = 3
    history_flat_dim = history_len * history_dim
    num_actions = 2

    teacher_obs = TensorDict(
        {
            "policy": torch.randn(6, current_dim + env_params_dim),
            "critic": torch.randn(6, current_dim + env_params_dim),
        },
        batch_size=[6],
    )
    teacher = EnvParamsConditionedEncoderActor(
        obs=teacher_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[7],
        critic_hidden_dims=[7],
        current_obs_dim=current_dim,
        env_params_dim=env_params_dim,
        env_params_encoder_hidden_dims=[6],
        env_params_latent_dim=latent_dim,
    )
    teacher.update_normalization(teacher_obs)
    teacher_path = Path(tmp_dir) / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, teacher_path)

    adapter = FootImuTcnEncoder(
        history_len=history_len,
        imu_dim=history_dim,
        channels=4,
        latent_dim=latent_dim,
        kernel_size=3,
        activation="elu",
    )
    history_normalizer = EmpiricalNormalization(history_flat_dim)
    history_normalizer.update(torch.randn(6, history_flat_dim) + 4.0)
    student_actor = copy.deepcopy(teacher.actor)
    with torch.no_grad():
        for parameter in student_actor.parameters():
            parameter.add_(0.25)

    adapter_payload = {
        "teacher_checkpoint": str(teacher_path),
        "adapter_state_dict": adapter.state_dict(),
        "student_actor_state_dict": student_actor.state_dict(),
        "history_normalizer_state_dict": history_normalizer.state_dict(),
        "objective": {"finetune_student_actor": True, "action_loss_weight": 0.5},
        "layout": {"kind": "joint_state", "history_len": history_len, "history_dim": history_dim},
        "dims": {
            "current_obs_dim": current_dim,
            "env_params_dim": env_params_dim,
            "latent_dim": latent_dim,
            "history_sample_dim": history_dim,
        },
    }
    adapter_path = Path(tmp_dir) / "adapter.pt"
    torch.save(adapter_payload, adapter_path)

    student_obs = TensorDict(
        {
            "policy": torch.randn(6, current_dim + history_flat_dim),
            "critic": torch.randn(6, current_dim + env_params_dim),
        },
        batch_size=[6],
    )
    student = ActorCriticFootImuTcn(
        obs=student_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[7],
        critic_hidden_dims=[7],
        current_obs_dim=current_dim,
        history_len=history_len,
        history_dim=history_dim,
        tcn_channels=4,
        tcn_latent_dim=latent_dim,
        tcn_kernel_size=3,
        tcn_activation="elu",
        asymmetric_actor_critic=True,
        env_params_dim=env_params_dim,
        env_params_encoder_hidden_dims=[6],
        env_params_latent_dim=latent_dim,
    )
    runner = SimpleNamespace(alg=SimpleNamespace(policy=student))
    return runner, teacher, adapter, student_actor, adapter_payload, adapter_path


def test_dagger_handoff_copies_actor_adapter_and_complete_privileged_critic_exactly():
    (
        resolve_path,
        checkpoint_state,
        copy_normalizer,
        copy_race_normalizer,
        initialize_student,
    ) = _load_train_definitions(
        "_resolve_existing_model_path",
        "_checkpoint_model_state_dict",
        "_copy_normalizer_state",
        "_copy_race_dagger_normalizer_state",
        "_initialize_student_from_dagger_adapter",
    )
    del resolve_path, checkpoint_state, copy_normalizer, copy_race_normalizer

    with tempfile.TemporaryDirectory() as tmp_dir:
        runner, teacher, adapter, student_actor, _, adapter_path = _make_exact_handoff_fixture(tmp_dir)
        initialize_student(runner, str(adapter_path))

        policy_state = runner.alg.policy.state_dict()
        for key, expected in adapter.state_dict().items():
            torch.testing.assert_close(policy_state[f"actor_imu_encoder.{key}"], expected)
        for key, expected in student_actor.state_dict().items():
            torch.testing.assert_close(policy_state[f"actor.{key}"], expected)
        for prefix in ("critic_env_params_encoder.", "critic.", "critic_obs_normalizer."):
            for key, expected in teacher.state_dict().items():
                if key.startswith(prefix):
                    torch.testing.assert_close(policy_state[key], expected)


def test_dagger_handoff_rejects_partial_finetuned_actor_transfer():
    (
        resolve_path,
        checkpoint_state,
        copy_normalizer,
        copy_race_normalizer,
        initialize_student,
    ) = _load_train_definitions(
        "_resolve_existing_model_path",
        "_checkpoint_model_state_dict",
        "_copy_normalizer_state",
        "_copy_race_dagger_normalizer_state",
        "_initialize_student_from_dagger_adapter",
    )
    del resolve_path, checkpoint_state, copy_normalizer, copy_race_normalizer

    with tempfile.TemporaryDirectory() as tmp_dir:
        runner, _, _, _, payload, _ = _make_exact_handoff_fixture(tmp_dir)
        first_key = next(iter(payload["student_actor_state_dict"]))
        payload["student_actor_state_dict"][first_key] = torch.zeros(1)
        bad_adapter_path = Path(tmp_dir) / "bad_adapter.pt"
        torch.save(payload, bad_adapter_path)

        with pytest.raises(RuntimeError, match="refusing a partial policy"):
            initialize_student(runner, str(bad_adapter_path))

        runner, _, _, _, payload, _ = _make_exact_handoff_fixture(tmp_dir)
        payload["student_actor_state_dict"].pop(next(iter(payload["student_actor_state_dict"])))
        truncated_adapter_path = Path(tmp_dir) / "truncated_adapter.pt"
        torch.save(payload, truncated_adapter_path)

        with pytest.raises(RuntimeError, match="actor transfer did not initialize"):
            initialize_student(runner, str(truncated_adapter_path))
