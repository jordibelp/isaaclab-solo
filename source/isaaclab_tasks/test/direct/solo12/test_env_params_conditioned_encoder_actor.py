# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict


def _load_actor_class():
    """Load the focused agents package without importing Isaac Sim task registration."""

    agents_dir = (
        Path(__file__).resolve().parents[3]
        / "isaaclab_tasks"
        / "direct"
        / "solo12_race"
        / "agents"
    )
    package_name = "_solo12_race_agents_test"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(agents_dir)]
        sys.modules[package_name] = package

    module_name = f"{package_name}.env_params_conditioned_encoder_actor"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name,
            agents_dir / "env_params_conditioned_encoder_actor.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].EnvParamsConditionedEncoderActor


EnvParamsConditionedEncoderActor = _load_actor_class()


def _make_model(
    *,
    share_latent: bool = False,
    normalize_observations: bool = False,
) -> EnvParamsConditionedEncoderActor:
    obs = TensorDict(
        {
            "policy": torch.zeros(4, 5),
            "critic": torch.zeros(4, 5),
        },
        batch_size=[4],
    )
    return EnvParamsConditionedEncoderActor(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=2,
        actor_obs_normalization=normalize_observations,
        critic_obs_normalization=normalize_observations,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        current_obs_dim=3,
        env_params_dim=2,
        env_params_encoder_hidden_dims=[4],
        env_params_latent_dim=2,
        actor_critic_share_latent_encoding=share_latent,
    )


def _encoder_state_by_suffix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def test_env_param_encoders_are_independent_by_default():
    model = _make_model()

    assert model.actor_critic_share_latent_encoding is False
    assert model.actor_env_params_encoder is not model.critic_env_params_encoder
    actor_parameter_ids = {id(parameter) for parameter in model.actor_env_params_encoder.parameters()}
    critic_parameter_ids = {id(parameter) for parameter in model.critic_env_params_encoder.parameters()}
    assert actor_parameter_ids.isdisjoint(critic_parameter_ids)


def test_shared_env_param_encoder_is_one_optimizer_parameter_set_and_round_trips():
    model = _make_model(share_latent=True)

    assert model.actor_env_params_encoder is model.critic_env_params_encoder
    encoder_parameters = list(model.actor_env_params_encoder.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    optimizer_parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]
    for encoder_parameter in encoder_parameters:
        assert sum(parameter is encoder_parameter for parameter in optimizer_parameters) == 1

    observations = TensorDict(
        {
            "policy": torch.randn(4, 5),
            "critic": torch.randn(4, 5),
        },
        batch_size=[4],
    )
    encoder_before_step = [parameter.detach().clone() for parameter in encoder_parameters]
    loss = model.act_inference(observations).square().sum() + model.evaluate(observations).square().sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(encoder_before_step, model.actor_env_params_encoder.parameters())
    )

    state_dict = model.state_dict()
    actor_state = _encoder_state_by_suffix(state_dict, "actor_env_params_encoder.")
    critic_state = _encoder_state_by_suffix(state_dict, "critic_env_params_encoder.")
    assert actor_state.keys() == critic_state.keys()
    for suffix in actor_state:
        torch.testing.assert_close(actor_state[suffix], critic_state[suffix])
        assert actor_state[suffix].data_ptr() == critic_state[suffix].data_ptr()

    restored = _make_model(share_latent=True)
    assert restored.load_state_dict(state_dict) is True
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1.0e-2)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_state = restored.state_dict()
    for key, expected in state_dict.items():
        torch.testing.assert_close(restored_state[key], expected)


def test_shared_encoder_keeps_branch_local_observation_normalizers():
    model = _make_model(share_latent=True, normalize_observations=True)
    assert model.actor_env_params_encoder is model.critic_env_params_encoder
    assert model.actor_obs_normalizer is not model.critic_obs_normalizer

    observations = TensorDict(
        {
            "policy": torch.zeros(4, 5),
            "critic": torch.full((4, 5), 3.0),
        },
        batch_size=[4],
    )
    model.update_normalization(observations)

    assert model.actor_obs_normalizer.count.item() == 4
    assert model.critic_obs_normalizer.count.item() == 4
    torch.testing.assert_close(model.actor_obs_normalizer.mean, torch.zeros(5))
    torch.testing.assert_close(model.critic_obs_normalizer.mean, torch.full((5,), 3.0))


def test_loading_separate_encoders_into_shared_model_preserves_actor_encoder(capsys):
    separate_model = _make_model()
    with torch.no_grad():
        for parameter in separate_model.actor_env_params_encoder.parameters():
            parameter.fill_(0.25)
        for parameter in separate_model.critic_env_params_encoder.parameters():
            parameter.fill_(-0.75)

    shared_model = _make_model(share_latent=True)
    assert shared_model.load_state_dict(separate_model.state_dict()) is False

    for parameter in shared_model.actor_env_params_encoder.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))
    assert "preserving the actor encoder" in capsys.readouterr().out


def test_shared_checkpoint_requires_matching_topology_for_safe_optimizer_resume():
    shared_model = _make_model(share_latent=True)
    separate_model = _make_model()

    with pytest.raises(ValueError, match="sharing topology"):
        separate_model.load_state_dict(shared_model.state_dict())


def test_legacy_separate_checkpoint_without_topology_marker_still_resumes_exactly():
    source = _make_model()
    legacy_state = source.state_dict()
    del legacy_state["_sharing_topology_marker"]

    restored = _make_model()
    assert restored.load_state_dict(legacy_state) is True
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)
