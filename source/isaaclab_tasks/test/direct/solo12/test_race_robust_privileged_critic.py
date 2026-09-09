# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU checks for the feedforward robust actor / teacher-compatible critic ablation."""

import copy
import importlib

import pytest
import torch
from tensordict import TensorDict

from test_env_params_conditioned_encoder_actor import EnvParamsConditionedEncoderActor


SharedActorCritic = importlib.import_module(
    EnvParamsConditionedEncoderActor.__module__.rsplit(".", 1)[0] + ".shared_actor_critic"
).SharedActorCritic


def _observations():
    current = torch.randn(8, 63)
    return TensorDict(
        {"policy": current, "critic": torch.cat((current, torch.randn(8, 16)), dim=-1)}, batch_size=[8]
    )


def _model(obs, *, asymmetric=True, **kwargs):
    return SharedActorCritic(
        obs=obs,
        obs_groups={"policy": ["policy"], "critic": ["critic" if asymmetric else "policy"]},
        num_actions=12,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        asymmetric_actor_critic=asymmetric,
        **kwargs,
    )


def test_privilege_changes_values_but_not_actor_and_critic_gradients_stay_separate():
    obs = _observations()
    model = _model(obs)
    changed = obs.clone()
    changed["critic"][:, 63:] += 10.0

    torch.testing.assert_close(model.act_inference(obs), model.act_inference(changed), rtol=0, atol=0)
    assert not torch.allclose(model.evaluate(obs), model.evaluate(changed))
    assert not hasattr(model, "actor_env_params_encoder")
    assert model.critic_env_params_encoder is not None

    loss = model.evaluate(obs).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.critic_env_params_encoder.parameters())
    assert all(p.grad is None for p in model.actor.parameters())


def test_critic_is_numerically_identical_to_teacher_after_copying_critic_state():
    obs = _observations()
    model = _model(obs)
    teacher_obs = TensorDict({"policy": obs["critic"]}, batch_size=[8])
    teacher = EnvParamsConditionedEncoderActor(
        obs=teacher_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=12,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        critic_obs_normalization=True,
    )
    teacher.update_normalization(teacher_obs)
    model.critic_env_params_encoder.load_state_dict(teacher.critic_env_params_encoder.state_dict(), strict=True)
    model.critic.load_state_dict(teacher.critic.state_dict(), strict=True)
    model.critic_obs_normalizer.load_state_dict(teacher.critic_obs_normalizer.state_dict(), strict=True)
    torch.testing.assert_close(model.evaluate(obs), teacher.evaluate(teacher_obs), rtol=0, atol=0)


@pytest.mark.parametrize("state_dependent_std", [False, True])
def test_default_actor_initialization_is_unchanged_and_inference_needs_no_critic(state_dependent_std):
    obs = _observations()
    torch.manual_seed(99)
    baseline = _model(obs, asymmetric=False, state_dependent_std=state_dependent_std)
    torch.manual_seed(99)
    asymmetric = _model(obs, state_dependent_std=state_dependent_std)
    policy_only = obs.select("policy")
    torch.testing.assert_close(baseline.act_inference(policy_only), asymmetric.act_inference(policy_only), rtol=0, atol=0)
    assert not any("env_params_encoder" in key for key in baseline.state_dict())


def test_asymmetric_checkpoint_and_optimizer_resume_exactly():
    obs = _observations()
    model = _model(obs)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.update_normalization(obs)
    loss = model.act_inference(obs).square().mean() + model.evaluate(obs).square().mean()
    loss.backward()
    optimizer.step()
    restored = _model(obs)
    assert restored.load_state_dict(copy.deepcopy(model.state_dict())) is True
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    torch.testing.assert_close(model.act_inference(obs), restored.act_inference(obs), rtol=0, atol=0)
    torch.testing.assert_close(model.evaluate(obs), restored.evaluate(obs), rtol=0, atol=0)
    for candidate, candidate_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        candidate_optimizer.zero_grad()
        (candidate.act_inference(obs).square().mean() + candidate.evaluate(obs).square().mean()).backward()
        candidate_optimizer.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_asymmetry_rejects_shared_networks_and_missing_privileged_observations():
    with pytest.raises(ValueError, match="shared_networks"):
        _model(_observations(), shared_networks=True)
    obs = _observations()
    obs["critic"] = obs["policy"]
    with pytest.raises(ValueError, match="privileged"):
        _model(obs)
