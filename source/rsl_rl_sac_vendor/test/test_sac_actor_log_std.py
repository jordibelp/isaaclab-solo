# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from tensordict import TensorDict

from rsl_rl_sac.models import SACActorModel


def _actor(state_dependent_std: bool) -> SACActorModel:
    obs = TensorDict({"policy": torch.zeros(2, 4)}, batch_size=[2])
    return SACActorModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=3,
        hidden_dims=[8],
        init_noise_std=0.15,
        state_dependent_std=state_dependent_std,
    )


def test_state_independent_log_std_is_a_learned_parameter():
    actor = _actor(state_dependent_std=False)
    obs = TensorDict({"policy": torch.stack((torch.zeros(4), torch.ones(4)))}, batch_size=[2])

    actor.sample_action_logp(obs)

    assert actor.mlp[-1].out_features == 3
    assert actor.log_std.shape == (3,)
    torch.testing.assert_close(actor.output_std[0], actor.output_std[1])
    torch.testing.assert_close(actor.output_std[0], torch.full((3,), 0.15), atol=1.0e-6, rtol=0.0)


def test_state_dependent_log_std_is_a_second_actor_output_head():
    actor = _actor(state_dependent_std=True)
    obs = TensorDict({"policy": torch.stack((torch.zeros(4), torch.ones(4)))}, batch_size=[2])

    actor.sample_action_logp(obs)

    assert actor.mlp[-2].out_features == 6
    assert not hasattr(actor, "log_std")
    torch.testing.assert_close(actor.output_std[0], actor.output_std[1])

    # The paper initializes this head with zero weights, so it starts state-independent.
    # Once its weights learn a nonzero value, sigma must vary with the observation.
    with torch.no_grad():
        actor.mlp[-2].weight[3:, 0] = 0.5
    actor.sample_action_logp(obs)
    assert not torch.allclose(actor.output_std[0], actor.output_std[1])


def test_deterministic_export_supports_both_log_std_parameterizations():
    obs = torch.randn(2, 4)
    for state_dependent_std in (False, True):
        exported = torch.jit.script(_actor(state_dependent_std).as_jit())
        output = exported(obs)
        assert output.shape == (2, 3)


def test_default_actor_initialization_remains_action_midpoint_centered():
    actor = _actor(state_dependent_std=True)
    actor.action_bias.copy_(torch.tensor([0.4, -0.8, 0.0]))
    actor.action_range.copy_(torch.tensor([1.4, 2.6, 0.9]))
    mean_bias_before = actor.mlp[-2].bias[:3].detach().clone()

    torch.testing.assert_close(mean_bias_before, torch.zeros(3))
    torch.testing.assert_close(actor._squash_and_scale(mean_bias_before), actor.action_bias)


def test_q_offset_centering_initializes_mean_for_zero_action():
    for state_dependent_std in (False, True):
        actor = _actor(state_dependent_std=state_dependent_std)
        actor.action_bias.copy_(torch.tensor([-0.4, 0.8, 0.0]))
        actor.action_range.copy_(torch.tensor([1.4, 2.6, 0.9]))
        log_std_before = None
        if state_dependent_std:
            log_std_before = actor.mlp[-2].bias[3:].detach().clone()

        actor.initialize_mean_head_for_action(torch.zeros(3))

        mean_bias = actor.mlp[-2].bias[:3] if state_dependent_std else actor.mlp[-1].bias
        expected_latent_mean = torch.atanh(-actor.action_bias / actor.action_range)
        torch.testing.assert_close(mean_bias, expected_latent_mean)
        torch.testing.assert_close(actor._squash_and_scale(mean_bias), torch.zeros(3), atol=1.0e-7, rtol=0.0)
        if state_dependent_std:
            torch.testing.assert_close(actor.mlp[-2].bias[3:], log_std_before)
