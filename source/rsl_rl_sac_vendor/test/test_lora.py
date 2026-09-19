# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard the low-rank adapters used for MJX SAC fine-tuning.

Three properties make an adapter useful, and each one fails quietly if broken:

* a freshly wrapped network must reproduce the pretrained network exactly, or fine-tuning
  starts from a policy that was never trained;
* only the adapter may receive gradients, or the "frozen" base drifts anyway; and
* merging must be exact, or the checkpoint does not describe the policy that was trained.
"""

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl_sac.models import SACActorModel
from rsl_rl_sac.modules import LoRALinear, apply_lora, merged_state_dict, selected_layer_indices

OBS_DIM = 6
ACTION_DIM = 3


def make_actor():
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(4, OBS_DIM)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    return SACActorModel(obs, groups, "actor", ACTION_DIM, hidden_dims=[16, 8], obs_normalization=True)


def sample_obs(batch=5):
    return TensorDict({"policy": torch.randn(batch, OBS_DIM)}, batch_size=[batch])


def test_wrapped_network_starts_identical_to_the_pretrained_one():
    actor = make_actor()
    obs = sample_obs()
    before = actor(obs)

    apply_lora(actor, rank=4, alpha=4.0)

    torch.testing.assert_close(actor(obs), before)


def test_only_the_adapter_receives_gradients():
    actor = make_actor()
    apply_lora(actor, rank=4, alpha=4.0)
    base_weights = {
        name: parameter.clone()
        for name, parameter in actor.named_parameters()
        if "lora_" not in name
    }

    actor(sample_obs()).sum().backward()
    for name, parameter in actor.named_parameters():
        if "lora_" in name:
            assert parameter.grad is not None, name
        else:
            assert parameter.requires_grad is False, name
            assert parameter.grad is None, name

    # A step on the trainable parameters must leave the pretrained weights untouched.
    trainable = [p for p in actor.parameters() if p.requires_grad]
    torch.optim.Adam(trainable, lr=1e-2).step()
    for name, parameter in actor.named_parameters():
        if "lora_" not in name:
            torch.testing.assert_close(parameter, base_weights[name])


def train_adapter(actor):
    """Move the adapter off its zero initialization so merging has something to fold in."""
    for name, parameter in actor.named_parameters():
        if "lora_b" in name:
            torch.nn.init.normal_(parameter, std=0.1)


def test_merged_checkpoint_reproduces_the_adapted_policy():
    """A LoRA run's checkpoint has to stay a plain SAC checkpoint describing the trained policy."""
    actor = make_actor()
    apply_lora(actor, rank=4, alpha=8.0)
    train_adapter(actor)
    obs = sample_obs()
    adapted = actor(obs)

    restored = make_actor()
    restored.load_state_dict(merged_state_dict(actor), strict=True)

    torch.testing.assert_close(restored(obs), adapted, atol=1e-6, rtol=1e-5)


def test_merging_works_after_a_forward_pass():
    """The actor caches its output distribution, which rules out copying the module to merge it."""
    actor = make_actor()
    apply_lora(actor, rank=2, alpha=2.0)
    actor(sample_obs())

    assert set(merged_state_dict(actor)) == set(make_actor().state_dict())


def test_merging_a_subset_leaves_untouched_layers_alone():
    actor = make_actor()
    apply_lora(actor, rank=2, alpha=2.0, layers="output")
    train_adapter(actor)

    state = merged_state_dict(actor)

    assert set(state) == set(make_actor().state_dict())
    torch.testing.assert_close(state["mlp.0.weight"], actor.mlp[0].weight)


@pytest.mark.parametrize(
    "mode,expected", [("all", 3), ("input", 1), ("output", 1), ("input_and_output", 2)]
)
def test_layer_selection_modes_wrap_the_expected_count(mode, expected):
    actor = make_actor()

    assert apply_lora(actor, rank=2, alpha=2.0, layers=mode) == expected
    assert sum(isinstance(m, LoRALinear) for m in actor.modules()) == expected


def test_layer_selection_matches_the_ppo_convention():
    assert selected_layer_indices(4, "all") == (0, 1, 2, 3)
    assert selected_layer_indices(4, "input") == (0,)
    assert selected_layer_indices(4, "output") == (3,)
    assert selected_layer_indices(4, "input_and_output") == (0, 3)
    # A one-layer network must not list the same layer twice.
    assert selected_layer_indices(1, "input_and_output") == (0,)


def test_adapter_trains_far_fewer_parameters():
    actor = make_actor()
    dense = sum(p.numel() for p in actor.parameters() if p.requires_grad)

    apply_lora(actor, rank=2, alpha=2.0)

    adapted = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    assert adapted < dense / 2


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(nn.Linear(3, 3), rank=0, alpha=1.0)
    with pytest.raises(ValueError, match="layers must be one of"):
        apply_lora(make_actor(), rank=2, alpha=2.0, layers="middle")
    with pytest.raises(ValueError, match="no linear layers"):
        apply_lora(nn.Sequential(nn.ReLU()), rank=2, alpha=2.0)
