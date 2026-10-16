# SPDX-License-Identifier: BSD-3-Clause
"""Check the Borinot PPO action-std bound without importing Isaac Sim."""
import importlib.util
from pathlib import Path

import pytest
import torch
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

PATH = Path(__file__).resolve().parents[2] / "borinot" / "agents" / "bounded_std_actor_critic.py"
spec = importlib.util.spec_from_file_location("bounded_std_actor_critic", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
BoundedStdActorCritic = module.BoundedStdActorCritic

OBS = TensorDict({"policy": torch.randn(4, 3)}, batch_size=[4])
GROUPS = {"policy": ["policy"], "critic": ["policy"]}
NETS = dict(actor_hidden_dims=[8], critic_hidden_dims=[8], init_noise_std=1.0)


@pytest.mark.parametrize("std_type,param", [("log", "log_std"), ("scalar", "std")])
def test_std_is_projected_into_bounds_and_can_shrink_again(std_type, param):
    policy = BoundedStdActorCritic(OBS, GROUPS, 2, noise_std_type=std_type, min_action_std=0.1, max_action_std=2.0, **NETS)
    with torch.no_grad():  # e.g. a resumed runaway checkpoint
        getattr(policy, param).copy_(torch.tensor([30.0, -30.0]) if std_type == "log" else torch.tensor([1e9, 1e-9]))
    policy.act(OBS)
    torch.testing.assert_close(policy.action_std[0], torch.tensor([2.0, 0.1]))

    optimizer = torch.optim.SGD([getattr(policy, param)], lr=1.0)
    (-policy.entropy.mean()).backward()  # entropy bonus pushes the std up
    assert (getattr(policy, param).grad != 0).all()  # gradient still reaches the parameter at the bound
    optimizer.step()
    policy.act(OBS)
    assert policy.action_std[0, 0] <= 2.0 + 1e-6

    optimizer.zero_grad()
    policy.entropy.mean().backward()  # a loss that prefers less noise
    optimizer.step()
    policy.act(OBS)
    assert policy.action_std[0, 0] < 2.0


def test_same_checkpoint_keys_as_plain_actor_critic():
    bounded = BoundedStdActorCritic(OBS, GROUPS, 2, noise_std_type="log", min_action_std=1e-6, max_action_std=13.0, **NETS)
    plain = ActorCritic(OBS, GROUPS, 2, noise_std_type="log", **NETS)
    assert bounded.state_dict().keys() == plain.state_dict().keys()
    bounded.load_state_dict(plain.state_dict())


def test_state_dependent_std_is_rejected():
    with pytest.raises(ValueError):
        BoundedStdActorCritic(OBS, GROUPS, 2, state_dependent_std=True, min_action_std=0.1, max_action_std=2.0, **NETS)
