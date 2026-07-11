from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from observation_permutation import (  # noqa: E402
    ObservationPermutationVecEnv,
    compute_permutation_aware_symmetry,
)


class FakeVecEnv(VecEnv):
    def __init__(self) -> None:
        self.num_envs = 3
        self.num_actions = 2
        self.max_episode_length = 100
        self.device = "cpu"
        self.cfg = object()
        self.unwrapped = self
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.obs = TensorDict(
            {
                "policy": torch.arange(18, dtype=torch.float32).reshape(3, 6),
                "critic": torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100.0,
            },
            batch_size=[self.num_envs],
        )

    def get_observations(self) -> TensorDict:
        return self.obs.clone()

    def step(self, actions: torch.Tensor):
        rewards = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.long)
        return self.get_observations(), rewards, dones, {}


def make_env(*, duration: int = 2, rounds: int = 3, symmetry_fn=None):
    return ObservationPermutationVecEnv(
        FakeVecEnv(),
        round_duration=duration,
        num_rounds=rounds,
        seed=123,
        symmetry_fn=symmetry_fn,
    )


def test_mapping_is_fixed_and_shared_across_parallel_environments() -> None:
    env = make_env()
    first = env.get_observations()
    second = env.get_observations()

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(env.unpermute_observations(first), env.env.get_observations())
    # A feature permutation never mixes the environment/batch dimension.
    torch.testing.assert_close(first["policy"][1] - first["policy"][0], torch.full((6,), 6.0))


def test_mapping_changes_only_after_a_complete_round() -> None:
    env = make_env(duration=2, rounds=3)
    canonical = env.env.get_observations()
    live_obs = env.get_observations()
    initial = live_obs.clone()

    assert not env.finish_learning_iteration(0, live_obs)
    torch.testing.assert_close(live_obs, initial)

    assert env.finish_learning_iteration(1, live_obs)
    assert env.round_index == 1
    assert not torch.equal(live_obs["policy"], initial["policy"])
    torch.testing.assert_close(env.unpermute_observations(live_obs), canonical)
    torch.testing.assert_close(live_obs, env.get_observations())

    # Once the configured final round is reached, the last mapping remains fixed.
    assert env.finish_learning_iteration(3, live_obs)
    final_mapping = live_obs.clone()
    assert not env.finish_learning_iteration(100, live_obs)
    torch.testing.assert_close(live_obs, final_mapping)


def test_resume_mapping_is_deterministic() -> None:
    first = make_env(duration=2, rounds=3)
    resumed = make_env(duration=2, rounds=3)
    first.set_learning_iteration(4)
    resumed.set_learning_iteration(4)

    assert first.round_index == resumed.round_index == 2
    torch.testing.assert_close(first.get_observations(), resumed.get_observations())


def test_symmetry_sees_semantic_order_then_reapplies_permutation() -> None:
    def fake_symmetry(env, obs, actions, obs_type):
        del env, obs_type
        symmetric_obs = TensorDict(
            {
                "policy": torch.cat((obs["policy"], -obs["policy"]), dim=0),
                "critic": torch.cat((obs["critic"], -obs["critic"]), dim=0),
            },
            batch_size=[2 * obs.batch_size[0]],
        )
        symmetric_actions = None if actions is None else torch.cat((actions, -actions), dim=0)
        return symmetric_obs, symmetric_actions

    env = make_env(symmetry_fn=fake_symmetry)
    permuted_obs = env.get_observations()
    actions = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    symmetric_obs, symmetric_actions = compute_permutation_aware_symmetry(
        env=env, obs=permuted_obs, actions=actions
    )

    semantic = env.unpermute_observations(symmetric_obs)
    canonical = env.env.get_observations()
    torch.testing.assert_close(semantic["policy"][:3], canonical["policy"])
    torch.testing.assert_close(semantic["policy"][3:], -canonical["policy"])
    torch.testing.assert_close(symmetric_actions, torch.cat((actions, -actions), dim=0))


@pytest.mark.parametrize("duration,rounds", [(0, 2), (2, 0), (-1, 2), (2, -1)])
def test_invalid_schedule_is_rejected(duration: int, rounds: int) -> None:
    with pytest.raises(ValueError):
        make_env(duration=duration, rounds=rounds)
