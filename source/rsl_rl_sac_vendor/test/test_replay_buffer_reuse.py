# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Guard replay-data reuse for sim-to-online fine-tuning (arXiv:2602.20220).

Two things have to hold for retained replay to mean anything:

* a snapshot must reproduce the transitions that were actually in the buffer, in order and
  without the zero padding or the overwritten slots of the circular storage; and
* a mini-batch must contain the requested share of retained samples, with that share moving
  along the configured anneal.

Both fail silently otherwise: training still runs, it just learns from padding or from the
wrong mixture.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.models import SACActorModel, SACCriticModel
from rsl_rl_sac.runners import OffPolicyRunner
from rsl_rl_sac.storage import MixedReplayBuffer, ReplayBuffer

OBS_DIM = 3
ACTION_DIM = 2


def make_buffer(num_envs=2, capacity_per_env=4, obs_dim=OBS_DIM, n_steps=1, gamma=0.9):
    obs = TensorDict({"policy": torch.zeros(num_envs, obs_dim)}, batch_size=[num_envs])
    return ReplayBuffer(
        num_envs,
        1,
        obs,
        (ACTION_DIM,),
        "cpu",
        buffer_size=num_envs * capacity_per_env,
        n_steps=n_steps,
        gamma=gamma,
    )


def push(buffer, value, done=0.0, bootstrap=0.0):
    """Append one transition per environment, tagged with ``value`` everywhere.

    ``buffer`` may be a mixture, in which case the write still goes through its own
    ``add_transition`` while the shapes come from the storage behind it.
    """
    storage = getattr(buffer, "online", buffer)
    num_envs = storage.num_envs
    obs_dim = storage.observations["policy"].shape[-1]
    transition = ReplayBuffer.Transition()
    transition.observations = TensorDict(
        {"policy": torch.full((num_envs, obs_dim), float(value))}, batch_size=[num_envs]
    )
    transition.next_observations = TensorDict(
        {"policy": torch.full((num_envs, obs_dim), float(value) + 0.5)}, batch_size=[num_envs]
    )
    transition.actions = torch.full((num_envs, ACTION_DIM), float(value))
    transition.rewards = torch.full((num_envs,), float(value))
    transition.dones = torch.full((num_envs,), float(done))
    transition.bootstrap = torch.full((num_envs,), float(bootstrap))
    buffer.add_transition(transition)


def stored_values(buffer):
    """The first observation feature of every slot, per environment."""
    return buffer.observations["policy"][:, :, 0]


def test_snapshot_of_partial_buffer_drops_unwritten_slots(tmp_path):
    buffer = make_buffer(capacity_per_env=4)
    for value in (1, 2, 3):
        push(buffer, value)

    info = buffer.save_snapshot(tmp_path / "replay_buffer.pt")
    loaded = ReplayBuffer.load_snapshot(tmp_path / "replay_buffer.pt", "cpu")

    assert info["transitions"] == 2 * 3
    assert loaded.buffer_size == 3
    assert loaded.num_transitions == 3
    assert stored_values(loaded).tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]


def test_snapshot_of_wrapped_buffer_keeps_newest_transitions_in_order(tmp_path):
    buffer = make_buffer(capacity_per_env=4)
    for value in range(6):
        push(buffer, value)
    # The circular buffer now holds 2..5 with the write pointer parked mid-array.
    assert buffer.step == 2
    assert stored_values(buffer)[0].tolist() == [4.0, 5.0, 2.0, 3.0]

    buffer.save_snapshot(tmp_path / "replay_buffer.pt")
    loaded = ReplayBuffer.load_snapshot(tmp_path / "replay_buffer.pt", "cpu")

    assert stored_values(loaded)[0].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert loaded.step == 0


def test_snapshot_round_trip_preserves_every_field(tmp_path):
    buffer = make_buffer(capacity_per_env=3)
    push(buffer, 1, done=0.0, bootstrap=0.0)
    push(buffer, 2, done=1.0, bootstrap=1.0)
    push(buffer, 3, done=1.0, bootstrap=0.0)

    buffer.save_snapshot(tmp_path / "replay_buffer.pt")
    loaded = ReplayBuffer.load_snapshot(tmp_path / "replay_buffer.pt", "cpu")

    for field in ("actions", "rewards", "dones", "bootstrap"):
        assert torch.equal(getattr(loaded, field), getattr(buffer, field)), field
    assert torch.equal(loaded.next_observations["policy"], buffer.next_observations["policy"])


def test_loaded_snapshot_uses_current_n_steps_and_gamma(tmp_path):
    buffer = make_buffer(capacity_per_env=4, n_steps=1, gamma=0.9)
    for value in range(4):
        push(buffer, value)
    buffer.save_snapshot(tmp_path / "replay_buffer.pt")

    loaded = ReplayBuffer.load_snapshot(tmp_path / "replay_buffer.pt", "cpu", n_steps=3, gamma=0.5)

    assert (loaded.n_steps, loaded.gamma) == (3, 0.5)
    # An n-step window must stay inside the snapshot, so only starts 0 and 1 are usable.
    _, starts = loaded._generate_valid_indices()
    assert sorted(set(starts.tolist())) == [0, 1]


def test_empty_buffer_refuses_to_snapshot(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        make_buffer().save_snapshot(tmp_path / "replay_buffer.pt")


def test_snapshot_write_is_atomic(tmp_path):
    path = tmp_path / "replay_buffer.pt"
    buffer = make_buffer(capacity_per_env=2)
    push(buffer, 1)
    buffer.save_snapshot(path)

    assert not (tmp_path / "replay_buffer.pt.tmp").exists()
    assert path.exists()


def make_mixture(offline_fraction=0.5, **kwargs):
    online = make_buffer(capacity_per_env=4)
    offline = make_buffer(capacity_per_env=4)
    for _ in range(4):
        push(online, 1)
        push(offline, 0)
    return MixedReplayBuffer(
        online, offline, initial_offline_fraction=offline_fraction, final_offline_fraction=0.0, **kwargs
    )


def test_mini_batch_holds_the_requested_share_of_retained_samples():
    mixture = make_mixture(offline_fraction=0.25)

    batches = list(mixture.mini_batch_generator(num_mini_batch=3, mini_batch_size=8))

    assert len(batches) == 3
    for obs, *_ in batches:
        values = obs["policy"][:, 0]
        assert values.numel() == 8
        assert int((values == 0.0).sum()) == 2
        assert int((values == 1.0).sum()) == 6


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_degenerate_shares_sample_a_single_buffer(fraction):
    mixture = make_mixture(offline_fraction=fraction)

    obs, *_ = next(mixture.mini_batch_generator(num_mini_batch=1, mini_batch_size=8))

    expected = 0.0 if fraction == 1.0 else 1.0
    assert torch.all(obs["policy"][:, 0] == expected)


def test_empty_online_buffer_falls_back_to_retained_data():
    online = make_buffer(capacity_per_env=4)
    offline = make_buffer(capacity_per_env=4)
    for _ in range(4):
        push(offline, 0)
    mixture = MixedReplayBuffer(online, offline, initial_offline_fraction=0.5)

    obs, *_ = next(mixture.mini_batch_generator(num_mini_batch=1, mini_batch_size=8))

    assert torch.all(obs["policy"][:, 0] == 0.0)


def test_offline_share_anneals_linearly_and_clamps():
    mixture = make_mixture(offline_fraction=0.5, anneal_iterations=10)

    assert mixture.set_iteration(0) == pytest.approx(0.5)
    assert mixture.set_iteration(5) == pytest.approx(0.25)
    assert mixture.set_iteration(10) == pytest.approx(0.0)
    assert mixture.set_iteration(99) == pytest.approx(0.0)


def test_transitions_are_written_only_to_the_online_buffer():
    mixture = make_mixture()
    offline_before = stored_values(mixture.offline).clone()

    push(mixture, 7)

    assert torch.equal(stored_values(mixture.offline), offline_before)
    assert 7.0 in stored_values(mixture.online)


def test_mismatched_observation_layout_is_rejected():
    online = make_buffer(obs_dim=OBS_DIM)
    offline = make_buffer(obs_dim=OBS_DIM + 1)

    with pytest.raises(ValueError, match="observation layout"):
        MixedReplayBuffer(online, offline)


def test_invalid_shares_are_rejected():
    online, offline = make_buffer(), make_buffer()

    with pytest.raises(ValueError, match=r"initial_offline_fraction"):
        MixedReplayBuffer(online, offline, initial_offline_fraction=1.5)
    with pytest.raises(ValueError, match=r"anneal_iterations"):
        MixedReplayBuffer(online, offline, anneal_iterations=0)


def build_sac(replay, **kwargs):
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(4, OBS_DIM)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", ACTION_DIM, hidden_dims=[8], init_noise_std=0.15)
    critic = SACCriticModel(obs, groups, "critic", 1, hidden_dims=[8], num_actions=ACTION_DIM)
    return SAC(actor, critic, replay, device="cpu", mini_batch_size=8, **kwargs)


def test_sac_update_runs_on_a_mixed_batch():
    """The concatenated batch has to survive the critic, alpha and actor steps unchanged."""
    alg = build_sac(make_mixture(offline_fraction=0.5), num_mini_batches=2, policy_frequency=1)
    before = [p.clone() for p in alg.actor.parameters()]

    losses = alg.update()

    assert all(math.isfinite(value) for value in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))


def test_actor_updates_are_delayed_by_policy_frequency():
    """``policy_frequency`` is M from arXiv:2602.20220: one actor step per M critic steps."""
    alg = build_sac(make_mixture(), num_mini_batches=40, policy_frequency=20)
    steps = []
    original_step = alg.actor_optimizer.step
    alg.actor_optimizer.step = lambda *a, **kw: (steps.append(1), original_step(*a, **kw))[1]

    alg.update()

    assert len(steps) == 2


def make_runner(save=True, every=500, rank=0, log_dir="/tmp/run"):
    runner = OffPolicyRunner.__new__(OffPolicyRunner)
    runner.save_replay_buffer = save
    runner.save_replay_buffer_every = every
    runner.gpu_global_rank = rank
    runner.logger = SimpleNamespace(log_dir=log_dir)
    return runner


def test_snapshot_cadence_covers_the_interval_and_the_final_iteration():
    runner = make_runner(every=100)

    assert runner._should_save_replay_buffer(99, 1000) is True
    assert runner._should_save_replay_buffer(98, 1000) is False
    # The last iteration always snapshots, so a finished run never leaves stale replay data.
    assert runner._should_save_replay_buffer(999, 1000) is True


def test_snapshots_are_disabled_by_default_and_off_rank():
    assert make_runner(save=False)._should_save_replay_buffer(99, 1000) is False
    assert make_runner(rank=1, every=100)._should_save_replay_buffer(99, 1000) is False
    assert make_runner(log_dir=None, every=100)._should_save_replay_buffer(99, 1000) is False
