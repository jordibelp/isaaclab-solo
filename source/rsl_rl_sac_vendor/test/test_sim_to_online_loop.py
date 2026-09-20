# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the whole sim-to-online data-reuse loop in one process, without a simulator.

Pretraining writes a replay snapshot, fine-tuning loads it as retained replay, and the
mixture feeds real SAC updates. The unit tests check each piece on its own; this one checks
that the runner wires them together, which is where the parts are actually used.
"""

import torch
import pytest
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.runners import OffPolicyRunner
from rsl_rl_sac.storage import MixedReplayBuffer, ReplayBuffer

OBS_DIM = 4
ACTION_DIM = 2


class StubEnv:
    """Smallest VecEnv the runner accepts: constant rewards and a fixed episode length."""

    def __init__(self, num_envs=4, episode_length=3, obs_scale=1.0):
        self.num_envs = num_envs
        self.num_actions = ACTION_DIM
        self.device = "cpu"
        self.cfg = {"episode_length_s": 1.0}
        self.max_episode_length = episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.obs_scale = obs_scale
        self.step_count = 0

    def get_observations(self):
        value = self.obs_scale * (1 + self.step_count % self.max_episode_length)
        return TensorDict(
            {"policy": torch.full((self.num_envs, OBS_DIM), float(value))}, batch_size=[self.num_envs]
        )

    def step(self, actions):
        self.step_count += 1
        obs = self.get_observations()
        timed_out = torch.full((self.num_envs,), self.step_count % self.max_episode_length == 0)
        extras = {"time_outs": timed_out, "time_outs_obs": obs.clone(), "log": {}}
        return obs, torch.ones(self.num_envs), timed_out, extras


def runner_cfg(log_every=1, **extra):
    cfg = dict(
        num_steps_per_env=2,
        save_interval=1000,
        log_interval=log_every,
        start_training=0,
        logger="tensorboard",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[8], init_noise_std=0.15),
        critic=dict(class_name="SACCriticModel", hidden_dims=[8]),
        algorithm=dict(
            class_name="SAC",
            replay_buffer_size=64,
            num_mini_batches=2,
            mini_batch_size=8,
            n_steps=1,
            rnd_cfg=None,
            symmetry_cfg=None,
        ),
    )
    cfg.update(extra)
    return cfg


def build_runner(monkeypatch, log_dir, env=None, **extra):
    monkeypatch.setattr(
        SAC, "_compute_action_scaling", lambda env, device: (torch.ones(ACTION_DIM), torch.ones(ACTION_DIM))
    )
    return OffPolicyRunner(env or StubEnv(), runner_cfg(**extra), log_dir=str(log_dir), device="cpu")


def test_pretraining_snapshot_feeds_fine_tuning(tmp_path, monkeypatch):
    pretrain_dir = tmp_path / "pretrain"
    pretrain_dir.mkdir()
    pretrainer = build_runner(
        monkeypatch, pretrain_dir, save_replay_buffer=True, save_replay_buffer_every=2
    )

    pretrainer.learn(num_learning_iterations=4)

    snapshot = pretrain_dir / "replay_buffer.pt"
    assert snapshot.exists()

    # Fine-tune in a second environment whose observations are on a different scale, which is
    # what the retained data is supposed to stabilize.
    finetune_dir = tmp_path / "finetune"
    finetune_dir.mkdir()
    finetuner = build_runner(monkeypatch, finetune_dir, env=StubEnv(obs_scale=10.0))
    offline = ReplayBuffer.load_snapshot(snapshot, "cpu", n_steps=1, gamma=0.99)
    finetuner.alg.replay_buffer = MixedReplayBuffer(
        finetuner.alg.replay_buffer,
        offline,
        initial_offline_fraction=0.5,
        final_offline_fraction=0.0,
        anneal_iterations=2,
    )
    before = [p.clone() for p in finetuner.alg.actor.parameters()]

    finetuner.learn(num_learning_iterations=3)

    assert any(not torch.equal(a, b) for a, b in zip(before, finetuner.alg.actor.parameters()))
    # Three iterations over a two-iteration anneal leaves the mixture on MJX data only.
    assert finetuner.alg.replay_buffer.offline_fraction == 0.0


def test_fine_tuning_snapshot_retains_only_the_new_environment(tmp_path, monkeypatch):
    """Saving from a mixture must record the online data, so trials chain without duplicates."""
    pretrain_dir = tmp_path / "pretrain"
    pretrain_dir.mkdir()
    pretrainer = build_runner(monkeypatch, pretrain_dir, save_replay_buffer=True, save_replay_buffer_every=2)
    pretrainer.learn(num_learning_iterations=2)

    finetune_dir = tmp_path / "finetune"
    finetune_dir.mkdir()
    finetuner = build_runner(
        monkeypatch,
        finetune_dir,
        env=StubEnv(obs_scale=10.0),
        save_replay_buffer=True,
        save_replay_buffer_every=2,
    )
    offline = ReplayBuffer.load_snapshot(pretrain_dir / "replay_buffer.pt", "cpu")
    finetuner.alg.replay_buffer = MixedReplayBuffer(finetuner.alg.replay_buffer, offline)

    finetuner.learn(num_learning_iterations=2)

    retained = ReplayBuffer.load_snapshot(finetune_dir / "replay_buffer.pt", "cpu")
    # obs_scale=10 marks the fine-tuning env; the pretraining env only ever produced 1..3.
    assert retained.observations["policy"].min() >= 10.0


def test_snapshots_are_not_written_unless_asked(tmp_path, monkeypatch):
    log_dir = tmp_path / "run"
    log_dir.mkdir()

    build_runner(monkeypatch, log_dir).learn(num_learning_iterations=2)

    assert not (log_dir / "replay_buffer.pt").exists()


def episode_schedule(warmup=5000, utd=1.25, fixed_updates=None):
    return dict(transitions_before_updates=warmup, utd=utd, fixed_updates=fixed_updates)


def record_updates(runner, monkeypatch):
    """Record real SAC optimizer work and exactly where it runs in the environment."""
    updates = []
    original = runner.alg.update

    def update():
        before = runner.alg.update_step
        result = original()
        updates.append((runner.env.step_count, runner.alg.update_step - before))
        return result

    monkeypatch.setattr(runner.alg, "update", update)
    return updates


def test_go1_schedule_updates_after_fifth_episode_without_backfilling(tmp_path, monkeypatch):
    runner = build_runner(
        monkeypatch, tmp_path, env=StubEnv(num_envs=1, episode_length=1000),
        num_steps_per_env=1000, update_schedule=episode_schedule(),
    )
    # Keep the replay small: warm-up counts collected transitions, not current buffer size.
    updates = record_updates(runner, monkeypatch)
    runner.learn(6)
    assert updates == [(5000, 1250), (6000, 1250)]


class EarlyTerminationEnv(StubEnv):
    def step(self, actions):
        obs, reward, _, extras = super().step(actions)
        done = torch.tensor([self.step_count in (3, 10, 14, 18)])
        extras["time_outs"] = torch.zeros_like(done)
        return obs, reward, done, extras


@pytest.mark.parametrize("start_iteration", [0, 3700])
def test_early_terminations_do_not_interrupt_collection(tmp_path, monkeypatch, start_iteration):
    """A fall resets the environment; it must not end the iteration.

    This env terminates at steps 3, 10, 14 and 18, so every 5-step period contains one.
    Each period must still collect its full 5 transitions and earn the same update budget.
    """
    runner = build_runner(
        monkeypatch, tmp_path, env=EarlyTerminationEnv(num_envs=1, episode_length=1000),
        num_steps_per_env=5, update_schedule=episode_schedule(warmup=8, utd=1.0),
    )
    runner.current_learning_iteration = start_iteration
    updates = record_updates(runner, monkeypatch)
    counts = []
    monkeypatch.setattr(runner.logger, "log", lambda **kw: counts.append(kw["collection_size_override"]))
    runner.learn(4)
    # Warm-up covers the first period only; every later period earns 5 updates.
    assert updates == [(10, 5), (15, 5), (20, 5)]
    assert counts == [5, 5, 5, 5]


def test_small_utd_carries_fractional_updates(tmp_path, monkeypatch):
    runner = build_runner(
        monkeypatch, tmp_path, env=StubEnv(num_envs=1, episode_length=2),
        num_steps_per_env=2, update_schedule=episode_schedule(warmup=0, utd=0.125),
    )
    updates = record_updates(runner, monkeypatch)
    runner.learn(8)
    assert updates == [(8, 1), (16, 1)]


def test_legacy_parallel_runner_keeps_iteration_warmup_and_fixed_budget(tmp_path, monkeypatch):
    runner = build_runner(monkeypatch, tmp_path, start_training=2)
    updates = record_updates(runner, monkeypatch)
    runner.learn(4)
    assert updates == [(6, 2), (8, 2)]


def test_retained_replay_annealing_starts_with_first_gradient_phase(tmp_path, monkeypatch):
    pretrain_dir = tmp_path / "pretrain"
    pretrain_dir.mkdir()
    pretrainer = build_runner(monkeypatch, pretrain_dir, save_replay_buffer=True)
    pretrainer.learn(2)
    finetune_dir = tmp_path / "finetune"
    finetune_dir.mkdir()
    runner = build_runner(
        monkeypatch, finetune_dir, env=StubEnv(num_envs=1, episode_length=2),
        num_steps_per_env=2, update_schedule=episode_schedule(warmup=6, utd=1),
    )
    offline = ReplayBuffer.load_snapshot(pretrain_dir / "replay_buffer.pt", "cpu")
    runner.alg.replay_buffer = MixedReplayBuffer(runner.alg.replay_buffer, offline, anneal_iterations=2)
    fractions = []
    original = runner.alg.update

    def update():
        fractions.append((runner.env.step_count, runner.alg.replay_buffer.offline_fraction))
        return original()

    monkeypatch.setattr(runner.alg, "update", update)
    runner.learn(5)
    assert fractions == [(6, 0.5), (8, 0.25), (10, 0.0)]


def test_short_episodes_wait_for_a_valid_n_step_window(tmp_path, monkeypatch):
    runner = build_runner(
        monkeypatch, tmp_path, env=StubEnv(num_envs=1, episode_length=1),
        num_steps_per_env=1, update_schedule=episode_schedule(warmup=0, utd=1),
    )
    runner.alg.replay_buffer.n_steps = 5
    updates = record_updates(runner, monkeypatch)
    runner.learn(6)
    assert updates == [(5, 5), (6, 1)]


def test_explicit_fixed_rollouts_count_all_parallel_transitions(tmp_path, monkeypatch):
    runner = build_runner(
        monkeypatch, tmp_path, env=StubEnv(num_envs=4, episode_length=3),
        num_steps_per_env=2, update_schedule=episode_schedule(warmup=10),
    )
    updates = record_updates(runner, monkeypatch)
    runner.learn(3)
    assert updates == [(4, 10), (6, 10)]


def test_fixed_update_budget_and_sparse_logging_with_early_terminations(tmp_path, monkeypatch):
    runner = build_runner(
        monkeypatch, tmp_path, env=EarlyTerminationEnv(num_envs=1, episode_length=1000),
        num_steps_per_env=5, log_interval=3, update_schedule=episode_schedule(warmup=0, fixed_updates=2),
    )
    updates = record_updates(runner, monkeypatch)
    counts = []
    monkeypatch.setattr(runner.logger, "log", lambda **kw: counts.append(kw["collection_size_override"]))
    runner.learn(4)
    # A fixed budget stays fixed, and each period ends on its step count, not on a fall.
    assert updates == [(5, 2), (10, 2), (15, 2), (20, 2)]
    assert counts == [5, 15]
