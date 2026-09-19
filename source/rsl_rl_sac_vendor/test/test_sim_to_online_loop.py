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
