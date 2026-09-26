# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The SAC update without per-mini-batch host syncs, with a sparse logged mirror loss, and compiled."""

import warnings
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC

NUM_ENVS = 16


def mirror(env=None, obs=None, actions=None):
    """A stand-in symmetry: the identity plus a sign flip, doubling the batch like Solo12's."""
    obs_aug = None
    if obs is not None:
        obs_aug = TensorDict({key: torch.cat((value, -value)) for key, value in obs.items()},
                             batch_size=[2 * obs.batch_size[0]], device=obs.device)
    actions_aug = None if actions is None else torch.cat((actions, -actions))
    return obs_aug, actions_aug


def build(monkeypatch, device="cpu", symmetry=None, critic_loss="two_hot", **algorithm):
    obs = TensorDict({"policy": torch.randn(NUM_ENVS, 6, device=device)}, batch_size=[NUM_ENVS], device=device)
    env = SimpleNamespace(num_actions=2, num_envs=NUM_ENVS)
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    if symmetry is not None:
        symmetry = {"use_data_augmentation": True, "use_mirror_loss": False, "mirror_loss_coeff": 0.1,
                    "data_augmentation_func": mirror, **symmetry}
    settings = dict(class_name="SAC", replay_buffer_size=NUM_ENVS * 32, num_mini_batches=3, mini_batch_size=8,
                    gamma=0.97, n_steps=3, policy_frequency=1, actor_learning_rate=1e-3,
                    critic_learning_rate=1e-3, alpha_learning_rate=1e-3, symmetry_cfg=symmetry)
    settings.update(algorithm)
    cfg = dict(
        num_steps_per_env=4,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[16, 16], activation="elu", obs_normalization=True),
        critic=dict(class_name="SACCriticModel", hidden_dims=[16, 16], activation="elu", obs_normalization=True,
                    layer_norm=True, distributional_loss=critic_loss, distributional_num_bins=51,
                    distributional_symlog_limit=5.0),
        algorithm=settings,
    )
    alg = SAC.construct_algorithm(obs, env, cfg, device)
    for _ in range(12):
        alg.act(obs)
        next_obs = TensorDict({"policy": torch.randn(NUM_ENVS, 6, device=device)}, batch_size=[NUM_ENVS],
                              device=device)
        dones = (torch.rand(NUM_ENVS, device=device) < 0.2).float()
        alg.process_env_step(next_obs, torch.randn(NUM_ENVS, device=device), dones, {})
        obs = next_obs
    return alg


@pytest.mark.parametrize("name,expected", [("adam", torch.optim.Adam), ("adamW", torch.optim.AdamW)])
def test_alpha_optimizer_is_configured_independently(monkeypatch, name, expected):
    alg = build(monkeypatch, alpha_optimizer=name, actor_optimizer="adamw", critic_optimizer="adamw")
    assert isinstance(alg.alpha_optimizer, expected)
    assert alg.alpha_optimizer.param_groups[0]["weight_decay"] == 0.0
    assert isinstance(alg.actor_optimizer, torch.optim.AdamW)
    assert isinstance(alg.critic_optimizer, torch.optim.AdamW)


def test_logged_mirror_loss_is_measured_once_per_interval(monkeypatch):
    alg = build(monkeypatch, symmetry={}, symmetry_log_interval=3)
    calls = []
    measure = alg._mirror_loss
    alg._mirror_loss = lambda *args: calls.append(1) or measure(*args)

    logged = ["symmetry" in alg.update() for _ in range(7)]

    # One mini-batch on updates 0, 3 and 6, instead of every mini-batch of every update.
    assert logged == [True, False, False, True, False, False, True]
    assert len(calls) == 3


@pytest.mark.parametrize("augment", [False, True])
def test_mirror_loss_that_trains_the_actor_still_runs_on_every_actor_update(monkeypatch, augment):
    alg = build(monkeypatch, symmetry=dict(use_data_augmentation=augment, use_mirror_loss=True),
                symmetry_log_interval=3)
    calls = []
    measure = alg._mirror_loss
    alg._mirror_loss = lambda *args: calls.append(1) or measure(*args)

    assert all("symmetry" in alg.update() for _ in range(2))
    assert len(calls) == 2 * alg.num_mini_batches


def test_a_timeout_without_a_done_is_rejected_when_stored(monkeypatch):
    alg = build(monkeypatch)
    obs = TensorDict({"policy": torch.randn(NUM_ENVS, 6)}, batch_size=[NUM_ENVS])
    alg.act(obs)
    time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
    time_outs[0] = True
    with pytest.raises(ValueError, match="time-out without a done"):
        alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS),
                             {"time_outs": time_outs, "time_outs_obs": {"policy": obs["policy"]}})


@pytest.mark.parametrize("critic_loss,q_reduction_method", [
    ("mse", "min"), ("two_hot", "mean"), ("c51", "mean_pi_q_none")
])
def test_compiled_update_matches_eager(monkeypatch, critic_loss, q_reduction_method):
    """Same seeds and replay give the same losses and weights, compiled or not."""
    pytest.importorskip("torch._inductor")
    import torch._inductor.config as inductor_config

    # Compiled graphs normally draw their own random numbers; fall back to eager RNG to compare.
    monkeypatch.setattr(inductor_config, "fallback_random", True)
    results = []
    for torch_compile in (False, True):
        torch.manual_seed(5)
        alg = build(monkeypatch, symmetry={}, critic_loss=critic_loss,
                    q_reduction_method=q_reduction_method, torch_compile=torch_compile)
        torch.manual_seed(6)
        losses = [alg.update() for _ in range(3)]
        results.append((losses, [p.detach().clone() for p in (*alg.actor.parameters(), *alg.critic.parameters())]))

    (eager_losses, eager_params), (compiled_losses, compiled_params) = results
    for eager, compiled in zip(eager_losses, compiled_losses):
        assert eager.keys() == compiled.keys()
        for key in eager:
            assert compiled[key] == pytest.approx(eager[key], rel=1e-4, abs=1e-5), key
    for eager, compiled in zip(eager_params, compiled_params):
        torch.testing.assert_close(compiled, eager, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_update_host_syncs_do_not_grow_with_mini_batches(monkeypatch):
    def count_syncs(num_mini_batches):
        alg = build(monkeypatch, device="cuda", symmetry={}, num_mini_batches=num_mini_batches)
        alg.update()  # the first call caches the symmetry constants on the GPU
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                alg.update()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        return sum("synchronizing" in str(warning.message) for warning in caught)

    assert count_syncs(2) == count_syncs(10)
