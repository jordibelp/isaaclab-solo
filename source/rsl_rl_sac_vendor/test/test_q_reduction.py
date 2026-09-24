# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Twin-critic reduction: clipped double Q ("min"), the FastSAC average ("mean"), or the FastSAC
reference code ("mean_pi_q_none": average in the actor, each critic keeps its own target)."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC, reduce_twin_q
from rsl_rl_sac.models import SACActorModel, SACCriticModel

METHODS = ["min", "mean", "mean_pi_q_none"]
LOSSES = ["mse", "two_hot", "hl_gauss", "mse_target_norm_popart"]


@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"))]
)
def device(request):
    return request.param


def models(device="cpu", loss="mse"):
    obs = TensorDict({"policy": torch.randn(8, 4, device=device)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[16, 8], init_noise_std=0.15).to(device)
    critic = SACCriticModel(
        obs, groups, "critic", 1, hidden_dims=[16, 8], num_actions=2, distributional_loss=loss,
        distributional_num_bins=51, distributional_symlog_limit=5.0,
    ).to(device)
    # Categorical heads start at zero, which makes both critics output exactly Q = 0. Random
    # heads keep Q1 != Q2, so "min" and "mean" really differ in every check below.
    with torch.no_grad():
        for network in (critic.critic1, critic.critic2):
            network[-1].weight.normal_(0.0, 0.5)
            network[-1].bias.normal_(0.0, 0.5)
    critic.init_target_networks()
    return obs, actor, critic


def one_batch(obs, device):
    actions = torch.randn(8, 2, device=device)
    rewards = torch.tensor([-10.0, -2.0, 0.045, -3.0, 1.0, 2.0, 3.0, 4.0], device=device)[:, None]
    dones = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=device)[:, None]
    timeouts = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)[:, None]
    steps = torch.tensor([1, 2, 5, 1, 3, 4, 2, 1], device=device)[:, None]
    return (obs, actions, rewards, obs, dones, timeouts, steps)


def test_reduce_twin_q_values_and_gradients():
    q1 = torch.tensor([[1.0], [-3.0], [2.0]], requires_grad=True)
    q2 = torch.tensor([[4.0], [-1.0], [0.5]], requires_grad=True)
    reduce_twin_q(q1, q2, "min").sum().backward()
    # Clipped double Q: each sample's gradient goes to the smaller critic only.
    torch.testing.assert_close(q1.grad, torch.tensor([[1.0], [1.0], [0.0]]))
    torch.testing.assert_close(q2.grad, torch.tensor([[0.0], [0.0], [1.0]]))

    q1.grad = q2.grad = None
    mean = reduce_twin_q(q1, q2, "mean")
    torch.testing.assert_close(mean, torch.tensor([[2.5], [-2.0], [1.25]]))
    mean.sum().backward()
    # The average sends half of every gradient to each critic.
    torch.testing.assert_close(q1.grad, torch.full_like(q1, 0.5))
    torch.testing.assert_close(q2.grad, torch.full_like(q2, 0.5))
    # The reference-code mode gives its actor the same average.
    torch.testing.assert_close(reduce_twin_q(q1, q2, "mean_pi_q_none"), mean)

    with pytest.raises(ValueError, match="q_reduction_method"):
        reduce_twin_q(q1, q2, "max")


def test_sac_rejects_an_unknown_reduction():
    obs, actor, critic = models()
    with pytest.raises(ValueError, match="q_reduction_method"):
        SAC(actor, critic, SimpleNamespace(), q_reduction_method="median")


@pytest.mark.parametrize("loss", LOSSES)
@pytest.mark.parametrize("method", METHODS)
def test_bellman_target_uses_the_configured_reduction(device, method, loss):
    torch.manual_seed(7)
    obs, actor, critic = models(device, loss=loss)
    batch = one_batch(obs, device)
    _, _, rewards, _, dones, timeouts, steps = batch
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch]))
    alg = SAC(actor, critic, replay, gamma=0.97, alpha=0.01, device=device, policy_frequency=1,
              q_reduction_method=method)
    # Rebuild the target by hand with the same action RNG, including the timeout bootstrap.
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    with torch.no_grad():
        next_actions, logp = actor.sample_action_logp(obs)
        q1, q2 = critic.evaluate_all_target_q(obs, next_actions)
        mask = 0.97 ** steps * (1 + timeouts - dones)
        expected = {
            "min": rewards + mask * (torch.minimum(q1, q2) - 0.01 * logp),
            "mean": rewards + mask * ((q1 + q2) / 2 - 0.01 * logp),
            # No reduction: column i is the target of critic i, from its own target network.
            "mean_pi_q_none": torch.cat(
                (rewards + mask * (q1 - 0.01 * logp), rewards + mask * (q2 - 0.01 * logp)), dim=-1
            ),
        }
    torch.random.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    # The reductions give different targets here, so this test can tell them apart.
    independent = expected["mean_pi_q_none"]
    assert not torch.allclose(expected["min"], expected["mean"])
    assert not torch.allclose(independent[:, :1], independent[:, 1:])
    # The two separate targets average to the shared "mean" target.
    torch.testing.assert_close(independent.mean(-1, keepdim=True), expected["mean"])

    original_loss = critic.losses_from_outputs
    seen = []

    def capture(output1, output2, y):
        seen.append(y.clone())
        return original_loss(output1, output2, y)

    critic.losses_from_outputs = capture
    losses = alg.update()
    # One shared target, or one column per critic, built from the configured reduction.
    torch.testing.assert_close(seen[0], expected[method])
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


@pytest.mark.parametrize("loss", LOSSES)
def test_each_critic_regresses_onto_its_own_target_column(device, loss):
    torch.manual_seed(3)
    obs, actor, critic = models(device, loss=loss)
    if critic.popart:
        critic.update_popart(torch.tensor(1.5, device=device), torch.tensor(6.0, device=device))
    output1, output2 = critic.critic_outputs(obs, torch.randn(8, 2, device=device))
    target1 = torch.linspace(-4.0, 3.0, 8, device=device)[:, None]
    target2 = torch.linspace(2.0, -1.0, 8, device=device)[:, None]
    loss1, loss2 = critic.losses_from_outputs(output1, output2, torch.cat((target1, target2), dim=-1))
    # Same numbers as training each critic alone on the shared path with its own target.
    torch.testing.assert_close(loss1, critic.losses_from_outputs(output1, output2, target1)[0], rtol=0, atol=0)
    torch.testing.assert_close(loss2, critic.losses_from_outputs(output1, output2, target2)[1], rtol=0, atol=0)
    # And not the other critic's target.
    assert not torch.isclose(loss1, critic.losses_from_outputs(output1, output2, target2)[0])
    assert not torch.isclose(loss2, critic.losses_from_outputs(output1, output2, target1)[1])


@pytest.mark.parametrize("loss", ["mse", "two_hot"])
@pytest.mark.parametrize("method", METHODS)
def test_actor_loss_uses_the_configured_reduction(device, method, loss):
    torch.manual_seed(11)
    obs, actor, critic = models(device, loss=loss)
    batch = one_batch(obs, device)
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch]))
    alg = SAC(actor, critic, replay, gamma=0.97, alpha=0.01, device=device, policy_frequency=1,
              q_reduction_method=method)

    # The second policy sample of an update feeds the alpha and actor losses; the critic
    # values at those actions are only computed for the actor loss.
    samples, values = [], []
    sample, evaluate = actor.sample_action_logp, critic.evaluate_all_q

    def record_sample(o):
        out = sample(o)
        samples.append(out[1].detach().clone())
        return out

    def record_values(o, a):
        out = evaluate(o, a)
        values.append(tuple(v.detach().clone() for v in out))
        return out

    actor.sample_action_logp, critic.evaluate_all_q = record_sample, record_values
    losses = alg.update()
    log_prob = samples[1]
    q1, q2 = values[0]
    assert not torch.allclose(q1, q2)
    # With one mini-batch, the final temperature is the one the actor loss used.
    alpha = alg.log_alpha.exp().detach()
    # Written out by hand, not with reduce_twin_q, so a wrong reduction cannot hide here.
    average = (alpha * log_prob - (q1 + q2) / 2).mean().item()
    expected = {"min": (alpha * log_prob - torch.minimum(q1, q2)).mean().item(), "mean": average,
                "mean_pi_q_none": average}
    assert expected["min"] != pytest.approx(expected["mean"], rel=1e-3)
    assert losses["actor"] == pytest.approx(expected[method], rel=1e-5, abs=1e-6)


def build_algorithm(monkeypatch, **algorithm):
    obs, _, _ = models()
    env = SimpleNamespace(num_actions=2, num_envs=8)
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    cfg = dict(
        num_steps_per_env=2,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[8]),
        critic=dict(class_name="SACCriticModel", hidden_dims=[8]),
        algorithm=dict(class_name="SAC", replay_buffer_size=64, **algorithm),
    )
    return SAC.construct_algorithm(obs, env, cfg, "cpu")


def test_runner_config_reaches_the_algorithm(monkeypatch):
    # agent.algorithm.q_reduction_method arrives as a key of the algorithm config dict.
    assert build_algorithm(monkeypatch).q_reduction_method == "min"
    assert build_algorithm(monkeypatch, q_reduction_method="mean").q_reduction_method == "mean"
    alg = build_algorithm(monkeypatch, q_reduction_method="mean_pi_q_none")
    assert alg.q_reduction_method == "mean_pi_q_none"
    with pytest.raises(ValueError, match="q_reduction_method"):
        build_algorithm(monkeypatch, q_reduction_method="Mean")


def test_checkpoint_records_the_reduction_and_resume_keeps_the_configured_one():
    obs, actor, critic = models()
    batch = one_batch(obs, "cpu")
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch]))
    trained = SAC(actor, critic, replay, q_reduction_method="mean")
    trained.update()
    saved = copy.deepcopy(trained.save())
    assert saved["q_reduction_method"] == "mean"

    # Like gamma or tau, the reduction is a training setting: the resuming config decides it.
    _, actor2, critic2 = models()
    resumed = SAC(actor2, critic2, replay)
    assert resumed.load(saved, load_cfg=None, strict=True)
    assert resumed.q_reduction_method == "min"
    assert resumed.save()["q_reduction_method"] == "min"

    # Checkpoints written before the key existed still load.
    del saved["q_reduction_method"]
    _, actor3, critic3 = models()
    assert SAC(actor3, critic3, replay, q_reduction_method="mean").load(saved, load_cfg=None, strict=True)
