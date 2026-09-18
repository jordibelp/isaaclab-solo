# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical and SAC-update checks, without starting Isaac Sim."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.models import DISTRIBUTION_STAT_NAMES, SACActorModel, SACCriticModel


@pytest.fixture(
    params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"))]
)
def device(request):
    return request.param


def models(device="cpu", ce=True, **kwargs):
    obs = TensorDict({"policy": torch.randn(8, 4, device=device)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[16, 8], init_noise_std=0.15).to(device)
    critic = SACCriticModel(
        obs, groups, "critic", 1, hidden_dims=[16, 8], num_actions=2,
        distributional_critic_ce=ce, **kwargs,
    ).to(device)
    return obs, actor, critic


def test_projection_edges_and_mean_in_reward_units(device):
    _, _, critic = models(device)
    b = critic.value_support
    values = torch.cat((b, (b[:-1] + b[1:]) / 2, b.new_tensor([-1e12, -10, -2, 0, 0.045, 1e12])))[:, None]
    values.requires_grad_()
    labels = critic.two_hot(values)
    assert not labels.requires_grad
    assert (labels >= 0).all()
    assert ((labels > 0).sum(-1) <= 2).all()
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(values[:, 0]))
    torch.testing.assert_close((labels * b).sum(-1, keepdim=True), values.clamp(b[0], b[-1]), atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(labels[:len(b)], torch.eye(len(b), device=device))
    # A rare catastrophic penalty retains its arithmetic mean, not a log-space mean.
    mixture = 0.99 * critic.two_hot(b.new_tensor([[1.0]])) + 0.01 * critic.two_hot(b.new_tensor([[-100.0]]))
    torch.testing.assert_close(critic.q_from_output(mixture.log()), b.new_tensor([[-0.01]]), atol=2e-6, rtol=0)


def test_logit_gradients_bounded_for_large_targets(device):
    _, _, critic = models(device)
    targets = torch.tensor([0.01, -2, -10, -1e6, 1e6, 1e12], device=device)[:, None]
    logits = torch.zeros(6, 255, device=device, requires_grad=True)
    labels = critic.two_hot(targets)
    loss = -(labels * logits.log_softmax(-1)).sum()
    grad, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(grad, logits.softmax(-1) - labels)
    assert grad.abs().max() <= 1
    assert (grad.abs().sum(-1) <= 2.00001).all()
    scalar = torch.zeros_like(targets, requires_grad=True)
    mse_grad, = torch.autograd.grad((scalar - targets).square().sum(), scalar)
    assert mse_grad[-1].abs() == 2e12


def test_scalar_default_is_exact_legacy_mse_and_state_dict(device):
    obs, _, critic = models(device, ce=False)
    actions = torch.randn(8, 2, device=device)
    target = torch.randn(8, 1, device=device)
    latent = torch.cat((critic.get_latent(obs), actions), -1)
    expected = (critic.critic1(latent), critic.critic2(latent))
    actual = critic.evaluate_all_q(obs, actions)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    old_loss = sum(torch.nn.functional.mse_loss(q, target) for q in expected) / 2
    new_loss = sum(critic.td_losses(obs, actions, target)) / 2
    params = [p for p in critic.parameters() if p.requires_grad]
    for old, new in zip(torch.autograd.grad(old_loss, params), torch.autograd.grad(new_loss, params)):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    assert "value_support" not in critic.state_dict()
    assert critic.critic1[-1].out_features == 1
    clone = copy.deepcopy(critic)
    clone.load_state_dict(critic.state_dict(), strict=True)


def test_initial_mean_exact_zero_and_action_gradients_after_learning(device):
    obs, _, critic = models(device)
    actions = torch.randn(8, 2, device=device, requires_grad=True)
    for q in (*critic.evaluate_all_q(obs, actions), *critic.evaluate_all_target_q(obs, actions)):
        assert q.shape == (8, 1)
        assert torch.count_nonzero(q) == 0
    optimizer = torch.optim.Adam([p for p in critic.parameters() if p.requires_grad], lr=2e-4)
    for _ in range(5):
        optimizer.zero_grad()
        sum(critic.td_losses(obs, actions.detach(), -2 + actions.detach()[:, :1])).backward()
        optimizer.step()
    q1, q2 = critic.evaluate_all_q(obs, actions)
    grad, = torch.autograd.grad((q1 + q2).sum(), actions)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    torch.testing.assert_close(critic(obs, actions=actions), q1)
    old = [p.clone() for p in critic.critic1_target.parameters()]
    critic.soft_update_target_networks(0.25)
    for before, online, after in zip(old, critic.critic1.parameters(), critic.critic1_target.parameters()):
        torch.testing.assert_close(after, 0.75 * before + 0.25 * online)
        assert not after.requires_grad and after.grad is None


def test_default_action_gradient_matches_float64_reference(device):
    torch.manual_seed(1)
    obs, _, critic = models(device)
    optimizer = torch.optim.Adam(critic.critic1.parameters(), lr=2e-4)
    x = torch.randn(8, 6, device=device, requires_grad=True)
    for _ in range(5):
        optimizer.zero_grad()
        logits = critic.critic1(x.detach())
        labels = critic.two_hot(torch.full((8, 1), -10., device=device))
        (-(labels * logits.log_softmax(-1)).sum(-1).mean()).backward()
        optimizer.step()
    grad, = torch.autograd.grad(critic.q_from_output(critic.critic1(x)).sum(), x)
    reference = copy.deepcopy(critic).double()
    x64 = x.detach().double().requires_grad_()
    mean64 = (reference.critic1(x64).softmax(-1) * reference.value_support).sum()
    grad64, = torch.autograd.grad(mean64, x64)
    torch.testing.assert_close(grad, grad64.float(), atol=2e-7, rtol=2e-3)


@pytest.mark.parametrize("ce", [False, True])
def test_complete_sac_update_bootstrap_and_checkpoint(device, ce):
    torch.manual_seed(21)
    obs, actor, critic = models(device, ce=ce)
    actions = torch.randn(8, 2, device=device)
    rewards = torch.tensor([-10., -2., 0.045, -1e9, 1., 2., 3., 4.], device=device)[:, None]
    dones = torch.tensor([1., 1., 0., 1., 0., 0., 0., 0.], device=device)[:, None]
    timeouts = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.], device=device)[:, None]
    steps = torch.tensor([1, 2, 5, 1, 3, 4, 2, 1], device=device)[:, None]
    batch = (obs, actions, rewards, obs, dones, timeouts, steps)
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([batch]))
    alg = SAC(actor, critic, replay, gamma=0.97, alpha=0.01, device=device, policy_frequency=1)
    # Compute the expected target with the same action RNG, including timeout bootstrap.
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    with torch.no_grad():
        next_actions, logp = actor.sample_action_logp(obs)
        q1, q2 = critic.evaluate_all_target_q(obs, next_actions)
        expected = rewards + 0.97 ** steps * (1 + timeouts - dones) * (torch.minimum(q1, q2) - 0.01 * logp)
    torch.random.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    original_loss = critic.losses_from_outputs
    seen = []

    def capture(output1, output2, y):
        seen.append(y.clone())
        return original_loss(output1, output2, y)

    critic.losses_from_outputs = capture
    before = [p.clone() for p in actor.parameters()]
    losses = alg.update()
    torch.testing.assert_close(seen[0], expected)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    if ce:
        assert losses["critic_target_clipped_fraction"] == 1 / 8
    else:
        assert "critic_target_clipped_fraction" not in losses
    assert all(p.grad is None for p in critic.critic1_target.parameters())
    saved = copy.deepcopy(alg.save())
    _, actor2, critic2 = models(device, ce=ce)
    resumed = SAC(actor2, critic2, replay, device=device)
    assert resumed.load(saved, load_cfg=None, strict=True)
    for original, restored in zip(critic.evaluate_all_q(obs, actions), critic2.evaluate_all_q(obs, actions)):
        torch.testing.assert_close(original, restored, atol=0, rtol=0)
    assert resumed.critic_optimizer.state_dict()["state"]
    assert all(torch.isfinite(torch.tensor(v)) for v in resumed.update().values())


@pytest.mark.parametrize("ce", [False, True])
def test_runner_flag_constructs_requested_head(monkeypatch, ce):
    obs, _, _ = models()
    env = SimpleNamespace(num_actions=2, num_envs=8)
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    cfg = dict(
        distributional_critic_ce=ce, num_steps_per_env=2,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[8]),
        critic=dict(class_name="SACCriticModel", hidden_dims=[8]),
        algorithm=dict(class_name="SAC", replay_buffer_size=64),
    )
    alg = SAC.construct_algorithm(obs, env, cfg, "cpu")
    assert alg.critic.distributional_critic_ce == ce
    assert alg.critic.critic1[-1].out_features == (255 if ce else 1)


@pytest.mark.parametrize("kwargs", [
    {"distributional_num_bins": 2}, {"distributional_num_bins": 254},
    {"distributional_symlog_limit": 0}, {"distributional_symlog_limit": float("nan")},
    {"distributional_symlog_limit": 90},
])
def test_invalid_support_rejected(kwargs):
    with pytest.raises(ValueError):
        models(**kwargs)


def test_namespaced_stats_bypass_the_loss_prefix():
    """CriticDist/* must reach the writer verbatim, while plain loss keys stay under Loss/."""
    from rsl_rl_sac.utils.logger import Logger

    tags = []
    logger = Logger.__new__(Logger)
    logger.writer = SimpleNamespace(add_scalar=lambda tag, value, step, **kw: tags.append(tag))
    logger.cfg = {"num_steps_per_env": 2, "algorithm": {"rnd_cfg": None}}
    logger.log_dir, logger.logger_type = None, "tensorboard"
    logger.num_envs, logger.gpu_world_size = 8, 1
    logger.tot_timesteps, logger.tot_time = 0, 1.0
    logger.ep_extras, logger.rewbuffer, logger.lenbuffer = [], [], []
    logger.log(
        it=1, start_it=0, total_it=2, collect_time=0.1, learn_time=0.1,
        loss_dict={"critic1": 0.5, DISTRIBUTION_STAT_NAMES[0]: -0.25},
        learning_rate=1e-3, action_std=torch.zeros(2), rnd_weight=None,
    )
    assert "Loss/critic1" in tags
    assert DISTRIBUTION_STAT_NAMES[0] in tags
    assert f"Loss/{DISTRIBUTION_STAT_NAMES[0]}" not in tags


def test_support_symlog_recovers_the_configured_grid(device):
    _, _, critic = models(device, distributional_num_bins=51, distributional_symlog_limit=5.0)
    expected = torch.linspace(-5.0, 5.0, 51, device=device)
    torch.testing.assert_close(critic.support_symlog(), expected, atol=2e-6, rtol=0)


def stats(critic, probs):
    """Run distribution_stats on an exact probability vector, for both critics."""
    logits = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
    return dict(zip(DISTRIBUTION_STAT_NAMES, critic.distribution_stats(logits, logits).tolist()))


def test_stats_on_a_single_atom_report_zero_spread(device):
    _, _, critic = models(device, distributional_num_bins=51, distributional_symlog_limit=5.0)
    atoms = critic.support_symlog()
    probs = torch.zeros(4, 51, device=device)
    probs[:, 40] = 1.0
    reported = stats(critic, probs)
    assert reported["CriticDist/active_atoms_p10"] == 1
    assert reported["CriticDist/effective_atoms"] == pytest.approx(1.0, abs=1e-5)
    assert reported["CriticDist/symlog_std_within_state"] == pytest.approx(0.0, abs=1e-5)
    assert reported["CriticDist/symlog_std_across_states"] == pytest.approx(0.0, abs=1e-6)
    assert reported["CriticDist/symlog_q05_q95_width"] == pytest.approx(0.0, abs=1e-6)
    for key in ("symlog_mean", "symlog_q05", "symlog_q50", "symlog_q95"):
        assert reported[f"CriticDist/{key}"] == pytest.approx(atoms[40].item(), abs=1e-5)
    assert reported["CriticDist/edge_mass"] == pytest.approx(0.0, abs=1e-30)


def test_stats_on_a_uniform_distribution_span_the_support(device):
    _, _, critic = models(device, distributional_num_bins=51, distributional_symlog_limit=5.0)
    reported = stats(critic, torch.full((4, 51), 1 / 51, device=device))
    assert reported["CriticDist/active_atoms_p10"] == 0  # 1/51 < 0.1: no single atom dominates
    assert reported["CriticDist/effective_atoms"] == pytest.approx(51.0, rel=1e-4)
    assert reported["CriticDist/edge_mass"] == pytest.approx(2 / 51, rel=1e-5)
    assert reported["CriticDist/symlog_mean"] == pytest.approx(0.0, abs=1e-5)
    # Uniform on [-5, 5] discretized to 51 atoms: the 5%/95% atoms sit one step inside +/-4.5.
    assert reported["CriticDist/symlog_q05"] == pytest.approx(-4.6, abs=0.11)
    assert reported["CriticDist/symlog_q95"] == pytest.approx(4.6, abs=0.11)


def test_stats_mix_the_twin_critics_and_separate_the_two_spreads(device):
    _, _, critic = models(device, distributional_num_bins=51, distributional_symlog_limit=5.0)
    atoms = critic.support_symlog()
    tiny = torch.finfo(torch.float32).tiny
    # Critic 1 and critic 2 each peak on a different atom, for two different states.
    logits1 = torch.full((2, 51), tiny, device=device).log()
    logits2 = logits1.clone()
    logits1[0, 20], logits2[0, 30] = 0.0, 0.0
    logits1[1, 24], logits2[1, 26] = 0.0, 0.0
    reported = dict(zip(DISTRIBUTION_STAT_NAMES, critic.distribution_stats(logits1, logits2).tolist()))
    assert reported["CriticDist/active_atoms_p10"] == 2  # the mixture puts 0.5 on each peak
    centers = torch.tensor([(atoms[20] + atoms[30]) / 2, (atoms[24] + atoms[26]) / 2])
    widths = torch.tensor([(atoms[30] - atoms[20]) / 2, (atoms[26] - atoms[24]) / 2])
    # Disagreement inside one state and variation across states are reported separately.
    assert reported["CriticDist/symlog_std_within_state"] == pytest.approx(widths.mean().item(), abs=1e-5)
    assert reported["CriticDist/symlog_std_across_states"] == pytest.approx(
        centers.std(unbiased=False).item(), abs=1e-5
    )
    assert reported["CriticDist/symlog_mean"] == pytest.approx(centers.mean().item(), abs=1e-5)


def test_stats_are_diagnostic_only_and_reach_the_loss_dict(device):
    obs, actor, critic = models(device)
    actions = torch.randn(8, 2, device=device, requires_grad=True)
    output1, output2 = critic.critic_outputs(obs, actions)
    reported = critic.distribution_stats(output1, output2)
    assert not reported.requires_grad
    assert torch.isfinite(reported).all()
    # The zero-initialized head is exactly symmetric: mean 0, and every atom equally used.
    assert reported[DISTRIBUTION_STAT_NAMES.index("CriticDist/symlog_mean")].item() == pytest.approx(0.0, abs=1e-5)

    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([(
        obs, actions.detach(), torch.zeros(8, 1, device=device), obs,
        torch.zeros(8, 1, device=device), torch.zeros(8, 1, device=device),
        torch.ones(8, 1, device=device),
    )]))
    losses = SAC(actor, critic, replay, device=device, policy_frequency=1).update()
    assert all(name in losses for name in DISTRIBUTION_STAT_NAMES)
    assert all(torch.isfinite(torch.tensor(losses[name])) for name in DISTRIBUTION_STAT_NAMES)
    _, actor2, scalar_critic = models(device, ce=False)
    scalar_losses = SAC(actor2, scalar_critic, replay, device=device, policy_frequency=1).update()
    assert not any(name in scalar_losses for name in DISTRIBUTION_STAT_NAMES)
