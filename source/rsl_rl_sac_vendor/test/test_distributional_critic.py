# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical and SAC-update checks, without starting Isaac Sim."""

import copy
import math
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


def models(device="cpu", loss="two_hot", **kwargs):
    obs = TensorDict({"policy": torch.randn(8, 4, device=device)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[16, 8], init_noise_std=0.15).to(device)
    critic = SACCriticModel(
        obs, groups, "critic", 1, hidden_dims=[16, 8], num_actions=2,
        distributional_loss=loss, **kwargs,
    ).to(device)
    return obs, actor, critic


CATEGORICAL = ["two_hot", "hl_gauss"]
POPART = "mse_target_norm_popart"


def test_c51_projects_full_distribution_and_terminal_to_reward(device):
    _, _, critic = models(device, loss="c51", c51_num_atoms=5, c51_v_min=-2, c51_v_max=2)
    support = critic.value_support
    probabilities = torch.tensor([[0.25, 0, 0, 0, 0.75], [0, 0, 1, 0, 0]], device=device)
    projected = critic.c51_project(probabilities, torch.tensor([[0.5], [-0.5]], device=device),
                                   torch.tensor([[0.5], [0]], device=device))
    expected = torch.tensor([[0, 0.125, 0.125, 0.375, 0.375], [0, 0.5, 0.5, 0, 0]], device=device)
    torch.testing.assert_close(projected, expected)
    torch.testing.assert_close(projected.sum(-1), torch.ones(2, device=device))
    torch.testing.assert_close((projected * support).sum(-1), torch.tensor([1.0, -0.5], device=device))
    assert (projected[0] > 0).sum() == 4  # not a two-hot projection of the scalar mean
    edge = critic.c51_project(probabilities[:1], torch.tensor([[100.]], device=device),
                              torch.ones(1, 1, device=device))
    torch.testing.assert_close(edge, torch.tensor([[0, 0, 0, 0, 1.]], device=device))


def test_c51_critic_decodes_mean_and_passes_action_gradients(device):
    obs, _, critic = models(device, loss="c51")
    actions = torch.randn(8, 2, device=device, requires_grad=True)
    with torch.no_grad():
        critic.critic1[-1].weight.normal_(0, 0.01)
    output = critic.critic_outputs(obs, actions)[0]
    expected = (output.softmax(-1) * critic.value_support).sum(-1, keepdim=True)
    torch.testing.assert_close(critic.evaluate_all_q(obs, actions)[0], expected)
    gradient, = torch.autograd.grad(expected.sum(), actions)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    labels = critic.c51_project(output.detach().softmax(-1), torch.zeros(8, 1, device=device),
                                torch.full((8, 1), 0.97, device=device))
    logits = torch.zeros_like(output, requires_grad=True)
    grad, = torch.autograd.grad(-(labels * logits.log_softmax(-1)).sum(), logits)
    torch.testing.assert_close(grad, logits.softmax(-1) - labels)
    assert grad.abs().max() <= 1


@pytest.mark.parametrize("method", ["min", "mean", "mean_pi_q_none"])
def test_c51_update_uses_projected_target_and_retains_sac_settings(device, method):
    torch.manual_seed(41)
    obs, actor, critic = models(device, loss="c51")
    with torch.no_grad():
        critic.critic1_target[-1].bias[0] = 3
        critic.critic2_target[-1].bias[-1] = 3
    actions = torch.randn(8, 2, device=device)
    rewards = torch.tensor([-10., -2., 0.045, -1e9, 1., 2., 3., 4.], device=device)[:, None]
    dones = torch.tensor([1., 1., 0., 1., 0., 0., 0., 0.], device=device)[:, None]
    timeouts = torch.tensor([0., 1., 0., 0., 0., 0., 0., 0.], device=device)[:, None]
    steps = torch.tensor([1, 2, 5, 1, 3, 4, 2, 1], device=device)[:, None]
    replay = SimpleNamespace(mini_batch_generator=lambda **kw: iter([(obs, actions, rewards, obs, dones, timeouts, steps)]))
    alg = SAC(actor, critic, replay, gamma=0.97, alpha=0.01, device=device, policy_frequency=1,
              q_reduction_method=method)
    rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    with torch.no_grad():
        next_actions, logp = actor.sample_action_logp(obs)
        t1, t2 = critic.target_outputs(obs, next_actions)
        p1, p2 = t1.softmax(-1), t2.softmax(-1)
        q1 = (p1 * critic.value_support).sum(-1, keepdim=True)
        q2 = (p2 * critic.value_support).sum(-1, keepdim=True)
        if method == "min":
            source1 = source2 = torch.where(q1 <= q2, p1, p2)
        elif method == "mean":
            source1 = source2 = (p1 + p2) / 2
        else:
            source1, source2 = p1, p2
        discount = 0.97 ** steps * (1 + timeouts - dones)
        reward = rewards - discount * 0.01 * logp
        expected1 = critic.c51_project(source1, reward, discount)
        expected2 = critic.c51_project(source2, reward, discount)
    torch.random.set_rng_state(rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    seen = []
    original = critic.c51_losses

    def capture(o1, o2, y1, y2):
        seen.append((y1.clone(), y2.clone()))
        return original(o1, o2, y1, y2)

    critic.c51_losses = capture
    losses = alg.update()
    torch.testing.assert_close(seen[0][0], expected1)
    torch.testing.assert_close(seen[0][1], expected2)
    assert all(math.isfinite(value) for value in losses.values())
    assert losses["critic_target_clipped_fraction"] > 0
    assert alg.gamma == 0.97 and alg.q_reduction_method == method
    saved = copy.deepcopy(alg.save())
    _, actor2, critic2 = models(device, loss="c51")
    resumed = SAC(actor2, critic2, replay, device=device)
    assert resumed.load(saved, load_cfg=None, strict=True)
    for actual, restored in zip(critic.evaluate_all_q(obs, actions), critic2.evaluate_all_q(obs, actions)):
        torch.testing.assert_close(actual, restored, atol=0, rtol=0)


def test_c51_checkpoint_rejects_a_different_support_or_scalar_target_mode():
    obs, actor, critic = models(loss="c51")
    alg = SAC(actor, critic, SimpleNamespace(), device="cpu")
    saved = copy.deepcopy(alg.save())
    assert saved["critic_distributional_loss"] == "c51"
    _, actor2, wrong_support = models(loss="c51", c51_v_min=-10, c51_v_max=10)
    with pytest.raises(ValueError, match="support differs"):
        SAC(actor2, wrong_support, SimpleNamespace(), device="cpu").load(saved, load_cfg=None, strict=True)
    _, actor3, scalar_target = models(loss="two_hot", distributional_num_bins=101)
    with pytest.raises(ValueError, match="C51 critic checkpoint"):
        SAC(actor3, scalar_target, SimpleNamespace(), device="cpu").load(saved, load_cfg=None, strict=True)


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


def test_hl_gauss_labels_are_a_normalized_gaussian_over_neighbouring_atoms(device):
    _, _, critic = models(device, loss="hl_gauss")
    b = critic.value_support
    values = torch.cat((b, (b[:-1] + b[1:]) / 2, b.new_tensor([-1e12, -10, -2, 0, 0.045, 1e12])))[:, None]
    values.requires_grad_()
    labels = critic.hl_gauss(values)
    assert not labels.requires_grad
    assert (labels >= 0).all()
    torch.testing.assert_close(labels.sum(-1), torch.ones_like(values[:, 0]))
    # sigma/spacing = 0.75 is meant to put the mass on roughly three neighbours, and that is
    # the whole difference from two-hot, which never uses more than two.
    spread = torch.exp(-(labels.clamp_min(1e-12) * labels.clamp_min(1e-12).log()).sum(-1))
    assert 2.5 < spread.mean() < 4.0
    assert (critic.two_hot(values) > 0).sum(-1).max() <= 2
    # Targets past the support land on its edge, exactly as two-hot clamps them.
    torch.testing.assert_close(labels[values[:, 0] == 1e12], labels[values[:, 0] == b[-1]])


def test_hl_gauss_decoded_mean_tracks_the_target_it_encodes(device):
    _, _, critic = models(device, loss="hl_gauss")
    b = critic.value_support
    assert critic.label_decode_bias() < 1e-3
    assert models(device, loss="two_hot")[2].label_decode_bias() < 1e-6
    # What survives the correction is a systematic outward skew, so measure it signed and on
    # one side only: a probe symmetric about zero would cancel it and look perfect. It has to
    # stay well under the sigma^2/2 the correction removes, since this part does compound.
    probe = torch.linspace(1.0, b[-1].item() * 0.5, 2000, device=device)[:, None]
    signed = (((critic.hl_gauss(probe) * b).sum(-1, keepdim=True) - probe) / probe).mean().item()
    assert 0 < signed < critic.hl_gauss_sigma**2 / 2 / 3
    # The property the whole scalar-target design rests on: a rare catastrophic penalty keeps
    # its arithmetic weight, so 99% of +1 and 1% of -100 still decodes to about -0.01.
    mixture = 0.99 * critic.hl_gauss(b.new_tensor([[1.0]])) + 0.01 * critic.hl_gauss(b.new_tensor([[-100.0]]))
    torch.testing.assert_close(critic.q_from_output(mixture.log()), b.new_tensor([[-0.01]]), atol=1e-3, rtol=0)


def test_hl_gauss_skew_correction_is_what_keeps_coarse_supports_usable(device):
    # A symlog-symmetric Gaussian is right-skewed in reward units. Without the correction the
    # decoded mean is multiplied by exp(sigma^2/2) on every backup, which compounds.
    coarse = dict(loss="hl_gauss", distributional_num_bins=51, distributional_symlog_limit=5.0)
    _, _, critic = models(device, **coarse)
    sigma = critic.hl_gauss_sigma
    probe = torch.linspace(-4.0, 4.0, 400, device=device)
    probe = (probe.sign() * probe.abs().expm1())[:, None]

    def decode(centers):
        cdf = torch.erf((critic.support_edges_symlog - centers) / (2**0.5 * sigma))
        labels = cdf[..., 1:] - cdf[..., :-1]
        return ((labels / labels.sum(-1, keepdim=True)) * critic.value_support).sum(-1)

    raw_centers = probe.sign() * probe.abs().log1p()
    gap_uncorrected = (decode(raw_centers) - probe[:, 0]).abs()
    gap_corrected = ((critic.hl_gauss(probe) * critic.value_support).sum(-1) - probe[:, 0]).abs()
    # Past |y| = 1 the error is multiplicative, and that is the part a Bellman backup compounds.
    large = probe[:, 0].abs() >= 1.0
    relative = lambda gap: (gap[large] / probe[large, 0].abs()).max().item()  # noqa: E731
    assert relative(gap_uncorrected) > 1e-2
    assert relative(gap_corrected) < relative(gap_uncorrected) / 5
    # Near zero the leftover is a fixed offset of the order of the sigma^2/2 shift itself,
    # rather than an error that grows with the target.
    assert gap_corrected[~large].max() <= sigma**2
    assert critic.label_decode_bias() >= relative(gap_corrected)


def test_hl_gauss_gradients_stay_bounded_and_checkpoints_stay_interchangeable(device):
    _, _, critic = models(device, loss="hl_gauss")
    targets = torch.tensor([0.01, -2, -10, -1e6, 1e6, 1e12], device=device)[:, None]
    logits = torch.zeros(6, 255, device=device, requires_grad=True)
    labels = critic.hl_gauss(targets)
    grad, = torch.autograd.grad(-(labels * logits.log_softmax(-1)).sum(), logits)
    torch.testing.assert_close(grad, logits.softmax(-1) - labels)
    assert grad.abs().max() <= 1
    assert (grad.abs().sum(-1) <= 2.00001).all()
    # Bin edges are derived, not saved, so a run can switch label scheme and resume.
    _, _, two_hot_critic = models(device, loss="two_hot")
    assert "support_edges_symlog" not in critic.state_dict()
    critic.load_state_dict(two_hot_critic.state_dict(), strict=True)


@pytest.mark.parametrize("ratio,expected", [(0.375, 2.0), (0.75, 3.3), (1.5, 6.3)])
def test_hl_gauss_sigma_ratio_sets_how_many_atoms_carry_mass(ratio, expected):
    _, _, critic = models(loss="hl_gauss", hl_gauss_sigma_ratio=ratio)
    labels = critic.hl_gauss(torch.tensor([[1.0], [-3.0], [0.2]]))
    spread = torch.exp(-(labels.clamp_min(1e-12) * labels.clamp_min(1e-12).log()).sum(-1))
    assert spread.mean().item() == pytest.approx(expected, abs=0.4)


def test_scalar_default_is_exact_legacy_mse_and_state_dict(device):
    obs, _, critic = models(device, loss="mse")
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


def test_popart_starts_as_the_scalar_mse_critic(device):
    torch.manual_seed(3)
    obs, _, scalar = models(device, loss="mse")
    torch.manual_seed(3)
    _, _, popart = models(device, loss=POPART)
    actions = torch.randn(8, 2, device=device)
    for a, b in zip(popart.evaluate_all_q(obs, actions), scalar.evaluate_all_q(obs, actions)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert set(popart.state_dict()) == set(scalar.state_dict()) | {"popart_mean", "popart_std"}
    assert not popart.distributional_critic_ce and popart.critic1[-1].out_features == 1


def test_popart_statistics_follow_the_paper_and_rescaling_keeps_every_q(device):
    # Proposition 1 is exact algebra, so check it in float64 across large jumps of the statistics.
    torch.manual_seed(4)
    obs, _, critic = models(device, loss=POPART, popart_beta=0.2)
    critic.double()
    obs = TensorDict({"policy": obs["policy"].double()}, batch_size=[8])
    actions = torch.randn(8, 2, device=device, dtype=torch.float64)
    with torch.no_grad():  # a lagging target, so two different functions must survive
        for p in (*critic.critic1_target.parameters(), *critic.critic2_target.parameters()):
            p.add_(0.1 * torch.randn_like(p))
    mean, nu = 0.0, 1.0  # the paper's Eq. 4 state: running first and second moments
    for batch_mean, batch_std in ((3.0, 0.5), (-40.0, 25.0), (-40.0, 1e-3), (1e3, 2e2)):
        before = (*critic.evaluate_all_q(obs, actions), *critic.evaluate_all_target_q(obs, actions))
        second = batch_mean**2 + batch_std**2
        critic.update_popart(*torch.tensor([batch_mean, second], dtype=torch.float64, device=device))
        mean, nu = 0.8 * mean + 0.2 * batch_mean, 0.8 * nu + 0.2 * second
        assert critic.popart_mean.item() == pytest.approx(mean, rel=1e-12)
        assert critic.popart_std.item() == pytest.approx(math.sqrt(nu - mean**2), rel=1e-9)
        after = (*critic.evaluate_all_q(obs, actions), *critic.evaluate_all_target_q(obs, actions))
        for old, new in zip(before, after):
            torch.testing.assert_close(new, old, atol=1e-9, rtol=1e-12)
    # Targets without spread cannot pull the scale to zero.
    critic.popart_beta = 1.0
    critic.update_popart(*torch.tensor([5.0, 25.0], dtype=torch.float64, device=device))
    assert critic.popart_mean.item() == 5.0 and critic.popart_std.item() == pytest.approx(1e-4)


def test_popart_loss_is_mse_on_normalized_targets(device):
    obs, _, critic = models(device, loss=POPART)
    critic.popart_mean.fill_(2.0)
    critic.popart_std.fill_(4.0)
    actions = torch.randn(8, 2, device=device)
    targets = torch.tensor([0.01, -2, -10, -1e3, 1, 2, 3, 4], device=device)[:, None]
    output1, output2 = critic.critic_outputs(obs, actions)
    loss1, loss2 = critic.losses_from_outputs(output1, output2, targets)
    torch.testing.assert_close(loss1, torch.nn.functional.mse_loss(output1, (targets - 2) / 4))
    torch.testing.assert_close(loss2, torch.nn.functional.mse_loss(output2, (targets - 2) / 4))
    # Every raw residual is divided by the same std: unlike CE's bounded p - t, a sample with a
    # 10x larger TD error keeps a 10x larger output gradient.
    q1 = critic.q_from_output(output1)
    torch.testing.assert_close(output1 - (targets - 2) / 4, (q1 - targets) / 4)


@pytest.mark.parametrize("loss", ["mse", *CATEGORICAL, POPART])
def test_complete_sac_update_bootstrap_and_checkpoint(device, loss):
    torch.manual_seed(21)
    obs, actor, critic = models(device, loss=loss)
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
    loss_time_std = []

    def capture(output1, output2, y):
        seen.append(y.clone())
        if critic.popart:
            loss_time_std.append(critic.popart_std.clone())
        return original_loss(output1, output2, y)

    critic.losses_from_outputs = capture
    before = [p.clone() for p in actor.parameters()]
    losses = alg.update()
    # PopArt rescales the target heads before the loss, so this also checks that it kept them.
    torch.testing.assert_close(seen[0], expected)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))
    if loss in CATEGORICAL:
        assert losses["critic_target_clipped_fraction"] == 1 / 8
    else:
        assert "critic_target_clipped_fraction" not in losses
    if loss == POPART:
        # Algorithm 1 order: the loss already used the statistics this batch produced.
        assert loss_time_std[0].item() == losses["PopArt/std"] != 1.0
        assert losses["PopArt/mean"] == critic.popart_mean.item() != 0.0
    else:
        assert "PopArt/std" not in losses
    assert all(p.grad is None for p in critic.critic1_target.parameters())
    saved = copy.deepcopy(alg.save())
    _, actor2, critic2 = models(device, loss=loss)
    resumed = SAC(actor2, critic2, replay, device=device)
    assert resumed.load(saved, load_cfg=None, strict=True)
    for original, restored in zip(critic.evaluate_all_q(obs, actions), critic2.evaluate_all_q(obs, actions)):
        torch.testing.assert_close(original, restored, atol=0, rtol=0)
    assert resumed.critic_optimizer.state_dict()["state"]
    assert all(torch.isfinite(torch.tensor(v)) for v in resumed.update().values())


def build_runner(monkeypatch, **cfg_extra):
    obs, _, _ = models()
    env = SimpleNamespace(num_actions=2, num_envs=8)
    monkeypatch.setattr(SAC, "_compute_action_scaling", lambda env, device: (torch.ones(2), torch.ones(2)))
    cfg = dict(
        num_steps_per_env=2,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name="SACActorModel", hidden_dims=[8]),
        critic=dict(class_name="SACCriticModel", hidden_dims=[8]),
        algorithm=dict(class_name="SAC", replay_buffer_size=64),
    )
    cfg.update(cfg_extra)
    return SAC.construct_algorithm(obs, env, cfg, "cpu")


@pytest.mark.parametrize("loss", ["mse", *CATEGORICAL, POPART])
def test_runner_flag_constructs_requested_head(monkeypatch, loss):
    alg = build_runner(monkeypatch, critic=dict(class_name="SACCriticModel", hidden_dims=[8],
                                                distributional_loss=loss, popart_beta=1e-3))
    assert alg.critic.distributional_loss == loss
    assert alg.critic.distributional_critic_ce == (loss in CATEGORICAL)
    assert alg.critic.popart == (loss == POPART)
    assert alg.critic.critic1[-1].out_features == (255 if loss in CATEGORICAL else 1)
    if loss == POPART:
        assert alg.critic.popart_beta == 1e-3


def test_c51_runner_flag_uses_reference_support_without_changing_other_sac_options(monkeypatch):
    alg = build_runner(monkeypatch, critic=dict(class_name="SACCriticModel", hidden_dims=[8],
                                                distributional_loss="c51"),
                       algorithm=dict(class_name="SAC", replay_buffer_size=64,
                                      q_reduction_method="mean_pi_q_none", target_entropy_scale=0.166))
    assert alg.critic.distributional_loss == "c51"
    assert alg.critic.critic1[-1].out_features == 101
    torch.testing.assert_close(alg.critic.value_support[[0, -1]], torch.tensor([-20., 20.]))
    assert alg.q_reduction_method == "mean_pi_q_none"


def test_deprecated_boolean_still_selects_two_hot(monkeypatch):
    alg = build_runner(monkeypatch, distributional_critic_ce=True)
    assert alg.critic.distributional_loss == "two_hot"
    with pytest.raises(ValueError):
        build_runner(monkeypatch, distributional_critic_ce=True,
                     critic=dict(class_name="SACCriticModel", hidden_dims=[8], distributional_loss="hl_gauss"))


@pytest.mark.parametrize("kwargs", [
    {"distributional_num_bins": 2}, {"distributional_num_bins": 254},
    {"distributional_symlog_limit": 0}, {"distributional_symlog_limit": float("nan")},
    {"distributional_symlog_limit": 90},
    {"loss": "hl_gauss", "hl_gauss_sigma_ratio": 0}, {"loss": "hl_gauss", "hl_gauss_sigma_ratio": -1},
    {"loss": "hl_gauss", "hl_gauss_sigma_ratio": float("inf")}, {"loss": "hl_gaus"},
    {"loss": POPART, "popart_beta": 0}, {"loss": POPART, "popart_beta": 1.5},
    {"loss": POPART, "popart_beta": float("nan")},
    {"loss": "c51", "c51_num_atoms": 1},
    {"loss": "c51", "c51_v_min": 2, "c51_v_max": 1},
    {"loss": "c51", "c51_v_min": float("nan")},
])
def test_invalid_support_rejected(kwargs):
    with pytest.raises(ValueError):
        models(**kwargs)


@pytest.mark.parametrize("envs", [1, 4])
def test_timeout_stores_the_pre_reset_observation_and_flags_the_bootstrap(envs):
    """A timeout must keep the state it reached, not the state the auto-reset produced."""
    obs = TensorDict({"policy": torch.randn(envs, 4)}, batch_size=[envs])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[8])
    critic = SACCriticModel(obs, groups, "critic", 1, hidden_dims=[8], num_actions=2)
    stored = {}
    replay = SimpleNamespace(add_transition=lambda t: stored.update(vars(t).copy()))
    alg = SAC(actor, critic, replay, device="cpu")
    alg.transition = SimpleNamespace(rewards=None, next_observations=None, dones=None,
                                     bootstrap=None, clear=lambda: None)

    pre_reset = TensorDict({"policy": torch.full((envs, 4), 7.0)}, batch_size=[envs])
    post_reset = TensorDict({"policy": torch.zeros(envs, 4)}, batch_size=[envs])
    dones = torch.zeros(envs, dtype=torch.long)
    dones[0] = 1
    # The wrapper publishes (num_envs,); squeezing that collapsed to a scalar for one env.
    time_outs = torch.zeros(envs, dtype=torch.bool)
    time_outs[0] = True
    alg.process_env_step(
        post_reset, torch.zeros(envs, 1), dones,
        {"time_outs": time_outs, "time_outs_obs": pre_reset},
    )

    kept = stored["next_observations"]["policy"]
    torch.testing.assert_close(kept[0], torch.full((4,), 7.0))  # timed out: pre-reset state
    if envs > 1:
        torch.testing.assert_close(kept[1], torch.zeros(4))  # still running: ordinary next obs
    # bootstrap must mark the timeout, or update() treats it as a terminal state worth zero.
    assert stored["bootstrap"].reshape(-1)[0] == 1
    assert stored["bootstrap"].shape == dones.shape


def test_missing_timeout_obs_falls_back_to_terminal_treatment():
    """Documents the failure mode: no time_outs_obs means every timeout looks terminal."""
    obs = TensorDict({"policy": torch.randn(2, 4)}, batch_size=[2])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", 2, hidden_dims=[8])
    critic = SACCriticModel(obs, groups, "critic", 1, hidden_dims=[8], num_actions=2)
    stored = {}
    replay = SimpleNamespace(add_transition=lambda t: stored.update(vars(t).copy()))
    alg = SAC(actor, critic, replay, device="cpu")
    alg.transition = SimpleNamespace(rewards=None, next_observations=None, dones=None,
                                     bootstrap=None, clear=lambda: None)

    dones = torch.tensor([1, 0])
    alg.process_env_step(obs, torch.zeros(2, 1), dones, {"time_outs": torch.tensor([True, False])})
    # bootstrap_mask = bootstrap + 1 - done is then zero at the timeout, zeroing its target.
    assert stored["bootstrap"].sum() == 0


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
    _, actor2, scalar_critic = models(device, loss="mse")
    scalar_losses = SAC(actor2, scalar_critic, replay, device=device, policy_frequency=1).update()
    assert not any(name in scalar_losses for name in DISTRIBUTION_STAT_NAMES)
