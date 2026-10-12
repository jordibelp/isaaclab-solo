"""Independent arithmetic, timeout observations, and logging-only parity checks."""
import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.models import SACActorModel, SACCriticModel
from rsl_rl_sac.storage import ReplayBuffer
from rsl_rl_sac.utils.return_metrics import (
    EpisodeReturnMetrics, install_return_metrics, theoretical_return_limits, timeout_bootstrap,
)


@pytest.fixture(params=['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU'))])
def device(request):
    return request.param


def config(**overrides):
    return SimpleNamespace(
        **{'track_lin_vel_xy_reward_scale': 1.5, 'track_ang_vel_z_reward_scale': 0.5,
           'two_feet_above_height_reward_scale': 1.5, 'undesired_contact_reward_scale': -2.25,
           'three_or_more_feet_contact_penalty_reward_scale': -100.0,
           'base_collision_terminal_penalty': -10.0, 'episode_length_s': 10.0,
           'three_or_more_feet_contact_triggers_reset': True,
           'three_or_more_feet_contact_triggers_after': 1.0, **overrides})


def test_limits_follow_active_scales_and_reset_semantics():
    cfg = config()
    limits = theoretical_return_limits(cfg, 0.97, 0.02)
    assert limits['Returns/theoretical_return_lim_pos'] == pytest.approx(7 / 3)
    assert limits['Returns/theoretical_return_lim_neg'] == pytest.approx(-12.18)
    assert theoretical_return_limits(vars(cfg), 0.97, 0.02) == limits
    cfg.two_feet_above_height_reward_scale = 2.0
    cfg.track_lin_vel_xy_reward_scale = 1.2
    assert theoretical_return_limits(cfg, 0.97, 0.02)['Returns/theoretical_return_lim_pos'] == pytest.approx(7.4 / 3)
    cfg.three_or_more_feet_contact_triggers_reset = False
    assert theoretical_return_limits(cfg, 0.97, 0.02)['Returns/theoretical_return_lim_neg'] == pytest.approx(-2.18 / 0.03)
    cfg.three_or_more_feet_contact_triggers_after = 11.0
    assert theoretical_return_limits(cfg, 0.97, 0.02)['Returns/theoretical_return_lim_neg'] == pytest.approx(-10.18)


def test_negated_positive_rewards_do_not_normalize_by_zero():
    limits = theoretical_return_limits(config(negate_positive_rewards=True), 0.97, 0.02)
    assert limits['Returns/theoretical_return_lim_pos'] == 0.0
    assert limits['Returns/theoretical_return_lim_neg'] < -12.18


def test_discounted_episodes_span_log_windows_and_reset_independently(device):
    tracker = EpisodeReturnMetrics(2, 0.5, device)
    tensor = lambda values: torch.tensor(values, device=device)
    tracker.step(tensor([1., 10.]), tensor([False, True]), tensor([False, False]), tensor([100., 100.]))
    assert tracker.collect()['Returns/actual_return'] == 10.0
    tracker.step(tensor([2., 4.]), tensor([False, False]), tensor([False, False]), tensor([0., 0.]))
    assert 'Returns/actual_return' not in tracker.collect()
    tracker.step(tensor([3., 8.]), tensor([True, True]), tensor([True, False]), tensor([6., 100.]))
    # Env 0: 1 + .5*2 + .25*3 + .125*6 = 3.5. Env 1: 4 + .5*8 = 8.
    metrics = tracker.collect()
    assert metrics['Returns/actual_return'] == pytest.approx(5.75)
    assert metrics['Returns/reward_only_return'] == pytest.approx((2.75 + 8) / 2)
    assert metrics['Returns/bootstrap_contribution'] == pytest.approx(0.75 / 2)
    assert metrics['Returns/actual_return_terminal'] == 8.0
    assert metrics['Returns/actual_return_truncated'] == 3.5
    assert metrics['Returns/actual_return_min'] == 3.5
    assert metrics['Returns/actual_return_max'] == 8.0
    assert metrics['Returns/completed_episodes'] == 2
    assert tracker.collect()['Returns/completed_episodes'] == 0


def test_sac_soft_return_excludes_initial_entropy_and_discounts_boundary(device):
    tracker = EpisodeReturnMetrics(1, 0.5, device, sac=True)
    t = lambda value: torch.tensor([value], device=device)
    tracker.step(t(1.), t(False), t(False), t(0.), t(100.), t(0.))
    tracker.step(t(2.), t(True), t(True), t(6.), t(4.), t(8.))
    metrics = tracker.collect()
    assert metrics['Returns/actual_return'] == 1 + 0.5 * 2 + 0.25 * 6
    assert metrics['Returns/actual_soft_return'] == 1 + 0.5 * (2 + 4) + 0.25 * (6 + 8)


class Writer:
    def __init__(self):
        self.rows = {}

    def add_scalar(self, name, value, step):
        self.rows[(name, step)] = float(value)


def make_algorithm(device, spacing, method, batch_norm=False):
    obs = TensorDict({'policy': torch.zeros(2, 3, device=device)}, batch_size=[2])
    groups = {'actor': ['policy'], 'critic': ['policy']}
    actor = SACActorModel(obs, groups, 'actor', 2, hidden_dims=[8]).to(device)
    critic = SACCriticModel(obs, groups, 'critic', 1, num_actions=2, hidden_dims=[8],
                           distributional_loss='mse' if spacing == 'mse' else 'two_hot',
                           distributional_num_bins=11, distributional_symlog_limit=3.,
                           distributional_linear_limit=10. if spacing == 'linear' else 0.,
                           batch_norm=batch_norm).to(device)
    replay = ReplayBuffer(2, 4, obs, (2,), device, 32, n_steps=1, gamma=0.5)
    return SAC(actor, critic, replay, device=device, gamma=0.5, num_mini_batches=2,
               mini_batch_size=4, q_reduction_method=method), obs


@pytest.mark.parametrize('spacing', ['symexp', 'linear', 'mse'])
@pytest.mark.parametrize('method', ['min', 'mean', 'mean_pi_q_none'])
def test_sac_timeout_uses_production_decoder_and_reduction(device, spacing, method):
    alg, obs = make_algorithm(device, spacing, method)
    final_obs = TensorDict({'policy': torch.full((2, 3), 7., device=device)}, batch_size=[2])
    timeout = torch.tensor([False, True], device=device)
    alg.actor(obs, stochastic_output=True)
    original_distribution = alg.actor.distribution
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device == 'cuda' else None
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == 'cuda' else []):
        action, _ = alg.actor.sample_action_logp(final_obs[timeout])
        latent = alg.critic.get_latent(final_obs[timeout])
        inputs = torch.cat((latent, action), dim=-1)
        outputs = [alg.critic.critic1(inputs), alg.critic.critic2(inputs)]
        if spacing == 'mse':
            qs = outputs
        else:
            qs = [(out.softmax(-1) * alg.critic.value_support).sum(-1, keepdim=True) for out in outputs]
        expected = torch.minimum(*qs) if method == 'min' else (qs[0] + qs[1]) / 2
    alg.actor.distribution = original_distribution
    value, _ = timeout_bootstrap(alg, final_obs, timeout)
    assert value[0] == 0
    torch.testing.assert_close(value[1], expected.reshape(()), rtol=1e-5, atol=1e-5)
    assert alg.actor.distribution is original_distribution
    assert torch.equal(torch.get_rng_state(), rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)


@pytest.mark.parametrize('batch_norm', [False, True])
def test_sac_logging_does_not_change_replay_normalizers_updates_or_rng(device, batch_norm):
    logged, obs = make_algorithm(device, 'symexp', 'min', batch_norm)
    baseline = copy.deepcopy(logged)
    env = SimpleNamespace(cfg=config(), step_dt=0.02, num_envs=2)
    env.unwrapped = env
    writer = Writer()
    logger = SimpleNamespace(writer=writer, log=lambda **kw: None, disable_logs=False)
    runner = SimpleNamespace(alg=logged, env=env, device=device, logger=logger)
    install_return_metrics(runner)
    for alg in (baseline, logged):
        torch.manual_seed(372)
        for step in range(3):
            alg.act(obs)
            extras = {'time_outs': torch.tensor([False, True], device=device),
                      'terminated': torch.tensor([False, False], device=device),
                      'time_outs_obs': obs + 7}
            alg.process_env_step(obs, torch.tensor([1., 2.], device=device),
                                 torch.tensor([0, 1], device=device), extras)
        metrics = alg.update()
        if alg is baseline:
            expected_metrics = metrics
            expected_rng = torch.get_rng_state()
            expected_cuda_rng = torch.cuda.get_rng_state(device) if device == 'cuda' else None
        else:
            assert metrics == expected_metrics
            assert torch.equal(torch.get_rng_state(), expected_rng)
            if expected_cuda_rng is not None:
                assert torch.equal(torch.cuda.get_rng_state(device), expected_cuda_rng)
    for model in ('actor', 'critic'):
        for name, value in getattr(baseline, model).state_dict().items():
            torch.testing.assert_close(value, getattr(logged, model).state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(baseline.log_alpha, logged.log_alpha, rtol=0, atol=0)
    for a, b in zip(baseline.replay_buffer.replay_buf, logged.replay_buffer.replay_buf):
        if isinstance(a, TensorDict):
            for key in a.keys():
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    logger.log(it=3, loss_dict=metrics)
    assert writer.rows[('Returns/theoretical_return_lim_neg', 3)] == pytest.approx(-12.18)
    assert ('Returns/actual_return', 3) in writer.rows
    assert writer.rows[('CriticQ/policy_mean_over_theoretical_pos', 3)] == pytest.approx(
        metrics['CriticQ/policy_mean'] / writer.rows[('Returns/theoretical_return_lim_pos', 3)])


def test_ppo_final_observation_and_simultaneous_termination_precedence(device):
    class Policy(torch.nn.Module):
        is_recurrent = False

        def evaluate(self, obs):
            return obs['policy'][:, :1]

    obs = TensorDict({'policy': torch.tensor([[6.], [900.]], device=device)}, batch_size=[2])
    seen = []
    alg = SimpleNamespace(policy=Policy(), gamma=0.5, transition=SimpleNamespace(values=torch.ones(2, 1, device=device)),
                          process_env_step=lambda *args: seen.append(args))
    env = SimpleNamespace(cfg=config(), step_dt=0.02, num_envs=2)
    env.unwrapped = env
    writer = Writer()
    runner = SimpleNamespace(alg=alg, env=env, device=device, writer=writer, log=lambda locs: None, disable_logs=False)
    install_return_metrics(runner)
    dones = torch.tensor([1, 1], device=device)
    alg.process_env_step(obs * 0, torch.tensor([2., 3.], device=device), dones,
                         {'time_outs': torch.tensor([True, True], device=device),
                          'terminated': torch.tensor([False, True], device=device), 'time_outs_obs': obs})
    runner.log({'it': 4})
    assert seen[0][2] is dones
    assert writer.rows[('Returns/actual_return_truncated', 4)] == 2 + 0.5 * 6
    assert writer.rows[('Returns/actual_return_terminal', 4)] == 3
    assert writer.rows[('Returns/bootstrap_contribution', 4)] == 1.5
    env.cfg.two_feet_above_height_reward_scale = 2.0
    env.cfg.track_lin_vel_xy_reward_scale = 1.2
    runner.log({'it': 5})
    assert writer.rows[('Returns/theoretical_return_lim_pos', 5)] == pytest.approx(3.7 * 0.02 / 0.5)
    assert ('Returns/actual_return', 5) not in writer.rows


@pytest.mark.parametrize('batch_norm', [False, True])
def test_real_ppo_logging_preserves_values_gae_normalization_and_updates(device, batch_norm):
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic

    obs = TensorDict({'policy': torch.zeros(2, 3, device=device)}, batch_size=[2])
    policy = ActorCritic(obs, {'policy': ['policy'], 'critic': ['policy']}, 2,
                         actor_hidden_dims=[8], critic_hidden_dims=[8],
                         actor_obs_normalization=True, critic_obs_normalization=True).to(device)
    if batch_norm:
        policy.critic.insert(1, torch.nn.BatchNorm1d(8).to(device))
    logged = PPO(policy, device=device, gamma=0.5, num_learning_epochs=1, num_mini_batches=1)
    logged.init_storage('rl', 2, 3, obs, [2])
    baseline = copy.deepcopy(logged)
    env = SimpleNamespace(cfg=config(), step_dt=0.02, num_envs=2)
    env.unwrapped = env
    runner = SimpleNamespace(alg=logged, env=env, device=device, writer=Writer(),
                             log=lambda locs: None, disable_logs=False)
    install_return_metrics(runner)
    for alg in (baseline, logged):
        torch.manual_seed(387)
        for _ in range(3):
            alg.act(obs)
            alg.process_env_step(obs, torch.tensor([1., 2.], device=device),
                                 torch.tensor([0, 1], device=device),
                                 {'time_outs': torch.tensor([False, True], device=device),
                                  'terminated': torch.zeros(2, device=device, dtype=torch.bool),
                                  'time_outs_obs': obs + 7})
        alg.compute_returns(obs)
        metrics = alg.update()
        if alg is baseline:
            expected_metrics, expected_rng = metrics, torch.get_rng_state()
            expected_cuda_rng = torch.cuda.get_rng_state(device) if device == 'cuda' else None
        else:
            assert metrics == expected_metrics
            assert torch.equal(torch.get_rng_state(), expected_rng)
            if expected_cuda_rng is not None:
                assert torch.equal(torch.cuda.get_rng_state(device), expected_cuda_rng)
    for name, value in baseline.policy.state_dict().items():
        torch.testing.assert_close(value, logged.policy.state_dict()[name], rtol=0, atol=0)
    runner.log({'it': 5})
    assert ('Returns/actual_return_truncated', 5) in runner.writer.rows
    assert ('CriticV/over_theoretical_pos', 5) in runner.writer.rows


@pytest.mark.parametrize('rnn_type', ['lstm', 'gru'])
def test_recurrent_ppo_timeout_preserves_environment_aligned_hidden_states(device, rnn_type):
    from rsl_rl.modules import ActorCriticRecurrent

    obs = TensorDict({'policy': torch.zeros(2, 3, device=device)}, batch_size=[2])
    policy = ActorCriticRecurrent(obs, {'policy': ['policy'], 'critic': ['policy']}, 2,
                                  actor_hidden_dims=[8], critic_hidden_dims=[8],
                                  rnn_hidden_size=4, rnn_type=rnn_type).to(device)
    with torch.no_grad():
        policy.act(obs)
        policy.evaluate(obs)
    saved = policy.get_hidden_states()
    baseline = copy.deepcopy(policy)
    final_obs = obs + 7
    with torch.no_grad():
        expected = baseline.eval().evaluate(final_obs).reshape(-1)
    value, _ = timeout_bootstrap(SimpleNamespace(policy=policy), final_obs,
                                 torch.tensor([False, True], device=device))
    assert value[0] == 0
    torch.testing.assert_close(value[1], expected[1])
    for before, after in zip(saved, policy.get_hidden_states()):
        assert before is after
    assert all(module.training for module in policy.modules())
