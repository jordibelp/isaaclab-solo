"""Solo12 discounted-return diagnostics, independent of training targets.

The positive limit uses bounded rewards and an infinite discounted continuation.
The negative limit is a collision-only estimate, NOT a bound on the full reward:
it excludes action-rate, torque, joint-limit, and contact-force costs. Both use
the active scalar configuration (after curriculum updates), not saved defaults.

actual_return = sum(t=0..T-1, gamma**t * reward[t]) + gamma**T * bootstrap.
True terminations have zero bootstrap. Timeouts use the final PRE-RESET state:
PPO V, or SAC Q for one sampled policy action, reduced like the actor objective.
SAC Q already includes future entropy; a truncated actual_return is therefore
a reward prefix plus an estimated soft-Q tail, not a fully observed reward G.
actual_soft_return adds rollout entropy at t>0 and boundary entropy at timeouts,
making it comparable to SAC Q(s_0,a_0). Neither estimate is a GAE/lambda return.
Completed episodes are pooled by count within each logging interval; unfinished
episodes carry across iterations. Parameters may change during an episode.
"""

from contextlib import contextmanager

import torch


RETURN_METRICS_CONFIG = {
    'negative_limit_kind': 'collision_only_estimate',
    'negative_limit_excludes': ['action_rate', 'torque', 'soft_joint_limits', 'foot_contact_force'],
    'positive_limit_horizon': 'infinite_discounted_continuation',
    'actual_return': 'sum(gamma**t * reward[t]) + gamma**T * Q_or_V(final_pre_reset_state)',
    'sac_bootstrap': 'one sampled policy action; current twin Q with actor reduction; no boundary entropy',
    'sac_soft_return': 'entropy at rollout t>0 and at the timeout boundary; comparable to initial Q(s,a)',
    'aggregation': 'completed episodes pooled by count within each logging interval',
}


def theoretical_return_limits(cfg, gamma: float, step_dt: float) -> dict[str, float]:
    """Return the positive ceiling and the explicitly approximate negative limit."""
    get = (lambda name, default=0.0: cfg.get(name, default)) if isinstance(cfg, dict) else (
        lambda name, default=0.0: getattr(cfg, name, default))
    scales = [float(get(name)) for name in (
        'track_lin_vel_xy_reward_scale', 'track_ang_vel_z_reward_scale',
        'two_feet_above_height_reward_scale', 'track_base_height_reward_scale',
    )]
    if not 0 <= gamma < 1:
        raise ValueError('Discounted theoretical return limits require 0 <= gamma < 1.')
    positive = sum(max(w, 0.0) for w in scales) * step_dt
    bounded_negative = sum(min(w, 0.0) for w in scales) * step_dt
    if get('negate_positive_rewards', False):
        bounded_negative -= positive
        positive = 0.0
    thighs = 4 * min(float(get('undesired_contact_reward_scale')), 0.0) * step_dt
    forbidden = min(float(get('three_or_more_feet_contact_penalty_reward_scale')), 0.0) * step_dt
    reset = bool(get('three_or_more_feet_contact_triggers_reset', False))
    if float(get('three_or_more_feet_contact_triggers_after', 0.0)) > float(get('episode_length_s', float('inf'))):
        forbidden = 0.0
    base = min(float(get('base_collision_terminal_penalty')), 0.0)
    terminal = base + forbidden + thighs + bounded_negative
    continuing = (thighs + bounded_negative + (0.0 if reset else forbidden)) / (1 - gamma)
    return {
        'Returns/theoretical_return_lim_pos': positive / (1 - gamma),
        'Returns/theoretical_return_lim_neg': min(terminal, continuing),
    }


def discounted_return_step(reward_sum, discount, reward, done, timeout, bootstrap, gamma, where):
    """Backend-neutral recurrence, shared by Torch and the MJX PPO scan."""
    reward_sum = reward_sum + discount * reward
    discount = discount * gamma
    tail = where(timeout, discount * bootstrap, 0.0)
    completed = reward_sum + tail
    return where(done, 0.0, reward_sum), where(done, 1.0, discount), completed, reward_sum, tail


@contextmanager
def diagnostic_inference(*models):
    """Preserve RNG, module modes, recurrent state, and actor distribution caches."""
    modules = [module for model in models for module in model.modules()]
    modes = [module.training for module in modules]
    caches = [(module, name, getattr(module, name)) for module in modules
              for name in ('distribution', 'hidden_state') if hasattr(module, name)]
    devices = sorted({p.device.index for model in models for p in model.parameters() if p.is_cuda})
    try:
        for module in modules:
            module.training = False
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            yield
    finally:
        for module, mode in zip(modules, modes):
            module.training = mode
        for module, name, value in caches:
            setattr(module, name, value)


def timeout_bootstrap(algorithm, final_obs, timeout):
    """Return full-size Q/V and boundary-entropy vectors without training side effects."""
    device = timeout.device
    value = torch.zeros(timeout.numel(), device=device)
    entropy = torch.zeros_like(value)
    if hasattr(algorithm, 'actor'):
        from rsl_rl_sac.algorithms.sac import reduce_twin_q

        with diagnostic_inference(algorithm.actor, algorithm.critic):
            obs = final_obs[timeout].to(device)
            actions, logp = algorithm.actor.sample_action_logp(obs)
            latent = algorithm.critic.get_latent(obs)
            inputs = torch.cat((latent, actions), dim=-1)
            q1 = algorithm.critic.q_from_output(algorithm.critic.critic1(inputs))
            q2 = algorithm.critic.q_from_output(algorithm.critic.critic2(inputs))
            value[timeout] = reduce_twin_q(q1, q2, algorithm.q_reduction_method).reshape(-1)
            entropy[timeout] = (-algorithm.log_alpha.exp() * logp).reshape(-1)
    else:
        policy = algorithm.policy
        with diagnostic_inference(policy):
            # Recurrent hidden states retain their full environment batch dimension.
            if policy.is_recurrent:
                value[timeout] = policy.evaluate(final_obs.to(device)).reshape(-1)[timeout]
            else:
                value[timeout] = policy.evaluate(final_obs[timeout].to(device)).reshape(-1)
    return value, entropy


class EpisodeReturnMetrics:
    """Device-side episode accumulators, pooled at writer cadence (not rollout cadence)."""

    def __init__(self, num_envs, gamma, device, *, sac=False):
        self.gamma, self.sac = float(gamma), sac
        self.reward_sum = torch.zeros(num_envs, device=device)
        self.soft_sum = torch.zeros_like(self.reward_sum)
        self.discount = torch.ones_like(self.reward_sum)
        self.steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        # count, G, reward prefix, tail, soft G, terminal G/count, timeout G/count
        self.sums = torch.zeros(9, device=device, dtype=torch.float64)
        self.extrema = torch.tensor([float('inf'), -float('inf')], device=device)
        self.values = torch.zeros(2, device=device, dtype=torch.float64)

    def step(self, reward, done, timeout, bootstrap, entropy=None, boundary_entropy=None, values=None):
        reward, done, timeout, bootstrap = [v.reshape(-1) for v in (reward, done, timeout, bootstrap)]
        done = done.bool()
        timeout = timeout.bool() & done
        previous_discount = self.discount
        self.reward_sum, self.discount, actual, prefix, tail = discounted_return_step(
            self.reward_sum, previous_discount, reward, done, timeout, bootstrap, self.gamma, torch.where)
        if entropy is not None:
            soft_reward = reward + torch.where(self.steps > 0, entropy.reshape(-1), 0.0)
            self.soft_sum, _, soft, _, _ = discounted_return_step(
                self.soft_sum, previous_discount, soft_reward, done, timeout,
                bootstrap + boundary_entropy.reshape(-1), self.gamma, torch.where)
        else:
            soft = actual
        terminal = done & ~timeout
        masked = lambda x, mask=done: torch.where(mask, x, 0.0).double().sum()
        self.sums += torch.stack((done.double().sum(), masked(actual), masked(prefix), masked(tail),
                                 masked(soft), masked(actual, terminal), terminal.double().sum(),
                                 masked(actual, timeout), timeout.double().sum()))
        self.extrema[0] = torch.minimum(self.extrema[0], torch.where(done, actual, float('inf')).amin())
        self.extrema[1] = torch.maximum(self.extrema[1], torch.where(done, actual, -float('inf')).amax())
        self.steps = torch.where(done, 0, self.steps + 1)
        if values is not None:
            self.values += torch.stack((values.double().sum(), values.new_tensor(values.numel()).double()))

    def collect(self):
        counts = self.sums.cpu().tolist()
        metrics = {'Returns/completed_episodes': counts[0], 'Returns/terminal_episodes': counts[6],
                   'Returns/truncated_episodes': counts[8]}
        if counts[0]:
            metrics.update(zip(('Returns/actual_return', 'Returns/reward_only_return',
                                'Returns/bootstrap_contribution'), [v / counts[0] for v in counts[1:4]]))
            metrics.update(zip(('Returns/actual_return_min', 'Returns/actual_return_max'), self.extrema.cpu().tolist()))
            if self.sac:
                metrics['Returns/actual_soft_return'] = counts[4] / counts[0]
        for name, total, count in (('terminal', counts[5], counts[6]), ('truncated', counts[7], counts[8])):
            if count:
                metrics[f'Returns/actual_return_{name}'] = total / count
        value_sum, value_count = self.values.cpu().tolist()
        if value_count:
            metrics['CriticV/mean'] = value_sum / value_count
        self.sums.zero_()
        self.extrema[:] = self.extrema.new_tensor([float('inf'), -float('inf')])
        self.values.zero_()
        return metrics


def install_return_metrics(runner):
    """Install for Isaac PPO/SAC and MJX SAC; no losses, targets, or replay changes."""
    raw_env = runner.env.unwrapped
    cfg = raw_env.cfg
    if not hasattr(cfg, 'two_feet_above_height_reward_scale'):
        return
    algorithm = runner.alg
    if hasattr(runner, 'cfg'):
        runner.cfg['return_metrics'] = RETURN_METRICS_CONFIG.copy()
    sac = hasattr(algorithm, 'actor')
    tracker = EpisodeReturnMetrics(runner.env.num_envs, algorithm.gamma, runner.device, sac=sac)
    runner.return_metrics = tracker
    original_step = algorithm.process_env_step

    def process_env_step(obs, reward, done, extras):
        original_done = done
        done = done.to(runner.device).bool().reshape(-1)
        timeout = extras.get('time_outs', torch.zeros_like(done)).to(runner.device).bool().reshape(-1)
        terminated = extras.get('terminated', getattr(raw_env, 'reset_terminated', None))
        if terminated is not None:
            timeout = timeout & ~terminated.to(runner.device).bool().reshape(-1)
        timeout = timeout & done
        bootstrap = torch.zeros_like(reward).reshape(-1)
        boundary_entropy = torch.zeros_like(bootstrap)
        if timeout.any():
            bootstrap, boundary_entropy = timeout_bootstrap(algorithm, extras['time_outs_obs'], timeout)
        entropy = values = None
        if sac:
            entropy = (-algorithm.log_alpha.exp().detach()
                       * algorithm.actor.executed_action_logp(algorithm.transition.actions)).reshape(-1)
        else:
            values = algorithm.transition.values.detach()
        tracker.step(reward, done, timeout, bootstrap, entropy, boundary_entropy, values)
        return original_step(obs, reward, original_done, extras)

    algorithm.process_env_step = process_env_step
    logger = runner.logger if sac else runner
    original_log = logger.log

    def log(*args, **kwargs):
        limits = theoretical_return_limits(cfg, algorithm.gamma, raw_env.step_dt)
        metrics = tracker.collect()
        metrics.update(limits)
        denominator = limits['Returns/theoretical_return_lim_pos']
        if denominator > 0:
            if sac:
                losses = kwargs.get('loss_dict', {})
                for name in ('policy_mean', 'reduced_mean'):
                    if f'CriticQ/{name}' in losses:
                        metrics[f'CriticQ/{name}_over_theoretical_pos'] = losses[f'CriticQ/{name}'] / denominator
            elif 'CriticV/mean' in metrics:
                metrics['CriticV/over_theoretical_pos'] = metrics['CriticV/mean'] / denominator
        result = original_log(*args, **kwargs)
        if logger.writer is not None and not getattr(logger, 'disable_logs', False):
            iteration = kwargs['it'] if sac else args[0]['it']
            for name, value in metrics.items():
                logger.writer.add_scalar(name, value, iteration)
        return result

    logger.log = log
