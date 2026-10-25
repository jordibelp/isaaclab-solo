# SPDX-License-Identifier: BSD-3-Clause
"""Real simulator checks of raw-action costs, reward delivery and reset isolation."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='Borinot-Goalkeeper-Direct-v0')
parser.add_argument('--mode', default='thrust_and_torque_control')
parser.add_argument('--sac', action='store_true')
parser.add_argument('--legacy-torque', action='store_true')
parser.add_argument('--hold-arm', action='store_true')
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

report = {'task': args.task, 'mode': args.mode, 'sac': args.sac, 'checks': [], 'passed': False}


def check(name, value):
    report['checks'].append({'name': name, 'passed': bool(value)})
    assert value, name


def expected(actions, cfg, hover, legacy):
    """Scalar oracle: relative distance between raw requests and feasible commands."""
    result = []
    for row in actions.double().cpu().tolist():
        thrust = torque = refs = 0.
        if legacy:
            hi = cfg.thrust_max if cfg.thrust_max > 0 else cfg.thrust_max_scale*hover
            thrust_scale = hi if cfg.thrust_mode == 'absolute' else min(hi, hover*(1+cfg.delta_thrust_scale))
            for value in row[:6]:
                bounded = min(1., max(-1., value))
                if cfg.thrust_mode == 'absolute':
                    request = cfg.thrust_min + (value+1)*.5*(hi-cfg.thrust_min)
                    command = cfg.thrust_min + (bounded+1)*.5*(hi-cfg.thrust_min)
                else:
                    request = hover*(1+cfg.delta_thrust_scale*value)
                    command = min(hi, max(cfg.thrust_min, hover*(1+cfg.delta_thrust_scale*bounded)))
                thrust += ((request-command)/thrust_scale)**2
            if cfg.des_joint_out:
                refs = sum((value-min(1., max(-1., value)))**2 for value in row[6:])
            else:
                for i, value in enumerate(row[6:]):
                    lo, hi = cfg.torque_min[i], cfg.torque_max[i]
                    bias = cfg.bias_torque_j2 if i == 1 else 0.
                    request = lo+(value+1)*.5*(hi-lo)+bias
                    command = lo+(min(1., max(-1., value))+1)*.5*(hi-lo)+bias
                    command = min(cfg.torque_lim, max(-cfg.torque_lim, command))
                    torque_scale = min(hi+bias, cfg.torque_lim)
                    torque += ((request-command)/torque_scale)**2
        elif cfg.control_mode in ('thrust_and_torque_control', 'thrust_and_PD_joint_pos_control'):
            thrust_scale = hover*cfg.thrust_to_weight
            if not cfg.sac_action_mapping and not cfg.remove_thrust_clipping:
                thrust_scale = min(thrust_scale, hover*(1+abs(cfg.rotor_action_scale*cfg.action_clamp)))
            for value in row[:6]:
                if cfg.sac_action_mapping:
                    request = hover*(1+(cfg.thrust_to_weight-1)*value if value >= 0 else 1+value)
                    command = min(hover*cfg.thrust_to_weight, max(0., request))
                else:
                    request = hover*(1+cfg.rotor_action_scale*value)
                    bounded = value if cfg.remove_thrust_clipping else min(cfg.action_clamp, max(-cfg.action_clamp, value))
                    command = max(0., hover*(1+cfg.rotor_action_scale*bounded))
                    if not cfg.remove_thrust_clipping:
                        command = min(hover*cfg.thrust_to_weight, command)
                thrust += ((request-command)/thrust_scale)**2
            if cfg.control_mode == 'thrust_and_torque_control' and not cfg.hold_arm:
                for value in row[6:]:
                    request = value*cfg.effort_lim_sim
                    torque_scale = cfg.effort_lim_sim
                    if not cfg.sac_action_mapping and not cfg.remove_thrust_clipping:
                        torque_scale *= min(1., cfg.action_clamp)
                    command = min(torque_scale, max(-torque_scale, request))
                    torque += ((request-command)/torque_scale)**2
            elif not cfg.sac_action_mapping and not cfg.remove_thrust_clipping and not cfg.hold_arm:
                refs = sum(max(abs(value)-cfg.action_clamp, 0.)**2 for value in row[-2:])
        elif cfg.control_mode == 'indi':
            reference = row[:-2] if cfg.hold_arm else row
            refs = sum(max(abs(value)-1., 0.)**2 for value in reference) if cfg.sac_action_mapping else 0.
        elif not cfg.remove_thrust_clipping:
            reference = row[:-2] if cfg.hold_arm else row
            refs = sum(max(abs(value)-cfg.action_clamp, 0.)**2 for value in reference)
        result.append([thrust, torque, refs])
    return actions.new_tensor(result)


env = None
try:
    legacy = args.task == 'Borinot-Direct-v0'
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=4)
    cfg.seed = 317
    for name in vars(cfg):
        if name.startswith('rwd_scale_'):
            setattr(cfg, name, 0.)
    cfg.rwd_scale_thrust_limits, cfg.rwd_scale_torque_limits, cfg.rwd_scale_action_limits = -1., -.25, -.5
    if legacy:
        source = Path(__file__).resolve().parents[5]
        cfg.robot_cfg.spawn.usd_path = str(source / 'isaaclab_assets/data/Robots/Borinot/borinotFlat.usd')
        cfg.ground_usd_path = str(source / 'borinotIsaacLab/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd')
        cfg.robot_cfg.init_state.pos = (0., 0., 1.5)
        if args.legacy_torque:
            cfg.des_joint_out, cfg.action_space, cfg.observation_space = False, 8, 32
            cfg.robot_cfg.actuators['leg_torque'].stiffness = 0.
            cfg.robot_cfg.actuators['leg_torque'].damping = 0.
    else:
        cfg.curriculum = False
        cfg.control_mode, cfg.sac_action_mapping = args.mode, args.sac
        cfg.hold_arm = args.hold_arm
        cfg.robot_home = (0., 0., 1.5)
        cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
        cfg.configure_control()
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    hover = cfg.hover_thrust_per_rotor if legacy else env._hover_per_rotor
    report['effective_hover_n'] = hover
    if not legacy:
        report['effective_thrust_to_weight'] = cfg.thrust_to_weight
    raw = torch.zeros_like(env.actions)
    for i, value in enumerate((-2., 0., 2., 20.)):
        raw[i, :] = value
    if legacy and cfg.des_joint_out:
        raw[1, 6:] = raw.new_tensor([1., 0., 1., 0.])
    wanted = expected(raw, cfg, hover, legacy)
    raw_copy = raw.clone()
    env._pre_physics_step(raw)
    check('cost_matches_independent_scalar_oracle_before_physics', torch.allclose(env._action_limit_costs, wanted, atol=2e-4, rtol=2e-6))
    check('caller_raw_actions_unchanged', torch.equal(raw, raw_copy))
    # Duplicate pre-physics calls do not accumulate the cost.
    env._pre_physics_step(raw)
    check('repeated_prephysics_does_not_accumulate', torch.equal(env._action_limit_costs, wanted) or torch.allclose(env._action_limit_costs, wanted, atol=2e-4, rtol=2e-6))
    obs, reward, terminated, truncated, info = env.step(raw)
    check('finite_real_transition', torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values()))
    key = 'action_limit_reward_terms' if legacy else 'reward_terms' if 'cmd-vel' in args.task else 'goalkeeper_reward_terms'
    terms = info[key]
    prefix = 'desirability/' if 'Simple' in args.task else ''
    weights = (-1., -.25, -.5)
    for i, name in enumerate(('thrust_limits', 'torque_limits', 'action_limits')):
        check(f'{name}_delivered_on_actual_transition', torch.allclose(terms[prefix+name], weights[i]*wanted[:, i], atol=2e-4, rtol=2e-6))
    if not legacy:
        check('agent_reward_is_sum_of_logged_terms', torch.allclose(reward, sum(terms.values()), atol=2e-4, rtol=2e-6))
        check('shared_helper_archived', any(Path(path).name == 'action_limits.py' for path in env.get_code_snapshot_files()))
    # Force a per-environment timeout with a nonzero cost on its final action.
    if not legacy:
        env._episode_steps[0] = env.max_episode_length - 1
    else:
        env.episode_length_buf[0] = env.max_episode_length - 1
    _, _, term, trunc, info = env.step(raw)
    check('forced_timeout_reaches_automatic_reset', bool(term[0] or trunc[0]))
    check('terminal_cost_survives_automatic_reset', torch.allclose(info[key][prefix+'thrust_limits'], -wanted[:, 0], atol=2e-4, rtol=2e-6))
    check('automatic_reset_clears_cost_for_selected_env', not env._action_limit_costs[0].any())
    if not legacy:
        tags = info.get('log', {})
        check('episode_logger_includes_all_three_terms', all('Episode_Reward/'+prefix+name in tags for name in ('thrust_limits', 'torque_limits', 'action_limits')))
    env._action_limit_costs.copy_(wanted)
    before = env._action_limit_costs.clone()
    env._reset_idx(torch.tensor([1], device=env.device))
    check('manual_partial_reset_clears_only_selected_env', not env._action_limit_costs[1].any() and torch.equal(env._action_limit_costs[[0, 2, 3]], before[[0, 2, 3]]))
    cfg.rwd_scale_thrust_limits = cfg.rwd_scale_torque_limits = cfg.rwd_scale_action_limits = 0.
    env._pre_physics_step(raw)
    check('all_weights_zero_disables_costs', not env._action_limit_costs.any())
    report['raw_actions'] = raw.cpu().tolist()
    report['expected_costs'] = wanted.cpu().tolist()
    report['passed'] = True
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if env is not None:
        env.close()
    app.close()
