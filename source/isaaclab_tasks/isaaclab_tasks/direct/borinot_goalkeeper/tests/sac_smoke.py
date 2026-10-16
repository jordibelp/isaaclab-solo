# SPDX-License-Identifier: BSD-3-Clause
"""Hydra + PhysX action contract, resolved recipe, and finite SAC inference."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="rsl_rl_sac_cfg_entry_point")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "rsl_rl_sac_vendor"))
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl_sac.runners import OffPolicyRunner


@hydra_task_config("Borinot-Goalkeeper-Direct-v0", args.agent)
def main(cfg, agent_cfg):
    report = {"checks": [], "recipe": agent_cfg.to_dict()}
    def check(name, condition):
        report["checks"].append({"name": name, "passed": bool(condition)})
        assert condition, name
        print(f"CHECK {name}: passed", flush=True)

    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
    cfg.curriculum = False
    agent_cfg.configure_environment(cfg)
    # Keep the test small; report['recipe'] above retains the true launch defaults.
    agent_cfg.actor.hidden_dims = agent_cfg.critic.hidden_dims = [32, 16]
    agent_cfg.algorithm.replay_buffer_size = 128
    agent_cfg.algorithm.torch_compile = False
    raw = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
    env = RslRlVecEnvWrapper(raw, clip_actions=None)
    try:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
        actor = runner.alg.actor
        limits = raw.robot.root_physx_view.get_dof_limits()[:, raw.joint_ids].to(env.device)
        expected_hard = torch.deg2rad(torch.tensor([[-90., 90.], [-160., 160.]], device=env.device))
        check("actual_PhysX_hard_limits", torch.allclose(limits, expected_hard.expand_as(limits), atol=1e-6))
        view = raw.robot.root_physx_view
        torque_mode = cfg.control_mode == "thrust_and_torque_control"
        check("actual_PD_stiffness", torch.allclose(view.get_dof_stiffnesses()[:, raw.joint_ids], torch.full((4, 2), 0.0 if torque_mode else cfg.kp)))
        check("actual_PD_damping", torch.allclose(view.get_dof_dampings()[:, raw.joint_ids], torch.full((4, 2), 0.0 if torque_mode else cfg.kd)))
        check("actual_effort_limit", torch.allclose(view.get_dof_max_forces()[:, raw.joint_ids], torch.full((4, 2), cfg.effort_lim_sim)))
        check("armature_preserved", torch.allclose(view.get_dof_armatures()[:, raw.joint_ids], torch.full((4, 2), .00036207)))
        check("actor_and_critic_observation_groups", actor.obs_dim == cfg.observation_space
              and runner.alg.critic.obs_dim == cfg.state_space)
        if runner.alg.critic.distributional_critic_ce:
            support = runner.alg.critic.value_support
            check("linear_101_atom_support", len(support) == 101 and support[50] == 0
                  and torch.allclose(support, torch.linspace(-20, 20, 101, device=env.device), atol=2e-6))
            report["support"] = support.cpu().tolist()
        else:
            check("scalar_MSE_head", runner.alg.critic.critic1[-1].out_features == 1)
        targets = []
        setter = "set_joint_effort_target" if torque_mode else "set_joint_position_target"
        original_set = getattr(raw.robot, setter)
        def record_target(value, **kwargs):
            targets.append(value.clone())
            original_set(value, **kwargs)
        setattr(raw.robot, setter, record_target)
        samples = []
        for u, expected_angles, ratio in ((-1., [-75., -200.], 0.), (0., [0., 0.], 1.), (1., [75., 200.], 3.5)):
            raw.actions[:] = actor.action_bias + actor.action_range * u
            raw._rotor_thrust[:] = raw._hover_per_rotor
            raw._apply_action()
            requested = raw._hover_per_rotor * ratio
            expected = raw._hover_per_rotor + raw._motor_alpha * (requested - raw._hover_per_rotor)
            check(f"rotors_u_{u:g}_with_motor_lag", torch.allclose(raw._rotor_thrust, torch.full_like(raw._rotor_thrust, expected), atol=2e-6))
            joint_target = targets[-1] if torque_mode else torch.rad2deg(targets[-1])
            expected_joint = [u * cfg.effort_lim_sim] * 2 if torque_mode else expected_angles
            check(f"joint_target_u_{u:g}", torch.allclose(joint_target, torch.tensor(expected_joint, device=env.device).expand_as(joint_target), atol=3e-5))
            samples.append({"u": u, "joint_torque_nm" if torque_mode else "joint_target_deg": joint_target[0].cpu().tolist(),
                            "requested_thrust_per_rotor_n": requested,
                            "lagged_thrust_per_rotor_n": raw._rotor_thrust[0].cpu().tolist()})
        if torque_mode:
            raw.actions[:, 6:] = torch.tensor([-1., 0.5], device=env.device)
            raw._apply_action()
            check("independent_joint_torques", torch.allclose(targets[-1], torch.tensor(
                [-cfg.effort_lim_sim, .5 * cfg.effort_lim_sim], device=env.device).expand_as(targets[-1])))
        setattr(raw.robot, setter, original_set)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(30):
                obs, rewards, dones, _ = env.step(actor(obs))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
        check("finite_inference_30_policy_steps", True)
        before = raw._rotor_thrust[1:].clone()
        with torch.inference_mode():
            raw._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_preserves_other_rotors", torch.equal(before, raw._rotor_thrust[1:]))
        check("hard_limits_survive_resets", torch.equal(limits.cpu(), raw.robot.root_physx_view.get_dof_limits()[:, raw.joint_ids]))
        check("gains_survive_resets", torch.allclose(view.get_dof_stiffnesses()[:, raw.joint_ids], torch.full((4, 2), 0.0 if torque_mode else cfg.kp))
              and torch.allclose(view.get_dof_dampings()[:, raw.joint_ids], torch.full((4, 2), 0.0 if torque_mode else cfg.kd)))
        report.update(action_samples=samples, robot_mass_kg=raw.robot_mass_kg,
                      control_mode=cfg.control_mode, kp=cfg.kp, kd=cfg.kd, effort_lim_sim=cfg.effort_lim_sim,
                      hover_per_rotor_n=raw._hover_per_rotor, actor_dimension=actor.obs_dim,
                      critic_dimension=runner.alg.critic.obs_dim)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"SAC_SMOKE: {len(report['checks'])} passed", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
