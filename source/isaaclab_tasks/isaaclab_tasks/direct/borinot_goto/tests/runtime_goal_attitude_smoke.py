# SPDX-License-Identifier: BSD-3-Clause
"""Hydra/PhysX proof of goal-facing physical resets; no training or learned-success claim."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-goto")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--live-goal", type=float, nargs=3)
parser.add_argument("--relative", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "rsl_rl_sac_vendor"))
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_apply_inverse
from pxr import Usd, UsdPhysics
from isaaclab_tasks.direct.borinot_goto.playback import BorinotGoToPlayEnv
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, args.agent)
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, args.device, 42
    cfg.curriculum = cfg.include_events_randomization = False
    cfg.forces_applied_to_base_curriculum = [0.0]
    # Deliberately incompatible home attitude and large noise must be ignored.
    cfg.robot_home_quat = (.5, .5, .5, .5)
    cfg.reset_rpy_noise_degs = (30., 40., 170.)
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(cfg)
    report = {"task": args.task, "agent": args.agent, "control_mode": cfg.control_mode,
              "enabled_after_hydra": cfg.init_base_attitude_points_to_goal,
              "live_goal": args.live_goal, "relative": args.relative, "checks": []}
    env = None

    def check(name, passed):
        passed = bool(passed)
        report["checks"].append({"name": name, "passed": passed})
        assert passed, name

    def check_facing(label, ids=None):
        ids = env.robot._ALL_INDICES if ids is None else ids
        data = env.robot.data
        delta = env._goal_position_w[ids] - data.root_pos_w[ids]
        delta[:, 2] = 0
        norm = delta.norm(dim=-1, keepdim=True)
        x = delta / norm.clamp_min(1e-12)
        x[norm[:, 0] == 0] = x.new_tensor([1., 0., 0.])
        rotation = matrix_from_quat(data.root_quat_w[ids])
        check(label + "/flat_and_facing", torch.allclose(rotation[:, :, 0], x, atol=2e-6, rtol=0)
              and torch.allclose(rotation[:, :, 2], x.new_tensor([0., 0., 1.]).expand_as(x), atol=2e-6, rtol=0))
        check(label + "/zero_reset_velocities", not data.root_state_w[ids, 7:].any())
        env._get_observations()
        check(label + "/actual_body_goal_observation", torch.allclose(env._commands[ids, :3],
              quat_apply_inverse(data.root_quat_w[ids], env._goal_position_w[ids] - data.root_pos_w[ids]), atol=2e-6))
        check(label + "/no_body_lateral_error", env._commands[ids, 1].abs().max() < 3e-6)

    try:
        check("hydra_flag_is_true", cfg.init_base_attitude_points_to_goal is True)
        env = (BorinotGoToPlayEnv(cfg, goal=args.live_goal, goal_relative=args.relative, reset_at_deadline=True)
               if args.live_goal else gym.make(args.task, cfg=cfg).unwrapped)
        obs, _ = env.reset()
        check_facing("initial")
        check("resolved_spaces", obs["policy"].shape == (8, cfg.observation_space)
              and obs["critic"].shape == (8, cfg.state_space))
        snapshots = [value.clone() for value in (env.robot.data.root_state_w, env._goal_position_w,
                                                env._goal_deadline_steps)]
        env._get_observations()
        env._get_observations()
        check("observation_reads_do_not_change_pose_or_goal", all(torch.equal(old, value) for old, value in
              zip(snapshots, (env.robot.data.root_state_w, env._goal_position_w, env._goal_deadline_steps))))
        ids = torch.tensor([5, 1], device=env.device)
        kept = torch.tensor([0, 2, 3, 4, 6, 7], device=env.device)
        values = (env.robot.data.root_state_w, env._goal_position_w, env._goal_deadline_steps,
                  env.actions, env._rotor_thrust)
        snapshots = [value[kept].clone() for value in values]
        count = 0
        sampler = env._sample_reset_goals

        def counted_sampler(*a):
            nonlocal count
            count += 1
            return sampler(*a)

        env._sample_reset_goals = counted_sampler
        env._reset_idx(ids)
        check("one_goal_sample_per_reset_batch", count == 1)
        check("partial_reset_isolation", all(torch.equal(value[kept], old) for value, old in zip(values, snapshots)))
        check_facing("partial", ids)
        for i in range(3):
            env._reset_idx(None)
            check_facing(f"full_{i}")
        env._goal_deadline_steps[:] = 1
        obs, reward, done, timeout, _ = env.step(torch.zeros(8, cfg.action_space, device=env.device))
        check("real_automatic_deadline_reset", done.all() and not timeout.any() and not env._episode_steps.any())
        check("finite_controller_step", torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all())
        check_facing("automatic")
        if args.live_goal:
            pose = env.robot.data.root_state_w.clone()
            env.set_play_goal((-.8, .6, 1.8), 2.0)
            check("new_live_goal_does_not_teleport", torch.equal(env.robot.data.root_state_w, pose))
            env._reset_idx(None)
            check_facing("new_live_goal_reset")
        else:
            cfg.goal_half_extents_m = tuple((0., 0., extent[2]) for extent in cfg.goal_half_extents_m)
            env._reset_idx(None)
            check_facing("vertical_goal")
            check("vertical_goal_yaw_zero", torch.equal(env.robot.data.root_quat_w,
                  torch.tensor([1., 0., 0., 0.], device=env.device).expand(8, -1)))
        report["sample_root_state_w"] = env.robot.data.root_state_w.cpu().tolist()
        report["sample_goal_position_w"] = env._goal_position_w.cpu().tolist()
        # Inspect neutral arm geometry in base axes after forwarding the new pose.
        env.sim.forward()
        env.scene.update(env.physics_dt)
        body_ids, names = env.robot.find_bodies(["base", "flying_arm_2__link1", "flying_arm_2__link2"], preserve_order=True)
        report["neutral_body_origins_b"] = {name: quat_apply_inverse(env.robot.data.root_quat_w[:1],
              env.robot.data.body_pos_w[:1, index] - env.robot.data.root_pos_w[:1])[0].cpu().tolist()
              for name, index in zip(names, body_ids)}
        stage = Usd.Stage.Open(cfg.robot.spawn.usd_path)
        axes = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joint = UsdPhysics.RevoluteJoint(prim)
                parent_name = joint.GetBody0Rel().GetTargets()[0].name
                parent_id = env.robot.find_bodies(parent_name)[0][0]
                local_q = joint.GetLocalRot0Attr().Get()
                q = env.robot.data.root_quat_w.new_tensor([[local_q.GetReal(), *local_q.GetImaginary()]])
                axis = q.new_zeros((1, 3))
                axis[0, "XYZ".index(joint.GetAxisAttr().Get())] = 1
                axis_w = quat_apply(env.robot.data.body_quat_w[:1, parent_id], quat_apply(q, axis))
                axes[prim.GetName()] = quat_apply_inverse(env.robot.data.root_quat_w[:1], axis_w)[0].cpu().tolist()
        report["neutral_joint_axes_b"] = axes
        report["passed"] = all(check["passed"] for check in report["checks"])
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
