# SPDX-License-Identifier: BSD-3-Clause
"""Small Hydra/PhysX proof, not a learned-controller benchmark."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Borinot-goto")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--capture", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "rsl_rl_sac_vendor"))
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.direct.borinot_goto.task_math import SuccessCurriculum


@hydra_task_config(args.task, args.agent)
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, args.device, 19
    cfg.curriculum = False
    configure = getattr(agent_cfg, "configure_environment", None)
    if configure is not None:
        configure(cfg)
    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array" if args.capture else None).unwrapped
    report = {"task": args.task, "agent": args.agent, "control_mode": cfg.control_mode, "checks": []}

    def check(name, passed):
        passed = bool(passed)
        report["checks"].append({"name": name, "passed": passed})
        assert passed, name

    try:
        obs, _ = env.reset()
        check("alias", gym.spec("Borinot-GoTo-Direct-v0").entry_point == gym.spec("Borinot-goto").entry_point)
        check("space_after_hydra", obs["policy"].shape == obs["critic"].shape == (8, cfg.observation_space))
        check("shared_observation_and_finite_horizon", torch.equal(obs["policy"], obs["critic"]) and cfg.is_finite_horizon)
        check("body_goal_error", torch.allclose(env._commands[:, :3], quat_apply_inverse(
            env.robot.data.root_quat_w, env._goal_position_w-env.robot.data.root_pos_w), atol=2e-6))
        check("normalized_time_one_at_reset", torch.equal(env._commands[:, 3], torch.ones(8, device=env.device)))
        check("seconds_at_reset", torch.allclose(obs["policy"][:, -1], env._goal_deadline_steps*env.step_dt))
        check("goal_is_visual_not_rigid", not env.scene.rigid_objects)
        before = env._goal_position_w.clone()
        reread = env._get_observations()["policy"]
        print("OBS_REREAD " + json.dumps({"max_abs":float((reread-obs["policy"]).abs().max()),
            "changed_columns":(reread != obs["policy"]).any(0).nonzero().flatten().cpu().tolist()}), flush=True)
        check("observation_reads_do_not_resample", torch.allclose(reread, obs["policy"], atol=2e-6, rtol=0)
              and torch.equal(before, env._goal_position_w))
        env.episode_length_buf[:] = 100000
        action = torch.zeros(8, cfg.action_space, device=env.device)
        for _ in range(10):
            obs, reward, done, timeout, info = env.step(action)
            check("finite_shared_controller_step", torch.isfinite(reward).all() and torch.isfinite(obs["policy"]).all())
            check("all_substeps_sampled", env._power_samples == cfg.decimation)
            check("no_pushes_no_premature_timeout", not env._base_push_force_z_w.any() and not timeout.any())
            check("no_terminal_cost_before_deadline", not info["reward_terms"]["terminal_distance"].any())
        # Partial reset: neither the task command nor any controller buffer may
        # change for other clones.
        kept = torch.tensor([0,2,3,4,5,6,7], device=env.device)
        buffers = (env._goal_position_w, env._goal_deadline_steps, env._episode_steps, env.actions,
                   env._rotor_thrust, env._goal_progress_factor, env.robot.data.root_state_w)
        snapshots = [v[kept].clone() for v in buffers]
        env._reset_idx(torch.tensor([1], device=env.device))
        check("partial_reset_isolation", all(torch.equal(v[kept], old) for v, old in zip(buffers, snapshots)))
        env.reset()
        # Force a known three-policy-step deadline, independent of any runner
        # episode counter. Do not teleport the goal between those transitions.
        env._goal_deadline_steps[:] = 3
        env._previous_goal_distance.copy_((env._goal_position_w-env.robot.data.root_pos_w).norm(dim=-1))
        for step in range(1,4):
            obs, reward, done, timeout, info = env.step(action)
            check("exact_deadline_not_timeout", bool(done.all()) == (step == 3) and not timeout.any())
            if step == 3:
                check("terminal_cost_uses_pre_reset_distance", torch.allclose(
                    info["reward_terms"]["terminal_distance"], cfg.rwd_scale_terminal_distance*info["goal"]["distance_m"], atol=1e-6))
                check("auto_reset_has_new_time_and_no_stale_steps", not env._episode_steps.any()
                      and (obs["policy"][:, -1] > 0).all() and (env._commands[:, 3] == 1).all())
                check("terminal_observations_have_zero_time", not info["time_outs_obs"]["policy"][:, -1].any())
                check("requested_rewards_only", all(not info["reward_terms"][name].any() for name in (
                    "terminal_speed", "terminal_success", "action_rate", "energy", "base_collision", "leave_area", "self_collision", "tilt")))
        # Real physics safety failure, before the goal clock, still pays a cost.
        env.reset()
        root = env.robot.data.root_state_w[0:1].clone()
        root[:, 2] = cfg.flight_height_range_m[1] + 1
        env.robot.write_root_state_to_sim(root, env_ids=torch.tensor([0], device=env.device))
        obs, reward, done, timeout, info = env.step(action)
        check("early_failure_is_terminal", done[0] and not timeout[0] and info["goal"]["failed"][0]
              and not info["goal"]["success"][0])
        check("failure_cannot_skip_goal_cost", info["reward_terms"]["terminal_distance"][0]
              <= cfg.rwd_scale_terminal_distance*cfg.failure_distance_floor_m + 1e-5)
        # Controlled success proves lifecycle/promotion, not policy skill.
        env.reset()
        cfg.curriculum, cfg.curriculum_min_steps = True, 2
        cfg.curriculum_min_episodes, cfg.curriculum_window_episodes = 2, 4
        env.goal_curriculum = SuccessCurriculum(cfg, env.device)
        env.goal_curriculum.start_tick = env.common_step_counter
        env._goal_episode_level.zero_()
        levels = []
        for step in range(4):
            env._goal_position_w.copy_(env.robot.data.root_pos_w)
            env._goal_deadline_steps[:] = 1
            env._episode_steps.zero_()
            _, _, done, timeout, info = env.step(action)
            levels.append(env.get_curriculum_global_idx())
            check("controlled_deadline_success", done.all() and info["goal"]["success"].all())
        check("one_promotion_per_dwell", levels == [0,1,1,2])
        check("logged_decision_precedes_selected_stage", info["log"]["Curriculum/evaluated_stage"] == 1
              and info["log"]["Curriculum/selected_stage"] == 2)
        if args.capture:
            from PIL import Image
            env.cfg.goal_debug_vis = True
            env._goal_position_w[0] = env.robot.data.root_pos_w[0] + torch.tensor([.8,0.,.3],device=env.device)
            env._get_observations()
            lookat = env.robot.data.root_pos_w[0].cpu().tolist()
            eye = [v+d for v,d in zip(lookat,(3.,-4.,1.5))]
            env.sim.set_camera_view(eye,lookat)
            env.render()  # first call creates the RGB annotator/render product
            for _ in range(12):
                env.sim.render()
            frame = env.render()
            check("nonempty_rendered_scene", frame is not None and frame.shape[2] == 3 and frame.std() > 5)
            args.capture.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(frame).save(args.capture)
        report.update({"observation_dim": cfg.observation_space, "action_dim": cfg.action_space,
                       "physics_hz": cfg.physics_hz, "policy_hz": cfg.policy_hz,
                       "nominal_mass_kg": env.robot_mass_kg, "controlled_levels": levels})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print("BORINOT_GOTO_RUNTIME " + json.dumps({"checks":len(report["checks"]), "passed":True,
              "control_mode":cfg.control_mode, "observation_dim":cfg.observation_space}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
