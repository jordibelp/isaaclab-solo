# SPDX-License-Identifier: BSD-3-Clause
"""GPU simulator smoke: finite rollouts, physical contacts, partial reset, curriculum."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg

cfg = BorinotGoalkeeperEnvCfg()
cfg.control_mode = "PI_rotors_PD_joint_pos_control"
cfg.physics_hz, cfg.policy_hz = 240, 60  # preserve the original contact-test baseline
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.curriculum = False
cfg.seed = 42
cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
cfg.robot_home = (0.0, 0.0, 1.5)  # known in-area baseline; do not rewrite local config tuning
cfg.robot.init_state.pos = cfg.robot_home
cfg.rwd_scale_energy = 0.0
env = gym.make("Borinot-Goalkeeper-Direct-v0", cfg=cfg).unwrapped
report = {"checks": [], "robot_mass_kg": env.robot_mass_kg, "physics_dt_s": env.physics_dt,
          "policy_dt_s": env.step_dt, "body_names": env.robot.body_names}


def check(name, condition, detail=None):
    report["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
    print(f"CHECK {name}: {bool(condition)} {detail if detail is not None else ''}", flush=True)
    if not condition:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        raise AssertionError(name)


try:
    obs, _ = env.reset()
    check("observation_contract", obs["policy"].shape == (args.num_envs, 63)
          and obs["critic"].shape == (args.num_envs, 63))
    check("finite_reset", all(torch.isfinite(value).all().item() for value in obs.values()))
    check("mass_preserved", abs(env.robot_mass_kg - 2.910538) < 1e-4)
    masses = env.robot.root_physx_view.get_masses()[0]
    check("paddle_mass", abs(float(masses[env.paddle_id]) - .100) < 1e-5)
    torch.testing.assert_close(env.robot.data.root_quat_w,
        torch.tensor(cfg.robot_home_quat, device=env.device).expand(args.num_envs, -1), atol=1e-6, rtol=0)
    check("reset_uses_goal_parallel_home_orientation", True)
    actions = torch.zeros(args.num_envs, 6, device=env.device)
    outcomes = torch.zeros(10, dtype=torch.long, device=env.device)
    max_contact = torch.zeros(6, device=env.device)
    save_contact_valid = True
    optional_rewards_off = True
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        check_finite = all(torch.isfinite(value).all().item() for value in obs.values())
        check_finite &= torch.isfinite(reward).all().item()
        if not check_finite:
            check("finite_rollout", False, step)
        codes = info["goalkeeper_outcome"]
        save_contact_valid &= bool(info["goalkeeper_had_paddle_hit"][codes == 7].all())
        optional_rewards_off &= bool((info["goalkeeper_reward_terms"]["optional_rates"] == 0).all())
        optional_rewards_off &= bool((info["goalkeeper_reward_terms"]["optional_hit"] == 0).all())
        outcomes += torch.bincount(codes[codes != 0], minlength=10)
        matrix = env.ball_contact.data.force_matrix_w[:, 0]
        max_contact = torch.maximum(max_contact, matrix.norm(dim=-1).amax(0))
    check("finite_rollout", True, {"steps": args.steps, "outcomes": outcomes.tolist()})
    check("physical_paddle_contacts", bool(max_contact[0] > .05), max_contact.tolist())
    check("baseline_can_save", bool(outcomes[7] > 0), outcomes.tolist())
    check("save_requires_physical_paddle_hit", save_contact_valid)
    # Correcting the paddle changes the rebound. Do not require a random floor save.
    check("additional_rewards_disabled_by_default", optional_rewards_off)

    # Real articulated poses: changing either joint must keep the face in Y-Z.
    from isaaclab.utils.math import quat_mul
    for shoulder, elbow in ((0., 0.), (-.8, 1.1), (.7, -.9)):
        env.reset()
        joint_position = env.robot.data.default_joint_pos.clone()
        joint_position[:, env.joint_ids] = torch.tensor([shoulder, elbow], device=env.device)
        env.robot.write_joint_state_to_sim(joint_position, torch.zeros_like(joint_position))
        joint_actions = actions.clone()
        joint_actions[:, 4:6] = torch.tensor([shoulder, elbow], device=env.device) / env._joint_scale
        env.step(joint_actions)
        normal = env._paddle_pose()[2]
        check(f"paddle_parallel_to_goal_{shoulder}_{elbow}", bool((normal[:, 0] > .9999).all()), normal.cpu().tolist())
    env.reset()

    # A partial reset must not change the other robot/ball states or erase their actuators.
    env.reset()
    for _ in range(4):
        env.step(actions)
    untouched_ball = env.ball.data.root_state_w[1:].clone()
    untouched_robot = env.robot.data.root_state_w[1:].clone()
    untouched_rotors = env._rotor_thrust[1:].clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    check("partial_reset_isolation", torch.equal(untouched_ball, env.ball.data.root_state_w[1:])
          and torch.equal(untouched_robot, env.robot.data.root_state_w[1:])
          and torch.equal(untouched_rotors, env._rotor_thrust[1:]))
    ball_masses = env.ball.root_physx_view.get_masses()[:, 0]
    check("ball_mass_randomization", bool(((ball_masses >= .020) & (ball_masses <= .050)).all()))
    inertia = env.ball.root_physx_view.get_inertias()
    expected = torch.eye(3).reshape(1, 9) * (0.4 * ball_masses * cfg.ball_radius_m**2)[:, None]
    check("ball_inertia_consistent", torch.allclose(inertia, expected, atol=1e-8, rtol=1e-4))
    materials = env.ball.root_physx_view.get_material_properties()
    check("restitution_readback", bool(((materials[:, :, 2] >= .35) & (materials[:, :, 2] <= .65)).all()))
    check("privileged_parameters_match_physics", torch.allclose(env._ball_parameters[:, 0].cpu(), ball_masses)
          and torch.allclose(env._ball_parameters[:, 1].cpu(), materials[:, 0, 2]))

    # Controlled events test outcome attribution through real PhysX, not proximity heuristics.
    def shoot_at(target, expected_code, expected_penalty, velocity_x=-3.0):
        env.reset()
        env.step(actions)
        position = target() + env.scene.env_origins
        state = env.ball.data.default_root_state.clone()
        state[:, :3] = position
        state[:, 7:10] = torch.tensor([velocity_x, 0.0, 0.0], device=env.device)
        state[:, 10:] = 0.0
        env.ball.write_root_state_to_sim(state)
        env._previous_ball.copy_(position - env.scene.env_origins)
        found = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        penalty_correct = True
        for _ in range(45):
            _, _, _, _, info = env.step(actions)
            selected = info["goalkeeper_outcome"] == expected_code
            found |= selected
            penalty_correct &= bool((info["goalkeeper_reward_terms"]["outcome"][selected] == expected_penalty).all())
        check(f"outcome_{expected_code}_penalty", penalty_correct)
        return int(found.sum())

    # Deliberately distinct values prove goal, ball/base hit, and robot crash are decoupled.
    cfg.rwd_scale_ball_base_collision = -2.0
    cfg.rwd_scale_goal = -3.0
    cfg.rwd_scale_robot_crash = -4.0
    base_hits = shoot_at(lambda: env.robot.data.root_pos_w - env.scene.env_origins
                         + torch.tensor([.38, 0.0, 0.0], device=env.device), 1, cfg.rwd_scale_ball_base_collision)
    check("base_hit_attribution", base_hits > 0, base_hits)
    arm1_id = env.robot.find_bodies("flying_arm_2__link1")[0][0]
    # With the new yaw, a shot at the upper-link COM hits a landing ski first.
    # Aim below that ski and above the paddle's top edge to expose the arm.
    arm_hits = shoot_at(lambda: env.robot.data.body_com_pos_w[:, arm1_id] - env.scene.env_origins
                        + torch.tensor([.22, 0.0, -.065], device=env.device), 2, cfg.rwd_scale_ball_arm_collision)
    check("arm_hit_attribution", arm_hits > 0, arm_hits)
    goals = shoot_at(lambda: torch.tensor([-.20, .85, 1.0], device=env.device).expand(args.num_envs, -1), 3, cfg.rwd_scale_goal)
    check("physical_goal_attribution", goals == args.num_envs, goals)
    floors = shoot_at(lambda: torch.tensor([1.0, .8, .04], device=env.device).expand(args.num_envs, -1), 4, cfg.rwd_scale_ball_floor, 0.0)
    check("live_ball_floor_attribution", floors == args.num_envs, floors)

    env.reset()
    tilted = env.robot.data.root_state_w.clone()
    tilted[:, 3:7] = torch.tensor([2**-.5, 2**-.5, 0., 0.], device=env.device)
    env.robot.write_root_state_to_sim(tilted)
    _, _, _, _, info = env.step(actions)
    check("tilt_crash_is_distinct_from_ball_base_hit", bool((info["goalkeeper_outcome"] == 8).all()))
    check("independent_robot_crash_scale", bool((info["goalkeeper_reward_terms"]["outcome"] == cfg.rwd_scale_robot_crash).all()))

    # A low, real paddle impact followed by floor contact in the same interval.
    saved_min = env._area_min.clone()
    env._area_min[2] = 0.0
    env.reset()
    low_root = env.robot.data.root_state_w.clone()
    # Keep the lower rim at the old fixture's height as the head is resized.
    low_root[:, 2] = env.scene.env_origins[:, 2] + .4 + cfg.paddle_diameter_m/2
    env.robot.write_root_state_to_sim(low_root)
    env.step(actions)
    low_ball = env.ball.data.default_root_state.clone()
    low_ball[:, :3] = env._paddle_pose()[0] + env.scene.env_origins
    low_ball[:, :3] += torch.tensor([.045, 0., -cfg.paddle_diameter_m/2 + .025], device=env.device)
    low_ball[:, 7:10] = torch.tensor([-3., 0., -2.], device=env.device)
    low_ball[:, 10:] = 0.
    env.ball.write_root_state_to_sim(low_ball)
    env._previous_ball.copy_(low_ball[:, :3] - env.scene.env_origins)
    _, _, _, _, info = env.step(actions)
    check("floor_save_still_pays_floor_penalty", bool(((info["goalkeeper_outcome"] == 7)
          & info["goalkeeper_had_paddle_hit"]
          & (info["goalkeeper_reward_terms"]["floor_after_save"] == cfg.rwd_scale_ball_floor)).any()))
    env._area_min.copy_(saved_min)

    # Contact and tilt reset switches must work independently in real physics.
    saved_min = env._area_min.clone()
    env._area_min[2] = -1.0  # isolate crash handling from the independent allowed-height termination
    for reset_contact in (False, True):
        for reset_tilt in (False, True):
            cfg.reset_robot_on_robot_crash = reset_contact
            cfg.reset_robot_on_tilt_threshold = reset_tilt
            for event in ("contact", "tilt"):
                env.reset()
                state = env.robot.data.root_state_w.clone()
                if event == "contact":
                    state[:, 2] = env.scene.env_origins[:, 2] + .025
                    folded = env.robot.data.default_joint_pos.clone()
                    folded[:, env.joint_ids] = torch.tensor([1.4, 1.57], device=env.device)
                    env.robot.write_joint_state_to_sim(folded, torch.zeros_like(folded))
                else:
                    roll = torch.tensor([2**-.5, 2**-.5, 0., 0.], device=env.device).expand(args.num_envs, -1)
                    state[:, 3:7] = quat_mul(state[:, 3:7], roll)
                env.robot.write_root_state_to_sim(state)
                _, _, terminated, _, info = env.step(actions)
                expected = reset_contact if event == "contact" else reset_tilt
                check(f"{event}_reset_contact_{reset_contact}_tilt_{reset_tilt}",
                      bool((terminated == expected).all()))
                check(f"{event}_penalty_contact_{reset_contact}_tilt_{reset_tilt}",
                      bool((info["goalkeeper_outcome"] == 8).all())
                      and bool((info["goalkeeper_reward_terms"]["outcome"] == cfg.rwd_scale_robot_crash).all()))
                check(f"{event}_detected_independently_contact_{reset_contact}_tilt_{reset_tilt}",
                      bool(info["goalkeeper_robot_contact_crash"].all()) == (event == "contact")
                      and bool(info["goalkeeper_tilt_threshold"].all()) == (event == "tilt"))
    cfg.reset_robot_on_robot_crash = cfg.reset_robot_on_tilt_threshold = True
    env._area_min.copy_(saved_min)

    original_threshold = cfg.robot_tilt_threshold_degs
    for threshold_deg, tilt_deg, expected in ((60.0, 45, False), (60.0, 75, True), (20.0, 30, True),
                                             (70.0, 69, False), (70.0, 71, True)):
        cfg.robot_tilt_threshold_degs = threshold_deg
        env.reset()
        state = env.robot.data.root_state_w.clone()
        half_angle = torch.tensor(tilt_deg * torch.pi / 360, device=env.device)
        roll = torch.zeros(args.num_envs, 4, device=env.device)
        roll[:, 0] = half_angle.cos()
        roll[:, 1] = half_angle.sin()
        state[:, 3:7] = quat_mul(state[:, 3:7], roll)
        env.robot.write_root_state_to_sim(state)
        _, _, terminated, _, info = env.step(actions)
        check(f"tilt_{tilt_deg}_threshold_{threshold_deg}_reset", bool((terminated == expected).all()))
        check(f"tilt_{tilt_deg}_threshold_{threshold_deg}_detection",
              bool((info["goalkeeper_tilt_threshold"] == expected).all())
              and not bool(info["goalkeeper_robot_contact_crash"].any()))
    cfg.robot_tilt_threshold_degs = original_threshold

    env.reset()
    state = env.robot.data.root_state_w.clone()
    state[:, 1] = env.scene.env_origins[:, 1] + cfg.goal_area_y_m[1] + .2
    env.robot.write_root_state_to_sim(state)
    _, _, terminated, truncated, info = env.step(actions)
    check("leaving_goal_area", bool((info["goalkeeper_outcome"] == 5).all()) and bool(terminated.all()))
    check("finite_horizon_not_bootstrapped", not bool(truncated.any()))

    # Stage-pure count aggregation, just-before/at dwell, and exclusion of older episodes.
    env.reset()
    env.cfg.curriculum = True
    env.cfg.curriculum_min_shots = 4
    env.cfg.curriculum_min_steps = 10
    env.shot_level = 0
    env._stage_start_step = 0
    env._stage_shots = 0
    env._curriculum_window.zero_()
    env._window_cursor = 0
    ids = torch.arange(4, device=env.device)
    env._shot_steps[ids] = 1
    env._episode_stage[ids] = 0
    env._shot_stage[ids] = 0
    env._outcome[ids] = 7
    env.common_step_counter = 9
    env._record_shots(ids)
    check("curriculum_dwell_boundary_before", env.shot_level == 0)
    env.common_step_counter = 10
    env._record_shots(ids)
    check("curriculum_dwell_boundary_at", env.shot_level == 1 and env._stage_shots == 0)
    check("curriculum_checkpoint_stage_restarts",
          env.get_curriculum_global_idx() == 1 and env.curriculum_stage_pure_episodes == 0)
    env.common_step_counter = 100
    env._record_shots(ids)  # these episodes still belong to stage zero
    check("curriculum_excludes_old_stage", env.shot_level == 1 and env._stage_shots == 0)
    check("curriculum_checkpoint_ignores_old_stage", env.curriculum_stage_pure_episodes == 0)
    env._outcome[ids] = 0
    env._record_shots(ids)
    check("curriculum_no_promotion_without_endings", env.shot_level == 1)
    report["outcome_counts"] = env.outcome_counts.tolist()
    report["passed"] = all(row["passed"] for row in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("GOALKEEPER_VALIDATION=" + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
