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
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.curriculum = False
cfg.seed = 42
cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
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
    check("observation_contract", obs["policy"].shape == (args.num_envs, 55)
          and obs["critic"].shape == (args.num_envs, 55))
    check("finite_reset", all(torch.isfinite(value).all().item() for value in obs.values()))
    check("mass_preserved", abs(env.robot_mass_kg - 2.910538) < 1e-4)
    masses = env.robot.root_physx_view.get_masses()[0]
    check("paddle_mass", abs(float(masses[env.paddle_id]) - .100) < 1e-5)
    actions = torch.zeros(args.num_envs, 6, device=env.device)
    outcomes = torch.zeros(10, dtype=torch.long, device=env.device)
    max_contact = torch.zeros(6, device=env.device)
    save_contact_valid = True
    floor_save_seen = False
    optional_rewards_off = True
    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(actions)
        check_finite = all(torch.isfinite(value).all().item() for value in obs.values())
        check_finite &= torch.isfinite(reward).all().item()
        if not check_finite:
            check("finite_rollout", False, step)
        codes = info["goalkeeper_outcome"]
        save_contact_valid &= bool(info["goalkeeper_had_paddle_hit"][codes == 7].all())
        floor_save_seen |= bool(((codes == 7) & (info["goalkeeper_reward_terms"]["floor_after_save"] == -.25)).any())
        optional_rewards_off &= bool((info["goalkeeper_reward_terms"]["optional_rates"] == 0).all())
        optional_rewards_off &= bool((info["goalkeeper_reward_terms"]["optional_hit"] == 0).all())
        outcomes += torch.bincount(codes[codes != 0], minlength=10)
        matrix = env.ball_contact.data.force_matrix_w[:, 0]
        max_contact = torch.maximum(max_contact, matrix.norm(dim=-1).amax(0))
    check("finite_rollout", True, {"steps": args.steps, "outcomes": outcomes.tolist()})
    check("physical_paddle_contacts", bool(max_contact[0] > .05), max_contact.tolist())
    check("baseline_can_save", bool(outcomes[7] > 0), outcomes.tolist())
    check("save_requires_physical_paddle_hit", save_contact_valid)
    check("floor_save_still_pays_floor_penalty", floor_save_seen)
    check("additional_rewards_disabled_by_default", optional_rewards_off)

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

    base_hits = shoot_at(lambda: env.robot.data.root_pos_w - env.scene.env_origins
                         + torch.tensor([.38, 0.0, 0.0], device=env.device), 1, -1.0)
    check("base_hit_attribution", base_hits > 0, base_hits)
    arm1_id = env.robot.find_bodies("flying_arm_2__link1")[0][0]
    arm_hits = shoot_at(lambda: env.robot.data.body_com_pos_w[:, arm1_id] - env.scene.env_origins
                        + torch.tensor([.22, 0.0, 0.0], device=env.device), 2, -.25)
    check("arm_hit_attribution", arm_hits > 0, arm_hits)
    goals = shoot_at(lambda: torch.tensor([-.20, .85, 1.0], device=env.device).expand(args.num_envs, -1), 3, -1.0)
    check("physical_goal_attribution", goals == args.num_envs, goals)
    floors = shoot_at(lambda: torch.tensor([1.0, .8, .04], device=env.device).expand(args.num_envs, -1), 4, -.25, 0.0)
    check("live_ball_floor_attribution", floors == args.num_envs, floors)

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
    env.cfg.curriculum_min_episodes = 4
    env.cfg.curriculum_min_steps = 10
    env.shot_level = 0
    env._stage_start_step = 0
    env._stage_episodes = 0
    env._curriculum_window.zero_()
    env._window_cursor = 0
    ids = torch.arange(4, device=env.device)
    env._shot_steps[ids] = 1
    env._episode_stage[ids] = 0
    env._outcome[ids] = 7
    env.common_step_counter = 9
    env._update_curriculum(ids)
    check("curriculum_dwell_boundary_before", env.shot_level == 0)
    env.common_step_counter = 10
    env._update_curriculum(ids)
    check("curriculum_dwell_boundary_at", env.shot_level == 1 and env._stage_episodes == 0)
    env.common_step_counter = 100
    env._update_curriculum(ids)  # these episodes still belong to stage zero
    check("curriculum_excludes_old_stage", env.shot_level == 1 and env._stage_episodes == 0)
    env._outcome[ids] = 0
    env._update_curriculum(ids)
    check("curriculum_no_promotion_without_endings", env.shot_level == 1)
    report["outcome_counts"] = env.outcome_counts.tolist()
    report["passed"] = all(row["passed"] for row in report["checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("GOALKEEPER_VALIDATION=" + json.dumps(report), flush=True)
finally:
    env.close()
    app.close()
