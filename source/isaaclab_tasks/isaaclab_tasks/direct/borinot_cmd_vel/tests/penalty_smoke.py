# SPDX-License-Identifier: BSD-3-Clause
"""Live floor, self, mixed-contact and terminal reward oracles (no learned policy)."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config("Borinot-cmd-vel", "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
    cfg.curriculum = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.
    cfg.reset_rpy_noise_degs = (0., 0., 0.)
    cfg.robot_home_quat = (1., 0., 0., 0.)
    env = gym.make("Borinot-cmd-vel", cfg=cfg).unwrapped
    report = {"control_mode": cfg.control_mode, "self_collisions": cfg.enabled_self_collisions,
              "policy_dt_s": env.step_dt, "folded_pose_degs": [40., 120.], "checks": [], "contacts": {}}

    def check(name, passed):
        passed = bool(passed)
        report["checks"].append({"name": name, "passed": passed})
        print(f"CHECK {name}: {passed}", flush=True)
        assert passed, name

    def pose(height=1.5, folded=(), position_x=0., quaternion=(1., 0., 0., 0.)):
        env.reset()
        root = env.robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins + torch.tensor([position_x, 0., height], device=env.device)
        root[:, 3:7] = torch.tensor(quaternion, device=env.device)
        root[:, 7:] = 0.
        q = torch.zeros_like(env.robot.data.default_joint_pos)
        if len(folded):
            q[list(folded)] = torch.deg2rad(torch.tensor([40., 120.], device=env.device))
        env.robot.write_root_state_to_sim(root)
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        env.scene.write_data_to_sim()
        env._episode_steps.fill_(1)
        env._power_sum_w.zero_()
        env._physics_samples = env._power_samples = 0
        for flag in env._events.values():
            flag.zero_()
        env._self_contact_bodies.zero_()

    def advance():
        # One policy interval of physics, with no policy/wrench. This isolates
        # collision attribution from controller choices and avoids auto-reset.
        for flag in env._events.values():
            flag.zero_()
        env._self_contact_bodies.zero_()
        env._power_sum_w.zero_()
        env._physics_samples = env._power_samples = 0
        for _ in range(cfg.decimation):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            env._physics_samples += 1
            env._sample_power()
            env._sample_physics_events()
        terminated, timeout = env._get_dones()
        reward = env._get_rewards()
        return terminated, timeout, reward

    def snapshot(name):
        data = env._body_contact_sensors[env._base_contact_column].data
        own = data.force_matrix_w[:, 0].sum(1)
        total = data.net_forces_w[:, 0]
        report["contacts"][name] = {"base_net_n": total.cpu().tolist(), "base_self_n": own.cpu().tolist(),
            "base_external_n": (total-own).cpu().tolist(),
            "self_contact_body_count": env._self_contact_bodies.sum(-1).cpu().tolist(),
            "penalties": {key: env.extras["reward_terms"][key].cpu().tolist()
                          for key in ("base_collision", "leave_area", "self_collision", "tilt")}}
        print("CONTACT_SNAPSHOT " + name + " " + json.dumps(report["contacts"][name]), flush=True)

    try:
        obs, _ = env.reset()
        check("four_explicit_self_filter_sources", len(env._body_contact_sensors) == 4
              and all(s.contact_physx_view.filter_count == 3 for s in env._body_contact_sensors))
        check("observation_contract_unchanged", obs["policy"].shape[-1] == cfg.observation_space)
        check("new_penalty_defaults", (cfg.rwd_scale_base_collision, cfg.rwd_scale_leave_area,
              cfg.rwd_scale_self_collision, cfg.rwd_scale_tilt) == (-2., -2., -2.25, -1.))
        check("self_cost_nonterminal_by_default", not cfg.self_collision_triggers_reset)
        # This overlap produces forces comfortably above the 0.6 N threshold.
        # The historical [70, -160] fixture only grazes at about 0.47 N.
        pose(folded=(0, 2))
        report["self_history"] = []
        for _ in range(24):
            terminated, _, _ = advance()
            report["self_history"].append({"count": env._self_contact_bodies.sum(-1).cpu().tolist(),
                "net_max_n": [float(s.data.net_forces_w.norm(dim=-1).max()) for s in env._body_contact_sensors],
                "q_degs": torch.rad2deg(env.robot.data.joint_pos).cpu().tolist()})
            if env._self_contact_bodies[[0, 2]].any(-1).all() or not cfg.enabled_self_collisions:
                break
        snapshot("self_only")
        if cfg.enabled_self_collisions:
            check("real_folded_self_contacts", env._self_contact_bodies.any(-1).tolist() == [True, False, True, False])
            check("self_does_not_create_external_base_crash", not env._events["base_collision"].any())
            check("self_only_keeps_episode_running", not terminated.any())
            expected = cfg.rwd_scale_self_collision * env._self_contact_bodies.sum(-1) * env.step_dt
            check("per_body_self_cost_exact_dt_scaling", torch.allclose(
                env.extras["reward_terms"]["self_collision"], expected, rtol=0, atol=1e-8))
            check("both_endpoints_count_not_pair_or_contact_points", (env._self_contact_bodies.sum(-1)[[0, 2]] >= 2).all())
        else:
            check("self_disabled_no_filtered_self_forces", not env._self_contact_bodies.any())
            check("self_disabled_zero_self_penalty", not env.extras["reward_terms"]["self_collision"].any())

        # Paddle-only floor contact is neither base contact nor self contact.
        pose(height=.40)
        paddle_column = env.contact_body_names.index("paddle")
        for _ in range(20):
            terminated, _, _ = advance()
            if env._body_contact_sensors[paddle_column].data.net_forces_w.norm(dim=-1).max() > 1:
                break
        snapshot("paddle_floor")
        check("real_paddle_ground_contact", env._body_contact_sensors[paddle_column].data.net_forces_w.norm(dim=-1).max() > 1)
        check("paddle_ground_not_self_or_base_collision", not env._events["base_collision"].any()
              and not env._self_contact_bodies.any())
        check("paddle_ground_has_no_unrequested_penalty", not env.extras["reward_terms"]["base_collision"].any()
              and not env.extras["reward_terms"]["self_collision"].any())

        # Isolate scene contact from height/tilt resets for penetrating fixtures.
        saved_height, saved_tilt = cfg.flight_height_range_m, cfg.reset_robot_on_tilt_threshold
        cfg.flight_height_range_m = (-10., 20.)
        cfg.reset_robot_on_tilt_threshold = False
        for label, fold in (("base_floor", ()), ("self_and_floor", (0, 1, 2, 3))):
            pose(height=.08, folded=fold)
            saw_both = False
            for _ in range(24):
                terminated, _, _ = advance()
                saw_both |= bool((env._events["base_collision"] & env._self_contact_bodies.any(-1)).any())
                if env._events["base_collision"].all() and (label == "base_floor" or saw_both or not cfg.enabled_self_collisions):
                    break
            snapshot(label)
            check(label+"_real_external_base_contact", env._events["base_collision"].all())
            check(label+"_base_crash_terminates", terminated.all())
            check(label+"_flat_two_point_penalty", torch.equal(env.extras["reward_terms"]["base_collision"],
                  torch.full((4,), -2., device=env.device)))
            cfg.reset_robot_on_robot_crash = False
            env._get_rewards()
            check(label+"_reset_switch_does_not_remove_penalty", not env._get_dones()[0].any()
                  and (env.extras["reward_terms"]["base_collision"] == -2.).all())
            cfg.reset_robot_on_robot_crash = True
            if label == "self_and_floor" and cfg.enabled_self_collisions:
                check("simultaneous_self_and_scene_contacts_are_retained", saw_both)
        cfg.flight_height_range_m, cfg.reset_robot_on_tilt_threshold = saved_height, saved_tilt

        pose(position_x=cfg.flight_xy_limit_m+1.)
        terminated, _, _ = advance()
        check("leave_area_terminates", terminated.all())
        check("leave_area_flat_two_points", torch.equal(env.extras["reward_terms"]["leave_area"], torch.full((4,), -2., device=env.device)))
        pose(quaternion=(2**-.5, 2**-.5, 0., 0.))
        terminated, _, _ = advance()
        check("tilt_terminates", terminated.all() and env._events["tilt"].all())
        check("tilt_flat_one_point", torch.equal(env.extras["reward_terms"]["tilt"], -torch.ones(4, device=env.device)))
        cfg.reset_robot_on_tilt_threshold = False
        env._get_rewards()
        check("tilt_reset_off_preserves_penalty", not env._get_dones()[0].any()
              and (env.extras["reward_terms"]["tilt"] == -1.).all())
        cfg.reset_robot_on_tilt_threshold = saved_tilt

        pose()
        env._episode_steps.fill_(env.max_episode_length)
        terminated, timeout = env._get_dones()
        env._get_rewards()
        check("clean_timeout_is_not_failure", not terminated.any() and timeout.all())
        check("clean_timeout_has_no_event_penalties", not any(env.extras["reward_terms"][name].any()
              for name in ("base_collision", "leave_area", "self_collision", "tilt")))

        for flag in env._events.values():
            flag.fill_(True)
        env._self_contact_bodies.fill_(True)
        env._reset_idx(torch.tensor([0], device=env.device))
        check("partial_reset_clears_selected_latches_only", all(not f[0] and f[1:].all() for f in env._events.values())
              and not env._self_contact_bodies[0].any() and env._self_contact_bodies[1:].all())

        # Use the real step/reset ordering, not only direct reward calls: the
        # agent must receive the final penalty even after buffers are reset.
        for label, kwargs, expected in (
            ("base", {"height": .08}, {"base_collision": -2.}),
            ("area", {"position_x": cfg.flight_xy_limit_m+1.}, {"leave_area": -2.}),
            ("tilt", {"quaternion": (2**-.5, 2**-.5, 0., 0.)}, {"tilt": -1.}),
            ("area_and_tilt", {"position_x": cfg.flight_xy_limit_m+1.,
                               "quaternion": (2**-.5, 2**-.5, 0., 0.)},
             {"leave_area": -2., "tilt": -1.}),
        ):
            cfg.flight_height_range_m = (-10., 20.) if label == "base" else saved_height
            cfg.reset_robot_on_tilt_threshold = label != "base"
            pose(**kwargs)
            for _ in range(8):
                _, reward, terminated, timeout, info = env.step(torch.zeros(4, cfg.action_space, device=env.device))
                if terminated.all():
                    break
            check(label+"_automatic_terminal_reset", terminated.all() and not timeout.any()
                  and not env._episode_steps.any())
            check(label+"_terminal_penalty_reaches_agent", all((info["reward_terms"][key] == value).all()
                  and info["log"]["Episode_Reward/"+key] == value for key, value in expected.items())
                  and torch.allclose(reward, sum(info["reward_terms"].values())))
            check(label+"_reset_clears_event_latches", not any(f.any() for f in env._events.values())
                  and not env._self_contact_bodies.any())
        cfg.flight_height_range_m, cfg.reset_robot_on_tilt_threshold = saved_height, saved_tilt
        env.reset()
        for _ in range(40):
            obs, reward, *_ = env.step(torch.zeros(4, cfg.action_space, device=env.device))
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
        check("finite_normal_rollout_with_contact_sensors", True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(f"PASSED {len(report['checks'])} checks", flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        env.close()


if __name__ == "__main__":
    main()
    app.close()
