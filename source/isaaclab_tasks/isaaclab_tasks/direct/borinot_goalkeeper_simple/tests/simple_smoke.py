# SPDX-License-Identifier: BSD-3-Clause
"""Check the simplified reward with real Hydra overrides, PhysX contacts and the playback env.

Run twice: once with defaults and once with env.minimize_energy_not_power=True.
"""
import argparse
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.borinot_goalkeeper_simple.playback import BorinotGoalkeeperSimplePlayEnv
from isaaclab_tasks.utils.hydra import hydra_task_config

TASK = "Borinot-Goalkeeper-Simple-Direct-v0"
HITS = ("task/hit_racket", "task/hit_arm", "task/hit_base")


@hydra_task_config(TASK, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = 8
    cfg.sim.device = args.device
    cfg.seed = 181018
    cfg.curriculum = cfg.reset_grid_enabled = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
    env = BorinotGoalkeeperSimplePlayEnv(cfg, keep_balls=2)
    n, dev = env.num_envs, env.device
    report = {"device": dev, "minimize_energy_not_power": cfg.minimize_energy_not_power, "checks": [], "data": {}}
    zero = torch.zeros(n, cfg.action_space, device=dev)

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        print(f"CHECK {name}: {bool(passed)}", flush=True)
        assert passed, name

    def quiet():
        """Fresh robots at home, no live ball, no automatic launches."""
        env.reset()
        everyone = torch.arange(n, device=dev)
        env._ball_active[:] = False
        env._park_ball(everyone)
        env._next_launch_steps[:] = 100000

    def shoot(ids, positions, velocities):
        ids = torch.tensor(ids, device=dev)
        state = env.ball.data.default_root_state[ids].clone()
        state[:, :3] = torch.stack(positions) + env.scene.env_origins[ids]
        state[:, 7:] = 0.0
        state[:, 7:10] = torch.tensor(velocities, device=dev)
        env.ball.write_root_state_to_sim(state, ids)
        env.ball_contact.reset(ids)
        env._previous_ball[ids] = torch.stack(positions)
        env._ball_active[ids] = True
        env._touched[ids] = env._ball_done[ids] = False
        env._shot_steps[ids] = 0

    def body(name):
        return env.robot.data.body_pos_w[:, env.robot.find_bodies(name)[0][0]] - env.scene.env_origins

    try:
        check("Hydra keeps physical self-collisions on", cfg.robot.spawn.articulation_props.enabled_self_collisions)
        check("filtered sensors resolved (ball 5, base 4, racket 1)", [
            s.contact_physx_view.filter_count for s in (env.ball_contact, env.base_contact, env.paddle_contact)] == [5, 4, 1])
        obs = env._get_observations()
        check("actor/critic sizes 53/63", (obs["policy"].shape[1], obs["critic"].shape[1]) == (cfg.observation_space,
              cfg.state_space) == (53, 63))

        # --- Task: real ball contacts and goals --------------------------------------------------------------
        quiet()
        racket, thigh = env._paddle_pose()[0], body("flying_arm_2__link1")
        arm = thigh + 0.5 * (body("flying_arm_2__link2") - thigh)
        base = body("base")
        x = torch.tensor([1.0, 0.0, 0.0], device=dev)
        fixtures = {0: (racket[0] + 0.15 * x, [-3.0, 0.0, 0.0]),  # racket face
                    1: (arm[1] + 0.15 * x, [-3.0, 0.0, 0.0]),  # thigh, clear of the racket
                    2: (base[2] + torch.tensor([0.45, 0.0, 0.04], device=dev), [-3.0, 0.0, 0.0]),  # rotor shield
                    3: (torch.tensor([-0.30, 0.8, 1.2], device=dev), [-10.0, 0.0, 0.0]),  # straight into the goal
                    4: (torch.tensor([-0.30, 1.5, 1.2], device=dev), [-10.0, 0.0, 0.0]),  # wide of the goal
                    # Back of the racket, beside the calf; it rebounds toward the goal.
                    5: (racket[5] + torch.tensor([-0.08, 0.05, 0.0], device=dev), [2.0, 0.0, 0.0]),
                    6: (torch.tensor([2.8, 0.0, 1.5], device=dev), [5.0, 0.0, 0.0])}  # leaves the tracking region
        shoot(list(fixtures), [p for p, _ in fixtures.values()], [v for _, v in fixtures.values()])
        history_before = env._history_used.sum(-1).clone()  # quiet() already retired the reset launches
        paid = {name: torch.zeros(n, device=dev) for name in (*HITS, "task/goal", "integrity/base_crash")}
        payments = {name: torch.zeros(n, device=dev, dtype=torch.long) for name in paid}
        outcome, finished_at, resets = torch.zeros(n, dtype=torch.long, device=dev), {}, []
        for k in range(60):
            _, _, terminated, _, info = env.step(zero)
            for i in terminated.nonzero().flatten().tolist():
                resets.append({"step": k, "env": i, "events": [name for name, v in info["goalkeeper_events"].items() if v[i]],
                               "base_xyz": (env.robot.data.root_pos_w[i] - env.scene.env_origins[i]).tolist()})
            terms = info["goalkeeper_reward_terms"]
            torch.testing.assert_close(sum(terms.values()), env.reward_buf)
            for name in paid:
                paid[name] += terms[name]
                payments[name] += terms[name] != 0
            for i in info["goalkeeper_shot_finished"].nonzero().flatten().tolist():
                outcome[i], finished_at[i] = info["goalkeeper_outcome"][i], k
        report["data"]["task"] = {"paid": {k: v.tolist() for k, v in paid.items()},
                                  "payments": {k: v.tolist() for k, v in payments.items()},
                                  "outcome": outcome.tolist(), "finished_at_step": finished_at, "resets": resets}
        expect = {0: ("task/hit_racket", 5.0), 1: ("task/hit_arm", 2.5), 2: ("task/hit_base", 1.25)}
        for i, (name, value) in expect.items():
            check(f"env {i}: real first contact pays only {name} once", payments[name][i] == 1 and paid[name][i] == value
                  and sum(int(payments[h][i]) for h in HITS) == 1)
        check("ball on the base is not a base crash", paid["integrity/base_crash"][2] == 0)
        check("real goal pays -5 once and ends the shot", payments["task/goal"][3] == 1 and paid["task/goal"][3] == -5.0
              and outcome[3] == 3)
        hit = sum(paid[h][5] for h in HITS)
        report["data"]["hit_then_goal_first_contact"] = [h for h in HITS if payments[h][5]]
        check("contact then goal: both paid once and summed", sum(int(payments[h][5]) for h in HITS) == 1
              and hit > 0 and payments["task/goal"][5] == 1 and paid["task/goal"][5] == -5.0 and outcome[5] == 3)
        check("wide and escaped balls end without reward", outcome[4] == 7 and outcome[6] == 7 and all(
            paid[name][[4, 6]].abs().sum() == 0 for name in (*HITS, "task/goal")))
        check("no goal for the clean hits", payments["task/goal"][:3].sum() == 0)
        # Zero actions are not a hover controller, so a robot may drift out of the box. Every reset
        # must come from an enabled safety reset, never from a ball contact or goal (reset_on_goal=False).
        check("ball events never reset the robot", all("leave_area" in r["events"] or "base_crash" in r["events"]
                                                       for r in resets))
        check("scored ball kept once in playback history", env._history_used[3].sum() == history_before[3] + 1)

        # --- Integrity: real self-contacts, base crash, area and tilt ----------------------------------------
        quiet()
        q = env.robot.data.default_joint_pos.clone()
        # Imposed overlapping poses from a 15-degree PhysX scan: 0 racket on base, 1 calf on base, 2 racket on thigh.
        poses = {0: (0.0, 150.0), 1: (-80.0, -90.0), 2: (40.0, 160.0), 3: (90.0, 0.0)}
        for i, (shoulder, elbow) in poses.items():
            q[i, env.joint_ids] = torch.deg2rad(torch.tensor([shoulder, elbow], device=dev))
        env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        root = env.robot.data.root_state_w.clone()
        root[3, :3] = env.scene.env_origins[3] + torch.tensor([0.2, 0.0, 0.20], device=dev)  # lands on its skis
        root[4, 0] = env.scene.env_origins[4, 0] + 0.95  # outside the goal-area box
        root[5, 3:7] = torch.tensor([math.cos(math.radians(40)), math.sin(math.radians(40)), 0.0, 0.0], device=dev)
        env.robot.write_root_state_to_sim(root)
        actions = zero.clone()
        actions[3, :6] = -1.0  # rotors off
        seen = {name: torch.zeros(n, dtype=torch.bool, device=dev) for name in ("base_crash", "self_collision",
                                                                                 "leave_area", "tilt")}
        columns = {"base-calf": False, "base-racket": False, "racket-thigh": False}
        crash_step = None
        for k in range(25):
            env._pre_physics_step(actions)  # keep the imposed state: step() would not reset these envs early
            for _ in range(cfg.decimation):
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                base_n = torch.linalg.vector_norm(env.base_contact.data.force_matrix_w[:, 0], dim=-1)
                columns["base-racket"] |= bool(base_n[0, 3] > 0.1)
                columns["base-calf"] |= bool(base_n[1, 2] > 0.1)
                columns["racket-thigh"] |= bool(torch.linalg.vector_norm(env.paddle_contact.data.force_matrix_w[2, 0]) > 0.1)
            terminated, _ = env._get_dones()
            env.reset_terminated[:] = terminated
            env.reset_buf = terminated
            env._get_rewards()
            terms, events = env.extras["goalkeeper_reward_terms"], env.extras["goalkeeper_events"]
            if k == 0:
                first = {"terms": {key: value.tolist() for key, value in terms.items()},
                         "terminated": terminated.tolist(), "events": {key: v.tolist() for key, v in events.items()}}
                report["data"]["integrity_first_step"] = first
                check("self-collisions pay -2.5 and keep the robot (default)", terms["integrity/self_collision"][:3].tolist()
                      == [-2.5] * 3 and not terminated[:3].any())
                check("self-contact is not a base crash", not events["base_crash"][:3].any())
                check("leaving the area pays -5 and resets", terms["integrity/leave_area"][4] == -5.0 and terminated[4])
                check("tilt alone has no penalty and no reset (default)", events["tilt"][5] and terms["integrity/base_crash"][5]
                      == terms["integrity/self_collision"][5] == terms["integrity/leave_area"][5] == 0 and not terminated[5])
                check("quiet environments have no events", not any(v[6:].any() for v in events.values()))
                cfg.self_collision_triggers_reset = cfg.reset_robot_on_tilt_threshold = True
                switched = env._get_dones()[0]
                cfg.self_collision_triggers_reset = cfg.reset_robot_on_tilt_threshold = False
                check("reset switches turn on self-collision and tilt resets", switched[[0, 1, 2, 5]].all())
            for name in seen:
                seen[name] |= events[name]
            if events["base_crash"][3] and crash_step is None:
                crash_step = k
                check("real base-floor contact pays -5 and resets", terms["integrity/base_crash"][3] == -5.0 and terminated[3]
                      and not events["leave_area"][3])
                break
        report["data"]["self_contact_columns"] = columns
        report["data"]["base_crash_step"] = crash_step
        check("all three robot/robot pairs were detected", all(columns.values()))
        check("landing robot crashed", crash_step is not None)

        # --- Desirability: power each step, or each shot's energy when the shot ends -------------------------
        # Envs 0 and 1 end one shot early; env 2 leaves the area on the last step; the rest hit the time limit.
        quiet()
        cfg.episode_length_s = 0.3
        shoot([0, 1], [torch.tensor([-0.30, 1.5, 1.2], device=dev), torch.tensor([-0.30, 0.8, 1.2], device=dev)],
              [[-10.0, 0.0, 0.0]] * 2)
        steps, sums, paid = env.max_episode_length, torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        joules, unpaid, shot_end = torch.zeros(n, device=dev), torch.zeros(n, device=dev), {}
        for k in range(steps):
            if k == steps - 1:
                root = env.robot.data.root_state_w[2:3].clone()
                root[0, 0] = env.scene.env_origins[2, 0] + 0.95  # outside the goal-area box
                env.robot.write_root_state_to_sim(root, torch.tensor([2], device=dev))
            _, _, terminated, truncated, info = env.step(zero)
            energy = info["goalkeeper_reward_terms"]["desirability/energy"]
            per_watt = cfg.rwd_scale_energy * info["power_mean_w"]["total_estimate"]
            sums += per_watt
            paid += energy
            joules += info["power_mean_w"]["total_estimate"] * env.step_dt
            for i in info["goalkeeper_shot_finished"].nonzero().flatten().tolist():
                shot_end[i] = k
            if cfg.minimize_energy_not_power:  # reference: pay the running sum when a shot ends or the env terminates
                unpaid += per_watt
                ends = info["goalkeeper_shot_finished"] | terminated
                torch.testing.assert_close(energy, unpaid * ends, rtol=1e-5, atol=1e-6)
                unpaid *= ~(ends | truncated)
            else:
                torch.testing.assert_close(energy, per_watt)
        others = torch.arange(n, device=dev) != 2
        check("env 2 terminated and the rest hit the time limit on the last step",
              terminated[2] and not terminated[others].any() and truncated[others].all())
        check("envs 0 and 1 ended their shot early", sorted(shot_end) == [0, 1] and max(shot_end.values()) < steps - 1)
        if cfg.minimize_energy_not_power:
            torch.testing.assert_close(paid[2], sums[2], rtol=1e-5, atol=1e-6)
            check("energy mode: termination pays the whole unpaid window", paid[2] < 0)
            check("energy mode: the shot end pays, the time-limit cut does not", (paid[:2] < 0).all()
                  and (paid[:2] > sums[:2]).all() and not paid[3:].any())
        else:
            check("power mode: each step pays rwd_scale_energy * mean power", True)
        torch.testing.assert_close(env.extras["log"]["Episode/energy_j"], joules.mean(), rtol=1e-5, atol=1e-4)
        check("Episode/energy_j logs the episode's energy", True)
        report["data"]["energy"] = {"episode_steps": steps, "shot_end_step": shot_end, "paid": paid.tolist(),
                                    "summed_power_terms": sums.tolist(), "joules": joules.tolist()}
        cfg.episode_length_s = 20.0

        # --- Partial reset and a random rollout ----------------------------------------------------------------
        env.reset()
        for _ in range(5):
            env.step(zero)
        before, unpaid_before = env._episode_energy_j.clone(), env._unpaid_energy_j.clone()
        env._reset_idx(torch.tensor([0], device=dev))
        check("partial reset clears only its own episode and unpaid energy", env._episode_energy_j[0] == 0
              and env._unpaid_energy_j[0] == 0 and torch.equal(env._episode_energy_j[1:], before[1:])
              and torch.equal(env._unpaid_energy_j[1:], unpaid_before[1:]))
        env.reset()
        torch.manual_seed(7)
        for _ in range(64):
            obs, reward, *_ = env.step(torch.randn(n, cfg.action_space, device=dev))
            assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
            torch.testing.assert_close(sum(env.extras["goalkeeper_reward_terms"].values()), reward)
        check("64 finite random-action steps; reward equals the sum of its terms", True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.close()


try:
    main()
finally:
    app.close()
