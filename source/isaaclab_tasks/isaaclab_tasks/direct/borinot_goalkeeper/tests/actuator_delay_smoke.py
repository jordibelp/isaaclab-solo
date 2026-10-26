# SPDX-License-Identifier: BSD-3-Clause
"""Live PhysX: a delayed policy command reaches the plant d physics steps late; feedback loops are not delayed.

Seeded GPU rollouts repeat bit for bit, so each environment is compared with its own neutral rollout.
Pass env.actuator_delay_range=[0.0,0.04] for the timing checks, or nothing for the default zero-delay checks.
./isaaclab.sh -p <this file> --task Borinot-Goalkeeper-Direct-v0 --output out.json --headless env.control_mode=indi
"""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Borinot-Goalkeeper-Direct-v0")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

STEP_AT, STEP_FOR, KICK_AT, AFTER_KICK = 5, 10, 4, 4  # policy steps
KICK = (0.6, -0.5, 0.4)  # rad/s, added to the world angular velocity


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    report = {"task": args.task, "control_mode": cfg.control_mode, "sac_action_mapping": cfg.sac_action_mapping,
              "physics_hz": cfg.physics_hz, "policy_hz": cfg.policy_hz, "hydra": hydra_args,
              "actuator_delay": cfg.actuator_delay, "actuator_delay_range": cfg.actuator_delay_range, "checks": []}

    def check(name, condition, **values):
        report["checks"].append({"name": name, "passed": bool(condition), **values})
        print(f"CHECK {name}: {bool(condition)} {values or ''}", flush=True)

    try:  # IsaacLab keeps the declared float type: write 0.0, not 0.
        type(cfg)().from_dict({"actuator_delay": 0})
        check("integer_literal_for_float_delay_is_rejected", False)
    except ValueError as error:
        check("integer_literal_for_float_delay_is_rejected", True, message=str(error))
    indi = cfg.control_mode == "indi"
    torque = cfg.control_mode == "thrust_and_torque_control"
    cfg.curriculum = False
    cfg.scene.num_envs = 9
    cfg.sim.device = args.device
    cfg.seed = 7
    env = gym.make(args.task, cfg=cfg).unwrapped
    n, dec = env.num_envs, cfg.decimation
    delay = env._actuator_delay
    report["bounds_steps"] = [delay.low, delay.high]
    check("hydra_value_reaches_the_environment", delay.high == round(max(
        cfg.actuator_delay, *(cfg.actuator_delay_range or (0,))) * cfg.physics_hz))
    timed = delay.high >= 8
    check("zero_delay_allocates_no_queue", timed or delay._queue is None)
    steps = torch.arange(n, device=env.device) if timed else torch.zeros(n, dtype=torch.long, device=env.device)

    log = []
    original = env._apply_action

    def recorded():
        data = env.robot.data
        state = torch.cat((data.root_state_w, data.joint_pos, data.joint_vel), -1).clone()
        original()
        signal = env._indi.requested_wrench[:, :1] if indi else env._actuator_commands[:, :6]
        feedback = env._indi.command_speed if indi else env._actuator_commands[:, :6]
        log.append((state, signal.clone(), feedback.clone()))

    env._apply_action = recorded
    neutral = torch.zeros(n, cfg.action_space, device=env.device)
    command = neutral.clone()
    if indi:
        command[:, 2] = 0.125 if cfg.sac_action_mapping else 1.0  # +1 m/s^2 world vertical acceleration
        command[:, 7:9] = 0.2 if cfg.sac_action_mapping else 0.3  # joint targets
    else:
        command[:, :6] = 0.2  # rotor thrust above hover
        command[:, 6:8] = 0.5 if torque else 0.3  # joint torque (PPO N*m, SAC normalized) or position target

    def rollout(ticks, step_command=False, kick=False, delays=steps):
        env.seed(7)
        env.reset()
        delay.steps.copy_(delays)  # deterministic delays; reset() already refilled the queues with neutral zeros
        log.clear()
        for tick in range(ticks):
            if kick and tick == KICK_AT:
                velocity = env.robot.data.root_vel_w.clone()
                velocity[:, 3:] += torch.tensor(KICK, device=env.device)
                env.robot.write_root_velocity_to_sim(velocity)
            env.step(command if step_command and tick >= STEP_AT else neutral)
        return [torch.stack([entry[k] for entry in log]) for k in range(3)]

    def first_difference(a, b, start):
        changed = (a != b).flatten(2).any(-1)  # [physics step, env]
        return [int(c.nonzero()[0]) - start if c.any() else None for c in changed.T]

    with torch.inference_mode():
        ticks = STEP_AT + STEP_FOR
        first_states, _, _ = rollout(ticks)
        neutral_states, neutral_signals, _ = rollout(ticks)
        repeat_states, _, _ = rollout(ticks)
        report.update(first_vs_second_rollout_first_difference=first_difference(first_states, neutral_states, 0),
                      first_vs_second_rollout_max_gap=float((first_states - neutral_states).abs().max()),
                      first_vs_second_rollout_gap_by_column=(first_states - neutral_states).abs().amax((0, 1)).tolist())
        check("seeded_rollouts_repeat_bit_for_bit", torch.equal(neutral_states, repeat_states))
        states, signals, _ = rollout(ticks, step_command=True)
        s0 = STEP_AT * dec
        arrival = first_difference(signals, neutral_signals, s0)
        divergence = first_difference(states, neutral_states, s0)
        report.update(delays_steps=steps.tolist(), command_arrival_steps=arrival, state_divergence_steps=divergence)
        check("command_reaches_controller_exactly_d_physics_steps_late", arrival == steps.tolist())
        check("robot_moves_bit_identically_until_one_physics_step_after_arrival", divergence == (steps + 1).tolist())

        # Feedback: neutral commands throughout, with and without an angular-velocity kick.
        ticks = KICK_AT + AFTER_KICK
        _, _, calm = rollout(ticks)
        kicked_states, _, kicked = rollout(ticks, kick=True)
        undelayed_states, _, undelayed = rollout(ticks, kick=True, delays=torch.zeros_like(steps))
        s1 = KICK_AT * dec
        reaction = first_difference(kicked, calm, s1)
        report.update(kick_reaction_steps=reaction,
                      kick_response_max=float((kicked - calm)[s1:].abs().max()))
        check("kick_response_is_bit_identical_with_and_without_the_command_delay",
              torch.equal(kicked, undelayed) and torch.equal(kicked_states, undelayed_states))
        if indi:
            check("indi_reacts_to_the_kick_in_the_same_physics_step_for_every_delay", reaction == [0] * n)
        else:
            check("direct_rotor_commands_ignore_the_kick_without_a_policy_reaction", reaction == [None] * n)

        if timed:  # every reset draws a new uniform delay; the episode log reports the finished episodes' mean
            draws = []
            for _ in range(40):
                env.reset()
                draws.append(delay.steps.clone())
            draws = torch.cat(draws)
            report["reset_draw_counts"] = torch.bincount(draws, minlength=delay.high + 1).tolist()
            check("every_whole_step_delay_in_the_range_is_drawn",
                  draws.min() == delay.low and draws.max() == delay.high
                  and bool((torch.bincount(draws - delay.low) > 0).all()))
            for _ in range(3):
                env.step(neutral)
            expected = float(delay.seconds.mean())
            env.reset()
            logged = float(env.extras["log"]["Episode/actuator_delay_s"])
            check("episode_log_reports_the_mean_delay_of_finished_episodes", abs(logged - expected) < 1e-6,
                  logged=logged, expected=expected)
    env.close()
    report["passed"] = all(c["passed"] for c in report["checks"])
    args.output.write_text(json.dumps(report, indent=1))
    print("ACTUATOR_DELAY_SMOKE=" + json.dumps({"passed": report["passed"], "task": args.task,
                                                "control_mode": cfg.control_mode}), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
