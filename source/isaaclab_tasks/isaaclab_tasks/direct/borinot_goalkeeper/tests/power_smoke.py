# SPDX-License-Identifier: BSD-3-Clause
"""GPU check of both Borinot power samplers, interval means, and reset boundaries."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["borinot", "goalkeeper"], required=True)
parser.add_argument("--direct-torque", action="store_true")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks

if args.task == "borinot":
    from isaaclab_tasks.direct.borinot.borinot_env_cfg import BorinotEnvCfg
    cfg = BorinotEnvCfg()
    task_id = "Borinot-Direct-v0"
    # The original config predates the source/ layout; override paths only for this check.
    source = Path(__file__).resolve().parents[5]
    cfg.robot_cfg.spawn.usd_path = str(source / "isaaclab_assets/data/Robots/Borinot/borinotFlat.usd")
    cfg.ground_usd_path = str(source / "borinotIsaacLab/isaacsim_assets/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd")
    cfg.robot_cfg.init_state.pos = (0.0, 0.0, 1.5)
    if args.direct_torque:
        cfg.des_joint_out = False
        cfg.action_space = 8
        cfg.observation_space = 32
        cfg.robot_cfg.actuators["leg_torque"].stiffness = 0.0
        cfg.robot_cfg.actuators["leg_torque"].damping = 0.0
else:
    from isaaclab_tasks.direct.borinot_goalkeeper.borinot_goalkeeper_env_cfg import BorinotGoalkeeperEnvCfg
    cfg = BorinotGoalkeeperEnvCfg()
    task_id = "Borinot-Goalkeeper-Direct-v0"
    cfg.curriculum = False
    cfg.reset_xy_noise_m = cfg.reset_z_noise_m = 0.0
assert cfg.energy_reward_scale == 0.0
assert cfg.arm_copper_loss_coefficient == 7.0
cfg.scene.num_envs = 4
cfg.sim.device = args.device
cfg.seed = 42
env = gym.make(task_id, cfg=cfg).unwrapped
report = {"task": task_id, "direct_torque": args.direct_torque, "physics_dt_s": env.physics_dt,
          "policy_dt_s": env.step_dt, "arm_copper_loss_coefficient_W_per_Nm2": cfg.arm_copper_loss_coefficient,
          "power_components": ["rotor_mechanical", "rotor_dissipative", "arm_mechanical", "arm_copper_loss"],
          "checks": [], "intervals": []}


def check(name, condition):
    report["checks"].append({"name": name, "passed": bool(condition)})
    assert condition, name


samples = []
original_sample = env._sample_power


def traced_sample():
    if env._power_samples < env._physics_samples:
        thrust = env._thrusts if args.task == "borinot" else env._rotor_thrust
        ids = env._leg_dof_ids if args.task == "borinot" else env.joint_ids
        # Independent reference using simulator actuator estimates, not the base's cached buffer.
        drag_torque = (cfg.km / cfg.kf) * thrust.double()
        speed = (thrust.double().clamp_min(0) / cfg.kf).sqrt()
        rotor = (drag_torque * speed).sum(-1)
        rotor_loss = rotor / cfg.p_el_motor_efficiency - rotor
        torque = env.robot.data.applied_torque[:, ids].double()
        arm = (torque * env.robot.data.joint_vel[:, ids].double()).sum(-1)
        current = torque / 0.225
        effective_resistance = cfg.arm_copper_loss_coefficient * 0.225**2
        copper = effective_resistance * current.square().sum(-1)
        samples.append(torch.stack((rotor, rotor_loss, arm, copper), dim=-1))
    original_sample()


env._sample_power = traced_sample


def step_and_check(actions):
    samples.clear()
    obs, reward, terminated, truncated, info = env.step(actions)
    check("one_sample_per_completed_physics_step", len(samples) == cfg.decimation
          and env._power_samples == cfg.decimation)
    expected = torch.stack(samples).mean(0)
    names = ("rotor_mechanical", "rotor_dissipative", "arm_mechanical", "arm_copper_loss")
    actual = torch.stack([info["power_mean_w"][name] for name in names], dim=-1)
    torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=1e-5)
    check("reward_and_observations_finite", torch.isfinite(reward).all()
          and all(torch.isfinite(value).all() for value in obs.values()))
    expected_reward = cfg.energy_reward_scale * expected.sum(-1)
    if args.task == "goalkeeper":
        torch.testing.assert_close(info["goalkeeper_reward_terms"]["energy"].double(),
                                   expected_reward, atol=1e-5, rtol=1e-5)
    else:
        torch.testing.assert_close(info["log"]["rewards/r_energy"].double(),
                                   expected_reward.mean(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(info["power_mean_w"]["total_estimate"],
                               info["power_mean_w"]["mechanical"] + info["power_mean_w"]["dissipative"])
    if args.task == "borinot" or bool((terminated | truncated).any()):
        for name in info["power_mean_w"]:
            check("numeric_power_logging", f"Power/{name}_mean_w" in info["log"])
    elif args.task == "goalkeeper":
        check("preserves_episode_only_log_packets", "log" not in info)
    report["intervals"].append({"energy_reward_scale": cfg.energy_reward_scale,
                                "substep_power_w": torch.stack(samples).cpu().tolist(),
                                "mean_power_w": actual.cpu().tolist()})
    return terminated | truncated, actual


try:
    env.reset()
    actions = torch.zeros(4, cfg.action_space, device=env.device)
    if args.task == "goalkeeper":
        actions[:, 4:6] = torch.tensor([0.2, -0.15], device=env.device)
    elif args.direct_torque:
        actions[:, 6:8] = torch.tensor([0.2, -0.15], device=env.device)
    else:
        actions[:, 6:10] = torch.tensor([0.98, 0.2, 0.99, -0.15], device=env.device)
    arm_nonzero = False
    for step in range(8):
        cfg.energy_reward_scale = 0.0 if step < 4 else -0.002
        actions[:, :1] = -0.2 if step % 2 else 0.0
        _, actual = step_and_check(actions)
        arm_nonzero |= bool((actual[:, 2].abs() > 0).any())
    check("arm_mechanical_power_is_active", arm_nonzero)
    check("arm_copper_loss_is_active", bool((actual[:, 3] > 0).any()))
    other_power = env._power_sum_w[1:].clone()
    env._reset_idx(torch.tensor([0], device=env.device))
    check("partial_reset_clears_only_selected_power", bool((env._power_sum_w[0] == 0).all())
          and torch.equal(env._power_sum_w[1:], other_power))
    step_and_check(actions)
    if args.task == "goalkeeper":
        env._shot_steps[:] = env.max_episode_length - 1
    else:
        env.episode_length_buf[:] = env.max_episode_length - 1
    done, actual = step_and_check(actions)
    check("terminal_interval_is_sampled_before_reset", bool(done.all()) and bool((actual[:, 0] > 0).all()))
    check("automatic_reset_clears_power_accumulators", bool((env._power_sum_w == 0).all()))
    step_and_check(actions)
    check("next_interval_has_no_previous_episode_samples", env._power_samples == cfg.decimation)
    print(f"PASS {task_id}: {len(report['checks'])} checks, {len(report['intervals'])} intervals", flush=True)
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    app.close()
