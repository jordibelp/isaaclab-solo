#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run two Solo12 checkpoints with a fixed command, save reusable timestep data, and plot comparisons.

Example:
./isaaclab.sh -p source/scripts/skrl/compare_solo12_checkpoints.py \
    --task solo12-v0 \
    --checkpoint_a logs/skrl/checkpoints/0326_vduw2o5j_best_agent.pt \
    --checkpoint_b logs/skrl/checkpoints/0326_sc1v0hs5_overnight_best_agent.pt \
    --cmd 1 1 1 \
    --duration_s 4 \
    --headless
"""

import argparse
import copy
import gc
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare two Solo12 skrl checkpoints under a fixed command.")
parser.add_argument("--task", type=str, default="solo12-v0", help="Task name.")
parser.add_argument("--agent", type=str, default=None, help="RL agent configuration entry point.")
parser.add_argument("--checkpoint_a", type=str, required=True, help="Path to the first checkpoint.")
parser.add_argument("--checkpoint_b", type=str, required=True, help="Path to the second checkpoint.")
parser.add_argument("--label_a", type=str, default=None, help="Legend label for checkpoint A.")
parser.add_argument("--label_b", type=str, default=None, help="Legend label for checkpoint B.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="ML framework used by the trained skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="RL algorithm used for training.",
)
parser.add_argument("--duration_s", type=float, default=4.0, help="How long to run each policy.")
parser.add_argument(
    "--cmd",
    type=float,
    nargs=3,
    default=(1.0, 1.0, 1.0),
    metavar=("VX", "VY", "WZ"),
    help="Fixed command (vx vy wz) applied to all environments.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for npz/csv/plots.")
parser.add_argument(
    "--force_transmited_through_joints_reward_scale",
    type=float,
    default=None,
    help="Override the environment reward scale for force_transmited_through_joints.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skrl
import torch
from packaging import version

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import borinotIsaacLab.tasks  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install with 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()

REWARD_TERM_ORDER = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "dof_torques_l2",
    "dof_acc_l2",
    "action_rate_l2",
    "feet_air_time",
    "undesired_contacts",
    "flat_orientation_l2",
    "force_transmited_through_joints",
    "foot_contact",
]


def _apply_fixed_command(raw_env, cmd):
    raw_env._commands[:, 0] = float(cmd[0])
    raw_env._commands[:, 1] = float(cmd[1])
    raw_env._commands[:, 2] = float(cmd[2])
    if hasattr(raw_env, "_command_steps_left"):
        raw_env._command_steps_left[:] = max(1, int(1e9))



def _compute_reward_terms(raw_env):
    commands = raw_env._commands
    robot = raw_env._robot
    joint_ids = raw_env._joint_ids

    lin_vel_error = torch.sum(torch.square(commands[:, :2] - robot.data.root_lin_vel_b[:, :2]), dim=1)
    yaw_rate_error = torch.square(commands[:, 2] - robot.data.root_ang_vel_b[:, 2])
    z_vel_error = torch.square(robot.data.root_lin_vel_b[:, 2])
    ang_vel_error = torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)
    joint_torques = torch.sum(torch.square(robot.data.applied_torque[:, joint_ids]), dim=1)
    joint_accel = torch.sum(torch.square(robot.data.joint_acc[:, joint_ids]), dim=1)
    action_rate = torch.sum(torch.square(raw_env._actions - raw_env._previous_actions), dim=1)
    flat_orientation = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)

    first_contact = raw_env._contact_sensor.compute_first_contact(raw_env.step_dt)[:, raw_env._feet_body_ids]
    last_air_time = raw_env._contact_sensor.data.last_air_time[:, raw_env._feet_body_ids]
    feet_air_time = torch.sum((last_air_time - raw_env.cfg.feet_air_time_threshold) * first_contact, dim=1)
    feet_air_time *= torch.norm(commands[:, :2], dim=1) > 0.1

    undesired_contacts = raw_env._compute_contact_count(raw_env._thigh_body_ids, raw_env.cfg.undesired_contact_threshold)
    force_transmited_through_joints_raw = raw_env._compute_force_transmited_through_joints()
    foot_contact_raw = raw_env._compute_foot_contact_penalty()

    return {
        "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / raw_env.cfg.tracking_std**2)
        * raw_env.cfg.track_lin_vel_xy_reward_scale
        * raw_env.step_dt,
        "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / raw_env.cfg.tracking_std**2)
        * raw_env.cfg.track_ang_vel_z_reward_scale
        * raw_env.step_dt,
        "lin_vel_z_l2": z_vel_error * raw_env.cfg.lin_vel_z_reward_scale * raw_env.step_dt,
        "ang_vel_xy_l2": ang_vel_error * raw_env.cfg.ang_vel_xy_reward_scale * raw_env.step_dt,
        "dof_torques_l2": joint_torques * raw_env.cfg.joint_torque_reward_scale * raw_env.step_dt,
        "dof_acc_l2": joint_accel * raw_env.cfg.joint_accel_reward_scale * raw_env.step_dt,
        "action_rate_l2": action_rate * raw_env.cfg.action_rate_reward_scale * raw_env.step_dt,
        "feet_air_time": feet_air_time * raw_env.cfg.feet_air_time_reward_scale * raw_env.step_dt,
        "undesired_contacts": undesired_contacts * raw_env.cfg.undesired_contact_reward_scale * raw_env.step_dt,
        "flat_orientation_l2": flat_orientation * raw_env.cfg.flat_orientation_reward_scale * raw_env.step_dt,
        "force_transmited_through_joints": force_transmited_through_joints_raw
        * raw_env.cfg.force_transmited_through_joints_reward_scale
        * raw_env.step_dt,
        "foot_contact": foot_contact_raw * raw_env.cfg.foot_contact_reward_scale * raw_env.step_dt,
        "force_transmited_through_joints_raw": force_transmited_through_joints_raw,
        "foot_contact_raw": foot_contact_raw,
        "joint_torque_sq_raw": joint_torques,
    }



def _sanitize_label(path_or_label: str) -> str:
    safe = str(path_or_label).strip().replace("/", "_")
    safe = safe.replace(" ", "_")
    return safe.replace(".pt", "")



def _run_single_checkpoint(env_cfg, experiment_cfg: dict, checkpoint_path: str, label: str, output_dir: Path):
    print(f"[COMPARE] Starting checkpoint: {label}", flush=True)
    local_env_cfg = copy.deepcopy(env_cfg)
    local_experiment_cfg = copy.deepcopy(experiment_cfg)

    local_env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else local_env_cfg.scene.num_envs
    local_env_cfg.sim.device = args_cli.device if args_cli.device is not None else local_env_cfg.sim.device

    if args_cli.seed is not None:
        local_experiment_cfg["seed"] = args_cli.seed
        local_env_cfg.seed = args_cli.seed

    local_env_cfg.command_resampling_time_s = 1.0e9
    local_env_cfg.standing_env_prob = 0.0
    local_env_cfg.command_lin_vel_x_range = (float(args_cli.cmd[0]), float(args_cli.cmd[0]))
    local_env_cfg.command_lin_vel_y_range = (float(args_cli.cmd[1]), float(args_cli.cmd[1]))
    local_env_cfg.command_ang_vel_z_range = (float(args_cli.cmd[2]), float(args_cli.cmd[2]))
    # Force a local plane terrain to avoid remote/default Omniverse USD dependencies during batch comparison.
    local_env_cfg.terrain.terrain_type = "plane"
    local_env_cfg.terrain.usd_path = None
    if args_cli.force_transmited_through_joints_reward_scale is not None:
        local_env_cfg.force_transmited_through_joints_reward_scale = float(args_cli.force_transmited_through_joints_reward_scale)

    env = gym.make(args_cli.task, cfg=local_env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    vec_env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    raw_env = env.unwrapped

    local_experiment_cfg["trainer"]["close_environment_at_exit"] = False
    local_experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    local_experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(vec_env, local_experiment_cfg)
    resume_path = os.path.abspath(checkpoint_path)
    print(f"[COMPARE] Loading checkpoint: {resume_path}", flush=True)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    obs, _ = vec_env.reset()
    _apply_fixed_command(raw_env, args_cli.cmd)

    dt = float(getattr(env, "step_dt", raw_env.step_dt))
    target_steps = max(1, int(round(float(args_cli.duration_s) / dt)))

    foot_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
    joint_names = list(raw_env.cfg.joint_names)

    times = np.zeros(target_steps, dtype=np.float64)
    commands = np.zeros((target_steps, 3), dtype=np.float32)
    base_lin_vel = np.zeros((target_steps, 3), dtype=np.float32)
    base_ang_vel = np.zeros((target_steps, 3), dtype=np.float32)
    contact_forces_xyz = np.zeros((target_steps, len(foot_names), 3), dtype=np.float32)
    contact_forces_norm = np.zeros((target_steps, len(foot_names)), dtype=np.float32)
    joint_torques = np.zeros((target_steps, len(joint_names)), dtype=np.float32)
    reward_terms = {name: np.zeros(target_steps, dtype=np.float32) for name in REWARD_TERM_ORDER}
    reward_terms["force_transmited_through_joints_raw"] = np.zeros(target_steps, dtype=np.float32)
    reward_terms["foot_contact_raw"] = np.zeros(target_steps, dtype=np.float32)
    reward_terms["joint_torque_sq_raw"] = np.zeros(target_steps, dtype=np.float32)
    reward_total = np.zeros(target_steps, dtype=np.float32)
    resets = np.zeros(target_steps, dtype=np.int32)

    print(f"[COMPARE] Running {target_steps} steps for {label} (dt={dt:.4f}s)", flush=True)
    for timestep in range(target_steps):
        _apply_fixed_command(raw_env, args_cli.cmd)
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(vec_env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in vec_env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = vec_env.step(actions)

        terms = _compute_reward_terms(raw_env)
        forces = raw_env._contact_sensor.data.net_forces_w[0, raw_env._feet_body_ids, :].detach().cpu().numpy()
        torques = raw_env._robot.data.applied_torque[0, raw_env._joint_ids].detach().cpu().numpy()
        lin_vel = raw_env._robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        ang_vel = raw_env._robot.data.root_ang_vel_b[0].detach().cpu().numpy()

        times[timestep] = (timestep + 1) * dt
        commands[timestep, :] = np.asarray(args_cli.cmd, dtype=np.float32)
        base_lin_vel[timestep, :] = lin_vel
        base_ang_vel[timestep, :] = ang_vel
        contact_forces_xyz[timestep, :, :] = forces
        contact_forces_norm[timestep, :] = np.linalg.norm(forces, axis=-1)
        joint_torques[timestep, :] = torques

        total_reward = 0.0
        for key, tensor in terms.items():
            value = float(tensor[0].item())
            reward_terms[key][timestep] = value
            if key in REWARD_TERM_ORDER:
                total_reward += value
        reward_total[timestep] = total_reward

        if timestep in {0, target_steps // 2, target_steps - 1}:
            print(
                f"[COMPARE] {label}: step {timestep + 1}/{target_steps}, "
                f"contact_sum={float(np.sum(contact_forces_norm[timestep, :])):.4f}, "
                f"reward_total={float(reward_total[timestep]):.6f}",
                flush=True,
            )

        terminated_any = bool(torch.as_tensor(terminated).any().item())
        truncated_any = bool(torch.as_tensor(truncated).any().item())
        if terminated_any or truncated_any:
            resets[timestep] = 1
            obs, _ = vec_env.reset()
            _apply_fixed_command(raw_env, args_cli.cmd)

    stem = _sanitize_label(label)
    npz_path = output_dir / f"{stem}_timeseries.npz"
    csv_path = output_dir / f"{stem}_timeseries.csv"
    summary_path = output_dir / f"{stem}_summary.csv"

    np.savez_compressed(
        npz_path,
        label=np.array(label),
        checkpoint=np.array(resume_path),
        dt=np.array(dt),
        times=times,
        commands=commands,
        foot_names=np.array(foot_names),
        joint_names=np.array(joint_names),
        contact_forces_xyz=contact_forces_xyz,
        contact_forces_norm=contact_forces_norm,
        joint_torques=joint_torques,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        reward_total=reward_total,
        resets=resets,
        **{f"reward__{k}": v for k, v in reward_terms.items()},
    )

    rows = []
    for i in range(target_steps):
        row = {
            "label": label,
            "checkpoint": resume_path,
            "step": i,
            "time_s": times[i],
            "cmd_vx": commands[i, 0],
            "cmd_vy": commands[i, 1],
            "cmd_wz": commands[i, 2],
            "base_lin_vel_x": base_lin_vel[i, 0],
            "base_lin_vel_y": base_lin_vel[i, 1],
            "base_lin_vel_z": base_lin_vel[i, 2],
            "base_ang_vel_x": base_ang_vel[i, 0],
            "base_ang_vel_y": base_ang_vel[i, 1],
            "base_ang_vel_z": base_ang_vel[i, 2],
            "contact_force_total": float(np.sum(contact_forces_norm[i, :])),
            "contact_force_FL": float(contact_forces_norm[i, 0]),
            "contact_force_FR": float(contact_forces_norm[i, 1]),
            "contact_force_RL": float(contact_forces_norm[i, 2]),
            "contact_force_RR": float(contact_forces_norm[i, 3]),
            "torque_abs_sum": float(np.sum(np.abs(joint_torques[i, :]))),
            "torque_sq_sum": float(np.sum(np.square(joint_torques[i, :]))),
            "reward_total": float(reward_total[i]),
            "reset": int(resets[i]),
        }
        for j, joint_name in enumerate(joint_names):
            row[f"torque_{joint_name}"] = float(joint_torques[i, j])
        for reward_name, reward_values in reward_terms.items():
            row[reward_name] = float(reward_values[i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary = {
        "label": label,
        "checkpoint": resume_path,
        "steps": target_steps,
        "duration_s": float(times[-1]),
        "dt": dt,
        "sum_contact_force_total": float(np.sum(np.sum(contact_forces_norm, axis=1))),
        "sum_tracking_reward": float(np.sum(reward_terms["track_lin_vel_xy_exp"])),
        "sum_energy": float(np.sum(reward_terms["dof_torques_l2"])),
        "sum_reward_total": float(np.sum(reward_total)),
        "sum_force_transmited_through_joints": float(np.sum(reward_terms["force_transmited_through_joints"])),
        "sum_force_transmited_through_joints_raw": float(np.sum(reward_terms["force_transmited_through_joints_raw"])),
        "sum_foot_contact": float(np.sum(reward_terms["foot_contact"])),
        "sum_foot_contact_raw": float(np.sum(reward_terms["foot_contact_raw"])),
        "sum_joint_torque_sq_raw": float(np.sum(reward_terms["joint_torque_sq_raw"])),
        "num_resets": int(np.sum(resets)),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"[COMPARE] Saved artifacts for {label} to {output_dir}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    try:
        vec_env.close()
    except Exception:
        pass
    del runner
    del vec_env
    del env
    gc.collect()
    sim = sim_utils.SimulationContext.instance()
    if sim is not None:
        try:
            sim.clear_all_callbacks()
        except Exception:
            pass
        try:
            sim.clear_instance()
        except Exception:
            pass
    print(f"[COMPARE] Cleaned up checkpoint: {label}", flush=True)
    return {
        "label": label,
        "checkpoint": resume_path,
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "times": times,
        "contact_forces_norm": contact_forces_norm,
        "reward_terms": reward_terms,
        "reward_total": reward_total,
        "summary": summary,
    }



def _make_plots(result_a: dict, result_b: dict, output_dir: Path):
    summary_df = pd.DataFrame([result_a["summary"], result_b["summary"]])
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)

    labels = summary_df["label"].tolist()
    contact_sums = summary_df["sum_contact_force_total"].to_numpy()
    tracking_sums = summary_df["sum_tracking_reward"].to_numpy()
    energy_sums = summary_df["sum_energy"].to_numpy()
    reward_sums = summary_df["sum_reward_total"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(contact_sums, tracking_sums, s=120)
    axes[0].set_xlabel("Sum contact forces across timesteps and feet")
    axes[0].set_ylabel("Sum tracking reward term")
    axes[0].set_title("Contact forces vs tracking reward")

    axes[1].scatter(contact_sums, energy_sums, s=120)
    axes[1].set_xlabel("Sum contact forces across timesteps and feet")
    axes[1].set_ylabel("Sum energy term (dof_torques_l2)")
    axes[1].set_title("Contact forces vs energy")

    axes[2].scatter(energy_sums, reward_sums, s=120)
    axes[2].set_xlabel("Sum energy term (dof_torques_l2)")
    axes[2].set_ylabel("Sum total reward")
    axes[2].set_title("Energy vs reward")

    for ax, xs, ys in [
        (axes[0], contact_sums, tracking_sums),
        (axes[1], contact_sums, energy_sums),
        (axes[2], energy_sums, reward_sums),
    ]:
        for label, x, y in zip(labels, xs, ys):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_scatter_metrics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    times_a = result_a["times"]
    times_b = result_b["times"]
    feet = ["FL", "FR", "RL", "RR"]
    for foot_idx, foot_name in enumerate(feet):
        axes[foot_idx].plot(times_a, result_a["contact_forces_norm"][:, foot_idx], label=result_a["label"], linewidth=1.5)
        axes[foot_idx].plot(times_b, result_b["contact_forces_norm"][:, foot_idx], label=result_b["label"], linewidth=1.5)
        axes[foot_idx].set_ylabel(f"{foot_name} |F| [N]")
        axes[foot_idx].grid(True, alpha=0.3)
        axes[foot_idx].legend(loc="upper right")

    axes[4].plot(times_a, np.sum(result_a["contact_forces_norm"], axis=1), label=result_a["label"], linewidth=1.8)
    axes[4].plot(times_b, np.sum(result_b["contact_forces_norm"], axis=1), label=result_b["label"], linewidth=1.8)
    axes[4].set_ylabel("Sum |F| all feet [N]")
    axes[4].set_xlabel("Time [s]")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="upper right")

    fig.suptitle("Contact forces over time")
    fig.tight_layout()
    fig.savefig(output_dir / "contact_forces_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    checkpoint_a = os.path.abspath(args_cli.checkpoint_a)
    checkpoint_b = os.path.abspath(args_cli.checkpoint_b)
    label_a = args_cli.label_a or Path(checkpoint_a).stem
    label_b = args_cli.label_b or Path(checkpoint_b).stem

    if args_cli.output_dir is None:
        output_dir = Path("/home/jordibelp/IsaacLab/logs/skrl/checkpoint_comparisons") / f"{_sanitize_label(label_a)}__vs__{_sanitize_label(label_b)}"
    else:
        output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_a = _run_single_checkpoint(env_cfg, experiment_cfg, checkpoint_a, label_a, output_dir)
    result_b = _run_single_checkpoint(env_cfg, experiment_cfg, checkpoint_b, label_b, output_dir)
    _make_plots(result_a, result_b, output_dir)

    print("[DONE] Comparison artifacts written to:", output_dir)
    print("[DONE] NPZ A:", result_a["npz_path"])
    print("[DONE] NPZ B:", result_b["npz_path"])
    print("[DONE] CSV A:", result_a["csv_path"])
    print("[DONE] CSV B:", result_b["csv_path"])
    print("[DONE] Summary:", output_dir / "comparison_summary.csv")
    print("[DONE] Plots:", output_dir / "comparison_scatter_metrics.png")
    print("[DONE] Plots:", output_dir / "contact_forces_over_time.png")


if __name__ == "__main__":
    main()
    simulation_app.close()
