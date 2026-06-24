#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Create comparison plots from two saved Solo12 checkpoint recordings.

Example:
python source/scripts/skrl/plot_solo12_checkpoint_comparison.py \
    --input_a logs/skrl/checkpoint_recordings/0326_vduw2o5j_best_agent/0326_vduw2o5j_best_agent_timeseries.npz \
    --input_b logs/skrl/checkpoint_recordings/0326_sc1v0hs5_overnight_best_agent/0326_sc1v0hs5_overnight_best_agent_timeseries.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Plot Solo12 checkpoint comparison from saved npz recordings.")
parser.add_argument("--input_a", type=str, required=True, help="NPZ file for checkpoint A.")
parser.add_argument("--input_b", type=str, required=True, help="NPZ file for checkpoint B.")
parser.add_argument("--label_a", type=str, default=None, help="Optional override label for checkpoint A.")
parser.add_argument("--label_b", type=str, default=None, help="Optional override label for checkpoint B.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots and combined summaries.")
args = parser.parse_args()


def _sanitize_label(path_or_label: str) -> str:
    safe = str(path_or_label).strip().replace("/", "_")
    safe = safe.replace(" ", "_")
    return safe.replace(".pt", "").replace(".npz", "")



def _load_recording(path: Path, label_override: str | None = None):
    data = np.load(path, allow_pickle=True)
    label = label_override or str(data["label"].item())
    checkpoint = str(data["checkpoint"].item())
    reward_terms = {}
    for key in data.files:
        if key.startswith("reward__"):
            reward_terms[key[len("reward__") :]] = data[key]
    summary = {
        "label": label,
        "checkpoint": checkpoint,
        "steps": int(len(data["times"])),
        "duration_s": float(data["times"][-1]),
        "dt": float(data["dt"].item()),
        "sum_contact_force_total": float(np.sum(np.sum(data["contact_forces_norm"], axis=1))),
        "sum_tracking_reward": float(np.sum(reward_terms["track_lin_vel_xy_exp"])),
        "sum_energy": float(np.sum(np.abs(reward_terms["dof_torques_l2"]))),
        "sum_reward_total": float(np.sum(data["reward_total"])),
        "sum_force_transmited_through_joints": float(np.sum(reward_terms["force_transmited_through_joints"])),
        "sum_force_transmited_through_joints_raw": float(np.sum(reward_terms["force_transmited_through_joints_raw"])),
        "sum_foot_contact": float(np.sum(reward_terms["foot_contact"])),
        "sum_foot_contact_raw": float(np.sum(reward_terms["foot_contact_raw"])),
        "sum_joint_torque_sq_raw": float(np.sum(reward_terms["joint_torque_sq_raw"])),
        "num_resets": int(np.sum(data["resets"])),
    }
    return {
        "label": label,
        "checkpoint": checkpoint,
        "times": data["times"],
        "contact_forces_norm": data["contact_forces_norm"],
        "contact_forces_xyz": data["contact_forces_xyz"],
        "joint_torques": data["joint_torques"],
        "reward_terms": reward_terms,
        "reward_total": data["reward_total"],
        "foot_names": data["foot_names"],
        "joint_names": data["joint_names"],
        "summary": summary,
    }



def _write_combined_timeseries_csv(result_a: dict, result_b: dict, output_dir: Path):
    rows = []
    for result in [result_a, result_b]:
        n = len(result["times"])
        for i in range(n):
            rows.append(
                {
                    "label": result["label"],
                    "checkpoint": result["checkpoint"],
                    "step": i,
                    "time_s": float(result["times"][i]),
                    "contact_force_total": float(np.sum(result["contact_forces_norm"][i, :])),
                    "contact_force_FL": float(result["contact_forces_norm"][i, 0]),
                    "contact_force_FR": float(result["contact_forces_norm"][i, 1]),
                    "contact_force_RL": float(result["contact_forces_norm"][i, 2]),
                    "contact_force_RR": float(result["contact_forces_norm"][i, 3]),
                    "reward_total": float(result["reward_total"][i]),
                    "tracking_reward": float(result["reward_terms"]["track_lin_vel_xy_exp"][i]),
                    "energy": float(abs(result["reward_terms"]["dof_torques_l2"][i])),
                    "force_transmited_through_joints": float(result["reward_terms"]["force_transmited_through_joints"][i]),
                    "force_transmited_through_joints_raw": float(result["reward_terms"]["force_transmited_through_joints_raw"][i]),
                    "foot_contact": float(result["reward_terms"]["foot_contact"][i]),
                    "foot_contact_raw": float(result["reward_terms"]["foot_contact_raw"][i]),
                    "joint_torque_sq_raw": float(result["reward_terms"]["joint_torque_sq_raw"][i]),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "comparison_timeseries.csv", index=False)



def _make_plots(result_a: dict, result_b: dict, output_dir: Path):
    summary_df = pd.DataFrame([result_a["summary"], result_b["summary"]])
    summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
    _write_combined_timeseries_csv(result_a, result_b, output_dir)

    labels = summary_df["label"].tolist()
    contact_sums = summary_df["sum_contact_force_total"].to_numpy()
    tracking_sums = summary_df["sum_tracking_reward"].to_numpy()
    energy_sums = summary_df["sum_energy"].to_numpy()
    reward_sums = summary_df["sum_tracking_reward"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(contact_sums, tracking_sums, s=120)
    axes[0].set_xlabel("Sum contact forces across timesteps and feet")
    axes[0].set_ylabel("Sum tracking reward term")
    axes[0].set_title("Contact forces vs tracking reward")

    axes[1].scatter(contact_sums, energy_sums, s=120)
    axes[1].set_xlabel("Sum contact forces across timesteps and feet")
    axes[1].set_ylabel("Sum abs energy term (|dof_torques_l2|)")
    axes[1].set_title("Contact forces vs abs energy")

    axes[2].scatter(energy_sums, reward_sums, s=120)
    axes[2].set_xlabel("Sum abs energy term (|dof_torques_l2|)")
    axes[2].set_ylabel("Sum tracking reward term")
    axes[2].set_title("Energy vs tracking reward")

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


input_a = Path(args.input_a).resolve()
input_b = Path(args.input_b).resolve()
result_a = _load_recording(input_a, args.label_a)
result_b = _load_recording(input_b, args.label_b)

if args.output_dir is None:
    output_dir = input_a.parent.parent / f"{_sanitize_label(result_a['label'])}__vs__{_sanitize_label(result_b['label'])}"
else:
    output_dir = Path(args.output_dir).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

_make_plots(result_a, result_b, output_dir)

print("[DONE] Comparison artifacts written to:", output_dir)
print("[DONE] Summary:", output_dir / "comparison_summary.csv")
print("[DONE] Timeseries:", output_dir / "comparison_timeseries.csv")
print("[DONE] Plots:", output_dir / "comparison_scatter_metrics.png")
print("[DONE] Plots:", output_dir / "contact_forces_over_time.png")
