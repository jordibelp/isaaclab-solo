"""Physics-rate H1a measurements and an explicit (not compiler-validated) Q11 wrap model."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

Q11_SCALE = 2048
Q11_MIN = -32768 / Q11_SCALE
Q11_MAX = 32767 / Q11_SCALE


def wrap_q11(acceleration: np.ndarray) -> np.ndarray:
    """Truncate toward zero, retain the low 16 bits, then decode as signed Q11.

    Do not use NumPy's out-of-range float-to-int16 cast: its behavior is not
    the hypothesis we want to specify. Real firmware casts remain unverified.
    """
    counts = np.trunc(np.asarray(acceleration) * Q11_SCALE)
    return ((counts + 32768) % 65536 - 32768) / Q11_SCALE


def q11_overflow(acceleration: np.ndarray) -> np.ndarray:
    counts = np.trunc(np.asarray(acceleration) * Q11_SCALE)
    return (counts < -32768) | (counts > 32767)


class ImuImpactRecorder:
    """Stream every physics sample to disk; diagnostics never alter simulator state."""

    def __init__(self, output_dir: Path, physics_dt: float, *, ill_wrapping_test: bool):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.dt = physics_dt
        self.wrap_enabled = ill_wrapping_test
        self.samples = 0
        self.episode_id = 0
        self.episode_samples = 0
        self.overflow_samples = 0
        self.overflow_without_base_contact = 0
        self.overflow_on_foot_contact = 0
        self.overflow_axes = np.zeros(3, dtype=int)
        self.max_abs_body = np.zeros(3)
        self.min_world_z = None
        self.max_world_z = None
        self.first_overflow = None
        self.episodes = []
        self.paths = []
        self.stream = (output_dir / "imu_acceleration.csv").open("w", newline="", encoding="utf-8")
        vector_fields = [f"{kind}_{axis}_mps2" for kind in
                         ("raw_world", "raw_body", "clipped_body", "wrapped_body", "delivered_body",
                          "clipped_world", "wrapped_world", "delivered_world") for axis in "xyz"]
        self.fields = ["time_s", "episode_id", "episode_time_s", *vector_fields,
                       *[f"overflow_body_{axis}" for axis in "xyz"],
                       "foot_ground_contacts", "foot_ground_normal_force_n", "base_ground_contact",
                       "base_height_m", "gravity_x_b"]
        self.writer = csv.writer(self.stream)
        self.writer.writerow(self.fields)

    def reset(self, reason: str) -> None:
        self.episodes.append({"episode_id": self.episode_id, "samples": self.episode_samples,
                              "end_time_s": self.samples * self.dt, "end_reason": reason})
        self.episode_id += 1
        self.episode_samples = 0

    def record(self, sim, rotation, raw_world, raw_body, clipped_body, wrapped_body, delivered_body) -> None:
        import mujoco

        self.samples += 1
        self.episode_samples += 1
        time_s = self.samples * self.dt
        overflow = q11_overflow(raw_body)
        foot_contacts, foot_force = 0, 0.
        # mj_step's contacts/forces describe the completed solve, not a new mj_forward.
        for index, contact in enumerate(sim.data.contact):
            if sim.ground_id not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == sim.ground_id else contact.geom1
            if sim.model.geom_bodyid[other] in sim.calf_body_ids:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(sim.model, sim.data, index, wrench)
                if wrench[0] > 0.:
                    foot_contacts += 1
                    foot_force += wrench[0]
        base_contact = sim._base_hit_ground_now()
        self.writer.writerow((time_s, self.episode_id, self.episode_samples * self.dt,
                              *raw_world, *raw_body, *clipped_body, *wrapped_body, *delivered_body,
                              *(rotation @ clipped_body), *(rotation @ wrapped_body), *(rotation @ delivered_body),
                              *overflow.astype(int), foot_contacts, foot_force, int(base_contact),
                              sim.base_height(), sim.gravity_x_b()))
        self.max_abs_body = np.maximum(self.max_abs_body, np.abs(raw_body))
        self.min_world_z = float(raw_world[2]) if self.min_world_z is None else min(self.min_world_z, float(raw_world[2]))
        self.max_world_z = float(raw_world[2]) if self.max_world_z is None else max(self.max_world_z, float(raw_world[2]))
        self.overflow_axes += overflow
        if overflow.any():
            self.overflow_samples += 1
            self.overflow_without_base_contact += int(not base_contact)
            self.overflow_on_foot_contact += int(foot_contacts > 0)
            if self.first_overflow is None:
                self.first_overflow = time_s
                print(f"[IMU] First Q11 overflow at t={time_s:.3f}s: body={raw_body} m/s²; "
                      f"world_z={raw_world[2]:.3f}; foot contacts={foot_contacts}; base contact={base_contact}", flush=True)
        if self.samples % round(5 / self.dt) == 0:
            self.stream.flush()
            print(f"[IMU] t={time_s:g}s: peak |body XYZ|={self.max_abs_body} m/s²; "
                  f"world Z=[{self.min_world_z:.2f}, {self.max_world_z:.2f}]; "
                  f"Q11 overflow={self.overflow_samples}/{self.samples} samples", flush=True)

    def close(self) -> list[Path]:
        if self.stream.closed:
            return self.paths
        self.stream.close()
        summary = {
            "samples": self.samples, "physics_dt_s": self.dt, "ill_wrapping_test": self.wrap_enabled,
            "q11_min_mps2": Q11_MIN, "q11_max_mps2": Q11_MAX,
            "first_overflow_time_s": self.first_overflow, "overflow_samples": self.overflow_samples,
            "overflow_fraction": self.overflow_samples / self.samples if self.samples else 0.,
            "overflow_samples_without_base_contact": self.overflow_without_base_contact,
            "overflow_samples_on_foot_ground_contact": self.overflow_on_foot_contact,
            "raw_world_z_min_mps2": self.min_world_z, "raw_world_z_max_mps2": self.max_world_z,
            "completed_episodes": self.episodes,
            "final_episode": {"episode_id": self.episode_id, "samples": self.episode_samples,
                              "end_reason": "run_end"},
        }
        for index, axis in enumerate("xyz"):
            summary[f"overflow_body_{axis}_samples"] = int(self.overflow_axes[index])
            summary[f"raw_body_{axis}_abs_max_mps2"] = float(self.max_abs_body[index])
        summary_path = self.output_dir / "imu_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.paths = [self.output_dir / "imu_config.json", self.output_dir / "imu_acceleration.csv", summary_path]
        self.paths = [path for path in self.paths if path.exists()]
        if self.samples:
            self.paths.append(self._plot())
        print(f"[IMU] Saved {self.samples} physics samples; Q11 overflow={self.overflow_samples}; "
              f"output={self.output_dir}", flush=True)
        return self.paths

    def _plot(self) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Full-rate numeric CSV is authoritative; do not average away impact peaks.
        data = np.genfromtxt(self.output_dir / "imu_acceleration.csv", delimiter=",", names=True, ndmin=1)
        time_s = data["time_s"]
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 9), constrained_layout=True)
        for kind, label in (("raw", "clean, unclipped"), ("clipped", "normal clip (counterfactual)"),
                            ("wrapped", "Q11 wrap (counterfactual)"), ("delivered", "delivered")):
            axes[0].plot(time_s, data[f"{kind}_world_z_mps2"], label=label, lw=.8)
        axes[0].set_ylabel("IMU point, world Z\nlinear acceleration [m/s²]")
        axes[0].legend(loc="upper right", ncol=2)
        for axis in "xyz":
            axes[1].plot(time_s, data[f"raw_body_{axis}_mps2"], label=f"body {axis.upper()}", lw=.7)
        for limit in (Q11_MIN, Q11_MAX):
            axes[1].axhline(limit, color="black", ls="--", lw=.8)
        axes[1].set_ylabel("Unclipped body XYZ\nlinear acceleration [m/s²]")
        axes[1].legend(loc="upper right")
        axes[2].plot(time_s, data["foot_ground_normal_force_n"], label="feet–ground normal force [N]")
        overflow = sum(data[f"overflow_body_{axis}"] for axis in "xyz") > 0
        axes[2].scatter(time_s[overflow], data["foot_ground_normal_force_n"][overflow],
                        s=5, color="red", label="any-axis Q11 overflow")
        base = data["base_ground_contact"] > 0
        axes[2].scatter(time_s[base], data["foot_ground_normal_force_n"][base],
                        s=7, color="black", marker="x", label="base–ground contact")
        axes[2].set_ylabel("Normal force [N]")
        axes[2].set_xlabel("Elapsed simulation time [s]; continuous across resets")
        axes[2].legend(loc="upper right")
        for ax in axes:
            for episode in self.episodes:
                ax.axvline(episode["end_time_s"], color="grey", alpha=.5, ls=":")
            ax.grid(alpha=.2)
        fig.suptitle(f"H1a IMU probe — {'Q11 wrapping' if self.wrap_enabled else 'normal observation path'}; "
                     "counterfactuals use this run's trajectory")
        path = self.output_dir / "imu_impacts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


def log_imu_csv(run, path: Path) -> int:
    """Keep all numeric samples in W&B, with named axes and their own time coordinate."""
    run.define_metric("imu/time_s")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for field in reader.fieldnames:
            if field != "time_s":
                run.define_metric(f"imu/{field}", step_metric="imu/time_s")
        count = 0
        for row in reader:
            run.log({f"imu/{key}": float(value) for key, value in row.items()})
            count += 1
    return count
