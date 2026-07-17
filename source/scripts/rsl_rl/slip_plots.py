# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Build per-foot slip plots from a ``--visualize-slip`` physics-rate CSV log.

This is a small standalone helper shared by ``play_direct_race_0423.py`` (which saves a
non-interactive PNG with the Agg backend) and by users who want to reopen the plot
interactively to zoom into the details:

    # Interactive window (zoom/pan), default backend:
    ./isaaclab.sh -p source/scripts/rsl_rl/slip_plots.py <slip_log.csv>

    # App-style picker: select one or more CSV files, one interactive window per file:
    ./isaaclab.sh -p source/scripts/rsl_rl/slip_plots.py
    ./isaaclab.sh -p source/scripts/rsl_rl/slip_plots.py --app

    # Just write a PNG next to the CSV, no window:
    ./isaaclab.sh -p source/scripts/rsl_rl/slip_plots.py <slip_log.csv> --no-show

The figure mirrors the manual ``tmp.py`` layout: a 2x2 grid (one subplot per foot) that
concatenates every contact episode end-to-end and overlays the friction/normal cone angle
against the dynamic/static friction-cone limits plus the friction-opposed contact speed on a twin axis.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FEET = ("FL", "FR", "RL", "RR")
SPEED_COL = "tangential_speed"
DEFAULT_GAP_S = None
DEFAULT_MIN_SAMPLES = 1
DEFAULT_EPS_DEG = 0.1
POLAR_DIRECTION_COLS = ("friction_direction_footprint_x", "friction_direction_footprint_y")
POLAR_TRACK_COLOR = "#4b5563"
POLAR_TRACK_ALPHA = 0.16
DEFAULT_SLIP_LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "skrl" / "slip_logs"


def _metadata_path(csv_path: str | os.PathLike[str]) -> Path:
    return Path(csv_path).with_suffix(".plot.json")


def save_plot_metadata(
    csv_path: str,
    *,
    task: str | None,
    title_extra: str | None,
    race_scene: str | None = None,
    finish_time_s: float | None = None,
) -> str:
    """Persist plot labels beside a slip CSV so it can be reopened without CLI title arguments."""
    path = _metadata_path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task": task,
                "title_extra": title_extra,
                "race_scene": race_scene,
                "finish_time_s": finish_time_s,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _load_plot_metadata(csv_path: str) -> dict:
    path = _metadata_path(csv_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not read plot metadata {path}: {type(exc).__name__}: {exc}")
        return {}


def _resolve_plot_labels(csv_path: str, title_extra: str | None) -> tuple[str | None, str | None]:
    metadata = _load_plot_metadata(csv_path)
    return metadata.get("task"), title_extra if title_extra is not None else metadata.get("title_extra")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _initial_dir(path: str | os.PathLike[str] | None) -> str:
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            return str(candidate)
    if DEFAULT_SLIP_LOG_DIR.is_dir():
        return str(DEFAULT_SLIP_LOG_DIR)
    return os.getcwd()


def _configure_interactive_backend() -> None:
    """Prefer a GUI backend for interactive windows when this process has a display."""
    if not _has_display():
        return

    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        return
    try:
        matplotlib.use("TkAgg")
    except Exception as exc:
        print(f"[WARN] Could not switch matplotlib to TkAgg ({type(exc).__name__}: {exc}).")


def _select_csv_paths_zenity(start_dir: str) -> list[str] | None:
    """Use Ubuntu's GTK file chooser through zenity when available."""
    zenity = shutil.which("zenity")
    if not zenity:
        return None

    try:
        result = subprocess.run(
            [
                zenity,
                "--file-selection",
                "--multiple",
                "--separator=\n",
                "--title=Open slip CSV plot(s)",
                "--file-filter=CSV files | *.csv",
                f"--filename={start_dir}/",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except Exception as exc:
        print(f"[WARN] Zenity file picker failed ({type(exc).__name__}: {exc}).")
        return None

    if result.returncode == 0:
        return [path for path in result.stdout.splitlines() if path]
    if result.returncode != 1:
        err = result.stderr.strip()
        print(f"[WARN] Zenity file picker failed with exit code {result.returncode}: {err}")
    return []


def _select_csv_paths_tk(start_dir: str) -> list[str] | None:
    """Fallback file picker for systems without zenity."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()
        paths = filedialog.askopenfilenames(
            title="Open slip CSV plot(s)",
            initialdir=start_dir,
            filetypes=(("Slip CSV files", "*.csv"), ("All files", "*")),
        )
        root.destroy()
        return list(paths)
    except Exception as exc:
        print(f"[WARN] Tk file picker failed ({type(exc).__name__}: {exc}).")
        return None


def select_csv_paths(initial_dir: str | os.PathLike[str] | None = None) -> list[str]:
    """Open a GUI file picker and return selected slip CSV paths.

    Prefer Ubuntu's GTK file chooser (via ``zenity``) because it has normal list sorting,
    including modified-date sorting, unlike the minimal Tk dialog.
    """
    start_dir = _initial_dir(initial_dir)
    if not _has_display():
        print("[WARN] No DISPLAY/WAYLAND_DISPLAY found; cannot open the slip CSV file picker.")
        return []

    for picker in (_select_csv_paths_zenity, _select_csv_paths_tk):
        paths = picker(start_dir)
        if paths is not None:
            return paths

    return []


def _resolve_gap_s(df, gap_s: float | None) -> float:
    if gap_s is not None:
        return max(0.0, float(gap_s))

    dt = df["sim_time_s"].drop_duplicates().sort_values().diff().dropna()
    if dt.empty:
        return 0.0
    return max(0.0, float(dt.median()))


def build_concatenated_contacts(df, foot, gap_s: float | None = DEFAULT_GAP_S, min_samples: int = DEFAULT_MIN_SAMPLES):
    """Concatenate all contact episodes for one foot end-to-end on a single time axis.

    Returns the concatenated DataFrame (with a ``t_concat`` column) and the list of vertical
    separator positions to draw between consecutive contact episodes.
    """
    import pandas as pd

    gap_s = _resolve_gap_s(df, gap_s)
    dff = df[(df["foot"] == foot) & (df["contact"].astype(bool))].copy()
    dff = dff.sort_values(["contact_id", "sim_time_s"])

    pieces = []
    boundaries = []
    offset = 0.0

    for contact_id, ep in dff.groupby("contact_id", sort=True):
        ep = ep.sort_values("sim_time_s").copy()

        if len(ep) < min_samples:
            continue

        # Time since touchdown for this episode.
        ep["t_contact"] = ep["sim_time_s"] - ep["sim_time_s"].iloc[0]

        # Concatenated contact time.
        ep["t_concat"] = ep["t_contact"] + offset
        ep["contact_id_concat"] = contact_id

        pieces.append(ep)

        # End of current contact episode.
        boundaries.append(ep["t_concat"].iloc[-1])

        # Offset for next episode.
        offset += ep["t_contact"].iloc[-1] + gap_s

    if len(pieces) == 0:
        return pd.DataFrame(), []

    out = pd.concat(pieces, ignore_index=True)

    # Drop the last boundary because there is no next contact after it.
    return out, boundaries[:-1]


def _numeric_column(df, column: str) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(df[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _friction_axis_speed_from_columns(df, speed_col: str = SPEED_COL) -> tuple[np.ndarray, str]:
    friction_cols = {"friction_force_x", "friction_force_y", "friction_force_z"}
    velocity_cols = {"vx", "vy", "vz"}
    if friction_cols.issubset(df.columns) and velocity_cols.issubset(df.columns):
        vel = np.column_stack([_numeric_column(df, col) for col in ("vx", "vy", "vz")])
        friction = np.column_stack(
            [_numeric_column(df, col) for col in ("friction_force_x", "friction_force_y", "friction_force_z")]
        )
        friction_norm = np.linalg.norm(friction, axis=1)
        signed_speed = np.divide(
            np.sum(vel * friction, axis=1),
            friction_norm,
            out=np.zeros_like(friction_norm),
            where=friction_norm > 1.0e-9,
        )
        return np.maximum(0.0, -signed_speed), "friction-opposed slip speed"

    if speed_col in df.columns:
        return np.abs(_numeric_column(df, speed_col)), "contact-point slip speed"
    if {"vx", "vy"}.issubset(df.columns):
        return np.hypot(_numeric_column(df, "vx"), _numeric_column(df, "vy")), "contact-point XY speed"
    return np.zeros(len(df), dtype=float), "contact-point slip speed"


def build_slip_figure(
    csv_path: str,
    *,
    fig,
    feet=FEET,
    speed_col: str = SPEED_COL,
    gap_s: float | None = DEFAULT_GAP_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_deg: float = DEFAULT_EPS_DEG,
    title_extra: str | None = None,
):
    """Populate ``fig`` with the per-foot concatenated-contacts slip plot.

    ``fig`` may be a pyplot-managed figure (for interactive display) or a bare
    ``matplotlib.figure.Figure`` with an Agg canvas (for headless PNG saving).
    """
    import pandas as pd

    task, title_extra = _resolve_plot_labels(csv_path, title_extra)
    df = pd.read_csv(csv_path)
    if {"normal_force", "friction_force"}.issubset(df.columns):
        normal_force = pd.to_numeric(df["normal_force"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        friction_force = pd.to_numeric(df["friction_force"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        df = df.copy()
        df["angle_deg"] = np.degrees(np.arctan2(friction_force, normal_force))
        if "effective_mu" in df.columns:
            df["effective_mu"] = np.divide(
                friction_force,
                normal_force,
                out=np.zeros_like(friction_force),
                where=normal_force > 1.0e-6,
            )
    gap_s = _resolve_gap_s(df, gap_s)
    axes = fig.subplots(2, 2, sharex=False).flat
    shared_legend = None

    for ax, foot in zip(axes, feet):
        dff, boundaries = build_concatenated_contacts(df, foot=foot, gap_s=gap_s, min_samples=min_samples)

        if dff.empty:
            ax.set_title(f"{foot}: no contact data")
            ax.axis("off")
            continue

        # Break every plotted curve at contact-episode boundaries by inserting NaNs. Each stance is
        # a maximal run of contact==1 (filtered normal force > base_contact_threshold); the airborne
        # phase between stances is filtered out, so there is genuinely no measurement in the gap. The
        # NaN break stops the line at liftoff and restarts it at the next touchdown instead of drawing
        # a misleading straight segment across the gap. ``dff`` itself is left intact for the stats.
        gid = dff["contact_id_concat"].to_numpy()
        breaks = np.flatnonzero(np.diff(gid) != 0) + 1

        def seg(values, _breaks=breaks):
            return np.insert(np.asarray(values, dtype=float), _breaks, np.nan)

        t = dff["t_concat"].to_numpy()
        t_seg = seg(dff["t_concat"].to_numpy())

        # Left axis: friction/normal cone angle and friction-cone limits. A tiny dot marks every per-substep
        # angle sample: at full zoom they blend into the line, but zooming in separates them so it
        # is clear exactly where each measurement falls.
        ax.plot(
            t_seg,
            seg(dff["angle_deg"]),
            color="tab:blue",
            label="cone angle",
            marker=".",
            markersize=5.6,
            markerfacecolor="tab:blue",
            markeredgecolor="none",
        )
        if "policy_action_update" in dff.columns:
            policy_decision = pd.to_numeric(dff["policy_action_update"], errors="coerce").fillna(0).to_numpy() > 0
        elif "substep" in dff.columns:
            policy_decision = pd.to_numeric(dff["substep"], errors="coerce").fillna(-1).to_numpy() == 0
        else:
            policy_decision = np.zeros(len(dff), dtype=bool)
        if np.any(policy_decision):
            ax.scatter(
                t[policy_decision],
                dff["angle_deg"].to_numpy(dtype=float)[policy_decision],
                color="black",
                edgecolors="white",
                linewidths=0.35,
                s=18,
                marker="o",
                label="policy decision",
                zorder=6,
            )
        # Keep the measured cone angle segmented at contact breaks, but join the cone-limit
        # samples across transitions so friction-limit changes are readable as one continuous trace.
        ax.plot(
            t,
            dff["angle_dyn_deg"].to_numpy(dtype=float),
            color="tab:orange",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            label="angle_dyn",
        )
        ax.plot(t, dff["angle_static_deg"].to_numpy(dtype=float), color="tab:green", label="angle_static")

        ax.set_xlabel("concatenated contact time [s]")
        ax.set_ylabel("cone angle [deg]")
        ax.set_title(f"{foot}: all contacts concatenated")
        # Only vertical (time) gridlines come from the left/angle axis. The horizontal gridlines
        # are drawn on the right/velocity axis below so they mark the tangential-speed ticks.
        # Keep the grid a light, solid neutral gray so it clearly reads as background and does not
        # get confused with the contact-boundary separators below when zooming in.
        ax.grid(True, axis="x", alpha=0.4, color="0.8", linestyle="-", linewidth=0.7)

        # Vertical separators between contact episodes. Use a saturated purple dash-dot so they are
        # distinguishable from the gray solid gridlines by both color and dash pattern at any zoom.
        for i, b in enumerate(boundaries):
            ax.axvline(
                b + gap_s / 2,
                color="tab:purple",
                linestyle=(0, (5, 2, 1, 2)),  # dash-dot, distinct from the solid gridlines
                linewidth=1.1,
                alpha=0.9,
                label="contact boundary" if i == 0 else "_nolegend_",
            )

        # Right axis: slip-speed magnitude. New logs include the PhysX friction-force vector, so
        # project contact-point velocity directly onto the direction opposed by friction. This
        # makes the speed curve use the same contact basis/sign convention as the dynamic limit.
        # Older logs fall back to their stored speed column (historically ``||v_xy||``).
        ax2 = ax.twinx()
        tangential_speed, speed_label = _friction_axis_speed_from_columns(dff, speed_col=speed_col)
        ax2.plot(
            t_seg,
            seg(tangential_speed),
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            marker=".",
            markersize=5.6,
            markerfacecolor="tab:red",
            markeredgecolor="none",
            label=speed_label,
        )
        if "tangential_speed_xy" in dff.columns:
            ax2.plot(
                t_seg,
                seg(np.abs(_numeric_column(dff, "tangential_speed_xy"))),
                color="tab:gray",
                linestyle="-.",
                linewidth=0.85,
                alpha=0.5,
                label="contact-point XY speed (old)",
            )
        # Faint baseline: the raw foot-body-origin speed (pre-omega x r). Where this rides high
        # while the red contact-point curve sits near zero, the foot is rolling/pivoting over a
        # planted contact, not sliding — that is why the cone angle stays in the static band.
        if "tangential_speed_origin" in dff.columns:
            ax2.plot(
                t_seg,
                seg(np.abs(dff["tangential_speed_origin"].to_numpy())),
                color="tab:gray",
                linestyle=":",
                linewidth=0.9,
                alpha=0.6,
                label="foot-origin speed (raw)",
            )
        ax2.set_ylabel("contact speed [m/s]", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        # Fixed 0.5 m/s spacing on the velocity axis (and matching y-gridlines) instead of auto ticks.
        from matplotlib.ticker import MultipleLocator  # noqa: PLC0415

        ax2.yaxis.set_major_locator(MultipleLocator(0.5))
        # Clip the velocity axis at 5 m/s; anything faster is off the top and intentionally not shown.
        ax2.set_ylim(0.0, 5.0)
        # Horizontal gridlines align with the velocity ticks (drawn behind the data lines).
        ax2.grid(True, axis="y", alpha=0.4, color="0.8", linestyle="-", linewidth=0.7)
        ax2.set_axisbelow(True)

        # Per-foot contact-time breakdown. Each row of ``dff`` is one physics substep while the
        # foot is in contact, and the physics rate is fixed, so the fraction of rows in a region
        # equals the fraction of contact time spent there.
        angle = dff["angle_deg"].to_numpy()
        angle_dyn = dff["angle_dyn_deg"].to_numpy()
        angle_static = dff["angle_static_deg"].to_numpy()
        total = len(dff)
        # a) cone angle above the dynamic cone (by more than eps) but still within the static cone.
        in_margin = (angle > angle_dyn + eps_deg) & (angle <= angle_static)
        # b) cone angle within +/- eps of the dynamic cone, i.e. riding the slip threshold.
        slipping = (angle > angle_dyn - eps_deg) & (angle <= angle_dyn + eps_deg)
        pct_margin = 100.0 * float(in_margin.sum()) / total
        pct_slipping = 100.0 * float(slipping.sum()) / total

        if shared_legend is None:
            handles_l, labels_l = ax.get_legend_handles_labels()
            handles_r, labels_r = ax2.get_legend_handles_labels()
            shared_legend = [
                (handle, label)
                for handle, label in zip(handles_l + handles_r, labels_l + labels_r)
                if not label.startswith("_")
            ]

        ax.text(
            0.98,
            0.98,
            f"dyn+{eps_deg:g}deg < angle <= static: {pct_margin:.1f}%\n"
            f"slipping (angle in dyn +/- {eps_deg:g}deg): {pct_slipping:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
        )

    stem = os.path.basename(csv_path).rsplit(".", 1)[0]
    title = f"Per-foot: all contact episodes concatenated — contact reaction force angle - {stem}"
    if task:
        title = f"{title}\ntask: {task}"
    if title_extra:
        title = f"{title}\n{title_extra}"
    fig.suptitle(title)
    if shared_legend:
        handles, labels = zip(*shared_legend)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(labels), fontsize=9)
    has_subtitle = bool(task or title_extra)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.88 if has_subtitle else 0.94))
    return fig


def save_slip_figure(
    csv_path: str,
    png_path: str,
    *,
    gap_s: float | None = DEFAULT_GAP_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_deg: float = DEFAULT_EPS_DEG,
    title_extra: str | None = None,
    dpi: int = 150,
) -> str:
    """Render the slip figure with the Agg backend and write it to ``png_path``.

    Uses the object-oriented Figure API (no pyplot) so it is safe to call from inside a
    running Isaac Sim / Kit process without touching any interactive GUI event loop.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 10))
    FigureCanvasAgg(fig)
    build_slip_figure(
        csv_path,
        fig=fig,
        gap_s=gap_s,
        min_samples=min_samples,
        eps_deg=eps_deg,
        title_extra=title_extra,
    )

    out_dir = os.path.dirname(os.path.abspath(png_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    return png_path


def _polar_contact_data(df, feet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return footprint-frame azimuth, cone angle, and friction limits for contact rows."""
    import pandas as pd

    missing = [column for column in POLAR_DIRECTION_COLS if column not in df.columns]
    if missing:
        raise ValueError(
            "slip CSV has no base-footprint force direction; generate a new log (missing " + ", ".join(missing) + ")"
        )

    contact = df["contact"].astype(bool)
    selected = df["foot"].isin(tuple(feet)) & contact
    dff = df.loc[selected]
    x = pd.to_numeric(dff[POLAR_DIRECTION_COLS[0]], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(dff[POLAR_DIRECTION_COLS[1]], errors="coerce").to_numpy(dtype=float)
    radius = np.deg2rad(pd.to_numeric(dff["angle_deg"], errors="coerce").to_numpy(dtype=float))
    dynamic = np.deg2rad(pd.to_numeric(dff["angle_dyn_deg"], errors="coerce").to_numpy(dtype=float))
    static = np.deg2rad(pd.to_numeric(dff["angle_static_deg"], errors="coerce").to_numpy(dtype=float))
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(radius) & np.isfinite(dynamic) & np.isfinite(static)
    valid &= np.hypot(x, y) > 1.0e-9
    # Angles beyond the static friction cone are force-estimation/contact-transition outliers.
    valid &= radius <= static
    return np.arctan2(y[valid], x[valid]), radius[valid], dynamic[valid], static[valid]


def _polar_contact_tracks(df, feet) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return chronological polar tracks, split by foot and contiguous contact episode."""
    tracks = []
    for foot in feet:
        foot_df = df[(df["foot"] == foot) & df["contact"].astype(bool)].copy()
        for _, episode in foot_df.groupby("contact_id", sort=False):
            episode = episode.sort_values("sim_time_s")
            theta, radius, _, _ = _polar_contact_data(episode, (foot,))
            if theta.size >= 2:
                tracks.append((theta, radius))
    return tracks


def _contact_timing_stats(df, feet) -> dict[str, dict[str, float | int]]:
    """Compute percentage of race time in contact and mean contiguous contact duration."""
    import pandas as pd

    dt_values = df["sim_time_s"].drop_duplicates().sort_values().diff().dropna()
    sample_dt = max(0.0, float(dt_values.median())) if not dt_values.empty else 0.0
    stats: dict[str, dict[str, float | int]] = {}
    selected_frames = []
    for foot in feet:
        foot_df = df[df["foot"] == foot]
        contact_df = foot_df[foot_df["contact"].astype(bool)]
        contact_count = int(contact_df[["foot", "contact_id"]].drop_duplicates().shape[0])
        stats[str(foot)] = {
            "contact_time_pct_of_race": 100.0 * len(contact_df) / len(foot_df) if len(foot_df) else float("nan"),
            "mean_contact_duration_s": sample_dt * len(contact_df) / contact_count if contact_count else float("nan"),
            "contact_episode_count": contact_count,
        }
        selected_frames.append(foot_df)

    selected_df = pd.concat(selected_frames, ignore_index=True) if selected_frames else df.iloc[0:0]
    selected_contact = selected_df[selected_df["contact"].astype(bool)]
    selected_contact_count = int(selected_contact[["foot", "contact_id"]].drop_duplicates().shape[0])
    stats["all_feet"] = {
        "contact_time_pct_of_race": (
            100.0 * len(selected_contact) / len(selected_df) if len(selected_df) else float("nan")
        ),
        "mean_contact_duration_s": (
            sample_dt * len(selected_contact) / selected_contact_count if selected_contact_count else float("nan")
        ),
        "contact_episode_count": selected_contact_count,
    }
    return stats


def _format_contact_timing_annotation(stats: dict[str, dict[str, float | int]], selected_feet) -> str:
    labels = ["all_feet", *selected_feet] if len(selected_feet) > 1 else list(selected_feet)
    lines = ["contact time (% of race) / mean contact"]
    for label in labels:
        values = stats[str(label)]
        pct = float(values["contact_time_pct_of_race"])
        duration = float(values["mean_contact_duration_s"])
        pct_text = "n/a" if not np.isfinite(pct) else f"{pct:.1f}%"
        duration_text = "n/a" if not np.isfinite(duration) else f"{duration * 1000.0:.1f} ms"
        display_label = "all feet" if label == "all_feet" else str(label)
        lines.append(f"{display_label}: {pct_text} / {duration_text}")
    return "\n".join(lines)


def _draw_polar_density(ax, theta, radius, dynamic, static, *, tracks=(), title: str, cmap: str = "plasma"):
    """Draw neutral chronological contact tracks beneath density-colored samples."""
    from matplotlib import colormaps  # noqa: PLC0415
    from matplotlib.collections import LineCollection  # noqa: PLC0415
    from matplotlib.colors import LogNorm  # noqa: PLC0415

    ax.set_title(title, pad=42)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        ["+x forward", "45°", "+y left", "135°", "-x rear", "225°", "-y right", "315°"],
    )
    ax.grid(alpha=0.35)

    if radius.size == 0:
        ax.text(0.5, 0.5, "no contact samples", transform=ax.transAxes, ha="center", va="center")
        return None

    radial_max = max(float(np.max(radius)), float(np.max(static)))
    radial_max = max(np.deg2rad(5.0), np.ceil(np.rad2deg(radial_max) / 5.0) * np.deg2rad(5.0))
    theta_edges = np.linspace(-np.pi, np.pi, 73)
    radius_edges = np.linspace(0.0, radial_max, 46)
    counts, _, _ = np.histogram2d(theta, radius, bins=(theta_edges, radius_edges))
    theta_bin = np.clip(np.searchsorted(theta_edges, theta, side="right") - 1, 0, counts.shape[0] - 1)
    radius_bin = np.clip(np.searchsorted(radius_edges, radius, side="right") - 1, 0, counts.shape[1] - 1)
    density = counts[theta_bin, radius_bin]
    norm = LogNorm(vmin=1.0, vmax=max(1.0, float(density.max())))
    cmap = colormaps[cmap]

    full_circle = np.linspace(0.0, 2.0 * np.pi, 361)
    dynamic_median = float(np.median(dynamic))
    static_median = float(np.median(static))
    ax.plot(
        full_circle,
        np.full_like(full_circle, dynamic_median),
        color="#00b8ff",
        linewidth=2.4,
        linestyle="-",
        label=rf"angle_dynamic = {np.rad2deg(dynamic_median):.1f}°",
        zorder=1,
    )
    ax.plot(
        full_circle,
        np.full_like(full_circle, static_median),
        color="#49e670",
        linewidth=2.4,
        label=rf"angle_static = {np.rad2deg(static_median):.1f}°",
        zorder=1,
    )
    for values, color in ((dynamic, "#00b8ff"), (static, "#49e670")):
        low, high = float(np.min(values)), float(np.max(values))
        if high - low > np.deg2rad(0.05):
            ax.fill_between(full_circle, low, high, color=color, alpha=0.10, linewidth=0, zorder=0)

    # Draw each foot/contact episode independently with one faint neutral style. Lines are
    # not part of the binned sample-density heatmap; repeated paths darken naturally where
    # their transparent segments overlap.
    all_segments = []
    for track_theta, track_radius in tracks:
        points = np.column_stack((np.unwrap(track_theta), track_radius))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        all_segments.extend(segments)
    if all_segments:
        collection = LineCollection(
            all_segments,
            colors=POLAR_TRACK_COLOR,
            alpha=POLAR_TRACK_ALPHA,
            linewidths=1.25,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
        ax.add_collection(collection)

    order = np.argsort(density, kind="stable")
    scatter = ax.scatter(
        theta[order],
        radius[order],
        c=density[order],
        cmap=cmap,
        norm=norm,
        s=9,
        linewidths=0,
        alpha=0.92,
        zorder=4,
    )

    ax.set_ylim(0.0, radial_max)
    ticks_deg = np.arange(10.0, np.rad2deg(radial_max) + 0.1, 10.0)
    if ticks_deg.size:
        ax.set_yticks(np.deg2rad(ticks_deg))
        ax.set_yticklabels([f"{value:g}°" for value in ticks_deg])
    ax.set_rlabel_position(22.5)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    return scatter


def save_polar_slip_figures(
    csv_path: str,
    output_stem: str,
    *,
    feet=FEET,
    title_extra: str | None = None,
    dpi: int = 170,
    cmap: str = "plasma",
) -> list[str]:
    """Save four per-foot and one all-feet footprint-frame polar density plots."""
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    task, title_extra = _resolve_plot_labels(csv_path, title_extra)
    df = pd.read_csv(csv_path)
    contact_timing = _contact_timing_stats(df, feet)
    saved = []
    plot_groups = [((foot,), foot) for foot in feet] + [(tuple(feet), "all_feet")]
    stem = os.path.basename(csv_path).rsplit(".", 1)[0]
    for selected_feet, suffix in plot_groups:
        theta, radius, dynamic, static = _polar_contact_data(df, selected_feet)
        tracks = _polar_contact_tracks(df, selected_feet)
        fig = Figure(figsize=(13.0, 9.5), facecolor="white")
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection="polar")
        label = selected_feet[0] if len(selected_feet) == 1 else "all feet"
        scatter = _draw_polar_density(
            ax,
            theta,
            radius,
            dynamic,
            static,
            tracks=tracks,
            title=f"{label}: contact-force direction vs reaction-force angle",
            cmap=cmap,
        )
        subtitle = f"base-footprint frame; dots are contact samples, lines follow them in time — {stem}"
        if task:
            subtitle += f"\ntask: {task}"
        if title_extra:
            subtitle += f"\n{title_extra}"
        fig.text(0.5, 0.96, subtitle, ha="center", va="top", fontsize=9)
        if scatter is not None:
            colorbar = fig.colorbar(scatter, ax=ax, pad=0.12, shrink=0.75)
            colorbar.set_label("samples in local angle/direction bin (warmer = denser)")
        fig.text(
            0.82,
            0.50,
            _format_contact_timing_annotation(contact_timing, selected_feet),
            ha="left",
            va="center",
            fontsize=10,
            linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.75"},
        )
        fig.subplots_adjust(left=0.06, right=0.72, bottom=0.15, top=0.82)
        output_path = f"{output_stem}_polar_{suffix}.png"
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
        saved.append(output_path)
    return saved


def save_straight_velocity_figure(
    csv_path: str,
    output_path: str,
    *,
    title_extra: str | None = None,
    dpi: int = 170,
) -> str:
    """Plot signed base velocity along the straight start-to-end direction."""
    import pandas as pd
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    task, title_extra = _resolve_plot_labels(csv_path, title_extra)
    metadata = _load_plot_metadata(csv_path)
    finish_time_s = metadata.get("finish_time_s")
    try:
        finish_time_s = float(finish_time_s)
    except (TypeError, ValueError):
        finish_time_s = None
    if finish_time_s is not None and not np.isfinite(finish_time_s):
        finish_time_s = None
    df = pd.read_csv(csv_path, usecols=["sim_time_s", "base_velocity_straight_mps"])
    data = df.drop_duplicates("sim_time_s").sort_values("sim_time_s")
    time_s = pd.to_numeric(data["sim_time_s"], errors="coerce").to_numpy(dtype=float)
    velocity = pd.to_numeric(data["base_velocity_straight_mps"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time_s) & np.isfinite(velocity)

    fig = Figure(figsize=(12.0, 6.5), facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.plot(time_s[valid], velocity[valid], color="tab:blue", linewidth=1.4)
    ax.axhline(0.0, color="0.35", linewidth=0.8)
    if finish_time_s is not None:
        ax.axvline(finish_time_s, color="tab:green", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            0.98,
            0.95,
            f"Race finish: {finish_time_s:.3f} s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "tab:green", "alpha": 0.9},
        )
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("base velocity along start → end [m/s]")
    title = "Base velocity projected onto the straight race direction"
    if task:
        title += f"\ntask: {task}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    return output_path


def open_slip_figures(
    csv_paths,
    *,
    gap_s: float | None = DEFAULT_GAP_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_deg: float = DEFAULT_EPS_DEG,
    title_extra: str | None = None,
    save_path: str | None = None,
    show: bool = True,
) -> int:
    """Open one interactive matplotlib figure per CSV path."""
    _configure_interactive_backend()

    import matplotlib.pyplot as plt

    opened = 0
    for csv_path in csv_paths:
        csv_path = os.path.expanduser(str(csv_path))
        fig = plt.figure(figsize=(16, 10))
        try:
            build_slip_figure(
                csv_path,
                fig=fig,
                gap_s=gap_s,
                min_samples=min_samples,
                eps_deg=eps_deg,
                title_extra=title_extra,
            )
        except Exception as exc:
            plt.close(fig)
            print(f"[WARN] Could not open slip plot for {csv_path}: {type(exc).__name__}: {exc}")
            continue

        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(os.path.basename(csv_path))

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved slip plot to {save_path}")

        opened += 1

    if show and opened > 0:
        plt.show()

    return opened


def launch_slip_plot_processes(
    csv_paths,
    *,
    gap_s: float | None = DEFAULT_GAP_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_deg: float = DEFAULT_EPS_DEG,
    title_extra: str | None = None,
) -> int:
    """Launch one detached plotting process per selected CSV path."""
    launched = 0
    script_path = str(Path(__file__).resolve())
    for csv_path in csv_paths:
        csv_path = os.path.expanduser(str(csv_path))
        cmd = [
            sys.executable,
            script_path,
            csv_path,
            "--min-samples",
            str(min_samples),
            "--eps-deg",
            str(eps_deg),
        ]
        if gap_s is not None:
            cmd.extend(("--gap-s", str(gap_s)))
        if title_extra:
            cmd.extend(("--title-extra", title_extra))
        try:
            subprocess.Popen(cmd, start_new_session=True)
        except Exception as exc:
            print(f"[WARN] Could not launch slip plot for {csv_path}: {type(exc).__name__}: {exc}")
            continue
        launched += 1
    return launched


def run_picker_app(
    initial_paths,
    *,
    initial_dir: str | os.PathLike[str] | None = None,
    gap_s: float | None = DEFAULT_GAP_S,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    eps_deg: float = DEFAULT_EPS_DEG,
    title_extra: str | None = None,
) -> None:
    """Run a tiny app window with an Open plot(s) button."""
    if not _has_display():
        print("[WARN] No DISPLAY/WAYLAND_DISPLAY found; cannot open the Slip Plot Opener app.")
        return

    _configure_interactive_backend()

    import tkinter as tk
    from tkinter import messagebox

    start_dir = _initial_dir(initial_dir)
    root = tk.Tk()
    root.title("Slip Plot Opener")
    root.geometry("360x140")
    root.minsize(360, 140)

    status = tk.StringVar(value=f"Default folder: {start_dir}")

    def open_selected(paths) -> None:
        paths = list(paths)
        if not paths:
            return
        launched = launch_slip_plot_processes(
            paths,
            gap_s=gap_s,
            min_samples=min_samples,
            eps_deg=eps_deg,
            title_extra=title_extra,
        )
        if launched > 0:
            status.set(f"Opening {launched} plot window(s).")
        else:
            status.set("No plots opened.")
            messagebox.showwarning("Slip Plot Opener", "No plot processes could be launched.")

    def choose_files() -> None:
        paths = select_csv_paths(start_dir)
        open_selected(paths)

    frame = tk.Frame(root, padx=16, pady=14)
    frame.pack(fill="both", expand=True)
    button = tk.Button(frame, text="Open plot(s)", command=choose_files, height=2)
    button.pack(fill="x")
    label = tk.Label(frame, textvariable=status, wraplength=320, justify="left", anchor="w")
    label.pack(fill="x", pady=(12, 0))

    if initial_paths:
        root.after(100, lambda: open_selected(initial_paths))

    root.mainloop()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot per-foot slip data from a --visualize-slip CSV log.")
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help=(
            "Path(s) to slip-log CSVs produced by --visualize-slip. If omitted, a multi-select file picker opens."
        ),
    )
    parser.add_argument(
        "--app",
        action="store_true",
        help="Open a small file-picker app with an 'Open plot(s)' button for repeated use.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Save the figure to this PNG path. Defaults to the CSV path with a .png suffix when --no-show is set.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; just write the PNG.",
    )
    parser.add_argument(
        "--initial-dir",
        default=None,
        help=f"Initial folder for the file picker. Defaults to {DEFAULT_SLIP_LOG_DIR}.",
    )
    parser.add_argument(
        "--gap-s",
        type=float,
        default=DEFAULT_GAP_S,
        help=(
            "Visual gap between contact episodes. Defaults to one CSV sample period; use 0 to remove artificial gap."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Ignore contact episodes with fewer than this many samples.",
    )
    parser.add_argument(
        "--eps-deg",
        type=float,
        default=DEFAULT_EPS_DEG,
        help=(
            "Half-width [deg] of the slip band around the dynamic cone. The legend reports the %% of"
            " contact time with (dyn+eps < angle <= static) and with (dyn-eps < angle <= dyn+eps)."
        ),
    )
    parser.add_argument(
        "--title-extra",
        default=None,
        help="Optional extra line appended to the figure title, for example friction settings.",
    )
    args = parser.parse_args(argv)

    if args.app:
        if args.no_show:
            parser.error("--app cannot be combined with --no-show.")
        if args.save:
            parser.error("--app cannot be combined with --save.")
        run_picker_app(
            args.csv_paths,
            initial_dir=args.initial_dir,
            gap_s=args.gap_s,
            min_samples=args.min_samples,
            eps_deg=args.eps_deg,
            title_extra=args.title_extra,
        )
        return

    csv_paths = list(args.csv_paths)
    selected_from_picker = False
    if not csv_paths:
        csv_paths = select_csv_paths(args.initial_dir)
        selected_from_picker = bool(csv_paths)
    if not csv_paths:
        print("[INFO] No slip CSV selected.")
        return

    if args.save and len(csv_paths) != 1:
        parser.error("--save can only be used with exactly one CSV path.")

    if args.no_show:
        for csv_path in csv_paths:
            png_path = args.save or (os.path.splitext(csv_path)[0] + ".png")
            save_slip_figure(
                csv_path,
                png_path,
                gap_s=args.gap_s,
                min_samples=args.min_samples,
                eps_deg=args.eps_deg,
                title_extra=args.title_extra,
            )
            print(f"[INFO] Saved slip plot to {png_path}")
        return

    if selected_from_picker and not args.save:
        launched = launch_slip_plot_processes(
            csv_paths,
            gap_s=args.gap_s,
            min_samples=args.min_samples,
            eps_deg=args.eps_deg,
            title_extra=args.title_extra,
        )
        print(f"[INFO] Opening {launched} slip plot window(s).")
        return

    open_slip_figures(
        csv_paths,
        gap_s=args.gap_s,
        min_samples=args.min_samples,
        eps_deg=args.eps_deg,
        title_extra=args.title_extra,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
