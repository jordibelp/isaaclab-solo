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
DEFAULT_EPS_DEG = 1.0
DEFAULT_SLIP_LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "skrl" / "slip_logs"


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
    if title_extra:
        title = f"{title}\n{title_extra}"
    fig.suptitle(title)
    if shared_legend:
        handles, labels = zip(*shared_legend)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(labels), fontsize=9)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.91 if title_extra else 0.94))
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
