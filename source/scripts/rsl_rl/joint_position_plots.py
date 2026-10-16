# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Low-rate, play-only Omni.UI plots of measured Solo12 joint angles."""

import math
from collections import deque


class JointPositionPlots:
    """Read env 0 only on sample ticks; never change policy or simulation state."""

    ANGLE_COLOR = 0xFFE8C44A
    SOFT_COLOR = 0xFF55B9FF
    PHYSICAL_COLOR = 0xFF6B6BFF

    def __init__(self, raw_env, interval_s=0.5):
        if not math.isfinite(interval_s) or interval_s <= 0.0:
            raise ValueError("Joint plot interval must be finite and positive.")
        self.env = raw_env
        self.interval_s = float(interval_s)
        self.next_sample_s = 0.0
        self.history = deque(maxlen=121)
        robot = raw_env._robot
        self.names = [robot.joint_names[i] for i in raw_env._joint_ids]
        self.physical_deg = [
            tuple(map(math.degrees, bounds))
            for bounds in robot.data.joint_pos_limits[0, raw_env._joint_ids].detach().cpu().tolist()
        ]
        # Use task limits, NOT IsaacLab's generic soft_joint_pos_limit_factor.
        self.soft_deg = [
            tuple(map(math.degrees, bounds))
            for bounds in raw_env._joint_soft_pos_limits[0].detach().cpu().tolist()
        ]
        self._build_window()

    def _build_window(self):
        import omni.ui as ui

        self.values = []
        self.plots = []
        self.window = ui.Window("Solo12 joint angles", width=940, height=740, visible=True)
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label(
                    f"Env 0 | measured q [deg] | {self.interval_s:g} s/sample (simulation time)",
                    height=22,
                )
                with ui.HStack(height=20):
                    ui.Label("Blue: q", style={"color": self.ANGLE_COLOR})
                    ui.Label("Orange: soft limits", style={"color": self.SOFT_COLOR})
                    ui.Label("Red: physical limits", style={"color": self.PHYSICAL_COLOR})
                self.time_label = ui.Label("Waiting for first sample", height=20)
                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=8, height=0):
                        for start in range(0, len(self.names), 3):
                            with ui.HStack(spacing=14, height=130):
                                for i in range(start, min(start + 3, len(self.names))):
                                    lo, hi = self.physical_deg[i]
                                    soft_lo, soft_hi = self.soft_deg[i]
                                    pad = 0.06 * (hi - lo)
                                    with ui.VStack(spacing=3):
                                        self.values.append(ui.Label(self.names[i], height=20))
                                        ui.Label(
                                            f"Physical: [{lo:.1f}, {hi:.1f}] deg", height=16,
                                            style={"color": self.PHYSICAL_COLOR, "font_size": 12},
                                        )
                                        ui.Label(
                                            f"Soft: [{soft_lo:.1f}, {soft_hi:.1f}] deg", height=16,
                                            style={"color": self.SOFT_COLOR, "font_size": 12},
                                        )
                                        with ui.ZStack(height=60):
                                            ui.Rectangle(style={"background_color": 0xFF242424})
                                            for bound, color in (
                                                (lo, self.PHYSICAL_COLOR), (hi, self.PHYSICAL_COLOR),
                                                (soft_lo, self.SOFT_COLOR), (soft_hi, self.SOFT_COLOR),
                                            ):
                                                ui.Plot(
                                                    ui.Type.LINE, lo - pad, hi + pad, bound, bound,
                                                    style={"color": color, "background_color": 0x00000000},
                                                )
                                            self.plots.append(ui.Plot(
                                                ui.Type.LINE, lo - pad, hi + pad,
                                                style={"color": self.ANGLE_COLOR, "background_color": 0x00000000},
                                            ))
                ui.Label("Orange/red angle = outside soft/physical limits. Hover for status.", height=18)
                ui.Label("History clears on reset. Closing this panel stops sampling.", height=18)
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def reset_history(self):
        """Do not connect an old episode to the new reset pose."""
        self.history.clear()

    def update(self, time_s):
        if not self.window.visible:
            # Avoid drawing a continuous trace across an unsampled hidden interval.
            self.history.clear()
            return False
        if time_s + 1.0e-9 < self.next_sample_s:
            return False
        # No device read or CPU transfer occurs before the sparse-rate gate above.
        q_deg = [
            math.degrees(q)
            for q in self.env._robot.data.joint_pos[0, self.env._joint_ids].detach().cpu().tolist()
        ]
        self.next_sample_s = time_s + self.interval_s
        self.history.append((time_s, q_deg))
        self.time_label.text = (
            f"t = {time_s:.1f} s | history: {self.history[0][0]:.1f} to {time_s:.1f} s "
            f"(left to right, {len(self.history)} samples)"
        )
        for i, (name, angle) in enumerate(zip(self.names, q_deg)):
            lo, hi = self.physical_deg[i]
            soft_lo, soft_hi = self.soft_deg[i]
            outside_physical = not lo <= angle <= hi
            outside_soft = not soft_lo <= angle <= soft_hi
            status = "PHYSICAL LIMIT" if outside_physical else "outside soft" if outside_soft else "inside soft"
            color = self.PHYSICAL_COLOR if outside_physical else self.SOFT_COLOR if outside_soft else self.ANGLE_COLOR
            self.values[i].text = f"{name.removesuffix('_joint')}: {angle:+.1f} deg"
            self.values[i].tooltip = f"{name}: {status}"
            self.values[i].style = {"color": color, "font_size": 13}
            self.plots[i].set_data(*(sample[i] for _, sample in self.history))
        return True

    def close(self):
        self.window.destroy()
