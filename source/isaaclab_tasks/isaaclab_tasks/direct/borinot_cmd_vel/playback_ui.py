# SPDX-License-Identifier: BSD-3-Clause
"""Live XYZ/yaw-rate controls and flight cameras, used only by the public player."""
import math

from .task_math import heading_rotation

CAMERAS = ("follow", "chase", "side", "top", "free")
RANGE_FIELDS = ("command_lin_vel_x_range", "command_lin_vel_y_range",
                "command_lin_vel_z_range", "command_ang_vel_z_range")


def camera_pose(mode, position, yaw=0.0):
    """World-aligned follow/side/top; yaw-aligned chase. Free leaves mouse control."""
    if mode == "free":
        return None
    offset = {"follow": (2.4, -2.8, 1.1), "chase": (-2.8, 0.0, 1.0),
              "side": (0.0, -3.5, 0.6), "top": (0.0, -0.01, 4.5)}[mode]
    if mode == "chase":
        c, s = math.cos(yaw), math.sin(yaw)
        offset = (c * offset[0] - s * offset[1], s * offset[0] + c * offset[1], offset[2])
    return tuple(p + d for p, d in zip(position, offset)), tuple(position)


class FlightControls:
    """UI callbacks queue intent; command/reset changes are applied before inference."""

    def __init__(self, env, command, ranges, camera="follow", show_ui=True):
        self.env, self.command, self.camera = env, list(command), camera
        self.paused = self.reset_requested = False
        self.manual_resets = 0
        self.window = None
        self._models, self._subscriptions = [], []
        self._next_refresh_s = 0.0
        if show_ui:
            self._build_window(ranges)

    def _build_window(self, ranges):
        import omni.ui as ui

        self.window = ui.Window("Borinot live command control", width=490, height=490,
                                dock_preference=ui.DockPreference.RIGHT_TOP)
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("BORINOT / VELOCITY COMMAND", height=28, style={"font_size": 20})
                ui.Label("vx, vy: heading axes · vz: world up · wz: world Z", height=22)
                for index, (label, bounds) in enumerate(zip(
                        ("vx [m/s]", "vy [m/s]", "vz [m/s]", "wz [rad/s]"), ranges)):
                    low, high = bounds
                    with ui.HStack(spacing=8, height=30):
                        ui.Label(label, width=100)
                        model = ui.SimpleFloatModel(self.command[index], min=low, max=high)
                        ui.FloatField(model, width=105)
                        ui.FloatSlider(model, min=low, max=high)
                        self._models.append(model)
                        self._subscriptions.append(model.subscribe_value_changed_fn(
                            lambda m, i=index: self.set_component(i, m.get_value_as_float())))
                ui.Label("Click a number to enter an exact value. Applies to every clone.", height=22)
                with ui.HStack(height=32):
                    ui.Button("Zero commands", clicked_fn=self.zero_commands)
                    ui.Button("Reset robot", clicked_fn=self.request_reset)
                    self.pause_button = ui.Button("Pause", clicked_fn=self.toggle_pause)
                ui.Label("CAMERA", height=22)
                with ui.HStack(height=30):
                    for mode in CAMERAS:
                        ui.Button(mode.title(), clicked_fn=lambda m=mode: self.set_camera(m))
                self.measured_label = ui.Label("", height=55, word_wrap=True)
                self.status_label = ui.Label("", height=55, word_wrap=True)
                ui.Label("Zero vz requests zero vertical speed, not altitude hold.\n"
                         "Flight bounds and crash resets remain enabled.", height=42, word_wrap=True)
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def set_component(self, index, value):
        if math.isfinite(value):
            self.command[index] = float(value)

    def zero_commands(self):
        self.command[:] = [0.0] * 4
        for model in self._models:
            model.set_value(0.0)

    def request_reset(self):
        self.reset_requested = True

    def toggle_pause(self):
        self.paused = not self.paused
        if self.window:
            self.pause_button.text = "Resume" if self.paused else "Pause"

    def set_camera(self, camera):
        self.camera = camera

    def update(self, time_s):
        data = self.env.robot.data
        # ViewportCameraController's "env" origin expects clone-local coordinates.
        controller = self.env.viewport_camera_controller
        if controller is not None and self.camera != "free":
            position = (data.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().tolist()
            yaw = 0.0
            if self.camera == "chase":
                w, x, y, z = data.root_quat_w[0].cpu().tolist()
                yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            eye, lookat = camera_pose(self.camera, position, yaw)
            controller.update_view_location(eye=eye, lookat=lookat)
        if self.window and (self.paused or time_s >= self._next_refresh_s):
            rotation = heading_rotation(data.root_quat_w[:1])[0]
            velocity = (rotation.T @ data.root_lin_vel_w[0]).cpu().tolist()
            wz = float(data.root_ang_vel_w[0, 2])
            height = float(data.root_pos_w[0, 2] - self.env.scene.env_origins[0, 2])
            self.measured_label.text = (f"Measured env 0: vx {velocity[0]:+.2f}, vy {velocity[1]:+.2f}, "
                                       f"vz {velocity[2]:+.2f} m/s\nwz {wz:+.2f} rad/s · height {height:.2f} m")
            force = float(self.env._force_limits[self.env.force_level])
            self.status_label.text = (f"{'PAUSED' if self.paused else 'PLAYING'} · {time_s:.1f} s · "
                                     f"{self.camera} · manual resets {self.manual_resets}\n"
                                     f"Force stage {self.env.force_level} · limit {force:g} N")
            self._next_refresh_s = time_s + 0.1

    def close(self):
        if self.window:
            self.window.destroy()
            self.window = None
        self._subscriptions.clear()
