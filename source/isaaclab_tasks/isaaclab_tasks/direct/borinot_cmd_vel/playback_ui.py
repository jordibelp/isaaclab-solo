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
        self.visuals = None
        self._viz_models = {}
        self._models, self._subscriptions = [], []
        self._next_refresh_s = 0.0
        if show_ui:
            from .playback_visuals import FlightVisuals
            self.visuals = FlightVisuals(env)
            self._build_window(ranges)

    def _build_window(self, ranges):
        import omni.ui as ui

        self.window = ui.Window("Borinot live command control", width=570, height=800,
                                dock_preference=ui.DockPreference.RIGHT_TOP)
        with self.window.frame, ui.ScrollingFrame() as self.scroll_frame:
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
                self._build_visual_controls(ui)
                self._build_push_controls(ui)
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def _build_visual_controls(self, ui):
        from .playback_visuals import OVERLAYS

        ui.Label("VISUAL OVERLAYS / ROBOT 0 / DEFAULT OFF", height=24)
        ui.Label("Arrows show vectors, including angular-velocity axes. Zero = no arrow.", height=32, word_wrap=True)
        cbrt = self.env.cfg.control_mode in ("cbrt", "cbrt_indi")
        for key, (label, color, _, _) in OVERLAYS.items():
            supported = key != "policy_rate" or cbrt
            with ui.HStack(height=24, spacing=7):
                model = ui.SimpleBoolModel(False)
                self._viz_models[key] = model
                ui.CheckBox(model=model, width=22, enabled=supported)
                rgba = sum(int(channel * 255) << (8 * i) for i, channel in enumerate((*color, 1.0)))
                ui.Label(label + (" (CBRT only)" if not supported else ""), style={"color": rgba})
                self._subscriptions.append(model.subscribe_value_changed_fn(
                    lambda m, k=key: self.visuals.set_enabled(k, m.get_value_as_bool())))
        with ui.HStack(height=28, spacing=7):
            ui.Label("Visual size multiplier", width=180)
            self._viz_size_model = ui.SimpleFloatModel(1.0, min=0.25, max=3.0)
            ui.FloatSlider(self._viz_size_model, min=0.25, max=3.0)
            self._subscriptions.append(self._viz_size_model.subscribe_value_changed_fn(
                lambda m: setattr(self.visuals, "size", m.get_value_as_float())))
            ui.Button("All off", width=70, clicked_fn=self.disable_visuals)
        self.visual_values_label = ui.Label("", height=0, word_wrap=True)

    def disable_visuals(self):
        for model in self._viz_models.values():
            model.set_value(False)

    def _build_push_controls(self, ui):
        ui.Label("BASE PUSHES / WORLD Z / EVERY CLONE", height=26)
        ui.Label("Training-style uniform magnitude, random +/- sign and disk point.", height=24)
        initial = float(self.env._force_limits[self.env.force_level])
        maximum = max(1.0, float(self.env._force_limits.max()), initial)
        self._push_models = {}
        for key, label, value, low, high in (
                ("limit", "Max |Fz| [N]", initial, 0., maximum),
                ("radius", "Point radius [m]", 0., 0., self.env.cfg.base_push_application_radius_m),
                ("angle", "Point azimuth [deg]", 0., -180., 180.)):
            with ui.HStack(height=26, spacing=6):
                ui.Label(label, width=150)
                model = ui.SimpleFloatModel(value, min=low, max=high)
                self._push_models[key] = model
                ui.FloatField(model, width=80)
                ui.FloatSlider(model, min=low, max=high)
                self._subscriptions.append(model.subscribe_value_changed_fn(lambda m: self._configure_pushes()))
        for key, label in (("automatic", "Automatic pushes (training timing)"),
                           ("random_point", "Sample a new application point for each push")):
            with ui.HStack(height=24, spacing=7):
                model = ui.SimpleBoolModel(True)
                self._push_models[key] = model
                ui.CheckBox(model=model, width=22)
                ui.Label(label)
                self._subscriptions.append(model.subscribe_value_changed_fn(lambda m: self._configure_pushes()))
        with ui.HStack(height=30, spacing=6):
            ui.Button("Sample & apply now", clicked_fn=lambda: self.apply_push("sample"))
            ui.Button("+Fmax", clicked_fn=lambda: self.apply_push("up"))
            ui.Button("-Fmax", clicked_fn=lambda: self.apply_push("down"))
            ui.Button("Stop pushes", clicked_fn=self.stop_pushes)
        cfg = self.env.cfg
        ui.Label(f"Duration {cfg.base_push_duration_range_s} s; start interval {cfg.base_push_interval_range_s} s.\n"
                 "Uncheck random point to use radius/azimuth in body XY. Z uses the training face.\n"
                 "Changes affect the next push. Apply while paused queues it for resume.", height=58, word_wrap=True)
        self.push_status_label = ui.Label("", height=40, word_wrap=True)

    def _configure_pushes(self):
        models = self._push_models
        point = None
        if not models["random_point"].get_value_as_bool():
            radius = models["radius"].get_value_as_float()
            angle = math.radians(models["angle"].get_value_as_float())
            point = (radius * math.cos(angle), radius * math.sin(angle))
        self.env.configure_play_pushes(models["limit"].get_value_as_float(),
                                       models["automatic"].get_value_as_bool(), point)

    def apply_push(self, mode="sample"):
        self._configure_pushes()
        self.env.request_play_push(mode)

    def stop_pushes(self):
        self._push_models["automatic"].set_value(False)
        self.env.stop_play_pushes()

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
        if self.visuals is not None:
            self.visuals.update()
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
                                     f"Curriculum stage {self.env.force_level} · force limit {force:g} N")
            self.visual_values_label.text = self.visuals.readout()
            point = self.env._base_push_points_b[0].cpu().tolist()
            force_z = float(self.env._base_push_force_z_w[0])
            remaining = float(self.env._base_push_steps_left[0]) * self.env.step_dt
            self.push_status_label.text = (f"Env 0: Fz {force_z:+.2f} N · remaining {remaining:.2f} s\n"
                                           f"Point body XYZ: {point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f} m")
            self._next_refresh_s = time_s + 0.1

    def close(self):
        if self.visuals is not None:
            self.visuals.close()
        if self.window:
            self.window.destroy()
            self.window = None
        self._subscriptions.clear()
