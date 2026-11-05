# SPDX-License-Identifier: BSD-3-Clause
"""Live goal/deadline controls and flight cameras, used only by the public go-to player."""
import math

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.direct.borinot_cmd_vel.playback_ui import camera_pose

from .task_math import training_deadline_interval

CAMERAS = ("goal", "follow", "chase", "side", "top", "free")
HORIZONTAL_FOV_DEG = 60.0  # Kit default perspective camera: 18.15 mm focal length, 20.955 mm aperture
REST_SPEED_M_S = 0.25  # training goals always start from rest


def goal_camera_pose(start, goal):
    """Static side view of one flight: frame the start and the goal with a margin."""
    delta = [g - s for g, s in zip(goal, start)]
    horizontal = math.hypot(delta[0], delta[1])
    side = (0.0, -1.0)  # vertical flight: same side as the "side" view
    if horizontal > 0.05:
        side = (delta[1] / horizontal, -delta[0] / horizontal)  # perpendicular to the flight line
        if side[1] > 0 or (side[1] == 0 and side[0] < 0):
            side = (-side[0], -side[1])  # prefer the -Y side, like the "side" view
    half_span = math.sqrt(sum(d * d for d in delta)) / 2
    distance = max(4.0, 1.2 * (half_span + 1.0) / math.tan(math.radians(HORIZONTAL_FOV_DEG / 2)))
    lookat = tuple((s + g) / 2 for s, g in zip(start, goal))
    eye = (lookat[0] + distance * side[0], lookat[1] + distance * side[1], lookat[2] + 0.35 * distance)
    return eye, lookat


def training_region_notes(cfg, difficulty, offset_b, goal_z, speed, deadline_s, deadline_range, step_dt):
    """Plain-language reasons why a goal leaves what this stage sampled. Empty: inside."""
    notes = []
    extent = cfg.goal_half_extents_m[difficulty]
    for axis, value, limit in zip("xyz", offset_b, extent):
        if abs(value) > limit:
            notes.append(f"body {axis} offset {value:+.2f} m is beyond the training +/-{limit:g} m")
    if cfg.include_base_height:  # the policy observes absolute height
        low = cfg.robot_home[2] - cfg.reset_z_noise_m - extent[2]
        high = cfg.robot_home[2] + cfg.reset_z_noise_m + extent[2]
        if not low <= goal_z <= high:
            notes.append(f"goal height {goal_z:.2f} m is outside the training heights {low:.2f}-{high:.2f} m")
    if speed > REST_SPEED_M_S:
        notes.append(f"robot moves at {speed:.2f} m/s; training goals start from rest")
    low, high = deadline_range
    if not low - 1e-6 <= deadline_s <= high + step_dt:  # training rounds deadlines up to one tick
        notes.append(f"deadline {deadline_s:.2f} s is outside the training {low:.2f}-{high:.2f} s for this distance")
    return notes


def describe_result(result, number):
    kind = {"goal": "Goal", "sampled": "Sampled goal", "hold": "Hold"}[result["kind"]]
    goal = ", ".join(f"{v:+.2f}" for v in result["goal_m"])
    if result["failed"]:
        outcome = f"FAILED after {result['elapsed_s']:.2f} s (safety reset)"
    else:
        outcome = "SUCCESS" if result["success"] else "MISSED"
    arrival = "never inside" if result["first_arrival_s"] is None else f"first inside {result['first_arrival_s']:.2f} s"
    return (f"{kind} #{number} ({goal}) m, deadline {result['deadline_s']:.2f} s: {outcome}. "
            f"Final distance {result['distance_m']:.3f} m, speed {result['speed_m_s']:.2f} m/s, "
            f"closest {result['min_distance_m']:.3f} m, {arrival}.")


def flight_state(env, time_s):
    """Read robot 0 without changing physics, commands or clocks. XYZ uses world axes."""
    data = env.robot.data
    position = data.root_pos_w[0] - env.scene.env_origins[0]
    goal = env._goal_position_w[0] - env.scene.env_origins[0]
    velocity = data.root_lin_vel_w[0]
    error = goal - position
    w, x, y, z = data.root_quat_w[0].tolist()
    return {
        "time_s": float(time_s), "position_m": position.tolist(), "velocity_m_s": velocity.tolist(),
        "goal_m": goal.tolist(), "error_m": error.tolist(), "distance_m": float(error.norm()),
        "speed_m_s": float(velocity.norm()),
        "yaw_deg": math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))),
        "remaining_s": float((env._goal_deadline_steps[0] - env._episode_steps[0]).clamp_min(0)) * env.step_dt,
        "phase": "HOLDING" if env._segment_hold[0] else "FLYING"}


def xyz_text(values):
    return "(" + ", ".join(f"{v:+.3f}" for v in values) + ")"


def describe_state(state):
    return (f"[STATE] t={state['time_s']:.2f} s robot 0 {state['phase']}: "
            f"position XYZ={xyz_text(state['position_m'])} m, "
            f"velocity XYZ={xyz_text(state['velocity_m_s'])} m/s, "
            f"goal XYZ={xyz_text(state['goal_m'])} m, error XYZ={xyz_text(state['error_m'])} m, "
            f"distance={state['distance_m']:.3f} m, speed={state['speed_m_s']:.3f} m/s, "
            f"yaw={state['yaw_deg']:+.1f} deg, left={state['remaining_s']:.2f} s")


class GoalControls:
    """UI callbacks queue intent; goal/reset changes are applied before inference."""

    def __init__(self, env, goal, deadline_s, ranges, camera="goal", show_ui=True, relative=False):
        self.env, self.camera = env, camera
        self.goal = list(goal)
        self.relative = bool(relative)
        self.auto_deadline = deadline_s is None
        self.deadline_s = 2.0 if deadline_s is None else float(deadline_s)
        self.go_requested = self.reset_requested = self.paused = False
        self.manual_resets = 0
        self.message = ""
        self.window = self._preview = self._preview_key = None
        self._models, self._subscriptions = [], []
        self._next_refresh_s = 0.0
        if show_ui:
            self._build_window(ranges)

    def _build_window(self, ranges):
        import omni.ui as ui

        self.window = ui.Window("Borinot live goal control", width=570, height=820,
                                dock_preference=ui.DockPreference.RIGHT_TOP)
        with self.window.frame, ui.ScrollingFrame():
            with ui.VStack(spacing=8, height=0):
                ui.Label("BORINOT / GO-TO GOAL", height=28, style={"font_size": 20})
                with ui.HStack(height=24, spacing=7):
                    self._relative_model = ui.SimpleBoolModel(self.relative)
                    ui.CheckBox(model=self._relative_model, width=22)
                    ui.Label("Relative to base position (off: world XYZ)")
                    self._subscriptions.append(self._relative_model.subscribe_value_changed_fn(
                        lambda m: self.set_relative(m.get_value_as_bool())))
                self.frame_label = ui.Label(self.frame_description(), height=58, word_wrap=True)
                self.state_label = ui.Label("", height=100, word_wrap=True)
                values = (*self.goal, self.deadline_s)
                for index, (label, (low, high)) in enumerate(zip(("x [m]", "y [m]", "z [m]", "deadline [s]"), ranges)):
                    if index == 3:
                        with ui.HStack(height=24, spacing=7):
                            self._auto_model = ui.SimpleBoolModel(self.auto_deadline)
                            ui.CheckBox(model=self._auto_model, width=22)
                            ui.Label("Auto deadline: middle of the training range for the distance")
                            self._subscriptions.append(self._auto_model.subscribe_value_changed_fn(
                                lambda m: setattr(self, "auto_deadline", m.get_value_as_bool())))
                    with ui.HStack(spacing=8, height=30):
                        ui.Label(label, width=100)
                        model = ui.SimpleFloatModel(values[index], min=low, max=high)
                        ui.FloatField(model, width=105)
                        ui.FloatSlider(model, min=low, max=high)
                        self._models.append(model)
                        self._subscriptions.append(model.subscribe_value_changed_fn(
                            lambda m, i=index: self.set_component(i, m.get_value_as_float())))
                ui.Label("Moving the deadline switches Auto off. Click a number to type an exact value.",
                         height=22, word_wrap=True)
                self.preview_label = ui.Label("", height=75, word_wrap=True)
                with ui.HStack(height=32):
                    ui.Button("Go to goal", clicked_fn=self.request_go)
                    ui.Button("Fly home", clicked_fn=self.fly_home)
                with ui.HStack(height=32):
                    ui.Button("Reset robot", clicked_fn=self.request_reset)
                    self.pause_button = ui.Button("Pause", clicked_fn=self.toggle_pause)
                ui.Label("CAMERA", height=22)
                with ui.HStack(height=30):
                    for mode in CAMERAS:
                        ui.Button(mode.title(), clicked_fn=lambda m=mode: self.set_camera(m))
                self.goal_label = ui.Label("", height=58, word_wrap=True)
                self.result_label = ui.Label("", height=95, word_wrap=True)
                self.status_label = ui.Label("", height=40, word_wrap=True)
                ui.Label(("Every deadline resets the robot to home and repeats the goal, as in training."
                          if self.env.reset_at_deadline else
                          "After the deadline the robot keeps the goal: a hold segment with the shortest training\n"
                          "deadline starts by itself. The policy is only paid at deadlines, so it may move a little\n"
                          "between them. A new goal starts from the current state, without a teleport.")
                         + "\nYellow sphere: goal not sent yet. Green sphere: active goal, success radius."
                         "\nFlight bounds and crash resets remain enabled.", height=75, word_wrap=True)
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def set_component(self, index, value):
        if not math.isfinite(value):
            return
        if index < 3:
            self.goal[index] = float(value)
        else:
            self.deadline_s = float(value)
            if self.auto_deadline and self.window:
                self._auto_model.set_value(False)
            self.auto_deadline = False

    def request_go(self):
        self.go_requested = True

    def set_relative(self, relative):
        self.relative = bool(relative)
        if self.window:
            self.frame_label.text = self.frame_description()

    def frame_description(self):
        if self.relative:
            return ("XYZ = displacement from the current base, along WORLD axes (not rotated by yaw). "
                    "Each Go uses the same numbers from the new position. The sent target stays fixed.")
        return "XYZ = world position in metres; z is height above ground. Switching mode does not send a goal."

    def preview_goal(self):
        """The next Go target for robot 0; a relative preview follows the base, not the active goal."""
        if self.relative:
            position = (self.env.robot.data.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().tolist()
            return [p + d for p, d in zip(position, self.goal)]
        return list(self.goal)

    def fly_home(self):
        self.set_relative(False)  # home is always an absolute target
        if self.window:
            self._relative_model.set_value(False)
        for index, value in enumerate(self.env.cfg.robot_home):
            self.set_component(index, value)
            if self._models:
                self._models[index].set_value(value)
        self.request_go()

    def request_reset(self):
        self.reset_requested = True

    def toggle_pause(self):
        self.paused = not self.paused
        if self.window:
            self.pause_button.text = "Resume" if self.paused else "Pause"

    def set_camera(self, camera):
        self.camera = camera

    def requested_deadline(self):
        return None if self.auto_deadline else self.deadline_s

    def update(self, time_s):
        env, data = self.env, self.env.robot.data
        origin = env.scene.env_origins[0]
        position = (data.root_pos_w[0] - origin).cpu().tolist()
        active = (env._goal_position_w[0] - origin).cpu().tolist()
        # ViewportCameraController's "env" origin expects clone-local coordinates.
        controller = env.viewport_camera_controller
        if controller is not None and self.camera != "free":
            if self.camera == "goal":
                eye, lookat = goal_camera_pose((env._segment_start_w[0] - origin).cpu().tolist(), active)
            else:
                w, x, y, z = data.root_quat_w[0].cpu().tolist()
                eye, lookat = camera_pose(self.camera, position, math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
            controller.update_view_location(eye=eye, lookat=lookat)
        if not self.window:
            return
        self._update_preview(active)
        if self.paused or time_s >= self._next_refresh_s:
            self._refresh_labels(time_s, position, active)
            self._next_refresh_s = time_s + 0.1

    def _update_preview(self, active):
        """Yellow marker at an edited goal that was not sent yet. Writes USD only on change."""
        goal = self.preview_goal()
        pending = max(abs(p - a) for p, a in zip(goal, active)) > 1e-4  # float32 goal buffer
        key = tuple(goal) if pending else None
        if key == self._preview_key:
            return
        self._preview_key = key
        if self._preview is None:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            # Opaque and emissive: a translucent sphere is nearly invisible on the grey floor.
            self._preview = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/BorinotGoTo/pending", markers={"pending": sim_utils.SphereCfg(
                    radius=self.env.cfg.goal_success_tolerance_m, visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.8, 0.1), emissive_color=(1.0, 0.8, 0.1)))}))
        self._preview.set_visibility(pending)
        if pending:
            origin = self.env.scene.env_origins[0]
            self._preview.visualize(translations=(origin + origin.new_tensor(goal))[None])

    def _refresh_labels(self, time_s, position, active):
        env, data, cfg = self.env, self.env.robot.data, self.env.cfg
        origin = env.scene.env_origins[0]
        goal = self.preview_goal()
        offset_w = origin.new_tensor(goal) + origin - data.root_pos_w[0]
        offset_b = quat_apply_inverse(data.root_quat_w[:1], offset_w[None])[0].cpu().tolist()
        distance = float(offset_w.norm())
        low, high = (float(v) for v in training_deadline_interval(
            cfg, env.play_difficulty, offset_w.new_tensor(distance)))
        deadline = (low + high) / 2 if self.auto_deadline else self.deadline_s
        speed = float(data.root_lin_vel_w[0].norm())
        notes = training_region_notes(cfg, env.play_difficulty, offset_b, goal[2], speed,
                                      deadline, (low, high), env.step_dt)
        self.preview_label.text = (
            f"If sent now: world XYZ {xyz_text(goal)} m, {distance:.2f} m from robot 0, "
            f"deadline {'auto ' if self.auto_deadline else ''}"
            f"{deadline:.2f} s (training {low:.2f}-{high:.2f} s for this distance).\n"
            + ("Outside the training region: " + "; ".join(notes) + "." if notes
               else f"Inside the training region of stage {env.goal_curriculum.level}."))
        remaining = float((env._goal_deadline_steps[0] - env._episode_steps[0]).clamp_min(0)) * env.step_dt
        current = flight_state(env, time_s)
        self.state_label.text = (
            f"CURRENT POSITION XYZ {xyz_text(current['position_m'])} m (world)\n"
            f"Velocity XYZ {xyz_text(current['velocity_m_s'])} m/s; speed {current['speed_m_s']:.3f} m/s\n"
            f"Goal error XYZ {xyz_text(current['error_m'])} m; distance {current['distance_m']:.3f} m\n"
            f"Yaw {current['yaw_deg']:+.1f} deg; time left {current['remaining_s']:.2f} s")
        state = "HOLDING" if env._segment_hold[0] else "FLYING"
        self.goal_label.text = (
            f"Active goal ({', '.join(f'{v:+.2f}' for v in active)}) m: {state}, "
            f"{remaining:.2f} of {int(env._goal_deadline_steps[0]) * env.step_dt:.2f} s left, "
            f"distance {float((env._goal_position_w[0] - data.root_pos_w[0]).norm()):.3f} m.\n"
            f"Input mode: {'RELATIVE displacement' if self.relative else 'WORLD position'}.")
        totals, results = env.play_totals, env.play_results
        self.result_label.text = (
            (describe_result(results[-1], len(results)) if results else "No goal finished yet.")
            + f"\nAll robots: goals {totals['goals']} ({totals['goal_successes']} inside "
              f"{cfg.goal_success_tolerance_m:g} m at the deadline, {totals['goal_failures']} failed); "
              f"holds {totals['holds']} ({totals['hold_successes']} inside, {totals['hold_failures']} failed, "
              f"worst {env.play_hold_worst_distance_m:.3f} m).")
        self.status_label.text = (f"{'PAUSED' if self.paused else 'PLAYING'} - {time_s:.1f} s - camera {self.camera}"
                                  f" - manual resets {self.manual_resets}" + (f"\n{self.message}" if self.message else ""))

    def close(self):
        if self._preview is not None:
            self._preview.set_visibility(False)
        if self.window:
            self.window.destroy()
            self.window = None
        self._subscriptions.clear()
