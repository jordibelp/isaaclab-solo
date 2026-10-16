# SPDX-License-Identifier: BSD-3-Clause
"""Small match panel and camera presets for goalkeeper playback."""

import math

CAMERA_MODES = ("overview", "front", "side", "top", "keeper", "follow", "orbit", "free")
OUTCOME_NAMES = ("Ongoing", "Base hit", "Arm hit", "Goal", "Ball on floor", "Left goal area",
                 "Shot deadline", "Blocked / save", "Robot crash", "Miss / escape")


def camera_pose(cfg, mode, robot=None, time_s=0.0):
    """Return env-local eye/target. Follow is world-aligned, never robot-yaw aligned."""
    gx, gy, gz = cfg.goal_center
    launch_x = max(cfg.launch_x_range)
    center = ((gx + launch_x) / 2, gy, gz)
    if mode == "overview":
        return (launch_x + 2.0, gy - 4.5, gz + 1.6), center
    if mode == "front":
        return (launch_x + 3.0, gy, gz), cfg.goal_center
    if mode == "side":
        return (center[0], gy - 5.0, gz + 0.6), center
    if mode == "top":
        return (center[0], gy - 0.01, gz + 5.0), center
    if mode == "keeper":
        # Behind the translucent goal, wide enough to see off-centre keeper spawns.
        return (gx - 2.0, gy, gz + 0.7), (0.6, gy, gz)
    if mode == "follow":
        target = tuple(robot) if robot is not None else tuple(cfg.robot_home)
        return (target[0] + 2.4, target[1] - 2.8, target[2] + 1.1), target
    if mode == "orbit":
        angle = time_s * 0.18
        return (center[0] + 4.5 * math.cos(angle), gy + 4.5 * math.sin(angle), gz + 1.8), center
    return None


class MatchScore:
    """Count resolved task outcomes, not touches or persistent-ball contacts."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.counts = [0] * len(OUTCOME_NAMES)
        self.last = "Waiting for the first shot"

    def record(self, code, finished):
        if finished and code:
            self.counts[code] += 1
            self.last = OUTCOME_NAMES[code]

    @property
    def shots(self):
        return sum(self.counts)

    @property
    def other(self):
        return self.shots - self.counts[3] - self.counts[7]


class PlayControls:
    """UI callbacks only queue intent. Physics mutations happen in the play loop."""

    def __init__(self, env, camera="overview", speed=1.0, show_ui=True):
        self.env = env
        self.camera = camera
        self.speed = speed
        self.paused = False
        self.reset_requested = False
        self.manual_resets = 0
        self.score = MatchScore()
        self.window = None
        self._last_camera = None
        self._next_refresh_s = 0.0
        self._free_pose = (tuple(env.cfg.viewer.eye), tuple(env.cfg.viewer.lookat))
        if show_ui:
            self._build_window()

    def _build_window(self):
        import omni.ui as ui

        self.window = ui.Window("Borinot Match", width=370, height=555,
                                dock_preference=ui.DockPreference.RIGHT_TOP)
        self.window.frame.set_style({
            "Label": {"font_size": 15, "color": 0xFFE9EDF4},
            "Button": {"background_color": 0xFF493B2B, "border_radius": 5, "margin": 2},
            "Button:hovered": {"background_color": 0xFF725538},
        })
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("BORINOT  /  GOALKEEPER", height=28, style={"font_size": 20})
                ui.Label("Match score · environment 0", height=20)
                with ui.HStack(height=65):
                    with ui.VStack():
                        ui.Label("GOALS", style={"color": 0xFF8985FF})
                        self.goals_label = ui.Label("0", style={"font_size": 32, "color": 0xFF8985FF})
                    with ui.VStack():
                        ui.Label("BLOCKED", style={"color": 0xFF9FE8A0})
                        self.saves_label = ui.Label("0", style={"font_size": 32, "color": 0xFF9FE8A0})
                self.summary_label = ui.Label("", height=23)
                self.last_label = ui.Label("", height=23)
                self.status_label = ui.Label("", height=35, word_wrap=True)
                with ui.HStack(height=32):
                    self.pause_button = ui.Button("Pause", clicked_fn=self.toggle_pause)
                    ui.Button("Reset Borinot", clicked_fn=self.request_reset)
                ui.Button("Clear score", height=26, clicked_fn=self.score.clear)
                ui.Label("CAMERA", height=22)
                for modes in (CAMERA_MODES[:4], CAMERA_MODES[4:]):
                    with ui.HStack(height=29):
                        for mode in modes:
                            ui.Button(mode.title(), clicked_fn=lambda m=mode: self.set_camera(m))
                ui.Label("PLAYBACK SPEED", height=22)
                with ui.HStack(height=29):
                    for speed in (0.25, 0.5, 1.0, 2.0):
                        ui.Button(f"{speed:g}×", clicked_fn=lambda s=speed: self.set_speed(s))
                ui.Label("Orange = live shot · blue = recent balls\n"
                         f"Keep last {self.env.keep_balls} balls (including live shot).\n"
                         "Old balls bounce off scenery only.\n"
                         "Reset keeps score; unfinished shots are not counted.",
                         word_wrap=True, height=80, style={"font_size": 13})
        self.window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.window:
            self.pause_button.text = "Resume" if self.paused else "Pause"

    def request_reset(self):
        self.reset_requested = True

    def set_camera(self, mode):
        self.camera = mode

    def set_speed(self, speed):
        self.speed = speed

    def update(self, time_s):
        dynamic = self.camera in ("follow", "orbit")
        if self.camera != self._last_camera or dynamic:
            if self.camera == "free":
                # Selecting Free stops tracking and leaves the current mouse view.
                pose = self._free_pose if self._last_camera is None else None
            else:
                root = (self.env.robot.data.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().tolist()
                pose = camera_pose(self.env.cfg, self.camera, root, time_s)
            controller = self.env.viewport_camera_controller
            if pose is not None and controller is not None:
                controller.update_view_location(eye=pose[0], lookat=pose[1])
            self._last_camera = self.camera
        if self.window and (self.paused or time_s >= self._next_refresh_s):
            score = self.score
            self.goals_label.text = str(score.counts[3])
            self.saves_label.text = str(score.counts[7])
            self.summary_label.text = (f"Shots {score.shots}  ·  Other {score.other}  ·  "
                                       f"Saves {100 * score.counts[7] / max(score.shots, 1):.0f}%")
            self.last_label.text = "Last: " + score.last
            self.status_label.text = (f"{'PAUSED' if self.paused else 'PLAYING'}  ·  {time_s:.1f} s  ·  "
                                      f"{self.speed:g}×  ·  {self.camera.title()}\n"
                                      f"Shot level {self.env.shot_level}  ·  Manual resets {self.manual_resets}")
            self._next_refresh_s = time_s + 0.1

    def close(self):
        if self.window:
            self.window.destroy()
            self.window = None
