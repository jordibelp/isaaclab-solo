"""Live command/force UI and follow cameras for the MuJoCo Solo12 sim-to-sim viewer.

Mirrors the play_direct_0325.py interaction model: clipped vx/vy/wz sliders, a body-frame
base force with Pulse/Hold/Release and a selectable application point, and heading-stable
follow cameras (side/front/free).
"""

from __future__ import annotations

import math
import threading

import mujoco
import numpy as np

BASE_HALF_EXTENTS = (0.2247476, 0.0986631, 0.0185328)
CAMERA_MODES = ("free", "side", "front")
# Heading-frame camera offsets, matching the IsaacLab follow cameras.
CAMERA_EYE_LOOKAT_B = {
    "side": ((0.0, -2.6, 1.1), (0.0, 0.0, 0.35)),
    "front": ((2.6, 0.0, 1.1), (0.0, 0.0, 0.35)),
}
# GLFW keycodes for the passive-viewer key callback.
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_A, KEY_C, KEY_D, KEY_F, KEY_R = 32, 65, 67, 68, 70, 82

KEYBOARD_HELP = (
    "viewer keys: arrows=vx/wz  A/D=vy  SPACE=zero cmd  R=reset robot  C=cycle camera  F=release force"
)


class LiveCommandState:
    """Thread-safe live command shared between the sim loop, sliders, and keyboard."""

    def __init__(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
        self._lock = threading.Lock()
        self.vx, self.vy, self.wz = float(vx), float(vy), float(wz)
        self._reset_requested = False

    def set_command(self, vx: float | None = None, vy: float | None = None, wz: float | None = None):
        with self._lock:
            if vx is not None: self.vx = float(vx)
            if vy is not None: self.vy = float(vy)
            if wz is not None: self.wz = float(wz)

    def adjust(self, dvx: float = 0.0, dvy: float = 0.0, dwz: float = 0.0):
        with self._lock:
            self.vx += dvx; self.vy += dvy; self.wz += dwz

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            return self.vx, self.vy, self.wz

    def get_clipped(self, ranges) -> tuple[float, float, float]:
        (vx_lo, vx_hi), (vy_lo, vy_hi), (wz_lo, wz_hi) = ranges
        vx, vy, wz = self.get()
        return (float(np.clip(vx, vx_lo, vx_hi)), float(np.clip(vy, vy_lo, vy_hi)),
                float(np.clip(wz, wz_lo, wz_hi)))

    def request_reset(self):
        with self._lock:
            self._reset_requested = True

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested


class BodyForceState:
    """Body-frame base force with Isaac-style Pulse/Hold/Release semantics.

    Slider values are captured when Pulse/Hold is clicked; ``get_active_force`` is called once
    per policy step and counts a pulse down.
    """

    def __init__(self, dt: float):
        self._lock = threading.Lock()
        self.dt = float(dt)
        self.magnitude = 0.0
        self.azimuth_deg = 0.0
        self.elevation_deg = 0.0
        self.point_b = (0.0, 0.0, 0.0)
        self.duration_s = 0.25
        self._active: tuple[float, float, float, float, float, float] | None = None
        self._pulse_steps_left: int | None = None  # None while holding

    def set(self, **fields):
        with self._lock:
            for name, value in fields.items():
                if name == "point_b":
                    self.point_b = tuple(float(v) for v in value)
                else:
                    setattr(self, name, float(value))

    def _selected_force(self) -> tuple[float, float, float, float, float, float]:
        azimuth = math.radians(self.azimuth_deg)
        elevation = math.radians(max(-90.0, min(90.0, self.elevation_deg)))
        fx = self.magnitude * math.cos(elevation) * math.cos(azimuth)
        fy = self.magnitude * math.cos(elevation) * math.sin(azimuth)
        fz = self.magnitude * math.sin(elevation)
        return (fx, fy, fz, *self.point_b)

    def pulse(self):
        with self._lock:
            self._active = self._selected_force()
            self._pulse_steps_left = max(1, int(round(self.duration_s / self.dt)))

    def hold(self):
        with self._lock:
            self._active = self._selected_force()
            self._pulse_steps_left = None

    def release(self):
        with self._lock:
            self._active = None
            self._pulse_steps_left = None

    def get_active_force(self) -> tuple[float, float, float, float, float, float] | None:
        with self._lock:
            if self._active is None:
                return None
            if self._pulse_steps_left is not None:
                if self._pulse_steps_left <= 0:
                    self._active = None
                    self._pulse_steps_left = None
                    return None
                self._pulse_steps_left -= 1
            return self._active


class FollowCamera:
    """Heading-stable follow camera, ported from the IsaacLab ChaseCamera.

    Heading comes from the horizontal projection of the body +y axis, which stays
    well-conditioned while the two-feet robot pitches to ~90 deg; anchor position and
    heading are EMA-smoothed (tau = 0.2 s).
    """

    def __init__(self, dt: float, mode: str = "free"):
        self.dt = float(dt)
        self.mode = mode
        self._anchor_w: np.ndarray | None = None
        self._heading: float | None = None
        self._smoothing_time_s = 0.20

    def cycle_mode(self) -> str:
        self.mode = CAMERA_MODES[(CAMERA_MODES.index(self.mode) + 1) % len(CAMERA_MODES)]
        return self.mode

    @staticmethod
    def _stable_heading_from_quat(q_wxyz, previous: float | None) -> float:
        w, x, y, z = [float(v) for v in q_wxyz]
        lateral_x = 2.0 * (x * y - w * z)
        lateral_y = 1.0 - 2.0 * (x * x + z * z)
        if math.hypot(lateral_x, lateral_y) > 1.0e-4:
            heading = math.atan2(lateral_y, lateral_x) - 0.5 * math.pi
        else:
            forward_x = 1.0 - 2.0 * (y * y + z * z)
            forward_y = 2.0 * (x * y + w * z)
            if math.hypot(forward_x, forward_y) <= 1.0e-4:
                return 0.0 if previous is None else previous
            heading = math.atan2(forward_y, forward_x)
        if previous is not None:
            heading = previous + math.atan2(math.sin(heading - previous), math.cos(heading - previous))
        return heading

    def update(self, base_pos_w, base_quat_wxyz):
        heading = self._stable_heading_from_quat(base_quat_wxyz, self._heading)
        pos = np.asarray(base_pos_w, dtype=float)
        if self._anchor_w is None:
            self._anchor_w, self._heading = pos.copy(), heading
        else:
            alpha = 1.0 - math.exp(-self.dt / self._smoothing_time_s)
            self._anchor_w += alpha * (pos - self._anchor_w)
            self._heading += alpha * (heading - self._heading)

    def apply(self, cam: mujoco.MjvCamera) -> bool:
        """Write lookat/azimuth/elevation/distance into the viewer camera. False in free mode."""
        if self.mode == "free" or self._anchor_w is None:
            return False
        eye_b, lookat_b = CAMERA_EYE_LOOKAT_B[self.mode]
        c, s = math.cos(self._heading), math.sin(self._heading)
        rot = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        eye_w = self._anchor_w + rot @ eye_b
        lookat_w = self._anchor_w + rot @ lookat_b
        view = lookat_w - eye_w
        distance = float(np.linalg.norm(view))
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat_w
        cam.distance = distance
        cam.azimuth = math.degrees(math.atan2(view[1], view[0]))
        cam.elevation = math.degrees(math.asin(view[2] / max(distance, 1e-9)))
        return True


def apply_body_force(sim, active_force) -> None:
    """Apply the body-frame force at its body-frame point, like the Isaac permanent wrench."""
    sim.data.qfrc_applied[:] = 0.0
    if active_force is None:
        return
    fx, fy, fz, px, py, pz = active_force
    from play_direct_mujoco import quat_rotation

    rotation = quat_rotation(sim.data.qpos[3:7])
    force_w = rotation @ (fx, fy, fz)
    point_w = sim.data.xpos[sim.base_id] + rotation @ (px, py, pz)
    mujoco.mj_applyFT(sim.model, sim.data, force_w, np.zeros(3), point_w, sim.base_id, sim.data.qfrc_applied)


def update_force_arrow(user_scn, sim, active_force) -> None:
    """Draw a red arrow pushing into the application point (empty when no force is active)."""
    user_scn.ngeom = 0
    if active_force is None:
        return
    fx, fy, fz, px, py, pz = active_force
    magnitude = math.sqrt(fx * fx + fy * fy + fz * fz)
    if magnitude < 1e-9:
        return
    from play_direct_mujoco import quat_rotation

    rotation = quat_rotation(sim.data.qpos[3:7])
    direction_w = rotation @ (fx, fy, fz) / magnitude
    tip_w = sim.data.xpos[sim.base_id] + rotation @ (px, py, pz)
    length = float(np.clip(0.08 * magnitude, 0.15, 0.6))
    geom = user_scn.geoms[0]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.array((0.85, 0.1, 0.1, 0.9), dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.014, tip_w - direction_w * length, tip_w)
    user_scn.ngeom = 1


def make_key_callback(cmd_state: LiveCommandState, force_state: BodyForceState | None,
                      camera: FollowCamera):
    def key_callback(keycode: int):
        if keycode == KEY_UP: cmd_state.adjust(dvx=0.05)
        elif keycode == KEY_DOWN: cmd_state.adjust(dvx=-0.05)
        elif keycode == KEY_LEFT: cmd_state.adjust(dwz=0.1)
        elif keycode == KEY_RIGHT: cmd_state.adjust(dwz=-0.1)
        elif keycode == KEY_A: cmd_state.adjust(dvy=0.05)
        elif keycode == KEY_D: cmd_state.adjust(dvy=-0.05)
        elif keycode == KEY_SPACE: cmd_state.set_command(0.0, 0.0, 0.0)
        elif keycode == KEY_R: cmd_state.request_reset()
        elif keycode == KEY_C: print(f"[INFO] Camera: {camera.cycle_mode()}", flush=True)
        elif keycode == KEY_F and force_state is not None: force_state.release()
    return key_callback


def start_control_panel(cmd_state: LiveCommandState, force_state: BodyForceState,
                        camera: FollowCamera, command_ranges, force_max: float) -> bool:
    """Launch the tkinter slider panel in a daemon thread. Returns False if tkinter is missing."""
    try:
        import tkinter as tk
    except Exception as exc:
        print(f"[WARN] tkinter unavailable ({exc}); keyboard control only. {KEYBOARD_HELP}", flush=True)
        return False

    (vx_lo, vx_hi), (vy_lo, vy_hi), (wz_lo, wz_hi) = command_ranges
    half_x, half_y, half_z = BASE_HALF_EXTENTS

    def run_panel():
        root = tk.Tk()
        root.title("Solo12 MuJoCo control")
        root.geometry("+40+40")

        def slider(parent, label, lo, hi, initial, on_change, resolution=0.01):
            frame = tk.Frame(parent); frame.pack(fill="x", padx=8)
            tk.Label(frame, text=label, width=14, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=initial)
            tk.Scale(frame, from_=lo, to=hi, resolution=resolution, orient="horizontal",
                     variable=var, length=260, command=lambda _: on_change(var.get())).pack(side="left", fill="x")
            return var

        tk.Label(root, text="Velocity command", font=("TkDefaultFont", 10, "bold")).pack(pady=(8, 0))
        vx0, vy0, wz0 = cmd_state.get()
        vx_var = slider(root, f"vx [{vx_lo:g},{vx_hi:g}]", vx_lo, vx_hi, vx0, lambda v: cmd_state.set_command(vx=v))
        vy_var = slider(root, f"vy [{vy_lo:g},{vy_hi:g}]", vy_lo, vy_hi, vy0, lambda v: cmd_state.set_command(vy=v))
        wz_var = slider(root, f"wz [{wz_lo:g},{wz_hi:g}]", wz_lo, wz_hi, wz0, lambda v: cmd_state.set_command(wz=v))
        buttons = tk.Frame(root); buttons.pack(pady=4)

        def zero():
            cmd_state.set_command(0.0, 0.0, 0.0)
        tk.Button(buttons, text="Zero cmd", command=zero).pack(side="left", padx=4)
        tk.Button(buttons, text="Reset robot", command=cmd_state.request_reset).pack(side="left", padx=4)

        tk.Label(root, text="Base force (body frame)", font=("TkDefaultFont", 10, "bold")).pack(pady=(10, 0))
        slider(root, f"magnitude [0,{force_max:g}] N", 0.0, force_max, 0.0, lambda v: force_state.set(magnitude=v), 0.1)
        slider(root, "azimuth [deg]", -180.0, 180.0, 0.0, lambda v: force_state.set(azimuth_deg=v), 1.0)
        slider(root, "elevation [deg]", -90.0, 90.0, 0.0, lambda v: force_state.set(elevation_deg=v), 1.0)
        slider(root, "pulse [s]", 0.05, 3.0, 0.25, lambda v: force_state.set(duration_s=v), 0.05)
        surfaces = tk.Frame(root); surfaces.pack()
        tk.Label(surfaces, text="point:").pack(side="left")
        for name, point in (("front", (half_x, 0, 0)), ("rear", (-half_x, 0, 0)), ("left", (0, half_y, 0)),
                            ("right", (0, -half_y, 0)), ("top", (0, 0, half_z)), ("center", (0, 0, 0))):
            tk.Button(surfaces, text=name, command=lambda p=point: force_state.set(point_b=p)).pack(side="left", padx=1)
        force_buttons = tk.Frame(root); force_buttons.pack(pady=4)
        tk.Button(force_buttons, text="Pulse", command=force_state.pulse).pack(side="left", padx=4)
        tk.Button(force_buttons, text="Hold", command=force_state.hold).pack(side="left", padx=4)
        tk.Button(force_buttons, text="Release", command=force_state.release).pack(side="left", padx=4)

        tk.Label(root, text="Camera", font=("TkDefaultFont", 10, "bold")).pack(pady=(10, 0))
        cam_var = tk.StringVar(value=camera.mode)
        cam_frame = tk.Frame(root); cam_frame.pack()
        for mode in CAMERA_MODES:
            tk.Radiobutton(cam_frame, text=mode, value=mode, variable=cam_var,
                           command=lambda: setattr(camera, "mode", cam_var.get())).pack(side="left", padx=4)
        tk.Label(root, text=KEYBOARD_HELP, wraplength=330, fg="#555").pack(padx=8, pady=(8, 8))

        def poll():
            # Reflect keyboard-driven changes back into the sliders and camera radio.
            vx, vy, wz = cmd_state.get()
            for var, value in ((vx_var, vx), (vy_var, vy), (wz_var, wz)):
                if abs(var.get() - value) > 1e-9:
                    var.set(value)
            if cam_var.get() != camera.mode:
                cam_var.set(camera.mode)
            root.after(100, poll)

        poll()
        root.mainloop()

    threading.Thread(target=run_panel, daemon=True, name="control-panel").start()
    return True
