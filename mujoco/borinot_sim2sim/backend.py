"""MuJoCo stepping, contact bookkeeping, viewer and rendering behind the Isaac Lab stand-ins.

One environment, one MuJoCo model. Each Isaac physics step (``sim.dt``) runs ``substeps``
MuJoCo steps with the same controls, like PhysX holds a command over one step. Quantities
that PhysX reports *for the last step* use the same window here:

* contact forces: mean force over the step, per body pair (ContactSensor);
* joint reaction wrenches: mean parent-on-child wrench, in the child joint frame;
* body accelerations: (velocity after - velocity before) / dt.
"""

from __future__ import annotations

import json
import math
import os
import time

import mujoco
import numpy as np

from . import usd_scene

# Command-line options, set by the AppLauncher stand-in before the environment exists.
OPTIONS = {"headless": False, "substeps": None, "default_substeps": 1, "show_viewer_ui": False,
           "max_fps": 30.0}
_current = None


def current() -> Backend:
    if _current is None:
        raise RuntimeError("The MuJoCo simulation context does not exist yet.")
    return _current


def camera_angles(eye, lookat) -> tuple[float, float, float]:
    """Isaac eye/lookat -> MuJoCo free-camera distance, azimuth and elevation (degrees)."""
    delta = np.asarray(lookat, dtype=float) - np.asarray(eye, dtype=float)
    distance = max(float(np.linalg.norm(delta)), 1e-6)
    azimuth = math.degrees(math.atan2(delta[1], delta[0]))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, delta[2] / distance))))
    return distance, azimuth, elevation


class Backend:
    """Owns the USD stage, the compiled model and the viewer."""

    def __init__(self, physics_dt: float, gravity):
        global _current
        _current = self
        self.stage = usd_scene.new_stage()
        self.physics_dt = float(physics_dt)
        self.gravity = tuple(float(g) for g in gravity)
        substeps = OPTIONS["substeps"]
        self.substeps = int(OPTIONS["default_substeps"] if substeps is None else substeps)
        if self.substeps < 1:
            raise ValueError("--mj_substeps must be a positive integer.")
        self.objects = []  # assets and sensors, in creation order
        self.markers = []
        self.model = self.data = self.index = None
        self.version = 0  # bumps whenever the MuJoCo state changes
        self.track_joint_wrench = False  # turned on by the first read of body_incoming_joint_wrench_b
        self.viewer = None
        self.closed = False
        self.view_pose = (np.array((4.0, 4.0, 3.0)), np.zeros(3))  # latest Isaac eye/lookat
        self._camera = None  # pose not yet applied to the interactive viewer
        self._renderer = None
        self._viewer_needs_model = True
        self._last_render_wall = -math.inf
        self.completed_steps = 0
        self._first_step_wall = self._last_step_wall = None

    # -- construction -------------------------------------------------------------

    def register(self, obj):
        if self.model is not None:
            raise RuntimeError("Assets and sensors must be created before the simulation starts.")
        self.objects.append(obj)

    def build(self):
        articulations = [o for o in self.objects if getattr(o, "kind", None) == "articulation"]
        rigid_objects = [o for o in self.objects if getattr(o, "kind", None) == "rigid_object"]
        self.spec, self.model, self.index = usd_scene.build_model(
            self.stage, [a.asset_path for a in articulations], [r.asset_path for r in rigid_objects],
            self.physics_dt, self.substeps, self.gravity,
            self_collisions={a.asset_path: a.self_collisions for a in articulations})
        self.data = mujoco.MjData(self.model)
        self._scratch = mujoco.MjData(self.model)
        n = self.model.nbody
        self.pair_force = np.zeros((n, n, 3))  # mean force on body i from body j over the last step
        self.joint_wrench = np.zeros((n, 6))  # mean parent-on-child [F, M] in the child joint frame
        self._force6 = np.zeros(6)
        for obj in self.objects:
            obj._initialize(self)
        self.forward()
        print(f"[MuJoCo] Model: {self.model.nbody - 1} bodies, {self.model.njnt} joints, {self.model.ngeom} geoms; "
              f"physics {1 / self.physics_dt:g} Hz x {self.substeps} substep(s) "
              f"(timestep {self.model.opt.timestep * 1000:g} ms)", flush=True)
        if not OPTIONS["headless"]:
            # Match the existing Solo12 player on GNOME Wayland. Native Wayland
            # GLFW emits positioning errors; Tk panels already use XWayland.
            if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
                os.environ.pop("WAYLAND_DISPLAY")
                print("[MuJoCo] Using XWayland for the viewer, like the Solo12 player.", flush=True)
            import mujoco.viewer as mujoco_viewer

            self.viewer = mujoco_viewer.launch_passive(
                self.model, self.data, show_left_ui=OPTIONS["show_viewer_ui"],
                show_right_ui=OPTIONS["show_viewer_ui"])

    def update_constants(self):
        """Refresh derived model constants after mass/inertia edits (scratch data only)."""
        self._viewer_needs_model = True
        mujoco.mj_setConst(self.model, self._scratch)
        self.forward()

    # -- stepping -----------------------------------------------------------------

    def forward(self):
        """Recompute every derived quantity after a state write."""
        mujoco.mj_forward(self.model, self.data)
        self.version += 1

    def step(self):
        """Advance one Isaac physics step; ``mj_step1`` already ran for the current state."""
        if self._first_step_wall is None:
            self._first_step_wall = time.perf_counter()
        m, d = self.model, self.data
        for obj in self.objects:
            obj._before_step()
        self.pair_force.fill(0.0)
        if self.track_joint_wrench:
            self.joint_wrench.fill(0.0)
        weight = 1.0 / self.substeps
        for _ in range(self.substeps):
            # mj_step2, split so that forces are read with the velocities they were computed from.
            mujoco.mj_fwdActuation(m, d)
            mujoco.mj_fwdAcceleration(m, d)
            mujoco.mj_fwdConstraint(m, d)
            mujoco.mj_checkAcc(m, d)
            d.qacc_warmstart[:] = d.qacc
            self._accumulate(weight)
            mujoco.mj_implicit(m, d)
            mujoco.mj_step1(m, d)  # kinematics, contacts and velocities of the new state
        self.version += 1
        for obj in self.objects:
            obj._after_step()
        self.completed_steps += 1
        self._last_step_wall = time.perf_counter()

    def _accumulate(self, weight):
        m, d = self.model, self.data
        for i in range(d.ncon):
            contact = d.contact[i]
            if contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, self._force6)
            # Contact frame rows: normal (geom1 -> geom2), then tangents. Force acts on geom2.
            force = contact.frame.reshape(3, 3).T @ self._force6[:3]
            b1, b2 = m.geom_bodyid[contact.geom1], m.geom_bodyid[contact.geom2]
            self.pair_force[b2, b1] += weight * force
            self.pair_force[b1, b2] -= weight * force
        if self.track_joint_wrench:
            mujoco.mj_rnePostConstraint(m, d)
            for obj in self.objects:
                if getattr(obj, "kind", None) == "articulation":
                    obj._accumulate_joint_wrench(weight)

    # -- viewer and rendering -------------------------------------------------------

    def set_camera(self, eye, lookat):
        self.view_pose = (np.asarray(eye, dtype=float), np.asarray(lookat, dtype=float))
        self._camera = self.view_pose  # applied once: the mouse may move a static view afterwards

    @staticmethod
    def _apply_camera(camera, pose):
        eye, lookat = pose
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance, camera.azimuth, camera.elevation = camera_angles(eye, lookat)

    def _draw_markers(self, scene):
        for marker in self.markers:
            for gtype, size, pos, mat, rgba, emission in marker.geoms():
                if scene.ngeom >= scene.maxgeom:
                    return
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(geom, gtype, size, pos, mat, rgba)
                geom.emission = emission
                scene.ngeom += 1

    def render(self):
        """Sync the viewer (camera, markers, state) and pump the control panel."""
        now = time.perf_counter()
        if now - self._last_render_wall < 1.0 / OPTIONS["max_fps"]:
            return
        self._last_render_wall = now
        if self.viewer is not None and self.viewer.is_running():
            with self.viewer.lock():
                if self._camera is not None:
                    self._apply_camera(self.viewer.cam, self._camera)
                    self._camera = None
                self.viewer.user_scn.ngeom = 0
                self._draw_markers(self.viewer.user_scn)
            # Visual meshes are large and do not change each frame. Copy only
            # integration state unless a physics/model setter changed the model.
            self.viewer.sync(state_only=not self._viewer_needs_model)
            self._viewer_needs_model = False
        from . import omni_ui_tk

        omni_ui_tk.pump()

    def render_rgb(self, width, height):
        """Offscreen RGB frame from the latest viewport eye/lookat."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._offscreen_camera = mujoco.MjvCamera()
        self._apply_camera(self._offscreen_camera, self.view_pose)
        self._renderer.update_scene(self.data, camera=self._offscreen_camera)
        self._draw_markers(self._renderer.scene)
        return self._renderer.render()

    def is_running(self):
        return not self.closed and (self.viewer is None or self.viewer.is_running())

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.completed_steps:
            elapsed = self._last_step_wall - self._first_step_wall
            simulated = self.completed_steps * self.physics_dt
            print("MUJOCO_PLAYBACK " + json.dumps({
                "physics_steps": self.completed_steps, "simulated_s": simulated,
                "rollout_wall_s": elapsed, "real_time_factor": simulated / elapsed,
                "mj_substeps": self.substeps, "viewer_enabled": self.viewer is not None,
                "viewer_max_fps": OPTIONS["max_fps"],
                "timing_window": "first physics step start to last physics step end; excludes startup",
            }), flush=True)
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        from . import omni_ui_tk

        omni_ui_tk.shutdown()


def is_running() -> bool:
    return _current is None or _current.is_running()


def shutdown():
    if _current is not None:
        _current.close()
