"""Physics checks for the Borinot MuJoCo backend (no policy, headless).

Run: ./isaaclab.sh -p -m pytest --confcutdir=mujoco mujoco/test_borinot_sim2sim.py -s -q
The module re-executes pytest once to load Isaac Sim's USD libraries (bootstrap.py).
"""

import importlib
from contextlib import nullcontext
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from borinot_sim2sim import bootstrap  # noqa: E402

bootstrap.bootstrap()

import mujoco  # noqa: E402
import torch  # noqa: E402

from borinot_sim2sim import backend  # noqa: E402

GRAVITY = np.array((0.0, 0.0, -9.81))


def make_env(kind, substeps=1, armature=None, **overrides):
    """Nominal single environment, built from the registered config class (no Hydra)."""
    backend.OPTIONS.update(headless=True, substeps=substeps)
    class_name = {"goto": "BorinotGoTo", "cmd_vel": "BorinotCmdVel", "goalkeeper": "BorinotGoalkeeper",
                  "goalkeeper_simple": "BorinotGoalkeeperSimple"}[kind]
    module = f"isaaclab_tasks.direct.borinot_{kind}.borinot_{kind}_env"
    Env = getattr(importlib.import_module(module), class_name + "Env")
    Cfg = getattr(importlib.import_module(module + "_cfg"), class_name + "EnvCfg")
    cfg = Cfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, "cpu", 0
    cfg.include_events_randomization = False
    if armature is not None:
        cfg.robot.actuators["arm"].armature = armature
    for key, value in overrides.items():
        setattr(cfg, key, value)
    env = Env(cfg)
    env.reset()
    return env


@pytest.fixture
def goto_env():
    env = make_env("goto", control_mode="wrench")
    yield env
    env.close()


def hold_still(env, height=1.5):
    """Robot at rest, no actuation: only gravity acts during the next step."""
    root = env.robot.data.default_root_state.clone()
    root[:, :3] = torch.tensor((0.0, 0.0, height))
    root[:, 7:] = 0.0
    env.robot.write_root_state_to_sim(root)
    env.robot.write_joint_state_to_sim(torch.zeros(1, 2), torch.zeros(1, 2))
    model, data = env.sim.backend.model, env.sim.backend.data
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0
    return model, data


def test_mass_properties_match_authored_stage(goto_env):
    """Every body keeps the task-edited USD mass, centre of mass and inertia tensor."""
    from pxr import UsdPhysics

    from borinot_sim2sim.usd_scene import _gf_quat, quat_to_matrix

    robot, stage = goto_env.robot, goto_env.sim.stage
    assert goto_env.robot_mass_kg == pytest.approx(2.993, abs=1e-5)  # short arm, inclusive base, no racket
    inertias = robot.root_physx_view.get_inertias()[0].reshape(-1, 3, 3).double().numpy()
    coms = robot.root_physx_view.get_coms()[0, :, :3].double().numpy()
    for index, name in enumerate(robot.body_names):
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}"))
        axes = quat_to_matrix(_gf_quat(api.GetPrincipalAxesAttr().Get()))
        authored = axes @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ axes.T
        assert robot.data.default_mass[0, index] == pytest.approx(api.GetMassAttr().Get(), rel=1e-6)
        np.testing.assert_allclose(coms[index], api.GetCenterOfMassAttr().Get(), atol=1e-6)
        np.testing.assert_allclose(inertias[index], authored, atol=1e-8)


def test_root_state_round_trip(goto_env):
    """Isaac root state: link pose and centre-of-mass velocity."""
    robot = goto_env.robot
    state = robot.data.default_root_state.clone()
    state[0, :7] = torch.tensor((0.3, -0.2, 1.4, 0.9238795, 0.0, 0.3826834, 0.0))
    state[0, 7:] = torch.tensor((0.5, -0.4, 0.3, 0.7, -0.2, 1.1))
    robot.write_root_state_to_sim(state)
    torch.testing.assert_close(robot.data.root_state_w, state, atol=1e-5, rtol=0)
    expected_b = (torch.tensor([[0.7071068, 0.0, -0.7071068], [0.0, 1.0, 0.0], [0.7071068, 0.0, 0.7071068]])
                  @ state[0, 10:])
    torch.testing.assert_close(robot.data.root_ang_vel_b[0], expected_b, atol=1e-5, rtol=0)


def test_free_fall_accelerations_and_zero_joint_wrench(goto_env):
    """Uniform gravity and no actuation: every body accelerates at g, joints carry no load."""
    hold_still(goto_env)
    robot, sim = goto_env.robot, goto_env.sim.backend
    robot.data.body_incoming_joint_wrench_b  # start measuring
    sim.step()
    np.testing.assert_allclose(robot.data.body_lin_acc_w[0].numpy(), np.tile(GRAVITY, (robot.num_bodies, 1)),
                               atol=1e-3)
    np.testing.assert_allclose(robot.data.body_incoming_joint_wrench_b[0].numpy(), 0.0, atol=1e-4)


def test_joint_wrench_matches_newton_euler(goto_env):
    """Incoming hip wrench = what the base must apply for the arm's measured motion."""
    model, data = hold_still(goto_env)
    robot, sim = goto_env.robot, goto_env.sim.backend
    ff = robot._ff_actuators
    data.ctrl[ff] = (0.8, -0.5)  # joint torques
    data.qvel[robot._root_dof:robot._root_dof + 6] = (0.2, -0.1, 0.3, 0.4, -0.6, 0.5)
    data.qvel[robot._dof_adr] = (1.5, -2.0)
    data.xfrc_applied[robot._body_ids[0]] = (1.0, -2.0, 25.0, 0.1, 0.05, -0.2)
    mujoco.mj_forward(model, data)
    sim.track_joint_wrench, sim.substeps = True, 1
    sim.joint_wrench.fill(0.0)
    for stage in (mujoco.mj_fwdActuation, mujoco.mj_fwdAcceleration, mujoco.mj_fwdConstraint):
        stage(model, data)  # the forward half of Backend.step, before integration
    sim._accumulate(1.0)
    thigh = robot._body_ids[robot.find_bodies("flying_arm_2__link1")[0][0]]
    subtree = [b for b in robot._body_ids if b != robot._body_ids[0]]
    force = np.zeros(3)
    centre = np.zeros(3)
    joint_pos, joint_rotation = next((p, r) for k, p, r in robot._joint_frames if robot._body_ids[k] == thigh)
    anchor = data.xpos[thigh] + data.xmat[thigh].reshape(3, 3) @ joint_pos
    moment = np.zeros(3)
    acceleration, velocity = np.zeros(6), np.zeros(6)
    for body in subtree:
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, body, acceleration, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        rotation = data.ximat[body].reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
        linear = model.body_mass[body] * acceleration[3:]  # objectAcceleration includes -g
        omega = velocity[:3]
        force += linear
        moment += np.cross(data.xipos[body] - anchor, linear) + inertia @ acceleration[:3] + np.cross(
            omega, inertia @ omega)
        centre += model.body_mass[body] * data.xipos[body]
    frame = data.xmat[thigh].reshape(3, 3) @ joint_rotation
    measured = sim.joint_wrench[thigh]
    np.testing.assert_allclose(measured[:3], frame.T @ force, atol=1e-6)
    np.testing.assert_allclose(measured[3:], frame.T @ moment, atol=1e-6)
    # About the hip axis (joint-frame X): the motor torque minus what accelerates the
    # joint-space armature, which no link carries (the same model as PhysX armature).
    hip_dof = robot._dof_adr[robot.find_joints("flying_arm_2__j_bl_link1")[0][0]]
    transmitted = data.qfrc_actuator[hip_dof] + data.qfrc_passive[hip_dof] - model.dof_armature[hip_dof] * data.qacc[
        hip_dof]
    assert measured[3] == pytest.approx(transmitted, abs=1e-6)


def test_ball_contact_force_and_restitution():
    """A resting ball reports its weight from the floor; drops bounce near the PhysX restitution."""
    env = make_env("goalkeeper", substeps=10)
    try:
        ball, sim = env.ball, env.sim.backend
        mass = float(ball.root_physx_view.get_masses()[0, 0])
        floor_column = 5  # paddle, base, arm1, arm2, goal, floor
        state = ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor((1.5, 0.8, env.cfg.ball_radius_m))
        state[0, 7:] = 0.0
        ball.write_root_state_to_sim(state)
        for _ in range(100):
            sim.step()
        weight = mass * 9.81
        np.testing.assert_allclose(env.ball_contact.data.net_forces_w[0, 0].numpy(), (0, 0, weight), atol=2e-3)
        np.testing.assert_allclose(env.ball_contact.data.force_matrix_w[0, 0, floor_column].numpy(), (0, 0, weight),
                                   atol=2e-3)
        results = {}
        for restitution in (0.35, 0.5, 0.65):
            materials = ball.root_physx_view.get_material_properties()
            materials[..., 2] = restitution
            ball.root_physx_view.set_material_properties(materials, torch.tensor([0]))
            state[0, :3] = torch.tensor((1.5, 0.8, 0.25))
            state[0, 7:10] = torch.tensor((0.0, 0.0, -5.0))
            ball.write_root_state_to_sim(state)
            impact = rebound = 0.0
            for _ in range(60):
                before = float(ball.data.root_lin_vel_w[0, 2])
                sim.step()
                impact = max(impact, -before)
                rebound = max(rebound, float(ball.data.root_lin_vel_w[0, 2]))
            results[restitution] = rebound / impact
        print("restitution target -> measured:", {k: round(v, 3) for k, v in results.items()})
        for target, measured in results.items():
            assert measured == pytest.approx(target, abs=0.03)
    finally:
        env.close()


def test_marker_shapes():
    """Every marker prototype used by the play scripts becomes a MuJoCo decoration."""
    from isaaclab.sim import ConeCfg, CylinderCfg, PreviewSurfaceCfg, SphereCfg

    from borinot_sim2sim.markers import VisualizationMarkers

    backend.OPTIONS.update(headless=True)
    sim = backend.Backend(0.005, (0.0, 0.0, -9.81))
    material = PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0), emissive_color=(1.0, 0.5, 0.0))
    markers = VisualizationMarkers(type("Cfg", (), {"markers": {
        "sphere": SphereCfg(radius=0.05, visual_material=material),
        "shaft": CylinderCfg(radius=1.0, height=1.0, visual_material=material),
        "head": ConeCfg(radius=1.0, height=1.0, visual_material=material)}})())
    markers.visualize(translations=torch.zeros(3, 3), scales=torch.full((3, 3), 0.1), marker_indices=[0, 1, 2])
    kinds = [g[0] for g in markers.geoms()]
    assert kinds == [mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_ARROW]
    assert all(math.isclose(g[5], 0.8) for g in markers.geoms())
    sim.markers.clear()


def test_reset_clears_applied_external_wrench(goto_env):
    robot, sim = goto_env.robot, goto_env.sim.backend
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=torch.tensor([[[1.0, 2.0, 3.0]]]), body_ids=[0])
    robot.write_data_to_sim()
    assert np.linalg.norm(sim.data.xfrc_applied) > 0
    robot.reset()
    robot.write_data_to_sim()
    np.testing.assert_array_equal(sim.data.xfrc_applied[robot._body_ids], 0.0)


def test_environment_close_closes_backend(goto_env):
    sim = goto_env.sim.backend
    goto_env.close()
    goto_env.close()  # idempotent, including the fixture teardown
    assert sim.closed
    assert not sim.is_running()


@pytest.mark.parametrize("substeps", [0, -1])
def test_invalid_substeps(substeps):
    backend.OPTIONS["substeps"] = substeps
    with pytest.raises(ValueError, match="positive integer"):
        backend.Backend(0.005, GRAVITY)


@pytest.mark.parametrize("kind", ["cmd_vel", "goto", "goalkeeper", "goalkeeper_simple"])
@pytest.mark.parametrize("mode", [
    "thrust_and_torque_control", "thrust_and_PD_joint_pos_control", "PI_rotors_PD_joint_pos_control",
    "indi", "wrench", "model_wrench", "model_wrench_indi", "model_wrench_yawless",
    "model_wrench_yawless_indi", "cbrt", "cbrt_indi",
])
def test_task_controller_rollout(kind, mode):
    """Run actual task observations, controller, physics, reward and reset for every contract."""
    options = {"control_mode": mode}
    if kind in ("cmd_vel", "goto"):
        options.update(obs_include_base_lin_acc=True, obs_include_base_lin_acc_steps=2)
    env = make_env(kind, substeps=10 if "goalkeeper" in kind else 1, **options)
    try:
        for _ in range(3):
            observation, reward, terminated, truncated, _ = env.step(torch.zeros(1, env.cfg.action_space))
            assert observation["policy"].shape == (1, env.cfg.observation_space)
            assert observation["critic"].shape == (1, env.cfg.state_space)
            assert all(torch.isfinite(value).all() for value in observation.values())
            assert torch.isfinite(reward).all()
            assert terminated.shape == truncated.shape == (1,)
        env.reset()
        assert torch.isfinite(env._get_observations()["policy"]).all()
    finally:
        env.close()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk controls need an X display")
@pytest.mark.parametrize("value", [1.5, 12.345, -8.901])
def test_slider_does_not_rewrite_exact_field(value):
    """Deferred Tk scale callbacks must not round or clamp a typed goal/deadline."""
    from borinot_sim2sim import omni_ui_tk as ui

    try:
        window = ui.Window("Borinot controls test", width=300, height=100)
        with window.frame:
            with ui.VStack():
                model = ui.SimpleFloatModel(1.5)
                slider = ui.FloatSlider(model, min=-0.1, max=5.9)
        ui.pump()
        assert model.as_float == 1.5
        model.set_value(value)
        ui.pump()
        assert model.as_float == value
        slider.widget.set(2.3)  # exercises the deferred Tk command used by dragging
        ui.pump()
        assert model.as_float == pytest.approx(float(slider.widget.get()))
    finally:
        ui.shutdown()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk controls need an X display")
def test_collapsible_frame_preserves_inputs_and_rebuilds_rows():
    from borinot_sim2sim import omni_ui_tk as ui

    try:
        window = ui.Window("Sequence widget test", width=400, height=300)
        builds = []

        def build_row():
            builds.append(ui.Label(f"Row {len(builds) + 1}"))

        with window.frame:
            with ui.CollapsableFrame("Optional sequence", collapsed=True) as section:
                field = ui.FloatField(ui.SimpleFloatModel(2.0))
                rows = ui.Frame(build_fn=build_row)
        ui.pump()
        assert section.collapsed and not section.inner.winfo_ismapped()
        section._header.invoke()
        ui.pump()
        assert not section.collapsed and section.inner.winfo_ismapped()
        field.widget.delete(0, "end")
        field.widget.insert(0, "2.75")
        field.widget.event_generate("<FocusOut>")
        section.collapsed = True
        section.collapsed = False
        ui.pump()
        assert field.model.as_float == 2.75
        old_label = builds[0]
        old_label.text = "Queued update before rebuild"
        rows.rebuild()  # callback is outside the original container context
        assert not old_label.widget.winfo_exists()
        builds[1].text = "Fresh row"
        ui.pump()
        assert builds[1].widget.cget("text") == "Fresh row"
        assert len(rows.inner.winfo_children()) == 1 and not ui._parents
    finally:
        ui.shutdown()


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Tk controls need an X display")
def test_live_goto_panel_keeps_drafts_and_builds_editable_sequence_rows(goto_env):
    from borinot_sim2sim import omni_ui_tk as ui
    from isaaclab_tasks.direct.borinot_goto.playback import BorinotGoToPlayEnv
    from isaaclab_tasks.direct.borinot_goto.playback_ui import GoalControls

    def button(widget, text):
        for child in widget.winfo_children():
            if child.winfo_class() == "Button" and child.cget("text") == text:
                return child
            found = button(child, text)
            if found is not None:
                return found
        return None

    goto_env.close()  # the adapter owns one simulation context at a time
    env = BorinotGoToPlayEnv(goto_env.cfg, goal=(0., 0., 1.5))
    panel = None
    try:
        env.reset()
        panel = GoalControls(env, (0., 0., 1.5), None, [(-6., 6.)] * 4, show_ui=True)
        ui.pump()
        target = env._goal_position_w.clone()
        panel._models[0].set_value(3.)
        ui.pump()
        assert not panel.go_requested and torch.equal(target, env._goal_position_w)
        panel.sequence_editor._header.invoke()
        button(panel.window.body, "Add waypoint from draft").invoke()
        panel._models[0].set_value(-3.)
        button(panel.window.body, "Add waypoint from draft").invoke()
        ui.pump()
        assert len(panel._sequence_models) == 10
        panel._sequence_models[4].set_value(2.)
        panel.remove_waypoint(1)
        assert len(panel._sequence_models) == 5 and panel.waypoints[0]["values"][4] == 2.
        button(panel.window.body, "Add waypoint from draft").invoke()
        button(panel.window.body, "Start sequence").invoke()
        wrapper = SimpleNamespace(get_observations=env._get_observations)
        observation, reset = panel.apply_requests(wrapper, env._get_observations(), 0.)
        assert not reset and panel.sequence_phase == "flying"
        assert env.play_goal.tolist() == [3., 0., 1.5]
        assert torch.isfinite(observation["policy"]).all()
        button(panel.window.body, "Stop sequence").invoke()
        assert panel._sequence is None and env.play_goal.tolist() == [3., 0., 1.5]
        button(panel.window.body, "Go to goal").invoke()
        panel._models[0].set_value(4.)  # later edits do not change the queued Go
        panel.apply_requests(wrapper, observation, 0.)
        assert env.play_goal.tolist() == [-3., 0., 1.5]
        panel.update(0.)
        ui.pump()
    finally:
        if panel is not None:
            panel.close()
        env.close()


@pytest.mark.parametrize("kind", ["cmd_vel", "goto", "goalkeeper", "goalkeeper_simple"])
@pytest.mark.parametrize("armature", [None, {
    "flying_arm_2__j_bl_link1": 0.0007, "flying_arm_2__j_link1_link2": 0.0012,
}])
def test_actuator_armature_matches_config_and_mass_matrix(kind, armature):
    """The actuator setting adds joint-space inertia, not an extra link/rotor mass."""
    env = make_env(kind, armature=armature)
    try:
        robot, sim = env.robot, env.sim.backend
        model, data = sim.model, sim.data
        expected = np.full(2, 0.00036207) if armature is None else np.array([
            armature[name] for name in robot.joint_names])
        np.testing.assert_array_equal(model.dof_armature[robot._dof_adr], expected)
        with_armature = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, with_armature, data.qM)
        model.dof_armature[robot._dof_adr] = 0.0
        sim.forward()
        without_armature = np.zeros_like(with_armature)
        mujoco.mj_fullM(model, without_armature, data.qM)
        delta = np.zeros_like(with_armature)
        delta[robot._dof_adr, robot._dof_adr] = expected
        np.testing.assert_allclose(with_armature - without_armature, delta, atol=1e-14, rtol=0)
        model.dof_armature[robot._dof_adr] = expected
        sim.forward()
        env.reset()
        env.step(torch.zeros(1, env.cfg.action_space))
        np.testing.assert_array_equal(model.dof_armature[robot._dof_adr], expected)
    finally:
        env.close()


def test_viewer_refresh_preserves_model_updates(goto_env, monkeypatch):
    """State-only refreshes are cheap; model edits must still reach the passive viewer."""
    sim = goto_env.sim.backend
    calls, now = [], [0.0]
    monkeypatch.setattr(backend, "time", SimpleNamespace(perf_counter=lambda: now[0]))
    sim.viewer = SimpleNamespace(
        cam=mujoco.MjvCamera(), user_scn=mujoco.MjvScene(sim.model, maxgeom=100),
        lock=nullcontext, is_running=lambda: True, close=lambda: None,
        sync=lambda state_only=False: calls.append(state_only),
    )
    sim.render()
    now[0] = 0.001
    sim.render()  # excess render calls do not trigger another model/state copy
    assert calls == [False]
    now[0] = 1.0
    sim.render()
    assert calls == [False, True]
    masses = goto_env.robot.root_physx_view.get_masses()
    goto_env.robot.root_physx_view.set_masses(masses, torch.tensor([0]))
    now[0] = 2.0
    sim.render()
    assert calls == [False, True, False]
