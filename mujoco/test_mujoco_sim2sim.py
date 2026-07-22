from pathlib import Path

import mujoco
import numpy as np
import pytest

import interactive
import play_direct_mujoco as sim2sim

CHECKPOINT = Path("/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt")


def make_env(**kwargs) -> sim2sim.Solo12Mujoco:
    return sim2sim.Solo12Mujoco(Path(__file__).with_name("solo12.xml"), **kwargs)


def test_model_contract():
    env = make_env()
    assert env.model.opt.timestep == sim2sim.PHYSICS_DT
    assert env.model.nu == 12
    assert np.isclose(env.model.body_mass.sum(), 3.028572, atol=1e-6)
    assert env.observation((0.5, 0.0, 0.0)).shape == (48,)
    assert np.allclose(env.data.qpos[env.joint_qpos], sim2sim.SAFE_Q)
    base_inertial_pos = env.model.body_ipos[env.base_id]
    assert np.allclose(base_inertial_pos, (-0.01, 0.0, 0.0)), "base COM must match the USD-authored value"


def test_visual_meshes_never_collide():
    """The Solo12 visual meshes are display-only; contact must come from the USD primitives."""
    env = make_env()
    model = env.model
    colliders = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0, (
                f"visual mesh {name} must not participate in contact"
            )
        elif model.geom_contype[i] or model.geom_conaffinity[i]:
            colliders[name] = int(model.geom_type[i])

    # ground plane + base box/2 rails + per leg (thigh box, knee sphere, foot cylinder)
    assert len(colliders) == 16, f"unexpected collider set: {sorted(colliders)}"
    for leg in ("FL", "FR", "RL", "RR"):
        assert colliders[f"{leg}_foot_geom"] == mujoco.mjtGeom.mjGEOM_CYLINDER, (
            "Solo12 feet are disks -- the contact geom must stay a cylinder, not a sphere"
        )


def test_actuator_matches_isaac_pd():
    env = make_env(kp=15.0, kd=0.5)
    assert np.allclose(env.model.actuator_gainprm[env.actuator_ids, 0], 15.0)
    assert np.allclose(env.model.actuator_biasprm[env.actuator_ids, 1], -15.0)
    assert np.allclose(env.model.actuator_biasprm[env.actuator_ids, 2], -0.5)
    assert np.allclose(env.model.actuator_forcerange[env.actuator_ids], [[-2.65, 2.65]] * 12)
    env.step(np.full(12, 10.0))  # saturating request
    assert np.max(np.abs(env.data.actuator_force)) <= sim2sim.EFFORT_LIMIT + 1e-9
    for _ in range(100):
        env.step(np.zeros(12))
    assert np.isfinite(env.data.qpos).all()
    assert np.max(np.abs(env.data.qvel[env.joint_dof])) < 20.0, "PD hold at the safe pose must stay stable"


def test_body_frame_velocity_observation():
    """Observation velocities must be R_body^T @ world COM velocities (Isaac contract), not ximat-frame."""
    env = make_env()
    rng = np.random.default_rng(7)
    quat = rng.normal(size=4); quat /= np.linalg.norm(quat)
    env.data.qpos[3:7] = quat
    env.data.qvel[:6] = rng.normal(size=6) * 0.5
    mujoco.mj_forward(env.model, env.data)

    h = 1e-6
    qpos_next = env.data.qpos.copy()
    mujoco.mj_integratePos(env.model, qpos_next, env.data.qvel, h)
    probe = mujoco.MjData(env.model)
    probe.qpos[:] = qpos_next
    mujoco.mj_forward(env.model, probe)

    rotation = sim2sim.quat_rotation(env.data.qpos[3:7])
    rotation_next = sim2sim.quat_rotation(qpos_next[3:7])
    lin_vel_w = (probe.xipos[env.base_id] - env.data.xipos[env.base_id]) / h
    omega_skew = (rotation_next - rotation) @ rotation.T / h
    ang_vel_w = np.array((omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]))

    obs = env.observation((0.0, 0.0, 0.0))
    np.testing.assert_allclose(obs[0:3], rotation.T @ lin_vel_w, atol=1e-4)
    np.testing.assert_allclose(obs[3:6], rotation.T @ ang_vel_w, atol=1e-4)
    np.testing.assert_allclose(obs[6:9], rotation.T @ (0, 0, -1.0), atol=1e-9)


def test_observation_layout():
    env = make_env()
    base = env.observation((0.0, 0.0, 0.0))
    with_cmd = env.observation((0.5, -0.3, 0.2))
    delta = with_cmd - base
    np.testing.assert_allclose(delta[9:12], (0.5, -0.3, 0.2))
    assert np.allclose(delta[:9], 0.0) and np.allclose(delta[12:], 0.0)
    action = np.linspace(-1.0, 1.0, 12)
    env.step(action)
    np.testing.assert_allclose(env.observation((0, 0, 0))[36:48], action)


def test_command_parser():
    assert sim2sim.parse_commands("0.5 0 0; 0 .3 0;") == [(0.5, 0.0, 0.0), (0.0, 0.3, 0.0)]


def test_env_override_consumption():
    overrides, ignored = sim2sim.consume_env_overrides([
        "env.kp=9.0", "env.kd=0.2", "env.command_lin_vel_x_range=[-0.6,0.6]",
        "env.tricky_terrain=False", "--disable_training_gain_sync",
    ])
    assert overrides == {"kp": 9.0, "kd": 0.2, "command_lin_vel_x_range": (-0.6, 0.6)}
    assert ignored == ["env.tricky_terrain=False", "--disable_training_gain_sync"]


def test_track_cmds_is_optional():
    args, unknown = sim2sim.build_parser().parse_known_args(["--checkpoint", "x"])
    assert args.track_cmds is None and args.kp is None and args.kd is None
    assert args.show_viewer_ui is False
    assert unknown == []


def test_body_force_state_pulse_hold_release():
    state = interactive.BodyForceState(dt=0.02)
    state.set(magnitude=4.0, azimuth_deg=90.0, elevation_deg=0.0, point_b=(0.1, 0.0, 0.0), duration_s=0.06)
    assert state.get_active_force() is None
    state.pulse()
    pulses = [state.get_active_force() for _ in range(5)]
    active = [force for force in pulses if force is not None]
    assert len(active) == 3  # 0.06 s / 0.02 s
    fx, fy, fz, px, _, _ = active[0]
    assert abs(fx) < 1e-9 and np.isclose(fy, 4.0) and abs(fz) < 1e-9 and np.isclose(px, 0.1)
    state.hold()
    assert all(state.get_active_force() is not None for _ in range(10))
    state.release()
    assert state.get_active_force() is None


def test_follow_camera_modes():
    camera = interactive.FollowCamera(dt=0.02, mode="side")
    for _ in range(300):  # converge the EMA on a static pose (identity quat -> heading 0)
        camera.update((1.0, 2.0, 0.4), (1.0, 0.0, 0.0, 0.0))
    cam = mujoco.MjvCamera()
    assert camera.apply(cam)
    assert np.isclose(cam.azimuth, 90.0, atol=0.5)  # side camera sits at -y, looking +y
    assert np.isclose(cam.elevation, -16.1, atol=0.5)
    assert np.allclose(cam.lookat, (1.0, 2.0, 0.75), atol=0.02)
    camera.mode = "front"
    camera.apply(cam)
    assert np.isclose(abs(cam.azimuth), 180.0, atol=0.5)
    camera.mode = "free"
    assert not camera.apply(cam)
    assert camera.cycle_mode() == "side"


def test_apply_body_force_world_mapping():
    env = make_env()
    env.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    interactive.apply_body_force(env, (0.0, 0.0, -5.0, 0.2247476, 0.0, 0.0))
    np.testing.assert_allclose(env.data.qfrc_applied[:3], (0.0, 0.0, -5.0), atol=1e-9)
    assert env.data.qfrc_applied[4] > 1.0  # downward force at the nose pitches the base
    interactive.apply_body_force(env, None)
    assert not env.data.qfrc_applied.any()


def test_force_arrow_geometry():
    env = make_env()
    scene = mujoco.MjvScene(env.model, 8)
    interactive.update_force_arrow(scene, env, (3.0, 0.0, 0.0, 0.2, 0.0, 0.0))
    assert scene.ngeom == 1
    interactive.update_force_arrow(scene, env, None)
    assert scene.ngeom == 0


def test_timestamped_run_directory(tmp_path):
    first = sim2sim.create_run_directory(tmp_path, "model_15008", "20260718_131500")
    second = sim2sim.create_run_directory(tmp_path, "model_15008", "20260718_131500")
    assert first == tmp_path / "model_15008" / "20260718_131500"
    assert second == tmp_path / "model_15008" / "20260718_131500_01"


def test_tracking_csv_logs_comparable_wandb_time_series(tmp_path):
    csv_path = tmp_path / "command_tracking.csv"
    csv_path.write_text(
        "time_s,cmd_vx,cmd_vy,cmd_wz,velocity_vx,velocity_vy,yaw_rate_wz,"
        "vxy_error_norm,wz_error_abs,reset,base_height_m,gravity_x_b\n"
        "0.02,0.5,-0.3,0.1,0.4,-0.1,-0.2,0.2236068,0.3,1,0.25,-0.9\n"
    )

    class Run:
        def __init__(self):
            self.metrics = []
            self.rows = []

        def define_metric(self, name, **kwargs):
            self.metrics.append((name, kwargs))

        def log(self, row):
            self.rows.append(row)

    run = Run()
    assert sim2sim.log_tracking_csv(run, csv_path) == 1
    assert ("tracking/error_vxy_norm_mps", {"step_metric": "tracking/time_s"}) in run.metrics
    assert run.rows[0]["tracking/time_s"] == pytest.approx(0.02)
    assert run.rows[0]["tracking/error_vx_abs_mps"] == pytest.approx(0.1)
    assert run.rows[0]["tracking/error_vy_abs_mps"] == pytest.approx(0.2)
    assert run.rows[0]["tracking/error_wz_abs_radps"] == pytest.approx(0.3)


def test_tracking_summary_distribution_fields():
    rows = [
        (0.02, 0.5, 0, 0, 0.4, 0, 0.1, 0.1, 0.1, 0, 0.35, -0.9),
        (0.04, 0.5, 0, 0, 0.3, 0, 0.2, 0.2, 0.2, 0, 0.34, -1.0),
    ]
    summary = sim2sim.summarize_rows(rows)
    assert summary["vxy_error_mean_mps"] == pytest.approx(0.15)
    assert summary["vxy_error_median_mps"] == pytest.approx(0.15)
    assert summary["vxy_error_p05_mps"] == pytest.approx(0.105)
    assert summary["vxy_error_p95_mps"] == pytest.approx(0.195)
    assert summary["vxy_error_std_mps"] == pytest.approx(0.05)
    assert summary["wz_error_std_radps"] == pytest.approx(0.05)
    assert summary["resets"] == 0
    assert summary["failures"] == 0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="real checkpoint unavailable")
def test_policy_normalization_matches_rsl_rl():
    import torch
    policy = sim2sim.Policy(CHECKPOINT)
    dims = [layer.in_features for layer in policy.layers] + [policy.layers[-1].out_features]
    assert dims == [48, 256, 128, 64, 12]
    obs = np.random.default_rng(0).normal(size=48)
    x = (torch.from_numpy(obs).float() - policy.obs_mean) / (policy.obs_std + sim2sim.OBS_NORM_EPS)
    for layer in policy.layers[:-1]:
        x = torch.nn.functional.elu(layer(x))
    np.testing.assert_allclose(policy(obs), policy.layers[-1](x).detach().numpy(), atol=1e-6)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="real checkpoint unavailable")
def test_rollout_stands_up_and_walks():
    policy = sim2sim.Policy(CHECKPOINT)
    env = make_env()
    command = (0.5, 0.0, 0.0)
    observation = env.observation(command)
    speeds, gravity_x = [], []
    for _ in range(int(10.0 / sim2sim.POLICY_DT)):
        env.step(policy(observation))
        observation = env.observation(command)
        vx, vy, _ = env.tracked_velocity()
        speeds.append(np.hypot(vx, vy))
        gravity_x.append(env.gravity_x_b())
        assert not env.base_hit_ground(), "base must not touch the ground during the rollout"
    assert min(gravity_x) < -0.6, "robot should reach the upright two-feet stance"
    assert np.mean(speeds[len(speeds) // 2:]) > 0.15, "robot should move once the command is active"
