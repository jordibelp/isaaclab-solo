"""Clean Solo12 IMU playback: clocks, rigid-body mechanics, layout and actor parity."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

import play_direct_mujoco as player


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/1109_qsc4btfx_best_model_curriculum_idx_2_4feet_IMU_SAC.pt")
IMU = dict(base_obs_from_imu=True, remove_root_lin_vel_b_from_actor_obs=True,
           include_lin_acc_steps_in_imu_obs=4)


def make_env(**overrides):
    return player.Solo12Mujoco(Path(__file__).with_name("solo12.xml"),
                               env_overrides={**IMU, **overrides})


def test_imu_cli_overrides_are_consumed():
    overrides, ignored = player.consume_env_overrides([
        "env.base_obs_from_imu=True", "env.remove_root_lin_vel_b_from_actor_obs=True",
        "env.include_lin_acc_steps_in_imu_obs=4", "env.base_imu_clip=False",
        "env.base_imu_gyro_clip=12.0", "env.base_imu_ekf_acc_clip=15.0",
    ])
    assert not ignored
    assert overrides == {**IMU, "base_imu_clip": False, "base_imu_gyro_clip": 12.,
                         "base_imu_ekf_acc_clip": 15.}


@pytest.mark.parametrize("token", [
    "env.base_obs_from_imu=maybe", "env.include_lin_acc_steps_in_imu_obs=-1",
    "env.include_lin_acc_steps_in_imu_obs=1.5", "env.base_imu_gyro_clip=nan",
    "env.base_imu_ekf_acc_clip=0",
])
def test_invalid_imu_settings_are_rejected(token):
    with pytest.raises(ValueError):
        player.consume_env_overrides([token])


@pytest.mark.parametrize("steps,include_velocity", [(0, False), (1, False), (4, False), (7, True)])
def test_imu_layout_and_reset(steps, include_velocity):
    env = make_env(include_lin_acc_steps_in_imu_obs=steps,
                   remove_root_lin_vel_b_from_actor_obs=not include_velocity)
    command = (.4, -.2, .1)
    obs = env.observation(command)
    start = 3 if include_velocity else 0
    assert obs.size == 45 + start + 3 * steps
    np.testing.assert_array_equal(obs[start+6:start+6+3*steps], 0.)
    np.testing.assert_array_equal(obs[start+6+3*steps:start+9+3*steps], command)
    env.step(np.linspace(-1, 1, 12))
    np.testing.assert_array_equal(env.observation(command)[-12:], env.action)
    before = env.observation(command)
    for _ in range(3):
        np.testing.assert_array_equal(env.observation(command), before)
    env.reset()
    np.testing.assert_array_equal(env.observation(command), obs)


def test_imu_samples_every_physics_step_oldest_first(monkeypatch):
    env = make_env(include_lin_acc_steps_in_imu_obs=7, base_imu_clip=False)
    original = mujoco.mj_step
    expected = []
    previous = np.zeros(6)

    def step_and_measure(model, data):
        nonlocal previous
        original(model, data)
        # Independent endpoint oracle using MuJoCo's COM velocity API.
        probe = mujoco.MjData(model)
        probe.qpos[:] = data.qpos
        probe.qvel[:] = data.qvel
        mujoco.mj_forward(model, probe)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, probe, mujoco.mjtObj.mjOBJ_BODY, env.base_id, velocity, 0)
        derivative = (velocity - previous) / player.PHYSICS_DT
        previous = velocity.copy()
        rotation = Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_matrix()
        lever = rotation @ (np.array((.01, 0, 0)) - model.body_ipos[env.base_id])
        acceleration = derivative[3:] + np.cross(derivative[:3], lever)
        acceleration += np.cross(velocity[:3], np.cross(velocity[:3], lever))
        expected.append(rotation.T @ acceleration)

    monkeypatch.setattr(mujoco, "mj_step", step_and_measure)
    for _ in range(2):
        env.step(np.linspace(-.6, .8, 12))
    assert len(expected) == 8
    np.testing.assert_allclose(env.observation((0, 0, 0))[6:27].reshape(7, 3), expected[-7:], atol=1e-11)


def test_freefall_acceleration_is_gravity_not_specific_force():
    env = make_env(base_imu_clip=False)
    env.data.qpos[2] = 10.
    env.model.dof_damping[:] = 0.
    env.model.dof_armature[:] = 0.
    env.data.qpos[3:7] = (1., 0., 0., 0.)
    mujoco.mj_forward(env.model, env.data)
    env.step(np.zeros(12))
    expected = np.tile((0., 0., -9.81), (4, 1))
    np.testing.assert_allclose(env.observation((0, 0, 0))[6:18].reshape(4, 3), expected, atol=1e-9)


def test_stationary_tilt_has_zero_acceleration_and_correct_gravity():
    env = make_env(include_lin_acc_steps_in_imu_obs=1)
    rotation = Rotation.from_euler("xyz", (.6, -.7, .9))
    env.data.qpos[3:7] = np.roll(rotation.as_quat(), 1)
    env._record_base_imu_sample()
    obs = env.observation((0, 0, 0))
    np.testing.assert_array_equal(obs[:3], 0.)
    np.testing.assert_allclose(obs[3:6], rotation.apply((0, 0, -1), inverse=True), atol=1e-14)
    np.testing.assert_array_equal(obs[6:9], 0.)


def test_rotating_base_uses_com_to_imu_lever_arm_and_clips():
    env = make_env(include_lin_acc_steps_in_imu_obs=1, base_imu_clip=False)
    # Fixed COM and constant world angular velocity: only centripetal acceleration remains.
    rotation = Rotation.from_euler("xyz", (.3, -.4, .7)).as_matrix()
    omega_b = np.array((1., 2., 3.))
    env.data.qpos[3:7] = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    env.data.qvel[3:6] = omega_b
    env.data.qvel[:3] = -rotation @ np.cross(omega_b, env.model.body_ipos[env.base_id])
    env._previous_imu_com_velocity = np.zeros(3)
    env._previous_imu_ang_velocity = rotation @ omega_b
    env._record_base_imu_sample()
    lever_b = np.array((.02, 0., 0.))
    expected = np.cross(omega_b, np.cross(omega_b, lever_b))
    np.testing.assert_allclose(env.observation((0, 0, 0))[6:9], expected, atol=1e-13)
    np.testing.assert_allclose(env.observation((0, 0, 0))[:3], omega_b, atol=1e-13)
    env.base_imu_clip = True
    env.base_imu_gyro_clip = .5
    env.base_imu_ekf_acc_clip = .1
    env._record_base_imu_sample()
    np.testing.assert_allclose(env.observation((0, 0, 0))[:3], .5)
    np.testing.assert_allclose(env.observation((0, 0, 0))[6:9], expected.clip(-.1, .1), atol=1e-13)


def test_clean_imu_matches_production_isaac_sensor_math():
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Solo12Env")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_read_base_imu_sample")
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)

    def rotate(q, v, inverse=False):
        return tensor(Rotation.from_quat(np.roll(q.numpy(), -1, axis=-1)).apply(v.numpy(), inverse=inverse))

    math_utils = SimpleNamespace(quat_apply=rotate,
                                 quat_apply_inverse=lambda q, v: rotate(q, v, inverse=True),
                                 quat_mul=lambda identity, q: q)
    namespace = {"torch": torch, "math_utils": math_utils}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    env = make_env(include_lin_acc_steps_in_imu_obs=1)
    rotation = Rotation.from_euler("xyz", (.6, -.7, .9))
    env.data.qpos[3:7] = np.roll(rotation.as_quat(), 1)
    env.data.qvel[:6] = (2., 3., -1., .5, -2., 3.)
    _, angular, linear = env._base_imu_kinematics()
    com_acceleration = np.array((4., -5., -3.))
    angular_acceleration = np.array((-2., 1., 6.))
    env._previous_imu_com_velocity = linear - player.PHYSICS_DT * com_acceleration
    env._previous_imu_ang_velocity = angular - player.PHYSICS_DT * angular_acceleration
    env._record_base_imu_sample()
    data = SimpleNamespace(
        body_link_quat_w=tensor(env.data.qpos[None, None, 3:7]),
        body_com_ang_vel_w=tensor(angular[None, None]),
        body_com_ang_acc_w=tensor(angular_acceleration[None, None]),
        body_com_lin_acc_w=tensor(com_acceleration[None, None]),
        body_com_pos_b=tensor(env.model.body_ipos[env.base_id][None, None]),
    )
    isaac = SimpleNamespace(
        _robot=SimpleNamespace(data=data), _base_wrench_body_ids=[0],
        _base_imu_offset_b=tensor(player.IMU_OFFSET_B[None]),
        _base_imu_heading_reference=tensor([[1., 0., 0., 0.]]),
        cfg=SimpleNamespace(imu_ekf_processed_inputs=True, base_imu_clip=True,
                            base_imu_gyro_clip=player.IMU_GYRO_CLIP,
                            base_imu_ekf_acc_clip=player.IMU_EKF_ACC_CLIP),
        _get_base_imu_orientation_obs=lambda q: rotate(q, tensor([[0., 0., -1.]]), inverse=True),
    )
    sample = namespace["_read_base_imu_sample"](isaac, corrupt=False).numpy()[0]
    actual = env.observation((0, 0, 0))
    np.testing.assert_allclose(actual[:9], np.r_[sample[:3], sample[6:9], sample[3:6]], atol=2e-12)


def test_imu_sampling_does_not_change_physics():
    imu = make_env()
    legacy = player.Solo12Mujoco(Path(__file__).with_name("solo12.xml"))
    for action in np.random.default_rng(10).uniform(-1., 1., (30, 12)):
        imu.step(action)
        legacy.step(action)
        np.testing.assert_array_equal(imu.data.qpos, legacy.data.qpos)
        np.testing.assert_array_equal(imu.data.qvel, legacy.data.qvel)


@pytest.mark.parametrize("overrides", [
    dict(base_obs_from_imu=False), dict(imu_ekf_processed_inputs=False),
])
def test_incompatible_imu_settings_fail_before_playback(overrides):
    with pytest.raises(ValueError):
        make_env(**overrides)


def test_layout_matches_production_isaac_observation_method():
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Solo12Env")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_simple_mlp_obs")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    env = make_env()
    env.step(np.linspace(-1., 1., 12))
    actual = env.observation((.5, -.2, .1))
    tensor = lambda x: torch.as_tensor(np.asarray(x)[None], dtype=torch.float64)
    isaac = SimpleNamespace(
        cfg=SimpleNamespace(action_obs_is_last_executed=False, base_obs_from_imu=True,
                            joint_pos_noise=(0., 0.), joint_vel_noise=(0., 0.)),
        _robot=SimpleNamespace(data=SimpleNamespace(joint_pos=tensor(env.data.qpos[env.joint_qpos]),
                                                    joint_vel=tensor(env.data.qvel[env.joint_dof]))),
        _joint_ids=list(range(12)), _q_offset_action_and_obs=tensor(player.SAFE_Q),
        _actions=tensor(env.action), _commands=tensor((.5, -.2, .1)),
        _imu_lin_acc_history=tensor(np.stack([env._imu_lin_acc_history]*2, axis=1)),
        _read_base_imu_sample=lambda **_: tensor(np.r_[actual[:3], np.zeros(3), actual[3:6]]),
    )
    expected = namespace["_get_simple_mlp_obs"](isaac, noisy=False, include_lin_vel=False)
    np.testing.assert_array_equal(actual, expected.numpy()[0])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="real IMU checkpoint unavailable")
def test_public_player_rejects_missing_observation_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["play_direct_mujoco.py", "--task=solo12-v0",
                                      f"--checkpoint={CHECKPOINT}", "--headless", "--track-cmds=0 0 0"])
    with pytest.raises(ValueError, match="checkpoint expects 57 inputs, but playback supplies 48"):
        player.main()


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="real IMU checkpoint unavailable")
def test_real_imu_actor_matches_production_sac_model():
    from tensordict import TensorDict
    from rsl_rl_sac.models import SACActorModel

    actor = player.Policy(CHECKPOINT)
    inputs = torch.randn(64, 57, generator=torch.Generator().manual_seed(9))
    obs = TensorDict({"policy": inputs}, batch_size=[64])
    reference = SACActorModel(obs, {"actor": ["policy"]}, "actor", 12,
                              hidden_dims=[1024, 512, 256], activation="swish", obs_normalization=True)
    reference.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["actor_state_dict"])
    reference.eval()
    with torch.inference_mode():
        # Compare batch-one inference on both sides, avoiding batch GEMM rounding differences.
        expected = np.concatenate([
            reference(TensorDict({"policy": x[None]}, batch_size=[1]), stochastic_output=False).numpy()
            for x in inputs
        ])
    actual = np.stack([actor(x.numpy()) for x in inputs])
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
