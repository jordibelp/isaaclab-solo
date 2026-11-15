"""Opt-in Q11 transport corruption on hardware-facing Solo12 acceleration inputs."""

from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(
    headless=True, kit_args="--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false"
).app

import gymnasium as gym
import hydra
import torch
from omegaconf import OmegaConf

import isaaclab.utils.math as math_utils
from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_strings_with_slices
from isaaclab_tasks.direct.solo12 import solo12_env as module
from isaaclab_tasks.direct.solo12.solo12_env_cfg import (
    Solo12TwoFeetEnvCfg, Solo12TwoFeetTorqueEnvCfg, Solo12TwoFeetImuTeacherEnvCfg,
    Solo12TwoFeetImuStudentDaggerEnvCfg, Solo12TwoFeetImuStudentRlEnvCfg,
)
from isaaclab_tasks.utils.hydra import register_task_to_hydra

TASKS = [
    "solo12-two-feet", "solo12-two-feet-torque", "solo12-two-feet-IMU-teacher",
    "solo12-two-feet-IMU-teacher-rl", "solo12-two-feet-IMU-student-dagger", "solo12-two-feet-IMU-student-rl",
]


@pytest.mark.parametrize("task", TASKS)
def test_all_two_feet_tasks_expose_default_off_hydra_override(task):
    cfg, _ = register_task_to_hydra(task, None)
    assert cfg.corrupt_imu_lin_acc_int16_wrap is False
    # Hydra registration serializes spaces in-place; compare against a fresh default config.
    baseline = type(cfg)()
    dims = (baseline.observation_space, baseline.state_space)
    with hydra.initialize(version_base="1.3", config_path=None):
        merged = hydra.compose(config_name=task, overrides=["env.corrupt_imu_lin_acc_int16_wrap=True"])
    values = replace_strings_with_slices(OmegaConf.to_container(merged, resolve=True))
    cfg.from_dict(values["env"])
    cfg = replace_strings_with_env_cfg_spaces(cfg)
    cfg.refresh_runtime_dependent_config()
    assert cfg.corrupt_imu_lin_acc_int16_wrap is True
    assert cfg.to_dict()["corrupt_imu_lin_acc_int16_wrap"] is True
    assert (cfg.observation_space, cfg.state_space) == dims


@pytest.mark.parametrize("value", [0, 1, "True", None])
def test_flag_requires_a_boolean(value):
    cfg = Solo12TwoFeetEnvCfg()
    cfg.corrupt_imu_lin_acc_int16_wrap = value
    with pytest.raises(ValueError, match="corrupt_imu_lin_acc_int16_wrap.*boolean"):
        cfg.refresh_runtime_dependent_config()


def test_raw_specific_force_cannot_use_the_processed_si_transport_model():
    cfg = Solo12TwoFeetImuStudentRlEnvCfg()
    cfg.corrupt_imu_lin_acc_int16_wrap = True
    cfg.imu_ekf_processed_inputs = False
    with pytest.raises(ValueError, match="imu_ekf_processed_inputs"):
        cfg.refresh_runtime_dependent_config()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_boundaries_truncation_and_multiple_wraps_match_integer_oracle(device, dtype):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    lsb = 1 / 2048
    values = torch.tensor([
        0., 17., -17., 16., -16., 16.-lsb, -16.-lsb, 49., -49., 50., -50.,
        .75*lsb, -.75*lsb, 16.-.5*lsb, -16.-.5*lsb,
    ], dtype=dtype, device=device)
    bits = [int(float(x)*2048) & 0xffff for x in values.cpu()]
    expected = torch.tensor([(v if v < 0x8000 else v-0x10000)/2048 for v in bits], dtype=dtype, device=device)
    before = values.clone()
    state = torch.get_rng_state()
    actual = module._wrap_imu_lin_acc_int16(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(values, before, rtol=0, atol=0)
    assert actual.dtype == dtype and actual.device == values.device
    assert torch.equal(torch.get_rng_state(), state)


def sensor_shell(cls=Solo12TwoFeetEnvCfg, *, wrap=True, n=2, **overrides):
    cfg = cls()
    if cfg.policy_model == "simple_mlp":
        cfg.base_obs_from_imu = True
        cfg.remove_root_lin_vel_b_from_actor_obs = True
        cfg.include_lin_acc_steps_in_imu_obs = 4
    cfg.noisy_actor_obs = False
    cfg.noisy_critic_obs = False
    cfg.corrupt_imu_lin_acc_int16_wrap = wrap
    for name, value in overrides.items():
        setattr(cfg, name, value)
    cfg.refresh_runtime_dependent_config()
    env = object.__new__(module.Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    env.scene = SimpleNamespace(num_envs=n)
    env.sim = SimpleNamespace(device="cpu")
    identity = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    env._robot = SimpleNamespace(data=SimpleNamespace(
        body_link_quat_w=identity[:, None], body_com_ang_vel_w=torch.zeros(n, 1, 3),
        body_com_ang_acc_w=torch.zeros(n, 1, 3), body_com_lin_acc_w=torch.zeros(n, 1, 3),
        body_com_pos_b=torch.zeros(n, 1, 3), root_lin_vel_b=torch.zeros(n, 3),
        root_ang_vel_b=torch.zeros(n, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
        joint_pos=torch.zeros(n, 12), joint_vel=torch.zeros(n, 12),
    ))
    env._base_wrench_body_ids = [0]
    env._base_imu_heading_reference = identity.clone()
    env._base_imu_offset_b = torch.tensor([[.01, 0., 0.]]).repeat(n, 1)
    env._base_imu_gravity_bias_w = torch.tensor([[0., 0., 9.81]])
    env._gravity_direction_w = torch.tensor([[0., 0., -1.]])
    env._imu_noise_scale = 1.
    env._base_imu_bias = torch.zeros(n, 6)
    env._base_imu_orientation_bias_rpy = torch.zeros(n, 3)
    env._base_imu_enabled = True
    env._base_imu_history_len = cfg.base_imu_history_length if cfg.imu_raw_inputs else 0
    env._base_imu_history = torch.zeros(n, env._base_imu_history_len, cfg.base_imu_history_sample_dim)
    steps = cfg.include_lin_acc_steps_in_imu_obs
    env._teacher_lin_acc_history = torch.zeros(n, steps if cfg.policy_model.startswith("base_imu_") else 0, 3)
    env._imu_lin_acc_history = torch.zeros(n, steps if cfg.base_obs_from_imu else 0, 2, 3)
    env._joint_ids = slice(None)
    env._q_offset_action_and_obs = torch.zeros(n, 12)
    env._actions = torch.zeros(n, 12)
    env._applied_actions = torch.zeros(n, 12)
    env._commands = torch.zeros(n, 3)
    env._base_push_forces_b = torch.zeros(n, 1, 3)
    env._feet_body_ids = [0, 1, 2, 3]
    env._contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 4, 3)))
    return env


@pytest.mark.parametrize("clip", [True, False])
@pytest.mark.parametrize("noise_controls", [
    {"noisy_actor_obs": False}, {"noisy_actor_obs": True, "noisy_imu": False},
    {"noisy_actor_obs": True, "imu_noise_scale": 0.},
])
def test_wrap_precedes_clip_and_does_not_depend_on_gaussian_noise(clip, noise_controls):
    env = sensor_shell(base_imu_clip=clip, **noise_controls)
    env._imu_noise_scale = env.cfg.imu_noise_scale
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([17., -17., 50.])
    observed = env._read_base_imu_sample(corrupt=True)
    clean = env._read_base_imu_sample(corrupt=False)
    torch.testing.assert_close(observed[:, 3:6], torch.tensor([[-15., 15., -14.]]).repeat(2, 1))
    expected_clean = torch.tensor([[17., -17., 50.]]).repeat(2, 1)
    if clip:
        expected_clean = expected_clean.clamp(-env.cfg.base_imu_ekf_acc_clip, env.cfg.base_imu_ekf_acc_clip)
    torch.testing.assert_close(clean[:, 3:6], expected_clean)
    torch.testing.assert_close(observed[:, :3], clean[:, :3])
    torch.testing.assert_close(observed[:, 6:], clean[:, 6:])


def test_wrapping_follows_sensor_bias_and_noise():
    env = sensor_shell(noisy_actor_obs=True, base_imu_gyro_noise_std=0., base_imu_ekf_acc_noise_std=0.)
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([15., -15., 0.])
    env._base_imu_bias[:, 3:6] = torch.tensor([2., -2., .25])
    sample = env._read_base_imu_sample(corrupt=True)
    torch.testing.assert_close(sample[:, 3:6], torch.tensor([[-15., 15., .25]]).repeat(2, 1))


def test_transport_operates_in_body_axes_not_world_z():
    env = sensor_shell()
    angle = torch.full((2,), torch.pi/2)
    zeros = torch.zeros(2)
    env._robot.data.body_link_quat_w[:, 0] = math_utils.quat_from_euler_xyz(zeros, angle, zeros)
    env._robot.data.body_com_lin_acc_w[:, 0, 2] = 17.
    observed = env._read_base_imu_sample(corrupt=True)
    assert torch.all(observed[:, 3] > 14.99)  # world +Z is body -X, so -17 wraps to +15.
    assert torch.all(observed[:, 4:6].abs() < .001)


def test_false_keeps_unquantized_values_and_wrap_only_changes_acceleration():
    env = sensor_shell(wrap=False, base_imu_clip=False)
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([17.12345, -19.23456, .123456])
    env._robot.data.body_com_ang_vel_w[:, 0] = torch.tensor([.3, -.7, 1.1])
    env._robot.data.body_link_quat_w[:, 0] = math_utils.quat_from_euler_xyz(
        torch.full((2,), .3), torch.full((2,), -.4), torch.full((2,), .6)
    )
    state = torch.get_rng_state()
    clean = env._read_base_imu_sample(corrupt=False)
    torch.testing.assert_close(env._read_base_imu_sample(corrupt=True), clean, rtol=0, atol=0)
    env.cfg.corrupt_imu_lin_acc_int16_wrap = True
    wrapped = env._read_base_imu_sample(corrupt=True)
    torch.testing.assert_close(wrapped[:, 3:6], module._wrap_imu_lin_acc_int16(clean[:, 3:6]), rtol=0, atol=0)
    torch.testing.assert_close(wrapped[:, :3], clean[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(wrapped[:, 6:], clean[:, 6:], rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("cls", [Solo12TwoFeetEnvCfg, Solo12TwoFeetTorqueEnvCfg])
def test_mlp_actor_keeps_wrapped_history_when_play_disables_noise_but_critic_is_clean(cls):
    env = sensor_shell(cls)
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([17., -17., 50.])
    for _ in range(4):
        env._record_base_imu_history_sample()
    obs = env._get_observations()
    torch.testing.assert_close(obs["policy"][:, 6:18], torch.tensor([[-15., 15., -14.]*4]).repeat(2, 1))
    limit = env.cfg.base_imu_ekf_acc_clip
    torch.testing.assert_close(obs["critic"][:, 9:21], torch.tensor([[limit, -limit, limit]*4]).repeat(2, 1))
    # Observation reads must not append or re-wrap history.
    history = env._imu_lin_acc_history.clone()
    torch.testing.assert_close(env._get_observations()["policy"], obs["policy"], rtol=0, atol=0)
    torch.testing.assert_close(env._imu_lin_acc_history, history, rtol=0, atol=0)


@pytest.mark.parametrize("cls", [Solo12TwoFeetImuStudentDaggerEnvCfg, Solo12TwoFeetImuStudentRlEnvCfg])
def test_student_histories_wrap_but_teacher_labels_and_privileged_acceleration_stay_clean(cls):
    env = sensor_shell(cls)
    for step in range(1, 7):
        env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([17.+step, -17.-step, 50.+step])
        env._record_base_imu_history_sample()
    clean = env._get_base_origin_lin_acc_b()
    expected = module._wrap_imu_lin_acc_int16(clean)
    torch.testing.assert_close(env._base_imu_history[:, -1, 27:30], expected)
    torch.testing.assert_close(env._teacher_lin_acc_history[:, -1], clean)
    obs = env._get_observations()
    torch.testing.assert_close(obs["critic"][:, 45:57], env._teacher_lin_acc_history.flatten(1))
    if cls is Solo12TwoFeetImuStudentDaggerEnvCfg:
        torch.testing.assert_close(obs["policy"][:, :67], obs["critic"])


def test_privileged_teacher_is_not_a_hardware_imu_sensor():
    env = sensor_shell(Solo12TwoFeetImuTeacherEnvCfg)
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.tensor([17., -17., 50.])
    env._record_base_imu_history_sample()
    wrapped_obs = env._get_observations()
    env.cfg.corrupt_imu_lin_acc_int16_wrap = False
    clean_obs = env._get_observations()
    torch.testing.assert_close(wrapped_obs["policy"], clean_obs["policy"], rtol=0, atol=0)
    torch.testing.assert_close(wrapped_obs["critic"], clean_obs["critic"], rtol=0, atol=0)


def test_vectorized_sensor_sampling_for_ten_thousand_envs():
    env = sensor_shell(n=10_000, base_imu_clip=False)
    env._robot.data.body_com_lin_acc_w[:, 0] = torch.arange(30_000).reshape(10_000, 3) / 100 - 150
    clean = env._read_base_imu_sample(corrupt=False)
    expected = module._wrap_imu_lin_acc_int16(clean[:, 3:6])
    state = torch.get_rng_state()
    actual = env._read_base_imu_sample(corrupt=True)
    torch.testing.assert_close(actual[:, 3:6], expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), state)


def _check_physx_sampling_actor_inputs_and_resets(task, cls):
    cfg = cls()
    cfg.corrupt_imu_lin_acc_int16_wrap = True
    cfg.scene.num_envs = 2
    cfg.seed = 71
    cfg.events = None
    cfg.include_events_randomization = False
    cfg.curriculum_two_feet = False
    cfg.tricky_terrain = False
    cfg.terrain.terrain_type = "plane"
    cfg.terrain.terrain_generator = None
    cfg.flat_terrain_grid_enabled = False
    cfg.noisy_actor_obs = cfg.noisy_imu = False
    cfg.base_imu_clip = False  # Expose the pre-transport oracle; separate tests cover clipping.
    if cfg.policy_model == "simple_mlp":
        cfg.base_obs_from_imu = True
        cfg.remove_root_lin_vel_b_from_actor_obs = True
        cfg.include_lin_acc_steps_in_imu_obs = 4
        cfg.noisy_critic_obs = False
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    try:
        env.reset()
        record = raw._record_base_imu_history_sample
        oracle = []

        def record_and_check():
            clean_sensor = raw._read_base_imu_sample(corrupt=False)[:, 3:6]
            wrapped = module._wrap_imu_lin_acc_int16(clean_sensor)
            origin = raw._get_base_origin_lin_acc_b()
            record()
            oracle.append(wrapped.clone())
            if raw.cfg.base_obs_from_imu:
                torch.testing.assert_close(raw._imu_lin_acc_history[:, -1, 0], wrapped, rtol=0, atol=0)
                torch.testing.assert_close(raw._imu_lin_acc_history[:, -1, 1], clean_sensor, rtol=0, atol=0)
            if raw._base_imu_history_len:
                torch.testing.assert_close(raw._base_imu_history[:, -1, 27:30], wrapped, rtol=0, atol=0)
                torch.testing.assert_close(raw._teacher_lin_acc_history[:, -1], origin, rtol=0, atol=0)

        raw._record_base_imu_history_sample = record_and_check
        for _ in range(16):
            obs, reward, _, _, _ = env.step(torch.zeros(2, 12, device=raw.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
            if raw.cfg.base_obs_from_imu:
                torch.testing.assert_close(obs["policy"][:, 6:18], raw._imu_lin_acc_history[:, :, 0].flatten(1))
                torch.testing.assert_close(obs["critic"][:, 9:21], raw._imu_lin_acc_history[:, :, 1].flatten(1))
        assert len(oracle) == 16 * cfg.decimation
        for name in ("_imu_lin_acc_history", "_base_imu_history", "_teacher_lin_acc_history"):
            before = getattr(raw, name).clone()
            raw._get_observations()
            torch.testing.assert_close(getattr(raw, name), before, rtol=0, atol=0)
        neighbor = {name:getattr(raw, name)[1].clone() for name in
                    ("_imu_lin_acc_history", "_base_imu_history", "_teacher_lin_acc_history")}
        raw._reset_idx(torch.tensor([0], device=raw.device))
        for name, expected in neighbor.items():
            assert getattr(raw, name)[0].count_nonzero() == 0
            torch.testing.assert_close(getattr(raw, name)[1], expected, rtol=0, atol=0)
        raw.episode_length_buf[0] = raw.max_episode_length - 1
        _, _, _, timed_out, extras = env.step(torch.zeros(2, 12, device=raw.device))
        assert timed_out[0]
        if raw.cfg.base_obs_from_imu:
            terminal = extras["time_outs_obs"]["policy"][0, 6:18].reshape(4, 3)
            torch.testing.assert_close(terminal, torch.stack(oracle[-4:], 1)[0], rtol=0, atol=0)
            assert raw._imu_lin_acc_history[0].count_nonzero() == 0
        else:
            assert raw._base_imu_history[0].count_nonzero() == 0
    finally:
        env.close()


def test_physx_sampling_actor_inputs_and_partial_and_timeout_resets():
    # One live scene per Kit process. The same check can cover other variants in isolated processes.
    _check_physx_sampling_actor_inputs_and_resets("solo12-two-feet", Solo12TwoFeetEnvCfg)
