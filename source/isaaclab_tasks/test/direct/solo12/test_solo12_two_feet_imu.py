"""Resolved task, privilege, force-frame, and physics-rate history checks."""

from types import SimpleNamespace

import pytest
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, kit_args="--/UJITSO/geometry=false --/UJITSO/textures=false --/UJITSO/materials=false").app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import (
    Solo12TwoFeetEnvCfg, Solo12TwoFeetImuTeacherEnvCfg,
    Solo12TwoFeetImuStudentDaggerEnvCfg, Solo12TwoFeetImuStudentRlEnvCfg,
)


@pytest.mark.parametrize("apply_tanh", [True, False])
@pytest.mark.parametrize("cls", [Solo12TwoFeetImuTeacherEnvCfg, Solo12TwoFeetImuStudentDaggerEnvCfg, Solo12TwoFeetImuStudentRlEnvCfg])
def test_one_hydra_field_controls_both_latent_ranges(cls, apply_tanh):
    cfg = cls()
    assert cfg.apply_tanh_to_latent is True
    cfg.from_dict({"apply_tanh_to_latent": apply_tanh})
    cfg.refresh_runtime_dependent_config()
    assert cfg.teacher_latent_activation == ("tanh" if apply_tanh else "identity")


@pytest.mark.parametrize("task, cls, actor_width", [
    ("solo12-two-feet-IMU-teacher", Solo12TwoFeetImuTeacherEnvCfg, 67),
    ("solo12-two-feet-IMU-student-dagger", Solo12TwoFeetImuStudentDaggerEnvCfg, 655),
    ("solo12-two-feet-IMU-student-rl", Solo12TwoFeetImuStudentRlEnvCfg, 588),
])
def test_new_tasks_inherit_the_two_feet_curriculum_without_reward_changes(task, cls, actor_width):
    cfg, base = cls(), Solo12TwoFeetEnvCfg()
    assert gym.spec(task).kwargs["env_cfg_entry_point"].endswith(cls.__name__)
    assert cfg.observation_space == actor_width
    assert cfg.teacher_encoder_obs_dim == 64
    assert cfg.teacher_critic_obs_dim == 67
    assert cfg.state_space == 67
    for name, value in base.to_dict().items():
        if "curriculum" in name or "reward_scale" in name or name in ("kp", "kd", "action_scale", "episode_length_s"):
            assert cfg.to_dict()[name] == value, name


@pytest.mark.parametrize("force_kind, extra", [("none", 0), ("norm", 4), ("vector", 12)])
@pytest.mark.parametrize("steps", [0, 1, 4, 7])
def test_privileged_dimensions_follow_overrides(force_kind, extra, steps):
    cfg = Solo12TwoFeetImuStudentRlEnvCfg()
    cfg.teacher_foot_force_obs = force_kind
    cfg.include_lin_acc_steps_in_imu_obs = steps
    cfg.refresh_runtime_dependent_config()
    assert cfg.teacher_critic_obs_dim == 48 + 3 + 3 * steps + extra
    assert cfg.observation_space == 588  # No privileged channel reaches the actor.


@pytest.mark.parametrize("steps", [-1, True, 1.5])
def test_invalid_acceleration_count_is_rejected(steps):
    cfg = Solo12TwoFeetImuTeacherEnvCfg()
    cfg.include_lin_acc_steps_in_imu_obs = steps
    with pytest.raises(ValueError, match="integer"):
        cfg.refresh_runtime_dependent_config()


def observation_shell(cfg, n=2):
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    env.scene = SimpleNamespace(num_envs=n)
    env._actions = torch.arange(n * 12, dtype=torch.float32).reshape(n, 12)
    env._applied_actions = -env._actions
    env._commands = torch.ones(n, 3)
    env._robot = SimpleNamespace(data=SimpleNamespace(
        projected_gravity_b=torch.ones(n, 3), root_lin_vel_b=torch.full((n, 3), 2.),
        root_ang_vel_b=torch.full((n, 3), 3.), root_quat_w=torch.tensor([[2**-0.5, 0., 0., 2**-0.5]]).repeat(n, 1),
    ))
    env._get_joint_state_obs = lambda corrupt: torch.full((n, 24), 4.) + (100 if corrupt and cfg.noisy_actor_obs else 0)
    env._teacher_lin_acc_history = torch.arange(n * 12, dtype=torch.float32).reshape(n, 4, 3)
    env._base_push_forces_b = torch.tensor([[[1., 2., 3.]], [[0., 0., 0.]]])
    env._feet_body_ids = [0, 1, 2, 3]
    env._contact_sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[3., 4., 0.]] * 4] * n)))
    env._base_imu_history = torch.ones(n, 13, 45)
    return env


def test_clean_teacher_labels_and_privileged_critic_remain_noiseless():
    for cls in (Solo12TwoFeetImuTeacherEnvCfg, Solo12TwoFeetImuStudentDaggerEnvCfg, Solo12TwoFeetImuStudentRlEnvCfg):
        cfg = cls()
        cfg.joint_pos_noise = cfg.joint_vel_noise = (100., 100.)
        env = observation_shell(cfg)
        obs = env._get_observations()
        assert obs["critic"].shape == (2, 67)
        torch.testing.assert_close(obs["critic"][:, 21:45], torch.full((2, 24), 4.))
        torch.testing.assert_close(obs["critic"][:, 45:57], env._teacher_lin_acc_history.flatten(1))
        torch.testing.assert_close(obs["critic"][:, 57:60], env._base_push_forces_b[:, 0])
        torch.testing.assert_close(obs["critic"][:, 60:64], torch.full((2, 4), 5.))
        if cls is Solo12TwoFeetImuTeacherEnvCfg:
            torch.testing.assert_close(obs["policy"], obs["critic"])
        if cls is Solo12TwoFeetImuStudentRlEnvCfg:
            assert obs["policy"].shape == (2, 588)
            torch.testing.assert_close(obs["policy"][:, :585], torch.ones(2, 585))


def test_vector_contact_forces_are_expressed_in_base_axes():
    cfg = Solo12TwoFeetImuTeacherEnvCfg()
    cfg.teacher_foot_force_obs = "vector"
    cfg.refresh_runtime_dependent_config()
    env = observation_shell(cfg)
    # A +90 degree yaw maps world (3,4,0) into base (4,-3,0).
    result = env._get_teacher_critic_obs(corrupt=False)
    torch.testing.assert_close(result[:, 60:72], torch.tensor([[4., -3., 0.] * 4] * 2))


def test_privileged_acceleration_history_records_once_and_zero_pads():
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = Solo12TwoFeetImuTeacherEnvCfg()
    env._base_imu_enabled = True
    env._base_imu_history_len = 0
    env._teacher_lin_acc_history = torch.zeros(2, 4, 3)
    env._update_base_imu_bias_random_walk = lambda: None
    oracle = []
    for value in range(1, 8):
        sample = torch.tensor([[value, value + 10, value + 20.]] * 2)
        env._get_base_origin_lin_acc_b = lambda sample=sample: sample
        env._record_base_imu_history_sample()
        oracle.append(sample)
        expected = torch.stack(([torch.zeros_like(sample)] * max(0, 4 - len(oracle)) + oracle[-4:]), dim=1)
        torch.testing.assert_close(env._teacher_lin_acc_history, expected, rtol=0, atol=0)


def test_live_physics_histories_push_forces_and_partial_reset():
    cfg = Solo12TwoFeetImuStudentRlEnvCfg()
    cfg.scene.num_envs = 2
    cfg.seed = 71
    env = gym.make("solo12-two-feet-IMU-student-rl", cfg=cfg)
    raw = env.unwrapped
    try:
        env.reset()
        assert raw._feet_body_names == ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        oracle = []
        record_original = raw._record_base_imu_history_sample

        def record_and_check():
            record_original()
            # Independent NumPy rotation and rigid-body point-acceleration calculation.
            data, i = raw._robot.data, raw._base_wrench_body_ids[0]
            quats = data.body_link_quat_w[:, i].cpu().numpy()
            values = []
            for e, (w, x, y, z) in enumerate(quats):
                rotation = np.array([
                    [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
                ])
                r = rotation @ -data.body_com_pos_b[e, i].cpu().numpy()
                omega = data.body_com_ang_vel_w[e, i].cpu().numpy()
                alpha = data.body_com_ang_acc_w[e, i].cpu().numpy()
                acc = data.body_com_lin_acc_w[e, i].cpu().numpy()
                values.append(rotation.T @ (acc + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))))
            oracle.append(torch.as_tensor(np.array(values), dtype=raw._teacher_lin_acc_history.dtype, device=raw.device))

        raw._record_base_imu_history_sample = record_and_check
        raw._base_push_forces_b[:, 0] = torch.tensor([0.4, -0.3, 0.2], device=raw.device)
        raw._base_push_steps_left[:] = 100
        for _ in range(3):
            _, _, _, _, extras = env.step(torch.zeros(2, 12, device=raw.device))
            assert not raw.reset_buf.any(), "History fixture unexpectedly reset"
            expected = torch.stack(oracle[-4:], dim=1)
            torch.testing.assert_close(raw._teacher_lin_acc_history, expected, atol=2e-4, rtol=2e-5)
            obs = raw._get_observations()
            torch.testing.assert_close(obs["critic"][:, 57:60], raw._base_push_forces_b[:, 0])
            contact = raw._contact_sensor.data.net_forces_w[:, raw._feet_body_ids].norm(dim=-1)
            torch.testing.assert_close(obs["critic"][:, 60:64], contact)
            before = raw._teacher_lin_acc_history.clone()
            raw._get_observations()
            raw._get_observations()
            torch.testing.assert_close(raw._teacher_lin_acc_history, before, rtol=0, atol=0)
        assert len(oracle) == 3 * cfg.decimation
        assert (raw._teacher_lin_acc_history[:, 1:] - raw._teacher_lin_acc_history[:, :-1]).abs().max() > 1e-4
        neighbor = raw._teacher_lin_acc_history[1].clone()
        raw._reset_idx(torch.tensor([0], device=raw.device))
        assert raw._teacher_lin_acc_history[0].count_nonzero() == 0
        assert raw._base_imu_history[0].count_nonzero() == 0
        assert raw._base_push_forces_b[0].count_nonzero() == 0
        torch.testing.assert_close(raw._teacher_lin_acc_history[1], neighbor, rtol=0, atol=0)
        # A real automatic timeout exposes the final four samples before clearing selected histories.
        raw.episode_length_buf[0] = raw.max_episode_length - 1
        _, _, _, timed_out, extras = env.step(torch.zeros(2, 12, device=raw.device))
        assert timed_out[0]
        terminal = extras["time_outs_obs"]["critic"][0, 45:57].reshape(4, 3)
        torch.testing.assert_close(terminal, torch.stack(oracle[-4:], 1)[0], atol=2e-4, rtol=2e-5)
        assert raw._teacher_lin_acc_history[0].count_nonzero() == 0
    finally:
        env.close()
