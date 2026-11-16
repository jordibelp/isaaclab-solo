"""Hardware-aligned hard limits and absolute soft-limit precedence."""

import math
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

from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
from isaaclab.utils import replace_strings_with_slices
from isaaclab_tasks.direct.solo12.agents.rsl_rl_sac_cfg import Solo12SAC
from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env, _joint_pos_limit_violation
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12TwoFeetEnvCfg
from isaaclab_tasks.utils.hydra import register_task_to_hydra


def limit_shell(*, dtype=torch.float32, device="cpu", n=2, **overrides):
    cfg = Solo12TwoFeetEnvCfg()
    for name, value in overrides.items():
        setattr(cfg, name, value)
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = cfg
    # Reversed simulator order catches accidental use of policy indices as articulation indices.
    env._joint_ids = list(reversed(range(12)))
    data = SimpleNamespace(joint_pos_limits=torch.zeros(n, 12, 2, dtype=dtype, device=device))
    env._robot = SimpleNamespace(data=data, writes=0)

    def write(limits, joint_ids):
        env._robot.writes += 1
        data.joint_pos_limits[:, joint_ids] = limits

    env._robot.write_joint_position_limit_to_sim = write
    return env


def hard_limits(env):
    return env._robot.data.joint_pos_limits[:, env._joint_ids]


def hardware_limits(*, device="cpu", dtype=torch.float32):
    # Numeric calibrated-coordinate thresholds from 1116_lab_solo12.yaml, in policy order.
    return torch.tensor([
        [-1.22, 2.35], [-3.5, 3.5], [-3.5, 3.5],
        [-2.35, 1.22], [-3.5, 3.5], [-3.5, 3.5],
        [-1.22, 2.35], [-3.5, 3.5], [-3.5, 3.5],
        [-2.35, 1.22], [-3.5, 3.5], [-3.5, 3.5],
    ], device=device, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_defaults_match_robot_yaml_and_keep_delta_fallback(dtype, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = limit_shell(dtype=dtype, device=device)
    env._configure_joint_position_limits()
    expected = hardware_limits(device=device, dtype=dtype).expand(2, -1, -1)
    torch.testing.assert_close(hard_limits(env), expected)
    delta = torch.tensor([math.radians(v) for v in [5., 10., 5.] * 4], device=device, dtype=dtype)
    expected_soft = hard_limits(env).clone()
    expected_soft[..., 0] += delta
    expected_soft[..., 1] -= delta
    torch.testing.assert_close(env._joint_soft_pos_limits, expected_soft)


@pytest.mark.parametrize("n", [1, 10000])
def test_legacy_explicit_settings_keep_exact_delta_result(n):
    env = limit_shell(
        n=n, use_asymmetric_hip_limits=False, joint_physical_limit_hip=(-65., 65.),
        use_asymmetric_thigh_limits=True, joint_physical_limit_front_thigh=(-145., 55.),
        joint_physical_limit_rear_thigh=(-55., 145.), joint_physical_limit_calf=(-179., 179.),
        joint_soft_limit_hip_delta=20., joint_soft_limit_thigh_delta=20., joint_soft_limit_calf_delta=24.,
    )
    env._configure_joint_position_limits()
    expected = torch.deg2rad(torch.tensor([
        [-65., 65.], [-145., 55.], [-179., 179.],
        [-65., 65.], [-145., 55.], [-179., 179.],
        [-65., 65.], [-55., 145.], [-179., 179.],
        [-65., 65.], [-55., 145.], [-179., 179.],
    ])).expand(n, -1, -1)
    assert torch.equal(hard_limits(env), expected)
    delta = torch.deg2rad(torch.tensor([20., 20., 24.] * 4))
    soft = expected.clone()
    soft[..., 0] += delta
    soft[..., 1] -= delta
    assert torch.equal(env._joint_soft_pos_limits, soft)


def test_absolute_common_and_side_ranges_override_deltas_independently_of_hard_flags():
    env = limit_shell(
        joint_soft_limit_hip=(-45., 45.), joint_soft_limit_left_hip=(-40., 55.),
        joint_soft_limit_thigh=(-90., 90.), joint_soft_limit_front_thigh=(-125., 35.),
        joint_soft_limit_rear_thigh=(-35., 125.), joint_soft_limit_calf=(-155., 155.),
        joint_soft_limit_front_calf=(-150., 10.),
        # Unused deltas must not influence validation or the result.
        joint_soft_limit_hip_delta=float("nan"), joint_soft_limit_thigh_delta=-5.,
        joint_soft_limit_calf_delta=999.,
    )
    env._configure_joint_position_limits()
    expected = torch.tensor([
        [-40., 55.], [-125., 35.], [-150., 10.],
        [-45., 45.], [-125., 35.], [-150., 10.],
        [-40., 55.], [-35., 125.], [-155., 155.],
        [-45., 45.], [-35., 125.], [-155., 155.],
    ])
    torch.testing.assert_close(env._joint_soft_pos_limits, torch.deg2rad(expected).expand(2, -1, -1))
    torch.testing.assert_close(hard_limits(env), hardware_limits().expand(2, -1, -1))


def test_partial_override_falls_back_per_joint_and_none_restores_delta():
    env = limit_shell(joint_soft_limit_front_thigh=(-125., 35.))
    env._configure_joint_position_limits()
    baseline = limit_shell()
    baseline._configure_joint_position_limits()
    expected = baseline._joint_soft_pos_limits.clone()
    expected[:, [1, 4]] = torch.deg2rad(torch.tensor([-125., 35.]))
    assert torch.equal(env._joint_soft_pos_limits, expected)
    env.cfg.joint_soft_limit_front_thigh = None
    env._configure_joint_position_limits()
    assert torch.equal(env._joint_soft_pos_limits, baseline._joint_soft_pos_limits)


def test_absolute_soft_limits_do_not_move_when_hard_limits_widen():
    env = limit_shell(joint_soft_limit_thigh=(-125., 125.))
    env._configure_joint_position_limits()
    before = env._joint_soft_pos_limits[:, [1, 4, 7, 10]].clone()
    env.cfg.joint_physical_limit_thigh = (-220., 220.)
    env._configure_joint_position_limits()
    assert torch.equal(env._joint_soft_pos_limits[:, [1, 4, 7, 10]], before)


@pytest.mark.parametrize("field,indices", [
    ("joint_soft_limit_right_hip", [3, 9]), ("joint_soft_limit_rear_calf", [8, 11]),
])
def test_remaining_side_overrides_only_change_selected_joints(field, indices):
    env = limit_shell(**{field: (-40., 40.)})
    baseline = limit_shell()
    env._configure_joint_position_limits()
    baseline._configure_joint_position_limits()
    expected = baseline._joint_soft_pos_limits.clone()
    expected[:, indices] = torch.deg2rad(torch.tensor([-40., 40.]))
    assert torch.equal(env._joint_soft_pos_limits, expected)


def test_left_right_soft_hip_overrides_work_with_common_physical_hips():
    env = limit_shell(
        use_asymmetric_hip_limits=False, joint_physical_limit_hip=(-65., 65.),
        joint_soft_limit_left_hip=(-45., 50.), joint_soft_limit_right_hip=(-50., 45.),
    )
    env._configure_joint_position_limits()
    torch.testing.assert_close(
        env._joint_soft_pos_limits[:, [0, 6]], torch.deg2rad(torch.tensor([-45., 50.])).expand(2, 2, -1)
    )
    torch.testing.assert_close(
        env._joint_soft_pos_limits[:, [3, 9]], torch.deg2rad(torch.tensor([-50., 45.])).expand(2, 2, -1)
    )


def test_sac_position_scaling_uses_resolved_absolute_soft_ranges():
    env = limit_shell(
        joint_soft_limit_hip=(-45., 45.), joint_soft_limit_front_thigh=(-125., 35.),
        joint_soft_limit_rear_thigh=(-35., 125.), joint_soft_limit_calf=(-155., 155.),
    )
    env._configure_joint_position_limits()
    env._q_offset_action_and_obs = torch.zeros(2, 12)
    upper, lower = Solo12SAC._compute_action_scaling(env, "cpu")
    torch.testing.assert_close(env.cfg.action_scale * upper, env._joint_soft_pos_limits[0, :, 1])
    torch.testing.assert_close(-env.cfg.action_scale * lower, env._joint_soft_pos_limits[0, :, 0])


@pytest.mark.parametrize("field", ["joint_physical_limit_left_hip", "joint_soft_limit_front_thigh"])
@pytest.mark.parametrize("value", [(-1.,), (-1., 0., 1.), (-1., -1.), (1., -1.),
                                       (float("nan"), 1.), (-1., float("inf")), "bad", 3.])
def test_invalid_active_pairs_fail_before_writing_physics(field, value):
    env = limit_shell(**{field: value})
    with pytest.raises(ValueError, match=field):
        env._configure_joint_position_limits()
    assert env._robot.writes == 0


@pytest.mark.parametrize("field,value", [
    ("joint_soft_limit_left_hip", (-70., 100.)),
    ("joint_soft_limit_right_hip", (-100., 70.)),
    ("joint_soft_limit_front_thigh", (-201., 0.)),
    ("joint_soft_limit_calf", (-10., 201.)),
])
def test_soft_ranges_must_fit_corresponding_hard_range(field, value):
    env = limit_shell(**{field: value})
    with pytest.raises(ValueError, match="inside.*physical|physical.*inside"):
        env._configure_joint_position_limits()
    assert env._robot.writes == 0


@pytest.mark.parametrize("value", [-1., float("nan"), 500.])
def test_used_delta_still_validated_even_with_another_explicit_range(value):
    env = limit_shell(joint_soft_limit_front_thigh=(-125., 35.), joint_soft_limit_thigh_delta=value)
    with pytest.raises(ValueError, match="soft.*limit|soft.*range"):
        env._configure_joint_position_limits()
    assert env._robot.writes == 0


def test_absolute_soft_bounds_may_equal_hard_bounds_and_feed_existing_penalty():
    env = limit_shell(joint_soft_limit_thigh=tuple(map(math.degrees, (-3.5, 3.5))))
    env._configure_joint_position_limits()
    assert torch.equal(env._joint_soft_pos_limits[:, [1, 4, 7, 10]], hard_limits(env)[:, [1, 4, 7, 10]])
    q = env._joint_soft_pos_limits.mean(-1)
    assert torch.count_nonzero(_joint_pos_limit_violation(q, env._joint_soft_pos_limits)) == 0
    q[:, 1] = env._joint_soft_pos_limits[:, 1, 1] + .25
    torch.testing.assert_close(_joint_pos_limit_violation(q, env._joint_soft_pos_limits), torch.full((2,), .25))


@pytest.mark.parametrize("task", [
    "solo12-two-feet", "solo12-two-feet-torque", "solo12-two-feet-IMU-teacher",
    "solo12-two-feet-IMU-teacher-rl", "solo12-two-feet-IMU-student-dagger", "solo12-two-feet-IMU-student-rl",
])
def test_hydra_accepts_new_settings_without_plus_prefix_and_serializes_them(task):
    cfg, _ = register_task_to_hydra(task, None)
    with hydra.initialize(version_base="1.3", config_path=None):
        merged = hydra.compose(config_name=task, overrides=[
            "env.use_asymmetric_hip_limits=True", "env.joint_physical_limit_left_hip=[-60,130]",
            "env.joint_physical_limit_right_hip=[-130,60]", "env.joint_soft_limit_hip=[-45,45]",
            "env.joint_soft_limit_front_thigh=[-125,35]", "env.joint_soft_limit_rear_thigh=[-35,125]",
            "env.joint_soft_limit_calf=[-155,155]",
        ])
    values = replace_strings_with_slices(OmegaConf.to_container(merged, resolve=True))
    cfg.from_dict(values["env"])
    cfg = replace_strings_with_env_cfg_spaces(cfg)
    cfg.refresh_runtime_dependent_config()
    saved = cfg.to_dict()
    assert saved["use_asymmetric_hip_limits"] is True
    assert list(saved["joint_physical_limit_left_hip"]) == [-60, 130]
    assert list(saved["joint_soft_limit_front_thigh"]) == [-125, 35]
    assert saved["joint_soft_limit_thigh"] is None
    env = limit_shell()
    env.cfg = cfg
    env._configure_joint_position_limits()
    torch.testing.assert_close(
        hard_limits(env)[:, 0], torch.deg2rad(torch.tensor([-60., 130.])).expand(2, -1)
    )


def test_physx_readback_explicit_soft_limits_and_partial_reset():
    cfg = Solo12TwoFeetEnvCfg()
    cfg.scene.num_envs = 2
    cfg.seed = 71
    cfg.events = None
    cfg.include_events_randomization = False
    cfg.curriculum_two_feet = False
    cfg.tricky_terrain = False
    cfg.terrain.terrain_type = "plane"
    cfg.terrain.terrain_generator = None
    cfg.flat_terrain_grid_enabled = False
    cfg.joint_soft_limit_hip = (-45., 45.)
    cfg.joint_soft_limit_front_thigh = (-125., 35.)
    cfg.joint_soft_limit_rear_thigh = (-35., 125.)
    cfg.joint_soft_limit_calf = (-155., 155.)
    env = gym.make("solo12-two-feet", cfg=cfg)
    raw = env.unwrapped
    try:
        env.reset()
        actual = raw._robot.root_physx_view.get_dof_limits()[:, raw._joint_ids].cpu()
        torch.testing.assert_close(actual, hardware_limits().expand(2, -1, -1))
        expected_soft = torch.deg2rad(torch.tensor([
            [-45., 45.], [-125., 35.], [-155., 155.],
            [-45., 45.], [-125., 35.], [-155., 155.],
            [-45., 45.], [-35., 125.], [-155., 155.],
            [-45., 45.], [-35., 125.], [-155., 155.],
        ]))
        torch.testing.assert_close(raw._joint_soft_pos_limits.cpu(), expected_soft.expand(2, -1, -1))
        for _ in range(12):
            obs, reward, _, _, _ = env.step(torch.zeros(2, 12, device=raw.device))
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(value).all() for value in obs.values())
        before = raw._joint_soft_pos_limits.clone()
        raw._reset_idx(torch.tensor([0], device=raw.device))
        assert torch.equal(raw._joint_soft_pos_limits, before)
        torch.testing.assert_close(raw._robot.root_physx_view.get_dof_limits()[:, raw._joint_ids].cpu(), actual)
    finally:
        env.close()
