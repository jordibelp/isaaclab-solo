from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict


MODULE_PATH = Path(__file__).parents[2] / "skrl" / "solo12_symmetry.py"
SPEC = importlib.util.spec_from_file_location("solo12_symmetry_imu_obs_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
symmetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(symmetry)

VECTOR_X = torch.tensor([1.0, -1.0, 1.0])
VECTOR_Y = torch.tensor([-1.0, 1.0, 1.0])
PSEUDOVECTOR_X = torch.tensor([-1.0, 1.0, -1.0])
PSEUDOVECTOR_Y = torch.tensor([1.0, -1.0, -1.0])
COMMAND_X = torch.tensor([1.0, -1.0, -1.0])
COMMAND_Y = torch.tensor([-1.0, 1.0, -1.0])


def make_env(*, base_obs_from_imu: bool = False, lin_acc_steps: int = 1, front_back_asymetry: bool = True):
    cfg = SimpleNamespace(
        policy_model="simple_mlp",
        base_imu_history_sample_dim=45,
        front_back_asymetry=front_back_asymetry,
        base_obs_from_imu=base_obs_from_imu,
        include_lin_acc_steps_in_imu_obs=lin_acc_steps,
    )
    # Wrapped like RslRlVecEnvWrapper -> env.unwrapped.cfg.
    return SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))


@pytest.mark.parametrize("lin_acc_steps", [1, 4])
@pytest.mark.parametrize(
    "transform, gyro_sign, vector_sign, command_sign",
    [
        (symmetry.transform_policy_obs_reflect_x, PSEUDOVECTOR_X, VECTOR_X, COMMAND_X),
        (symmetry.transform_policy_obs_reflect_y, PSEUDOVECTOR_Y, VECTOR_Y, COMMAND_Y),
    ],
)
def test_imu_acceleration_layout_mirrors_each_signal_by_its_type(
    lin_acc_steps, transform, gyro_sign, vector_sign, command_sign
):
    acc_end = 6 + 3 * lin_acc_steps
    obs = torch.randn(5, 45 + 3 * lin_acc_steps)  # gyro, gravity, accelerations, commands, joints, actions

    mirrored = transform(obs, lin_acc_steps=lin_acc_steps)

    torch.testing.assert_close(mirrored[:, 0:3], obs[:, 0:3] * gyro_sign)
    torch.testing.assert_close(mirrored[:, 3:6], obs[:, 3:6] * vector_sign)
    torch.testing.assert_close(mirrored[:, 6:acc_end], obs[:, 6:acc_end] * vector_sign.repeat(lin_acc_steps))
    torch.testing.assert_close(mirrored[:, acc_end : acc_end + 3], obs[:, acc_end : acc_end + 3] * command_sign)
    # Without the acceleration columns this is exactly the 45-D layout without linear velocity.
    without_acc = torch.cat((obs[:, :6], obs[:, acc_end:]), dim=-1)
    torch.testing.assert_close(torch.cat((mirrored[:, :6], mirrored[:, acc_end:]), dim=-1), transform(without_acc))


def test_imu_accelerations_follow_gravity_when_linear_velocity_is_observed():
    slices = symmetry._get_observation_slices(48 + 3 * 4, lin_acc_steps=4)

    assert (slices.lin_vel, slices.ang_vel, slices.gravity, slices.lin_acc, slices.commands) == (
        slice(0, 3),
        slice(3, 6),
        slice(6, 9),
        slice(9, 21),
        slice(21, 24),
    )
    with pytest.raises(ValueError, match="got 45"):
        symmetry._get_observation_slices(45, lin_acc_steps=1)


@pytest.mark.parametrize("obs_dim, lin_acc_steps", [(45, 0), (48, 0), (48, 1), (51, 1), (57, 4), (60, 5)])
def test_policy_transforms_are_involutions(obs_dim, lin_acc_steps):
    obs = torch.randn(4, obs_dim)
    for transform in (
        symmetry.transform_policy_obs_reflect_x,
        symmetry.transform_policy_obs_reflect_y,
        symmetry.transform_policy_obs_rotate_180,
    ):
        torch.testing.assert_close(transform(transform(obs, lin_acc_steps), lin_acc_steps), obs)


def test_env_flags_resolve_the_ambiguous_layouts():
    obs = torch.randn(3, 48)

    legacy, _ = symmetry.compute_symmetric_observations_actions(env=make_env(), obs=obs)
    imu, _ = symmetry.compute_symmetric_observations_actions(env=make_env(base_obs_from_imu=True), obs=obs)
    imu_no_acc, _ = symmetry.compute_symmetric_observations_actions(
        env=make_env(base_obs_from_imu=True, lin_acc_steps=0), obs=obs
    )

    torch.testing.assert_close(legacy[3:], symmetry.transform_policy_obs_reflect_x(obs))
    torch.testing.assert_close(imu[3:], symmetry.transform_policy_obs_reflect_x(obs, lin_acc_steps=1))
    torch.testing.assert_close(imu_no_acc, legacy)
    assert not torch.allclose(imu[3:], legacy[3:])


@pytest.mark.parametrize("front_back_asymetry, copies", [(True, 2), (False, 4)])
def test_privileged_linear_velocity_is_mirrored_with_the_policy_batch(front_back_asymetry, copies):
    policy = torch.randn(2, 45)
    lin_vel = torch.randn(2, 3)
    actions = torch.randn(2, 12)
    obs = TensorDict({"policy": policy, symmetry.PRIVILEGED_OBS_GROUP: lin_vel}, batch_size=[2])
    env = make_env(front_back_asymetry=front_back_asymetry)

    obs_aug, actions_aug = symmetry.compute_symmetric_observations_actions(env=env, obs=obs, actions=actions)

    assert obs_aug.batch_size == torch.Size([2 * copies])
    assert actions_aug.shape == (2 * copies, 12)
    expected = [lin_vel, lin_vel * VECTOR_X, lin_vel * VECTOR_Y, lin_vel * VECTOR_X * VECTOR_Y][:copies]
    torch.testing.assert_close(obs_aug[symmetry.PRIVILEGED_OBS_GROUP], torch.cat(expected, dim=0))
    torch.testing.assert_close(obs_aug["policy"][2:4], symmetry.transform_policy_obs_reflect_x(policy))


def test_noise_free_critic_group_uses_the_policy_layout_with_linear_velocity():
    policy = torch.randn(2, 45 + 3 * 4)  # IMU gyro, gravity, 4 accelerations; no linear velocity
    critic = torch.randn(2, 48 + 3 * 4)  # the same terms plus the linear velocity in front
    obs = TensorDict({"policy": policy, "critic": critic}, batch_size=[2])
    env = make_env(base_obs_from_imu=True, lin_acc_steps=4)

    obs_aug, _ = symmetry.compute_symmetric_observations_actions(env=env, obs=obs)

    expected = symmetry.transform_policy_obs_reflect_x(critic, lin_acc_steps=4)
    torch.testing.assert_close(obs_aug["critic"][2:], expected)
    torch.testing.assert_close(obs_aug["critic"][2:, 0:3], critic[:, 0:3] * VECTOR_X)


def test_cfg_lookup_keeps_the_previous_defaults():
    assert symmetry._uses_front_back_symmetry(SimpleNamespace()) is True
    assert symmetry._get_policy_model(SimpleNamespace()) is None
    assert symmetry._policy_obs_imu_lin_acc_steps(SimpleNamespace()) == 0
    assert symmetry._policy_obs_imu_lin_acc_steps(make_env(base_obs_from_imu=False, lin_acc_steps=4)) == 0

    env = make_env(base_obs_from_imu=True, lin_acc_steps=4, front_back_asymetry=False)
    assert symmetry._uses_front_back_symmetry(env) is True
    assert symmetry._get_policy_model(env) == "simple_mlp"
    assert symmetry._get_base_imu_history_sample_dim(env) == 45
    assert symmetry._policy_obs_imu_lin_acc_steps(env) == 4
