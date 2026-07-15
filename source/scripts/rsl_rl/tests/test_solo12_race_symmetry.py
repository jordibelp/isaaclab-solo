from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[2] / "skrl" / "solo12_race_symmetry.py"
SPEC = importlib.util.spec_from_file_location("solo12_race_symmetry_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
symmetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(symmetry)


def test_left_right_augmentation_only_doubles_batch() -> None:
    obs = torch.arange(2 * 57, dtype=torch.float32).reshape(2, 57)
    actions = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12)

    obs_aug, actions_aug = symmetry.compute_left_right_symmetric_observations_actions(
        obs=obs, actions=actions
    )

    assert obs_aug.shape == (4, 57)
    assert actions_aug.shape == (4, 12)
    torch.testing.assert_close(obs_aug[:2], obs)
    torch.testing.assert_close(obs_aug[2:], symmetry.transform_policy_obs_reflect_x(obs))
    torch.testing.assert_close(actions_aug[:2], actions)
    torch.testing.assert_close(actions_aug[2:], symmetry.transform_actions_reflect_x(actions))


def test_left_right_reflection_is_an_involution() -> None:
    obs = torch.randn(3, 57)
    actions = torch.randn(3, 12)

    mirrored_obs = symmetry.transform_policy_obs_reflect_x(obs)
    mirrored_actions = symmetry.transform_actions_reflect_x(actions)

    torch.testing.assert_close(symmetry.transform_policy_obs_reflect_x(mirrored_obs), obs)
    torch.testing.assert_close(symmetry.transform_actions_reflect_x(mirrored_actions), actions)


def test_full_race_augmentation_remains_fourfold() -> None:
    obs = torch.randn(2, 57)
    actions = torch.randn(2, 12)

    obs_aug, actions_aug = symmetry.compute_symmetric_observations_actions(obs=obs, actions=actions)

    assert obs_aug.shape == (8, 57)
    assert actions_aug.shape == (8, 12)
