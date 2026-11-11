"""First-push sampling, existing gap semantics, and real reset/wrench checks."""

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
from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg, Solo12TwoFeetEnvCfg
from isaaclab_tasks.utils.hydra import register_task_to_hydra


@pytest.mark.parametrize("task", [
    "solo12-two-feet", "solo12-two-feet-IMU-teacher",
    "solo12-two-feet-IMU-student-dagger", "solo12-two-feet-IMU-student-rl",
])
@pytest.mark.parametrize("override, expected", [
    ("[0.0,5.0]", [0.0, 5.0]), ("[0.0,0.0]", [0.0, 0.0]), ("null", None),
])
def test_registered_task_hydra_roundtrip(task, override, expected):
    cfg, _ = register_task_to_hydra(task, None)
    assert cfg.base_push_initial_delay_range_s == (0.0, 5.0)
    with hydra.initialize(version_base="1.3", config_path=None):
        merged = hydra.compose(config_name=task, overrides=[f"env.base_push_initial_delay_range_s={override}"])
    values = replace_strings_with_slices(OmegaConf.to_container(merged, resolve=True))
    cfg.from_dict(values["env"])
    cfg = replace_strings_with_env_cfg_spaces(cfg)
    cfg.refresh_runtime_dependent_config()
    actual = cfg.base_push_initial_delay_range_s
    assert (None if actual is None else list(actual)) == expected
    assert cfg.base_push_interval_range_s == (2.0, 5.0)
    assert cfg.base_push_duration_range_s == (0.5, 2.0)
    assert cfg.forces_applied_to_base_curriculum_by_phase == (0.0, 0.0, 0.0, 5.0, 8.0)


@pytest.mark.parametrize("delay", [[-1.0, 5.0], [5.0, 0.0], [float("nan"), 5.0],
                                   [0.0, float("inf")], [], [1.0], "early", 1.0])
def test_invalid_initial_delay_fails_before_simulation(delay):
    cfg = Solo12TwoFeetEnvCfg()
    cfg.base_push_initial_delay_range_s = delay
    with pytest.raises(ValueError, match="base_push_initial_delay_range_s"):
        cfg.refresh_runtime_dependent_config()


def push_shell(delay, count=4):
    env = object.__new__(Solo12Env)
    env._is_closed = True
    env.cfg = Solo12EnvCfg()
    env.cfg.base_push_initial_delay_range_s = delay
    env.cfg.base_push_force_xy_range = (1.0, 1.0)
    env.cfg.base_push_force_z_range = (1.0, 1.0)
    env.scene = SimpleNamespace(num_envs=count)
    env.sim = SimpleNamespace(device="cpu")
    env._base_wrench_body_ids = [0]
    env._base_push_application_half_extents = torch.tensor(env.cfg.base_push_application_half_extents)
    env._base_push_interval_step_range = (100, 250)
    env._base_push_duration_step_range = (100, 100)
    env._base_push_initial_delay_step_range = (
        env._base_push_interval_step_range if delay is None else env._seconds_range_to_steps(delay)
    )
    env._base_push_steps_left = torch.zeros(count, dtype=torch.long)
    env._base_push_steps_until_next = torch.zeros(count, dtype=torch.long)
    env._base_push_forces_b = torch.zeros(count, 1, 3)
    env._base_push_application_points_b = torch.zeros(count, 1, 3)
    env._robot = SimpleNamespace(permanent_wrench_composer=SimpleNamespace(
        reset=lambda *args: None, set_forces_and_torques=lambda **kwargs: None,
    ))
    return env


def test_initial_sampling_covers_early_and_late_times_and_is_per_reset():
    env = push_shell((0.0, 5.0), 10_000)
    ids = torch.arange(env.num_envs)
    torch.manual_seed(19)
    env._reset_base_pushes(ids)
    first = env._base_push_steps_until_next.clone()
    assert 1800 < int((first <= 50).sum()) < 2200
    assert 3800 < int((first <= 100).sum()) < 4200
    assert int(first.min()) == 1 and int(first.max()) == 250
    env._reset_base_pushes(ids)
    assert not torch.equal(first, env._base_push_steps_until_next)


def test_none_preserves_legacy_draws_and_rng_and_plain_solo12_default():
    assert Solo12EnvCfg().base_push_initial_delay_range_s is None
    env = push_shell(None, 100)
    torch.manual_seed(27)
    expected = torch.randint(100, 251, (100,))
    expected_rng = torch.get_rng_state()
    torch.manual_seed(27)
    env._reset_base_pushes(torch.arange(100))
    assert torch.equal(env._base_push_steps_until_next, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)


def test_zero_delay_starts_immediately_without_concatenating_later_pushes():
    env = push_shell((0.0, 0.0), 1)
    env._base_push_interval_step_range = (100, 100)
    env._reset_base_pushes(torch.tensor([0]))
    starts, active_ticks = [], []
    start = env._start_base_pushes

    def record(ids):
        starts.append(tick)
        start(ids)

    env._start_base_pushes = record
    for tick in range(200):
        env._update_base_push_wrench()
        active_ticks.append(bool(env._base_push_forces_b.any()))
    assert starts == [0, 199]  # Existing one-policy-tick convention: about 4 s, not 2 s.
    assert all(active_ticks[:100])
    assert not any(active_ticks[100:199])


def test_partial_reset_clears_only_selected_pushes():
    env = push_shell((0.0, 0.0))
    env._base_push_forces_b.fill_(7)
    env._base_push_application_points_b.fill_(8)
    env._base_push_steps_left.fill_(9)
    env._base_push_steps_until_next.fill_(10)
    selected, untouched = torch.tensor([0, 2]), torch.tensor([1, 3])
    env._reset_base_pushes(selected)
    assert not env._base_push_forces_b[selected].any()
    assert not env._base_push_application_points_b[selected].any()
    assert not env._base_push_steps_left[selected].any()
    assert torch.all(env._base_push_steps_until_next[selected] == 1)
    assert torch.all(env._base_push_forces_b[untouched] == 7)
    assert torch.all(env._base_push_application_points_b[untouched] == 8)
    assert torch.all(env._base_push_steps_left[untouched] == 9)
    assert torch.all(env._base_push_steps_until_next[untouched] == 10)


def test_live_first_step_force_and_manual_and_automatic_resets():
    cfg = Solo12EnvCfg()
    cfg.scene.num_envs = 2
    cfg.seed = 71
    cfg.include_events_randomization = False
    cfg.base_push_initial_delay_range_s = (0.0, 0.0)
    cfg.base_push_force_xy_range = (1.0, 1.0)
    cfg.base_push_force_z_range = (1.0, 1.0)
    cfg.forces_applied_to_base_curriculum = []
    env = gym.make("solo12-v0", cfg=cfg)
    raw = env.unwrapped
    try:
        env.reset()
        assert raw._base_push_initial_delay_step_range == (1, 1)
        assert not raw._base_push_forces_b.any()
        action = torch.zeros(2, 12, device=raw.device)
        env.step(action)
        assert not raw.reset_buf.any()
        torch.testing.assert_close(raw._base_push_forces_b, torch.ones(2, 1, 3, device=raw.device))
        composer = raw._robot.permanent_wrench_composer
        torch.testing.assert_close(composer.composed_force_as_torch[:, raw._base_wrench_body_ids],
                                   raw._base_push_forces_b)
        neighbor = raw._base_push_steps_left[1].clone()
        raw._reset_idx(torch.tensor([0], device=raw.device))
        assert not raw._base_push_forces_b[0].any()
        assert not composer.composed_force_as_torch[0].any()
        assert raw._base_push_steps_left[1] == neighbor
        env.step(action)
        assert raw._base_push_forces_b[0].any()
        raw.episode_length_buf[0] = raw.max_episode_length - 1
        _, _, _, timeouts, _ = env.step(action)
        assert timeouts[0]
        assert not raw._base_push_forces_b[0].any()
        assert not composer.composed_force_as_torch[0].any()
        assert raw._base_push_steps_until_next[0] == 1
        env.step(action)
        assert raw._base_push_forces_b[0].any()
    finally:
        env.close()
