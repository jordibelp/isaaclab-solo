from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app


SCRIPT_DIR = Path(__file__).resolve().parents[1]
SKRL_SCRIPT_DIR = SCRIPT_DIR.parent / "skrl"
for path in (SCRIPT_DIR, SKRL_SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from solo12_rnd import configure_solo12_rnd, load_checkpoint_with_optional_fresh_rnd  # noqa: E402
from solo12_symmetry import compute_symmetric_observations_actions  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env  # noqa: E402
from isaaclab_tasks.direct.solo12.solo12_env_cfg import Solo12EnvCfg, Solo12TwoFeetEnvCfg  # noqa: E402


def make_agent_cfg():
    return SimpleNamespace(
        class_name="OnPolicyRunner",
        algorithm=SimpleNamespace(rnd_cfg=None),
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
    )


def test_rnd_defaults_are_disabled_for_all_solo_tasks() -> None:
    for cfg in (Solo12EnvCfg(), Solo12TwoFeetEnvCfg()):
        assert cfg.rnd_network is False
        assert cfg.beta_curiosity == 0.0


def test_disabled_rnd_does_not_change_environment_observations() -> None:
    observations = {"policy": torch.ones(2, 48)}
    fake_env = SimpleNamespace(
        cfg=SimpleNamespace(rnd_network=False),
        _get_rnd_curiosity_state=lambda: pytest.fail("disabled RND should not build a curiosity state"),
    )

    actual = Solo12Env._with_optional_rnd_state(fake_env, observations)

    assert actual is observations
    assert list(actual) == ["policy"]


def test_rnd_setup_uses_task_focused_robotics_configuration() -> None:
    env_cfg = SimpleNamespace(rnd_network=True, beta_curiosity=1.25)
    agent_cfg = make_agent_cfg()

    result = configure_solo12_rnd(env_cfg, agent_cfg)

    assert result.enabled is True
    assert result.beta == pytest.approx(1.25)
    assert agent_cfg.obs_groups["rnd_state"] == ["rnd_state"]
    assert agent_cfg.algorithm.rnd_cfg.weight == pytest.approx(1.25)
    assert agent_cfg.algorithm.rnd_cfg.state_normalization is True
    assert agent_cfg.algorithm.rnd_cfg.reward_normalization is False
    assert agent_cfg.algorithm.rnd_cfg.learning_rate == pytest.approx(1.0e-3)
    assert agent_cfg.algorithm.rnd_cfg.num_outputs == 1
    assert agent_cfg.algorithm.rnd_cfg.target_hidden_dims == [5]
    assert agent_cfg.algorithm.rnd_cfg.predictor_hidden_dims == [5, 5]


@pytest.mark.parametrize(
    ("enabled", "beta", "message"),
    (
        (True, 0.0, "requires env.beta_curiosity > 0.0"),
        (False, 1.0, "requires env.rnd_network=True"),
        (True, -1.0, "must be finite and non-negative"),
        (True, float("nan"), "must be finite and non-negative"),
    ),
)
def test_rnd_setup_rejects_silent_noop_or_invalid_combinations(enabled, beta, message) -> None:
    with pytest.raises(ValueError, match=message):
        configure_solo12_rnd(SimpleNamespace(rnd_network=enabled, beta_curiosity=beta), make_agent_cfg())


def test_resume_old_checkpoint_keeps_fresh_rnd_while_standard_runner_loads_policy(tmp_path) -> None:
    checkpoint = tmp_path / "old_checkpoint.pt"
    torch.save({"model_state_dict": {}, "optimizer_state_dict": {}, "iter": 123, "infos": {}}, checkpoint)
    fresh_rnd = object()
    fresh_rnd_optimizer = object()
    seen = []

    def load(path):
        seen.append((path, runner.alg.rnd, runner.alg.rnd_optimizer))
        return {"loaded": True}

    runner = SimpleNamespace(
        alg=SimpleNamespace(rnd=fresh_rnd, rnd_optimizer=fresh_rnd_optimizer),
        load=load,
    )

    infos, initialized_fresh = load_checkpoint_with_optional_fresh_rnd(runner, str(checkpoint))

    assert infos == {"loaded": True}
    assert initialized_fresh is True
    assert seen == [(str(checkpoint), None, None)]
    assert runner.alg.rnd is fresh_rnd
    assert runner.alg.rnd_optimizer is fresh_rnd_optimizer


def test_resume_rnd_checkpoint_uses_standard_exact_restore(tmp_path) -> None:
    checkpoint = tmp_path / "rnd_checkpoint.pt"
    torch.save({"rnd_state_dict": {}, "rnd_optimizer_state_dict": {}}, checkpoint)
    rnd = object()
    rnd_optimizer = object()
    runner = SimpleNamespace(
        alg=SimpleNamespace(rnd=rnd, rnd_optimizer=rnd_optimizer),
        load=lambda path: {"path": path},
    )

    infos, initialized_fresh = load_checkpoint_with_optional_fresh_rnd(runner, str(checkpoint))

    assert infos == {"path": str(checkpoint)}
    assert initialized_fresh is False
    assert runner.alg.rnd is rnd
    assert runner.alg.rnd_optimizer is rnd_optimizer


@pytest.mark.parametrize("present_key", ("rnd_state_dict", "rnd_optimizer_state_dict"))
def test_resume_rejects_partial_rnd_checkpoint(tmp_path, present_key) -> None:
    checkpoint = tmp_path / "partial_rnd_checkpoint.pt"
    torch.save({present_key: {}}, checkpoint)
    runner = SimpleNamespace(
        alg=SimpleNamespace(rnd=object(), rnd_optimizer=object()),
        load=lambda path: pytest.fail("partial RND checkpoint must not be passed to runner.load"),
    )

    with pytest.raises(ValueError, match="incomplete RND state"):
        load_checkpoint_with_optional_fresh_rnd(runner, str(checkpoint))


def test_curiosity_state_contains_clean_posture_and_contact_configuration() -> None:
    projected_gravity = torch.tensor(((0.1, 0.2, -0.9), (0.3, 0.4, -0.8)))
    joint_pos = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    joint_offset = torch.full((2, 12), 0.5)
    contacts = torch.tensor(((True, False, True, False), (False, True, False, True)))
    fake_env = SimpleNamespace(
        _joint_ids=list(range(12)),
        _q_offset_action_and_obs=joint_offset,
        _robot=SimpleNamespace(
            data=SimpleNamespace(projected_gravity_b=projected_gravity, joint_pos=joint_pos)
        ),
        cfg=SimpleNamespace(feet_ground_contact_threshold=1.0),
        _get_feet_contact_mask=lambda threshold: contacts,
    )

    actual = Solo12Env._get_rnd_curiosity_state(fake_env)
    expected = torch.cat((projected_gravity, joint_pos - joint_offset, contacts.float()), dim=-1)

    assert actual.shape == (2, 19)
    torch.testing.assert_close(actual, expected)


def test_symmetry_repeats_rnd_state_without_treating_it_as_policy_features() -> None:
    policy = torch.arange(96, dtype=torch.float32).reshape(2, 48)
    rnd_state = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    observations = TensorDict(
        {"policy": policy, "rnd_state": rnd_state},
        batch_size=[2],
    )
    env = SimpleNamespace(cfg=SimpleNamespace(policy_model="simple_mlp", front_back_asymetry=True))

    augmented, _ = compute_symmetric_observations_actions(env=env, obs=observations)

    assert augmented.batch_size == torch.Size([4])
    torch.testing.assert_close(augmented["rnd_state"], torch.cat((rnd_state, rnd_state), dim=0))
