"""Hardware position-error clipping: default parity, sampling, and PD saturation."""

import ast
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import play_direct_mujoco as sim2sim


ROOT = Path(__file__).resolve().parents[1]


def isaac_apply_action():
    """Exercise the production method without loading Kit in the unit-test process."""
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "Solo12Env")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_action")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_action"]


@pytest.mark.parametrize("clip_rad", [0.0, 0.2])
def test_isaac_clips_delayed_targets_without_changing_action_history(clip_rad):
    q_all = torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.]])
    joint_ids = [3, 0, 2]  # policy order differs from articulation order
    error = torch.tensor([[.5, -.5, .1], [-.2, .2, -.3]])
    delayed = q_all[:, joint_ids] + error
    sent = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(bringup_clip_rad=clip_rad, action_scale=.25),
        _processed_actions=torch.zeros_like(delayed),  # deliberately NOT the delayed command
        _q_offset_action_and_obs=torch.ones_like(delayed),
        _action_delay_buffer=SimpleNamespace(compute=lambda _: delayed),
        _robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q_all),
                               set_joint_position_target=lambda target, **kw: sent.append(target.clone())),
        _joint_ids=joint_ids, _record_bringup_errors=True,
    )
    isaac_apply_action()(env)
    expected = error if clip_rad == 0 else error.clamp(-clip_rad, clip_rad)
    torch.testing.assert_close(sent[0] - q_all[:, joint_ids], expected)
    torch.testing.assert_close(env._applied_actions, (delayed - 1.) / .25)
    torch.testing.assert_close(env._bringup_requested_error, error)
    torch.testing.assert_close(env._bringup_applied_error, expected)
    # An automatic reset must not corrupt the just-recorded, pre-physics sample.
    delayed.zero_()
    q_all.zero_()
    torch.testing.assert_close(env._bringup_requested_error, error)


def test_mujoco_zero_clip_is_bit_exact_with_legacy_rollout():
    path = Path(__file__).with_name("solo12.xml")
    actual = sim2sim.Solo12Mujoco(path, bringup_clip_rad=0.0)
    legacy = sim2sim.Solo12Mujoco(path)
    rng = np.random.default_rng(51)
    for action in rng.uniform(-2., 2., (10, 12)):
        legacy.action = action.copy()
        legacy.data.ctrl[legacy.actuator_ids] = sim2sim.SAFE_Q + sim2sim.ACTION_SCALE * action
        for _ in range(sim2sim.DECIMATION):
            mujoco.mj_step(legacy.model, legacy.data)
        actual.step(action)
        np.testing.assert_array_equal(actual.data.qpos, legacy.data.qpos)
        np.testing.assert_array_equal(actual.data.qvel, legacy.data.qvel)
        np.testing.assert_array_equal(actual.data.actuator_force, legacy.data.actuator_force)


def test_mujoco_reclips_each_substep_and_preserves_damping_and_total_torque_cap(monkeypatch):
    env = sim2sim.Solo12Mujoco(Path(__file__).with_name("solo12.xml"), kp=9., kd=.2,
                             bringup_clip_rad=.2, env_overrides={"effort_limit_sim": 2.25})
    action = np.array([2., -2., .1] * 4)
    env.data.qvel[env.joint_dof] = -np.sign(action)
    original_step = mujoco.mj_step
    calls = []

    def check_then_step(model, data):
        q = data.qpos[env.joint_qpos].copy()
        error = data.ctrl[env.actuator_ids] - q
        raw = sim2sim.SAFE_Q + sim2sim.ACTION_SCALE * action - q
        np.testing.assert_allclose(error, np.clip(raw, -.2, .2), atol=1e-14)
        mujoco.mj_forward(model, data)
        expected = np.clip(9. * error - .2 * data.qvel[env.joint_dof], -2.25, 2.25)
        np.testing.assert_allclose(data.actuator_force[env.actuator_ids], expected, atol=1e-13)
        calls.append((q, data.actuator_force[env.actuator_ids].copy()))
        original_step(model, data)

    monkeypatch.setattr(mujoco, "mj_step", check_then_step)
    env.step(action)
    assert len(calls) == sim2sim.DECIMATION
    assert not np.array_equal(calls[0][0], calls[-1][0])
    assert calls[0][1][0] == pytest.approx(2.0)  # damping adds to kp * .2 = 1.8 Nm
    np.testing.assert_array_equal(env.action, action)


@pytest.mark.parametrize("value", [-.2, float("nan"), float("inf")])
def test_invalid_clip_is_rejected(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        sim2sim.Solo12Mujoco(Path(__file__).with_name("solo12.xml"), bringup_clip_rad=value)


def test_mujoco_cli_accepts_both_flag_spellings_and_training_effort_limit():
    parser = sim2sim.build_parser()
    assert parser.parse_args(["--checkpoint=x"]).bringup_clip_rad == 0.
    for flag in ("--bringup_clip_rad=.2", "--bringup-clip-rad=.2"):
        assert parser.parse_args(["--checkpoint=x", flag]).bringup_clip_rad == .2
    overrides, ignored = sim2sim.consume_env_overrides(["env.effort_limit_sim=2.25"])
    assert overrides == {"effort_limit_sim": 2.25}
    assert not ignored
