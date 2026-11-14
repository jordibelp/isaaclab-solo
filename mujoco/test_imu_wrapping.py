"""H1a: deterministic Q11 wrapping, physics-rate logging, and opt-in isolation."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import play_direct_mujoco as player

MODEL = Path(__file__).with_name("solo12.xml")
IMU = dict(base_obs_from_imu=True, remove_root_lin_vel_b_from_actor_obs=True,
           include_lin_acc_steps_in_imu_obs=4)


def test_wrapping_flag_is_explicit():
    args = player.build_parser().parse_args(["--checkpoint=x.pt", "--ill-wrapping-test"])
    assert args.ill_wrapping_test
    assert not player.build_parser().parse_args(["--checkpoint=x.pt"]).ill_wrapping_test


def test_q11_signed_boundaries_and_multi_wrap():
    from solo12_imu_diagnostics import wrap_q11, q11_overflow
    lsb = 1 / 2048
    values = np.array([0., 17., -17., 16., -16., 16.-lsb, -16.-lsb, 49., -49., .75*lsb, -.75*lsb])
    np.testing.assert_array_equal(wrap_q11(values),
                                  [0., -15., 15., -16., -16., 16.-lsb, 16.-lsb, -15., 15., 0., 0.])
    np.testing.assert_array_equal(q11_overflow(values),
                                  [False, True, True, True, False, False, True, True, True, False, False])


def inject_world_acceleration(env, world_acc, rotation):
    env.data.qpos[3:7] = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    env.data.qvel[:] = 0.
    env._previous_imu_com_velocity = -player.PHYSICS_DT * np.asarray(world_acc)
    env._previous_imu_ang_velocity[:] = 0.
    env._record_base_imu_sample()


@pytest.mark.parametrize("clip", [True, False])
def test_wrap_before_clip_and_history_in_body_axes(clip):
    env = player.Solo12Mujoco(MODEL, env_overrides={**IMU, "base_imu_clip": clip}, ill_wrapping_test=True)
    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    inject_world_acceleration(env, [0., 0., 17.], rotation)
    np.testing.assert_allclose(env.observation((0., 0., 0.))[6:18].reshape(4, 3)[-1], [15., 0., 0.], atol=.001)
    before = env.observation((0., 0., 0.))
    np.testing.assert_array_equal(env.observation((0., 0., 0.)), before)
    env.reset()
    np.testing.assert_array_equal(env.observation((0., 0., 0.))[6:18], 0.)


def test_wrap_rejects_policy_without_acceleration():
    with pytest.raises(ValueError, match="acceleration"):
        player.Solo12Mujoco(MODEL, ill_wrapping_test=True)
    with pytest.raises(ValueError, match="acceleration"):
        player.Solo12Mujoco(MODEL, policy_model="base_imu_teacher", ill_wrapping_test=True)


def test_telemetry_does_not_change_rollout_and_keeps_every_substep(tmp_path):
    from solo12_imu_diagnostics import ImuImpactRecorder
    baseline = player.Solo12Mujoco(MODEL, env_overrides=IMU)
    logged = player.Solo12Mujoco(MODEL, env_overrides=IMU)
    logged.imu_diagnostics = ImuImpactRecorder(tmp_path, player.PHYSICS_DT, ill_wrapping_test=False)
    for _ in range(3):
        baseline.step(np.zeros(12))
        logged.step(np.zeros(12))
        np.testing.assert_array_equal(logged.data.qpos, baseline.data.qpos)
        np.testing.assert_array_equal(logged.observation((0., 0., 0.)), baseline.observation((0., 0., 0.)))
    logged.reset()
    logged.step(np.zeros(12))
    logged.imu_diagnostics.close()
    rows = list(csv.DictReader((tmp_path / "imu_acceleration.csv").open()))
    assert len(rows) == 16
    np.testing.assert_allclose([float(r["time_s"]) for r in rows], np.arange(1, 17)*.005)
    assert rows[-1]["episode_id"] == "1"
    assert float(rows[12]["episode_time_s"]) == pytest.approx(.005)
    summary = json.loads((tmp_path / "imu_summary.json").read_text())
    assert summary["samples"] == 16
    assert (tmp_path / "imu_impacts.png").is_file()
    assert logged.imu_diagnostics.close()  # idempotent for main's finally


def test_logger_has_world_z_body_xyz_counterfactuals_and_contacts(tmp_path):
    from solo12_imu_diagnostics import ImuImpactRecorder
    env = player.Solo12Mujoco(MODEL, env_overrides=IMU)
    env.imu_diagnostics = ImuImpactRecorder(tmp_path, player.PHYSICS_DT, ill_wrapping_test=False)
    rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    inject_world_acceleration(env, [0., 0., 17.], rotation)
    env.imu_diagnostics.close()
    row = next(csv.DictReader((tmp_path / "imu_acceleration.csv").open()))
    assert float(row["raw_world_z_mps2"]) == pytest.approx(17.)
    assert float(row["raw_body_x_mps2"]) == pytest.approx(-17.)
    assert float(row["wrapped_body_x_mps2"]) == pytest.approx(15.)
    assert float(row["delivered_body_x_mps2"]) == pytest.approx(-player.IMU_EKF_ACC_CLIP)
    assert float(row["wrapped_world_z_mps2"]) == pytest.approx(-15.)
    assert int(row["overflow_body_x"]) == 1
    assert int(row["base_ground_contact"]) == 0
    summary = json.loads((tmp_path / "imu_summary.json").read_text())
    assert summary["overflow_samples"] == 1
    assert summary["first_overflow_time_s"] == pytest.approx(.005)
    assert summary["raw_world_z_max_mps2"] == pytest.approx(17.)


def test_student_wraps_sensor_history_but_teacher_stays_clean():
    from solo12_imu_diagnostics import wrap_q11
    clean = player.Solo12Mujoco(MODEL, policy_model="base_imu_student_rl")
    wrapped = player.Solo12Mujoco(MODEL, policy_model="base_imu_student_rl", ill_wrapping_test=True)
    inject_world_acceleration(clean, [17., -19., 20.], np.eye(3))
    inject_world_acceleration(wrapped, [17., -19., 20.], np.eye(3))
    obs = wrapped.observation((0., 0., 0.))[:-3].reshape(-1, 45)
    np.testing.assert_allclose(obs[-1, 27:30], wrap_q11(np.array([17., -19., 20.])))
    np.testing.assert_array_equal(wrapped._teacher_lin_acc_history, clean._teacher_lin_acc_history)


def test_empty_recording_can_close(tmp_path):
    from solo12_imu_diagnostics import ImuImpactRecorder
    recorder = ImuImpactRecorder(tmp_path, .005, ill_wrapping_test=False)
    recorder.close()
    assert json.loads((tmp_path / "imu_summary.json").read_text())["samples"] == 0


def test_wandb_logs_every_numeric_imu_column(tmp_path):
    from solo12_imu_diagnostics import ImuImpactRecorder, log_imu_csv
    env = player.Solo12Mujoco(MODEL, env_overrides=IMU)
    env.imu_diagnostics = ImuImpactRecorder(tmp_path, .005, ill_wrapping_test=False)
    env.step(np.zeros(12))
    env.imu_diagnostics.close()
    rows, metrics = [], []
    run = SimpleNamespace(log=rows.append, define_metric=lambda *a, **kw: metrics.append((a, kw)))
    assert log_imu_csv(run, tmp_path / "imu_acceleration.csv") == 4
    assert len(rows) == 4
    assert "imu/raw_world_z_mps2" in rows[0]
    assert "imu/overflow_body_x" in rows[0]
    assert "imu/foot_ground_normal_force_n" in rows[0]


def test_world_z_above_16_does_not_imply_sensor_axis_overflow(tmp_path):
    from solo12_imu_diagnostics import ImuImpactRecorder
    env = player.Solo12Mujoco(MODEL, env_overrides=IMU)
    env.imu_diagnostics = ImuImpactRecorder(tmp_path, .005, ill_wrapping_test=False)
    inject_world_acceleration(env, [0., 0., 17.], Rotation.from_euler("y", 45, degrees=True).as_matrix())
    env.imu_diagnostics.close()
    summary = json.loads((tmp_path / "imu_summary.json").read_text())
    assert summary["raw_world_z_max_mps2"] == pytest.approx(17.)
    assert summary["overflow_samples"] == 0


def test_quantizer_matches_independent_integer_bit_oracle():
    from solo12_imu_diagnostics import wrap_q11
    values = np.random.default_rng(42).uniform(-200., 200., 1000)
    expected = []
    for value in values:
        bits = int(value * 2048) & 0xffff
        expected.append((bits if bits < 0x8000 else bits - 0x10000) / 2048)
    np.testing.assert_array_equal(wrap_q11(values), expected)


def test_live_dispatch_auto_logs_and_finalizes_on_interrupt(tmp_path, monkeypatch):
    import sys
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"checkpoint stub; actor loader replaced only in this test")
    monkeypatch.setattr(player, "Policy", lambda _: SimpleNamespace(input_dim=57, checkpoint_format="ppo"))

    def interrupted_live(args, sim, policy, command_ranges):
        assert sim.ill_wrapping_test
        sim.step(np.zeros(12))
        raise KeyboardInterrupt

    monkeypatch.setattr(player, "run_live", interrupted_live)
    monkeypatch.setattr(sys, "argv", ["play_direct_mujoco.py", f"--checkpoint={checkpoint}",
                                    "--ill-wrapping-test", f"--output-dir={tmp_path}/out",
                                    "env.base_obs_from_imu=True", "env.include_lin_acc_steps_in_imu_obs=4",
                                    "env.remove_root_lin_vel_b_from_actor_obs=True"])
    with pytest.raises(KeyboardInterrupt):
        player.main()
    summaries = list((tmp_path / "out").glob("*/*/imu_summary.json"))
    assert len(summaries) == 1
    assert json.loads(summaries[0].read_text())["samples"] == 4
    config = json.loads(summaries[0].with_name("imu_config.json").read_text())
    assert config["ill_wrapping_test"]
    assert config["input_dim"] == 57
