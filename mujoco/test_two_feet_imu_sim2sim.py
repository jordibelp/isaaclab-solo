"""Teacher/DAgger/PPO loading and physics-rate signal contracts without Isaac Sim."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from rsl_rl.networks import EmpiricalNormalization
from scipy.spatial.transform import Rotation

import play_direct_mujoco as player
from solo12_imu_policy import EncodedPolicy, TASK_MODELS, production_modules


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(__file__).with_name("solo12.xml")


@pytest.fixture(params=[True, False], ids=["bounded", "unbounded"])
def checkpoints(tmp_path, request):
    models, _ = production_modules()
    torch.manual_seed(372)
    bounded = request.param
    obs = {name: torch.randn(12, 67) for name in ("policy", "critic")}
    teacher = models.Solo12BaseImuTeacherActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
        teacher_encoder_obs_dim=64, teacher_latent_dim=7, teacher_encoder_hidden_dims=[13, 9],
        actor_hidden_dims=[11, 8], critic_hidden_dims=[15, 10],
        actor_obs_normalization=True, critic_obs_normalization=True,
        teacher_latent_activation="tanh" if bounded else "identity",
    )
    teacher.update_normalization(obs)
    teacher.eval()
    latent_metadata = {"apply_tanh_to_latent": bounded, "teacher_latent_activation": teacher.teacher_latent_activation}
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict(), "infos": {"solo12_imu_latent": latent_metadata}}, teacher_path)
    adapter = models.BaseImuTcnEncoder(history_len=9, sample_dim=45, channels=8, latent_dim=7,
                                      kernel_size=3, apply_tanh_to_latent=bounded).eval()
    history_normalizer = EmpiricalNormalization(405)
    history_normalizer.update(torch.randn(16, 405))
    history_normalizer.eval()
    adapter_payload = {
        "teacher_checkpoint": str(teacher_path), "layout": {"kind": "base_imu", "apply_tanh_to_latent": bounded},
        "observation_contract": {"apply_tanh_to_latent": bounded},
        "adapter_state_dict": adapter.state_dict(), "history_normalizer_state_dict": history_normalizer.state_dict(),
        "dims": {"history_len": 9, "history_sample_dim": 45, "teacher_obs_dim": 67, "latent_dim": 7},
    }
    adapter_path = tmp_path / "adapter.pt"
    torch.save(adapter_payload, adapter_path)
    student_obs = {"policy": torch.randn(12, 408), "critic": obs["critic"]}
    student = models.Solo12BaseImuStudentActorCritic(
        student_obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
        history_len=9, history_sample_dim=45, teacher_critic_obs_dim=67,
        actor_hidden_dims=[11, 8], critic_hidden_dims=[15, 10],
        tcn_channels=8, tcn_latent_dim=7, tcn_kernel_size=3,
        actor_obs_normalization=True, critic_obs_normalization=True, apply_tanh_to_latent=bounded,
    )
    student.update_normalization(student_obs)
    student.eval()
    student_path = tmp_path / "student.pt"
    torch.save({"model_state_dict": student.state_dict(), "infos": {"solo12_imu_latent": latent_metadata}}, student_path)
    return SimpleNamespace(**locals())


@pytest.mark.parametrize("stage,inputs", [("teacher", 67), ("student-dagger", 408), ("student-rl", 408)])
def test_loaded_actor_matches_production_batch_one(checkpoints, stage, inputs):
    c = checkpoints
    path = {"teacher": c.teacher_path, "student-dagger": c.adapter_path, "student-rl": c.student_path}[stage]
    policy = EncodedPolicy(path, f"solo12-two-feet-IMU-{stage}", {})
    assert policy.input_dim == inputs
    assert policy.contract["apply_tanh_to_latent"] is c.bounded
    observations = torch.randn(16, inputs, generator=torch.Generator().manual_seed(32))
    with torch.inference_mode():
        for observation in observations:
            if stage == "student-dagger":
                teacher_obs = torch.zeros(1, 67)
                teacher_obs[:, -3:] = observation[-3:]
                commands = c.teacher.actor_obs_normalizer(teacher_obs)[:, -3:]
                latent = c.adapter(c.history_normalizer(observation[None, :-3]))
                expected = c.teacher.actor(torch.cat((latent, commands), -1)).numpy()[0]
            else:
                reference = c.teacher if stage == "teacher" else c.student
                expected = reference.act_inference({"policy": observation[None]}).numpy()[0]
            np.testing.assert_array_equal(policy(observation.numpy()), expected)


@pytest.mark.parametrize("normalize_teacher,normalize_history", [(False, False), (True, False), (False, True)])
def test_dagger_preserves_identity_normalization(checkpoints, normalize_teacher, normalize_history):
    c = checkpoints
    if not normalize_teacher:
        c.teacher.actor_obs_normalizer = torch.nn.Identity()
        torch.save({"model_state_dict": c.teacher.state_dict()}, c.teacher_path)
    if not normalize_history:
        c.adapter_payload["history_normalizer_state_dict"] = None
    torch.save(c.adapter_payload, c.adapter_path)
    policy = EncodedPolicy(c.adapter_path, "solo12-two-feet-IMU-student-dagger", {})
    obs = torch.randn(1, 408)
    teacher_obs = torch.zeros(1, 67)
    teacher_obs[:, -3:] = obs[:, -3:]
    with torch.inference_mode():
        history = c.history_normalizer(obs[:, :-3]) if normalize_history else obs[:, :-3]
        expected = c.teacher.actor(torch.cat((c.adapter(history), c.teacher.actor_obs_normalizer(teacher_obs)[:, -3:]), -1))
    np.testing.assert_array_equal(policy(obs.numpy()[0]), expected.numpy()[0])


def test_missing_teacher_can_be_relocated_explicitly(checkpoints):
    c = checkpoints
    c.adapter_payload["teacher_checkpoint"] = "/unavailable/cluster/teacher.pt"
    torch.save(c.adapter_payload, c.adapter_path)
    with pytest.raises(ValueError, match="--teacher-checkpoint"):
        EncodedPolicy(c.adapter_path, "solo12-two-feet-IMU-student-dagger", {})
    policy = EncodedPolicy(c.adapter_path, "solo12-two-feet-IMU-student-dagger", {}, c.teacher_path)
    assert policy.teacher_checkpoint == str(c.teacher_path)


def test_dagger_player_uses_the_modified_student_head(checkpoints):
    c = checkpoints
    actor = copy.deepcopy(c.teacher.actor)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.15)
    c.adapter_payload["student_actor_state_dict"] = actor.state_dict()
    c.adapter_payload["objective"] = {"finetune_student_actor": True}
    torch.save(c.adapter_payload, c.adapter_path)
    policy = EncodedPolicy(c.adapter_path, "solo12-two-feet-IMU-student-dagger", {})
    obs = torch.randn(1, 408)
    teacher_obs = torch.zeros(1, 67)
    teacher_obs[:, -3:] = obs[:, -3:]
    with torch.inference_mode():
        latent = c.adapter(c.history_normalizer(obs[:, :-3]))
        command = c.teacher.actor_obs_normalizer(teacher_obs)[:, -3:]
        expected = actor(torch.cat((latent, command), -1))
        old = c.teacher.actor(torch.cat((latent, command), -1))
    assert not torch.equal(expected, old)
    np.testing.assert_array_equal(policy(obs.numpy()[0]), expected.numpy()[0])
    c.adapter_payload.pop("student_actor_state_dict")
    torch.save(c.adapter_payload, c.adapter_path)
    with pytest.raises(ValueError, match="actor-fine-tuning"):
        EncodedPolicy(c.adapter_path, "solo12-two-feet-IMU-student-dagger", {})


def test_wrong_task_checkpoint_pair_fails(checkpoints):
    c = checkpoints
    for path, stage in ((c.teacher_path, "student-rl"), (c.adapter_path, "teacher"), (c.student_path, "teacher")):
        with pytest.raises(ValueError):
            EncodedPolicy(path, f"solo12-two-feet-IMU-{stage}", {})


@pytest.mark.parametrize("stage", ["teacher", "student-rl"])
def test_ppo_without_normalizers_stays_identity(checkpoints, stage):
    c = checkpoints
    reference = c.teacher if stage == "teacher" else c.student
    path = c.teacher_path if stage == "teacher" else c.student_path
    reference.actor_obs_normalizer = torch.nn.Identity()
    reference.critic_obs_normalizer = torch.nn.Identity()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["model_state_dict"] = reference.state_dict()
    torch.save(payload, path)
    policy = EncodedPolicy(path, f"solo12-two-feet-IMU-{stage}", {})
    observation = torch.randn(1, policy.input_dim)
    with torch.inference_mode():
        expected = reference.act_inference({"policy": observation}).numpy()[0]
    np.testing.assert_array_equal(policy(observation.numpy()[0]), expected)


def test_saved_semantics_and_parameter_free_activation(checkpoints):
    c = checkpoints
    policy = EncodedPolicy(c.teacher_path, "solo12-two-feet-IMU-teacher", {"apply_tanh_to_latent": not c.bounded})
    assert policy.contract["apply_tanh_to_latent"] is c.bounded
    params = c.tmp_path / "params"
    params.mkdir()
    (params / "solo12_imu_contract.json").write_text(json.dumps({
        "observation_contract": {"include_lin_acc_steps_in_imu_obs": 4}}))
    with pytest.raises(ValueError, match="conflicts with the saved"):
        EncodedPolicy(c.teacher_path, "solo12-two-feet-IMU-teacher", {"include_lin_acc_steps_in_imu_obs": 0})


@pytest.mark.parametrize("override", [
    {"base_obs_from_imu": True}, {"remove_root_lin_vel_b_from_actor_obs": True},
    {"imu_ekf_processed_inputs": False}, {"use_rotMat_on_imu_encoder": True},
    {"teacher_foot_force_obs": "invalid"}, {"teacher_include_base_force": False},
    {"include_foot_height_obs": True}, {"decimation": 2},
])
def test_bad_observation_contract_fails_before_rollout(checkpoints, override):
    with pytest.raises(ValueError):
        EncodedPolicy(checkpoints.teacher_path, "solo12-two-feet-IMU-teacher", override)


def test_task_and_override_parser():
    for task in TASK_MODELS:
        assert player.build_parser().parse_args(["--checkpoint=x.pt", f"--task={task}"]).task == task
    values, ignored = player.consume_env_overrides([
        "env.teacher_foot_force_obs=vector", "env.include_foot_height_obs=True",
        "env.teacher_include_base_force=False", "env.apply_tanh_to_latent=False",
        "env.base_imu_history_policy_steps=2", "env.use_rotMat_on_imu_encoder=False",
    ])
    assert not ignored
    assert values["teacher_foot_force_obs"] == "vector"
    assert values["base_imu_history_policy_steps"] == 2


def test_student_physics_history_matches_independent_endpoint_oracle(monkeypatch):
    env = player.Solo12Mujoco(MODEL, policy_model="base_imu_student_rl", env_overrides={"base_imu_clip": False})
    original = mujoco.mj_step
    expected = []
    _, angular, linear = env._base_imu_kinematics()
    previous = np.r_[angular, linear]

    def step_and_measure(model, data):
        nonlocal previous
        original(model, data)
        probe = mujoco.MjData(model)
        probe.qpos[:] = data.qpos
        probe.qvel[:] = data.qvel
        mujoco.mj_forward(model, probe)
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(model, probe, mujoco.mjtObj.mjOBJ_BODY, env.base_id, velocity, 0)
        derivative = (velocity - previous) / .005
        previous = velocity.copy()
        rotation = Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_matrix()
        lever = rotation @ (np.array((.01, 0., 0.)) - model.body_ipos[env.base_id])
        acc = derivative[3:] + np.cross(derivative[:3], lever) + np.cross(velocity[:3], np.cross(velocity[:3], lever))
        expected.append(np.r_[data.qpos[env.joint_qpos] - player.SAFE_Q, data.qvel[env.joint_dof],
                              rotation.T @ velocity[:3], rotation.T @ acc, rotation.T @ [0., 0., -1.], env.action])

    monkeypatch.setattr(mujoco, "mj_step", step_and_measure)
    for index in range(5):
        env.step(np.linspace(-.3, .7, 12) + index / 100)
    assert len(expected) == 20
    obs = env.observation((.2, -.1, .3))
    np.testing.assert_allclose(obs[:-3].reshape(13, 45), expected[-13:], atol=2e-11, rtol=1e-11)
    np.testing.assert_array_equal(obs[-3:], (.2, -.1, .3))
    for _ in range(4):
        np.testing.assert_array_equal(env.observation((.2, -.1, .3)), obs)
    env.reset()
    np.testing.assert_array_equal(env.observation((0., 0., 0.)), 0.)
    env.step(np.zeros(12))
    assert np.isfinite(env.observation((0., 0., 0.))).all()


@pytest.mark.parametrize("force_kind", ["none", "norm", "vector"])
@pytest.mark.parametrize("height", [False, True])
def test_teacher_layout_matches_production_observation(force_kind, height):
    env = player.Solo12Mujoco(MODEL, policy_model="base_imu_teacher", env_overrides={
        "teacher_foot_force_obs": force_kind, "include_foot_height_obs": height})
    env.step(np.linspace(-.8, .4, 12))
    env._base_push_force_b[:] = (1., -3., 2.)
    env._calf_forces_w[:] = np.arange(12).reshape(4, 3) - 2
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py"
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Solo12Env")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_get_teacher_encoder_obs")
    tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float64)
    rotation, angular, linear = env._base_imu_kinematics()
    math_utils = SimpleNamespace(quat_apply_inverse=lambda q, f: f @ tensor(rotation))
    namespace = {"torch": torch, "math_utils": math_utils}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    probe = mujoco.MjData(env.model)
    probe.qpos[:] = env.data.qpos
    mujoco.mj_kinematics(env.model, probe)
    data = SimpleNamespace(projected_gravity_b=tensor([rotation.T @ [0., 0., -1.]]),
                           root_lin_vel_b=tensor([rotation.T @ linear]), root_ang_vel_b=tensor([rotation.T @ angular]),
                           root_quat_w=tensor(env.data.qpos[None, 3:7]))
    reference = SimpleNamespace(
        cfg=SimpleNamespace(include_foot_height_obs=height, action_obs_is_last_executed=False,
                            include_lin_acc_steps_in_imu_obs=4, teacher_include_base_force=True,
                            teacher_foot_force_obs=force_kind),
        _robot=SimpleNamespace(data=data), _actions=tensor(env.action[None]), num_envs=1,
        _get_joint_state_obs=lambda **_: tensor([np.r_[env.data.qpos[env.joint_qpos] - player.SAFE_Q, env.data.qvel[env.joint_dof]]]),
        _get_foot_height_obs=lambda: tensor(probe.xpos[None, env.calf_body_ids, 2]),
        _teacher_lin_acc_history=tensor(env._teacher_lin_acc_history[None]),
        _base_push_forces_b=tensor(env._base_push_force_b[None, None]), _feet_body_ids=list(range(4)),
        _contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=tensor(env._calf_forces_w[None]))),
    )
    expected = namespace["_get_teacher_encoder_obs"](reference, corrupt=False).numpy()[0]
    np.testing.assert_allclose(env.observation((.4, -.2, .1)), np.r_[expected, (.4, -.2, .1)], atol=1e-14)


def test_teacher_acceleration_uses_origin_not_imu_and_is_not_clipped():
    env = player.Solo12Mujoco(MODEL, policy_model="base_imu_teacher")
    rotation = Rotation.from_euler("xyz", (.3, -.6, .8)).as_matrix()
    omega_b = np.array((5., -8., 10.))
    env.data.qpos[3:7] = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
    env.data.qvel[3:6] = omega_b
    env.data.qvel[:3] = -rotation @ np.cross(omega_b, env.model.body_ipos[env.base_id])
    env._previous_imu_com_velocity = np.zeros(3)
    env._previous_imu_ang_velocity = rotation @ omega_b
    env._record_base_imu_sample()
    expected = np.cross(omega_b, np.cross(omega_b, -env.model.body_ipos[env.base_id]))
    np.testing.assert_allclose(env._teacher_lin_acc_history[-1], expected, atol=1e-12)
    sensor = np.cross(omega_b, np.cross(omega_b, player.IMU_OFFSET_B - env.model.body_ipos[env.base_id]))
    assert not np.allclose(sensor, expected)


def test_teacher_observes_held_push_and_reset_clears_all_privileged_history():
    import interactive

    env = player.Solo12Mujoco(MODEL, policy_model="base_imu_teacher")
    rotation = Rotation.from_euler("xyz", (.3, -.6, .8))
    env.data.qpos[3:7] = np.roll(rotation.as_quat(), 1)
    mujoco.mj_forward(env.model, env.data)
    interactive.apply_body_force(env, (1., -2., 3., .1, .2, 0.))
    env.step(np.ones(12) * .2)
    np.testing.assert_allclose(env._base_push_force_b, (1., -2., 3.), atol=1e-14)
    np.testing.assert_allclose(env.observation((0., 0., 0.))[-10:-7], (1., -2., 3.), atol=1e-14)
    env.reset()
    np.testing.assert_array_equal(env._base_push_force_b, 0.)
    np.testing.assert_array_equal(env._teacher_lin_acc_history, 0.)
    np.testing.assert_array_equal(env._calf_forces_w, 0.)
    np.testing.assert_array_equal(env.data.qfrc_applied, 0.)


def test_contact_sign_and_non_ground_body_support_match_known_weight():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001"/>
      <worldbody><body name="support"><geom type="plane" size="2 2 .1"/></body>
      <body name="calf" pos="0 0 .1"><freejoint/><geom type="sphere" size=".1" mass="2"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    calf = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calf")
    support = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "support")
    raw = SimpleNamespace(model=model, data=data, calf_body_ids=np.array([calf, support, -10, -11]),
                          _calf_forces_w=np.zeros((4, 3)))
    player.Solo12Mujoco._record_calf_contacts(raw)
    np.testing.assert_allclose(raw._calf_forces_w[0], [0., 0., 19.62], atol=1e-7)
    np.testing.assert_allclose(raw._calf_forces_w[1], [0., 0., -19.62], atol=1e-7)


def test_encoded_sampling_does_not_modify_legacy_dynamics():
    simple = player.Solo12Mujoco(MODEL)
    teacher = player.Solo12Mujoco(MODEL, policy_model="base_imu_teacher")
    student = player.Solo12Mujoco(MODEL, policy_model="base_imu_student_rl")
    for action in np.random.default_rng(11).uniform(-1., 1., (40, 12)):
        for env in (simple, teacher, student):
            env.step(action)
        np.testing.assert_array_equal(simple.data.qpos, teacher.data.qpos)
        np.testing.assert_array_equal(simple.data.qpos, student.data.qpos)
        np.testing.assert_array_equal(simple.data.qvel, teacher.data.qvel)
        np.testing.assert_array_equal(simple.data.qvel, student.data.qvel)
