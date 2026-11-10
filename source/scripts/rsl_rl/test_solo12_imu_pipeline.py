"""CPU checks for real base-IMU networks and the public DAgger-to-PPO loader."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.networks import EmpiricalNormalization
from tensordict import TensorDict

import solo12_imu_pipeline as pipeline

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/base_imu_actor_critic.py"
spec = importlib.util.spec_from_file_location("solo12_base_imu_test_models", MODEL_PATH)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)
sym_spec = importlib.util.spec_from_file_location("solo12_pipeline_symmetry", ROOT / "source/scripts/skrl/solo12_symmetry.py")
symmetry = importlib.util.module_from_spec(sym_spec)
sym_spec.loader.exec_module(symmetry)


def train_definitions():
    path = Path(__file__).with_name("train.py")
    names = {
        "_configure_student_policy_from_dagger_adapter", "_checkpoint_model_state_dict",
        "_initialize_student_from_dagger_adapter", "_copy_normalizer_state", "_copy_dagger_actor_normalizer_state",
    }
    nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    ns = {"torch": torch, "_resolve_existing_model_path": lambda path: path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


class ContractEnv(SimpleNamespace):
    def refresh_runtime_dependent_config(self):
        self.teacher_encoder_obs_dim = (
            45 + 4 * self.include_foot_height_obs + 3 * self.include_lin_acc_steps_in_imu_obs
            + 3 * self.teacher_include_base_force + {"none": 0, "norm": 4, "vector": 12}[self.teacher_foot_force_obs]
        )
        self.teacher_critic_obs_dim = self.teacher_encoder_obs_dim + 3
        self.base_imu_history_sample_dim = 24 + 12 + (
            (15 if self.use_rotMat_on_imu_encoder else 9) if self.imu_ekf_processed_inputs else 6
        )
        self.base_imu_history_length = self.base_imu_history_policy_steps * self.decimation + 1


def contract_env():
    cfg = ContractEnv(
        policy_model="base_imu_student_rl", include_foot_height_obs=False, include_lin_acc_steps_in_imu_obs=4,
        teacher_include_base_force=True, teacher_foot_force_obs="norm", teacher_latent_activation="identity", action_obs_is_last_executed=False,
        base_imu_history_action_is_last_executed=False, imu_ekf_processed_inputs=True,
        use_rotMat_on_imu_encoder=False, decimation=4, sim=SimpleNamespace(dt=0.005),
        base_imu_history_policy_steps=2, base_imu_tcn_channels=8, base_imu_tcn_kernel_size=3,
        base_imu_tcn_activation="elu", front_back_asymetry=True,
    )
    cfg.refresh_runtime_dependent_config()
    return cfg


@pytest.fixture
def chain(tmp_path):
    torch.manual_seed(71)
    cfg = contract_env()
    obs = TensorDict({"policy": torch.randn(6, 67), "critic": torch.randn(6, 67)}, batch_size=[6])
    teacher = models.Solo12BaseImuTeacherActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
        teacher_encoder_obs_dim=64, teacher_latent_dim=7, teacher_encoder_hidden_dims=[17, 11],
        actor_hidden_dims=[16, 12], critic_hidden_dims=[19, 13], actor_obs_normalization=True,
        critic_obs_normalization=True, noise_std_type="log",
    )
    teacher.update_normalization(obs)
    teacher.eval()
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, teacher_path)
    adapter = models.BaseImuTcnEncoder(history_len=9, sample_dim=45, channels=8, latent_dim=7, kernel_size=3)
    history = torch.randn(6, 405)
    normalizer = EmpiricalNormalization(405)
    normalizer.update(history)
    normalizer.eval()
    dims = {"history_len": 9, "history_sample_dim": 45, "teacher_obs_dim": 67, "latent_dim": 7}
    checkpoint = {
        **pipeline.adapter_metadata(cfg, SimpleNamespace(activation="elu"), str(teacher_path),
                                    supervise_action=True, action_loss_coef=0.25),
        "adapter_state_dict": adapter.state_dict(), "history_normalizer_state_dict": normalizer.state_dict(),
        "dims": dims,
    }
    return SimpleNamespace(cfg=cfg, teacher=teacher, teacher_path=teacher_path, adapter=adapter,
                           normalizer=normalizer, checkpoint=checkpoint, history=history, obs=obs, tmp_path=tmp_path)


@pytest.mark.parametrize("normalize_history", [True, False])
@pytest.mark.parametrize("normalize_teacher", [True, False])
def test_public_handoff_restores_exact_actions_critic_and_all_tensors(chain, normalize_history, normalize_teacher):
    c = chain
    if not normalize_history:
        c.checkpoint["history_normalizer_state_dict"] = None
    if not normalize_teacher:
        c.teacher.actor_obs_normalizer = torch.nn.Identity()
        c.teacher.critic_obs_normalizer = torch.nn.Identity()
        torch.save({"model_state_dict": c.teacher.state_dict()}, c.teacher_path)
    adapter_path = c.tmp_path / "adapter.pt"
    torch.save(c.checkpoint, adapter_path)
    env_cfg = contract_env()
    env_cfg.base_imu_history_policy_steps = 3
    env_cfg.base_imu_tcn_channels = 64
    policy_cfg = SimpleNamespace()
    ns = train_definitions()
    ns["_configure_student_policy_from_dagger_adapter"](env_cfg, policy_cfg, str(adapter_path))
    assert env_cfg.base_imu_history_length == 9
    assert policy_cfg.actor_hidden_dims == [16, 12]
    assert policy_cfg.critic_hidden_dims == [19, 13]
    student_obs = TensorDict({"policy": torch.cat((c.history, c.obs["policy"][:, -3:]), -1),
                             "critic": c.obs["critic"]}, batch_size=[6])
    student = models.Solo12BaseImuStudentActorCritic(
        student_obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(policy_cfg)
    )
    ns["_initialize_student_from_dagger_adapter"](SimpleNamespace(alg=SimpleNamespace(policy=student)), str(adapter_path))
    student.eval()
    assert student.critic_history_encoder is None
    history = c.normalizer(c.history) if normalize_history else c.history
    command = c.teacher.actor_obs_normalizer(c.obs["policy"])[:, -3:]
    expected_actions = c.teacher.actor(torch.cat((c.adapter(history), command), -1))
    torch.testing.assert_close(student.act_inference(student_obs), expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(student.evaluate(student_obs), c.teacher.evaluate(c.obs), rtol=0, atol=0)
    for key, value in c.adapter.state_dict().items():
        torch.testing.assert_close(student.actor_history_encoder.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(student.log_std, c.teacher.log_std, rtol=0, atol=0)
    # Policy actions cannot depend on clean privileged signals; values can.
    other_obs = student_obs.clone()
    other_obs["critic"] += 100
    torch.testing.assert_close(student.act_inference(other_obs), expected_actions, rtol=0, atol=0)


def test_missing_adapter_tensor_is_not_silently_skipped(chain):
    c = chain
    c.checkpoint["adapter_state_dict"].pop("to_latent.1.bias")
    path = c.tmp_path / "broken.pt"
    torch.save(c.checkpoint, path)
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    obs = TensorDict({"policy": torch.zeros(2, 408), "critic": torch.zeros(2, 67)}, batch_size=[2])
    student = models.Solo12BaseImuStudentActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(cfg))
    with pytest.raises(RuntimeError, match="exact transfer"):
        train_definitions()["_initialize_student_from_dagger_adapter"](SimpleNamespace(alg=SimpleNamespace(policy=student)), str(path))


def test_contract_sidecar_and_teacher_architecture_round_trip(chain):
    c = chain
    cfg = SimpleNamespace(actor_hidden_dims=[16, 12], critic_hidden_dims=[19, 13], activation="elu")
    pipeline.save_training_contract(str(c.tmp_path), c.cfg, cfg)
    c.cfg.policy_model = "base_imu_student_dagger"
    c.cfg.teacher_foot_force_obs = "vector"
    c.cfg.include_lin_acc_steps_in_imu_obs = 1
    cfg.teacher_encoder_obs_dim = 45
    cfg.teacher_encoder_hidden_dims = [256, 128, 64]
    cfg.teacher_latent_dim = 32
    pipeline.configure_teacher_for_dagger(c.cfg, cfg, str(c.teacher_path))
    assert c.cfg.teacher_foot_force_obs == "norm"
    assert c.cfg.include_lin_acc_steps_in_imu_obs == 4
    assert cfg.teacher_encoder_hidden_dims == [17, 11]
    assert cfg.teacher_latent_dim == 7


@pytest.mark.parametrize("bad_length", [0, 8, 12])
def test_history_endpoint_convention_rejects_bad_metadata(chain, bad_length):
    chain.checkpoint["dims"]["history_len"] = bad_length
    with pytest.raises(ValueError, match="Endpoint-inclusive"):
        pipeline.configure_base_imu_student(chain.cfg, SimpleNamespace(), chain.checkpoint, str(chain.teacher_path))


def test_action_loss_updates_only_the_history_encoder(chain):
    c = chain
    for parameter in c.teacher.parameters():
        parameter.requires_grad_(False)
    z = c.adapter(c.history)
    action = c.teacher.actor(torch.cat((z, c.obs["policy"][:, -3:]), -1))
    action.square().mean().backward()
    assert all(p.grad is None for p in c.teacher.parameters())
    assert c.adapter.conv_stack[0].conv.weight.grad.abs().sum() > 0


def test_new_teacher_can_match_the_students_bounded_latent_without_changing_old_teacher_outputs(chain):
    teacher = chain.teacher
    inputs = chain.obs["policy"] * 1000
    original = teacher.teacher_encoder(inputs[:, :64])
    torch.testing.assert_close(teacher.encode_teacher(inputs), original, rtol=0, atol=0)
    teacher.teacher_latent_activation = "tanh"
    bounded = teacher.encode_teacher(inputs)
    torch.testing.assert_close(bounded, torch.tanh(original), rtol=0, atol=0)
    assert bounded.abs().max() <= 1


@pytest.mark.parametrize("foot_kind", ["none", "norm", "vector"])
@pytest.mark.parametrize("front_back", [False, True])
def test_extended_symmetry_preserves_frames_and_leg_identity(foot_kind, front_back):
    cfg = contract_env()
    cfg.teacher_foot_force_obs = foot_kind
    cfg.front_back_asymetry = not front_back
    cfg.refresh_runtime_dependent_config()
    env = SimpleNamespace(cfg=cfg)
    privileged = torch.arange(1, 1 + cfg.teacher_critic_obs_dim, dtype=torch.float32).reshape(1, -1)
    history = torch.randn(1, 408)
    obs = TensorDict({"policy": history, "critic": privileged}, batch_size=[1])
    augmented, _ = symmetry.compute_symmetric_observations_actions(env, obs=obs)
    assert augmented.batch_size[0] == (4 if front_back else 2)
    mirrored = augmented[1:2]
    twice, _ = symmetry.compute_symmetric_observations_actions(env, obs=mirrored)
    torch.testing.assert_close(twice["critic"][1], privileged[0], rtol=0, atol=0)
    expected_vectors = privileged[:, 45:60].reshape(1, 5, 3) * torch.tensor([1., -1., 1.])
    torch.testing.assert_close(mirrored["critic"][:, 45:60], expected_vectors.flatten(1))
    if foot_kind != "none":
        feet = privileged[:, 60:-3].reshape(1, 4, -1)[:, [1, 0, 3, 2]]
        if foot_kind == "vector":
            feet = feet * torch.tensor([1., -1., 1.])
        torch.testing.assert_close(mirrored["critic"][:, 60:-3], feet.flatten(1))
    if front_back:
        fore_aft, _ = symmetry.compute_symmetric_observations_actions(env, obs=augmented[2:3])
        torch.testing.assert_close(fore_aft["critic"][2], privileged[0], rtol=0, atol=0)
        vector_y = privileged[:, 45:60].reshape(1, 5, 3) * torch.tensor([-1., 1., 1.])
        torch.testing.assert_close(augmented["critic"][2:3, 45:60], vector_y.flatten(1))
    # The teacher actor and the student's critic use the same privileged transform.
    cfg.policy_model = "base_imu_teacher"
    teacher_obs = TensorDict({"policy": privileged, "critic": privileged}, batch_size=[1])
    teacher_aug, _ = symmetry.compute_symmetric_observations_actions(env, obs=teacher_obs)
    torch.testing.assert_close(teacher_aug["policy"], augmented["critic"], rtol=0, atol=0)


def test_legacy_base_adapter_without_race_layout_is_supported(chain):
    chain.checkpoint.pop("layout")
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(chain.cfg, cfg, chain.checkpoint, str(chain.teacher_path))
    assert cfg.history_len == 9
    assert cfg.tcn_channels == 8


def test_single_endpoint_history_is_a_valid_no_history_ablation(chain):
    c = chain
    c.checkpoint["dims"]["history_len"] = 1
    c.checkpoint["layout"]["history_len"] = 1
    adapter = models.BaseImuTcnEncoder(history_len=1, sample_dim=45, channels=8, latent_dim=7, kernel_size=3)
    c.checkpoint["adapter_state_dict"] = adapter.state_dict()
    c.checkpoint["history_normalizer_state_dict"] = None
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    assert c.cfg.base_imu_history_policy_steps == 0
    path = c.tmp_path / "single.pt"
    torch.save(c.checkpoint, path)
    obs = TensorDict({"policy": torch.randn(2, 48), "critic": torch.randn(2, 67)}, batch_size=[2])
    student = models.Solo12BaseImuStudentActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(cfg))
    train_definitions()["_initialize_student_from_dagger_adapter"](SimpleNamespace(alg=SimpleNamespace(policy=student)), str(path))
    assert torch.isfinite(student.act_inference(obs)).all()


def test_adapter_cannot_initialize_a_race_or_teacher_task(chain):
    chain.cfg.policy_model = "base_imu_teacher"
    with pytest.raises(ValueError, match="base_imu_student_rl"):
        pipeline.configure_base_imu_student(chain.cfg, SimpleNamespace(), chain.checkpoint, str(chain.teacher_path))
