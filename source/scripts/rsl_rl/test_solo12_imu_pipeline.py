"""CPU checks for real base-IMU networks and the public DAgger-to-PPO loader."""

import ast
import argparse
import copy
import importlib.util
import math
import sys
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


@pytest.mark.parametrize("apply_tanh", [True, False])
def test_tcn_latent_switch_changes_only_the_final_activation(apply_tanh):
    encoder = models.BaseImuTcnEncoder(
        history_len=9, sample_dim=45, channels=8, latent_dim=7, kernel_size=3,
        apply_tanh_to_latent=apply_tanh,
    )
    with torch.no_grad():
        encoder.to_latent[1].weight.zero_()
        encoder.to_latent[1].bias.fill_(3.)
    expected = torch.tanh(torch.tensor(3.)) if apply_tanh else torch.tensor(3.)
    torch.testing.assert_close(encoder(torch.zeros(2, 405)), expected.expand(2, 7))


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
        self.teacher_latent_activation = "tanh" if self.apply_tanh_to_latent else "identity"
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
        teacher_include_base_force=True, teacher_foot_force_obs="norm", apply_tanh_to_latent=True, action_obs_is_last_executed=False,
        base_imu_history_action_is_last_executed=False, imu_ekf_processed_inputs=True,
        use_rotMat_on_imu_encoder=False, decimation=4, sim=SimpleNamespace(dt=0.005),
        base_imu_history_policy_steps=2, base_imu_tcn_channels=8, base_imu_tcn_kernel_size=3,
        base_imu_tcn_activation="elu", front_back_asymetry=True,
    )
    cfg.refresh_runtime_dependent_config()
    return cfg


@pytest.fixture(params=[True, False], ids=["bounded", "unbounded"])
def chain(tmp_path, request):
    torch.manual_seed(71)
    cfg = contract_env()
    cfg.apply_tanh_to_latent = request.param
    cfg.refresh_runtime_dependent_config()
    obs = TensorDict({"policy": torch.randn(6, 67), "critic": torch.randn(6, 67)}, batch_size=[6])
    teacher = models.Solo12BaseImuTeacherActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
        teacher_encoder_obs_dim=64, teacher_latent_dim=7, teacher_encoder_hidden_dims=[17, 11],
        actor_hidden_dims=[16, 12], critic_hidden_dims=[19, 13], actor_obs_normalization=True,
        critic_obs_normalization=True, noise_std_type="log",
        teacher_latent_activation=cfg.teacher_latent_activation,
    )
    teacher.update_normalization(obs)
    teacher.eval()
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, teacher_path)
    adapter = models.BaseImuTcnEncoder(
        history_len=9, sample_dim=45, channels=8, latent_dim=7, kernel_size=3,
        apply_tanh_to_latent=cfg.apply_tanh_to_latent,
    )
    history = torch.randn(6, 405)
    normalizer = EmpiricalNormalization(405)
    normalizer.update(history)
    normalizer.eval()
    dims = {"history_len": 9, "history_sample_dim": 45, "teacher_obs_dim": 67, "latent_dim": 7}
    checkpoint = {
        **pipeline.adapter_metadata(cfg, SimpleNamespace(activation="elu"), str(teacher_path),
                                    action_loss_coef=0.25),
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
    assert policy_cfg.apply_tanh_to_latent == c.cfg.apply_tanh_to_latent
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


def dagger_cli(monkeypatch, *args):
    path = Path(__file__).with_name("train_solo12_base_imu_dagger.py")
    nodes = ast.parse(path.read_text()).body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "parser" for t in node.targets))
    stop = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Attribute) and t.attr == "argv" for t in node.targets))
    monkeypatch.setattr(sys, "argv", [str(path), *args])
    ns = {"argparse": argparse, "math": math,
          "cli_args": SimpleNamespace(add_rsl_rl_args=lambda parser: None),
          "AppLauncher": SimpleNamespace(add_app_launcher_args=lambda parser: None)}
    exec(compile(ast.Module(body=nodes[start:stop], type_ignores=[]), str(path), "exec"), ns)
    assert not ns["hydra_args"]
    return ns["args_cli"]


@pytest.mark.parametrize("args,finetune,supervise", [
    ([], False, True), (["--action-loss-coef", "0"], False, False),
    (["--finetune-student-actor"], True, True),
    (["--supervise-action", "--action-loss-coef", "0"], False, False),
])
def test_action_coefficient_controls_supervision_without_a_separate_flag(monkeypatch, args, finetune, supervise):
    parsed = dagger_cli(monkeypatch, *args)
    assert parsed.finetune_student_actor is finetune
    assert parsed.supervise_action is supervise


@pytest.mark.parametrize("args", [
    ["--finetune-student-actor", "--action-loss-coef", "0"],
    ["--action-loss-coef", "-0.1"], ["--action-loss-coef", "nan"], ["--action-loss-coef", "inf"],
])
def test_invalid_action_objective_fails_before_simulator_launch(monkeypatch, args):
    with pytest.raises(SystemExit) as exc:
        dagger_cli(monkeypatch, *args)
    assert exc.value.code == 2


@pytest.mark.parametrize("finetune,coefficient", [(False, 0.), (False, 0.25), (True, 0.25)])
def test_real_supervised_loop_updates_only_selected_networks(chain, finetune, coefficient):
    c = chain
    c.teacher.requires_grad_(False)
    student_actor = copy.deepcopy(c.teacher.actor).requires_grad_(True) if finetune else None
    parameters = list(c.adapter.parameters()) + (list(student_actor.parameters()) if finetune else [])
    teacher_before = {k: v.clone() for k, v in c.teacher.state_dict().items()}
    adapter_before = {k: v.clone() for k, v in c.adapter.state_dict().items()}
    actor_before = {k: v.clone() for k, v in student_actor.state_dict().items()} if finetune else None
    batch = (c.history, torch.randn(6, 7), torch.randn(6, 3), torch.randn(6, 12))
    path = Path(__file__).with_name("train_solo12_base_imu_dagger.py")
    main = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop = next(n for n in ast.walk(main) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "range(int(updates))")
    ns = {"torch": torch, "nn": torch.nn, "teacher": c.teacher, "adapter": c.adapter,
          "student_actor": student_actor, "history_normalizer": None, "device": torch.device("cpu"),
          "optimizer": torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.),
          "trainable_parameters": parameters, "updates": 1, "mini_batch_size": 6,
          "buffer": SimpleNamespace(sample=lambda *_: batch),
          "args_cli": SimpleNamespace(action_loss_coef=coefficient, supervise_action=coefficient != 0., max_grad_norm=1.),
          "loss_sum": 0., "latent_loss_sum": 0., "action_loss_sum": 0.}
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), ns)
    assert all(torch.equal(v, c.teacher.state_dict()[k]) for k, v in teacher_before.items())
    assert all(p.grad is None for p in c.teacher.parameters())
    assert any(not torch.equal(v, c.adapter.state_dict()[k]) for k, v in adapter_before.items())
    assert (ns["action_loss_sum"] > 0.) is (coefficient != 0.)
    if finetune:
        assert any(not torch.equal(v, student_actor.state_dict()[k]) for k, v in actor_before.items())


def test_public_handoff_uses_modified_actor_but_original_privileged_critic(chain):
    c = chain
    actor = copy.deepcopy(c.teacher.actor)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.12)
    c.checkpoint["student_actor_state_dict"] = actor.state_dict()
    c.checkpoint["objective"]["finetune_student_actor"] = True
    path = c.tmp_path / "finetuned.pt"
    torch.save(c.checkpoint, path)
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    obs = TensorDict({"policy": torch.cat((c.history, c.obs["policy"][:, -3:]), -1),
                      "critic": c.obs["critic"]}, batch_size=[6])
    student = models.Solo12BaseImuStudentActorCritic(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(cfg))
    train_definitions()["_initialize_student_from_dagger_adapter"](SimpleNamespace(alg=SimpleNamespace(policy=student)), str(path))
    student.eval()
    command = c.teacher.actor_obs_normalizer(c.obs["policy"])[:, -3:]
    expected = actor(torch.cat((c.adapter(c.normalizer(c.history)), command), -1))
    torch.testing.assert_close(student.act_inference(obs), expected, rtol=0, atol=0)
    torch.testing.assert_close(student.evaluate(obs), c.teacher.evaluate(c.obs), rtol=0, atol=0)
    for k, v in c.teacher.critic.state_dict().items():
        torch.testing.assert_close(student.critic.state_dict()[k], v, rtol=0, atol=0)
    for k, v in actor.state_dict().items():
        torch.testing.assert_close(student.actor.state_dict()[k], v, rtol=0, atol=0)
    torch.testing.assert_close(student.log_std, c.teacher.log_std, rtol=0, atol=0)


def dagger_definitions(*names):
    path = Path(__file__).with_name("train_solo12_base_imu_dagger.py")
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"torch": torch}
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_collection_uses_modified_student_head_but_keeps_teacher_labels_fixed(chain):
    c = chain
    c.teacher.requires_grad_(False)
    actor = copy.deepcopy(c.teacher.actor)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.1)
    collect = dagger_definitions("_teacher_targets_and_student_action")["_teacher_targets_and_student_action"]
    args = dict(teacher=c.teacher, adapter=c.adapter, history_normalizer=c.normalizer,
                teacher_obs_raw=c.obs["policy"], history_raw=c.history, stochastic_actions=False)
    original = collect(**args)
    modified = collect(**args, student_actor=actor)
    for index in (0, 1, 2):
        torch.testing.assert_close(original[index], modified[index], rtol=0, atol=0)
    assert not torch.equal(original[3], modified[3])
    expected = actor(torch.cat((c.adapter(c.normalizer(c.history)), modified[1]), -1))
    torch.testing.assert_close(modified[3], expected, rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [
    {"objective": {"finetune_student_actor": True}},
    {"objective": {"finetune_student_actor": False}, "student_actor_state_dict": {}},
    {"student_actor_state_dict": 123},
])
def test_inconsistent_saved_actor_is_never_silently_ignored(checkpoint):
    with pytest.raises(ValueError, match="student_actor_state_dict"):
        pipeline.adapter_student_actor_state(checkpoint)


@pytest.mark.parametrize("finetune,coefficient", [(False, 0.), (False, 0.25), (True, 0.25)])
def test_saved_objective_records_supervision_and_head_training_separately(chain, finetune, coefficient):
    metadata = pipeline.adapter_metadata(chain.cfg, SimpleNamespace(), str(chain.teacher_path),
                                         action_loss_coef=coefficient, finetune_student_actor=finetune)
    assert metadata["objective"] == {"supervise_action": coefficient != 0.,
                                     "action_loss_coef": coefficient, "finetune_student_actor": finetune}


def test_isaac_adapter_player_uses_saved_modified_head(chain):
    c = chain
    actor = copy.deepcopy(c.teacher.actor)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.12)
    c.checkpoint["student_actor_state_dict"] = actor.state_dict()
    c.checkpoint["objective"]["finetune_student_actor"] = True
    path = Path(__file__).with_name("play_direct_0325.py")
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {"BaseImuDaggerAdapterPolicy", "_build_base_imu_dagger_adapter_policy"}]
    ns = {"torch": torch, "TensorDict": TensorDict, "BaseImuTcnEncoder": models.BaseImuTcnEncoder,
          "EmpiricalNormalization": EmpiricalNormalization,
          "_resolve_dagger_teacher_checkpoint": lambda _: str(c.teacher_path),
          "_load_base_imu_teacher_policy": lambda **_: copy.deepcopy(c.teacher)}
    exec(compile(ast.Module(body=ast.parse("from __future__ import annotations").body + nodes,
                            type_ignores=[]), str(path), "exec"), ns)
    cfg = SimpleNamespace(**vars(c.cfg))
    cfg.base_imu_history_flat_dim = 405
    obs = TensorDict({"policy": torch.cat((c.obs["policy"], c.history, c.obs["policy"][:, -3:]), -1)}, batch_size=[6])
    env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu", cfg=cfg), num_actions=12,
                          get_observations=lambda: obs)
    policy = ns["_build_base_imu_dagger_adapter_policy"](
        adapter_checkpoint=c.checkpoint, vec_env=env, agent_cfg=SimpleNamespace(policy=None), resume_path="adapter.pt")
    command = c.teacher.actor_obs_normalizer(c.obs["policy"])[:, -3:]
    expected = actor(torch.cat((c.adapter(c.normalizer(c.history)), command), -1))
    torch.testing.assert_close(policy(obs), expected, rtol=0, atol=0)


def test_new_teacher_can_match_the_students_bounded_latent_without_changing_old_teacher_outputs(chain):
    teacher = chain.teacher
    teacher.teacher_latent_activation = "identity"
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
    adapter = models.BaseImuTcnEncoder(
        history_len=1, sample_dim=45, channels=8, latent_dim=7, kernel_size=3,
        apply_tanh_to_latent=c.cfg.apply_tanh_to_latent,
    )
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


def test_latent_activation_survives_teacher_adapter_and_ppo_metadata(chain):
    c = chain
    expected = c.cfg.apply_tanh_to_latent
    pipeline.save_training_contract(str(c.tmp_path), c.cfg, SimpleNamespace())
    c.cfg.apply_tanh_to_latent = not expected
    pipeline.restore_latent_activation_from_checkpoint(c.cfg, str(c.teacher_path))
    assert c.cfg.apply_tanh_to_latent == expected
    assert c.cfg.teacher_latent_activation == ("tanh" if expected else "identity")
    adapter_path = c.tmp_path / "adapter.pt"
    torch.save(c.checkpoint, adapter_path)
    c.cfg.apply_tanh_to_latent = not expected
    pipeline.restore_latent_activation_from_checkpoint(c.cfg, str(adapter_path))
    assert c.cfg.apply_tanh_to_latent == expected


def test_legacy_adapter_defaults_to_its_original_tanh(chain):
    c = chain
    c.checkpoint["layout"].pop("apply_tanh_to_latent")
    c.checkpoint["observation_contract"].pop("apply_tanh_to_latent")
    c.cfg.apply_tanh_to_latent = False
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    assert cfg.apply_tanh_to_latent is True


def test_inconsistent_latent_metadata_is_rejected(chain):
    c = chain
    c.checkpoint["layout"]["apply_tanh_to_latent"] = not c.cfg.apply_tanh_to_latent
    with pytest.raises(ValueError, match="disagree on apply_tanh_to_latent"):
        pipeline.configure_base_imu_student(c.cfg, SimpleNamespace(), c.checkpoint, str(c.teacher_path))


def test_standalone_ppo_checkpoint_keeps_activation_without_mutating_existing_infos(chain):
    c = chain
    expected = c.cfg.apply_tanh_to_latent
    existing = {"best_model_value": 12.3}
    runner = SimpleNamespace(save=lambda path, infos=None: torch.save({"infos": infos}, path))
    pipeline.attach_latent_contract_to_runner(runner, c.cfg)
    path = c.tmp_path / "standalone" / "policy.pt"
    path.parent.mkdir()
    runner.save(str(path), infos=existing)
    assert existing == {"best_model_value": 12.3}
    assert torch.load(path, weights_only=False)["infos"]["best_model_value"] == 12.3
    c.cfg.apply_tanh_to_latent = not expected
    pipeline.restore_latent_activation_from_checkpoint(c.cfg, str(path))
    assert c.cfg.apply_tanh_to_latent == expected
