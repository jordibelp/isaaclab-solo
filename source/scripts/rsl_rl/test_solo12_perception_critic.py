"""CPU integration checks for shared perception inputs and the critic-only handoff."""

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import solo12_imu_pipeline as pipeline
import test_solo12_imu_pipeline as base
import torch
from tensordict import TensorDict

chain = base.chain
ROOT = base.ROOT
spec = importlib.util.spec_from_file_location(
    "solo12_perception_test_ppo", ROOT / "source/isaaclab_tasks/isaaclab_tasks/direct/solo12/agents/solo12_ppo.py"
)
ppo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppo)


def make_policy(kind, *, includes=True, detached=True, normalize=True):
    obs = TensorDict(
        {"policy": torch.randn(8, 408 if kind == "student" else 67), "critic": torch.randn(8, 67)}, batch_size=[8]
    )
    common = dict(
        actor_hidden_dims=[16],
        critic_hidden_dims=[19],
        actor_obs_normalization=normalize,
        critic_obs_normalization=normalize,
        noise_std_type="log",
        priv_critic_includes_perception_encoding=includes,
        priv_critic_detached_perception_encoding=detached,
    )
    if kind == "student":
        policy = base.models.Solo12BaseImuStudentActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            12,
            history_len=9,
            history_sample_dim=45,
            teacher_critic_obs_dim=67,
            tcn_channels=8,
            tcn_latent_dim=7,
            **common,
        )
    else:
        policy = base.models.Solo12BaseImuTeacherActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            12,
            teacher_encoder_obs_dim=64,
            teacher_latent_dim=7,
            teacher_encoder_hidden_dims=[17],
            **common,
        )
    policy.update_normalization(obs)
    return policy, obs


@pytest.mark.parametrize("kind", ["teacher", "student"])
@pytest.mark.parametrize("includes", [False, True])
@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_critic_uses_actual_actor_encoding_and_detaches_only_critic_gradients(kind, includes, detached, normalize):
    policy, obs = make_policy(kind, includes=includes, detached=detached, normalize=normalize)
    encoded_actor_input = policy._actor_input(policy.actor_obs_normalizer(policy.get_actor_obs(obs)))
    privileged = policy.critic_obs_normalizer(obs["critic"])
    expected = (
        torch.cat((privileged[:, :-3], encoded_actor_input[:, :-3], privileged[:, -3:]), -1) if includes else privileged
    )
    seen = []
    hook = policy.critic.register_forward_pre_hook(lambda _module, args: seen.append(args[0].detach().clone()))
    values = policy.evaluate(obs)
    hook.remove()
    torch.testing.assert_close(seen[0], expected.detach(), rtol=0, atol=0)
    values.square().mean().backward()
    encoder = policy.actor_history_encoder if kind == "student" else policy.teacher_encoder
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.critic.parameters())
    assert all(p.grad is None for p in policy.actor.parameters())
    assert policy.log_std.grad is None
    if includes and not detached:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    else:
        assert all(p.grad is None for p in encoder.parameters())
    policy.zero_grad(set_to_none=True)
    policy.act_inference(obs).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    if kind == "student":
        assert policy.critic_history_encoder is None


@pytest.mark.parametrize("kind", ["teacher", "student"])
@pytest.mark.parametrize("includes", [False, True])
def test_same_privileged_features_can_have_history_dependent_values(kind, includes):
    policy, obs = make_policy(kind, includes=includes)
    other = obs.clone()
    other["policy"] += torch.randn_like(other["policy"])
    assert not torch.equal(policy.act_inference(obs), policy.act_inference(other))
    if includes:
        assert not torch.equal(policy.evaluate(obs), policy.evaluate(other))
    else:
        torch.testing.assert_close(policy.evaluate(obs), policy.evaluate(other), rtol=0, atol=0)


@pytest.mark.parametrize("detached", [False, True])
def test_public_handoff_expands_critic_without_changing_pretrained_functions(chain, detached):
    c = chain
    c.cfg.priv_critic_includes_perception_encoding = True
    c.cfg.priv_critic_detached_perception_encoding = detached
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    assert cfg.priv_critic_includes_perception_encoding
    assert cfg.priv_critic_detached_perception_encoding == detached
    path = c.tmp_path / "adapter-perception.pt"
    torch.save(c.checkpoint, path)
    obs = TensorDict(
        {"policy": torch.cat((c.history, c.obs["policy"][:, -3:]), -1), "critic": c.obs["critic"]}, batch_size=[6]
    )
    student = base.models.Solo12BaseImuStudentActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(cfg)
    )
    base.train_definitions()["_initialize_student_from_dagger_adapter"](
        SimpleNamespace(alg=SimpleNamespace(policy=student)), str(path)
    )
    expected = c.teacher.actor(
        torch.cat((c.adapter(c.normalizer(c.history)), c.teacher.actor_obs_normalizer(c.obs["policy"])[:, -3:]), -1)
    )
    torch.testing.assert_close(student.act_inference(obs), expected, rtol=0, atol=0)
    torch.testing.assert_close(student.evaluate(obs), c.teacher.evaluate(c.obs), rtol=1e-6, atol=1e-7)
    first = student.critic.state_dict()["0.weight"]
    assert torch.count_nonzero(first[:, 64:71]) == 0
    torch.testing.assert_close(first[:, :64], c.teacher.critic.state_dict()["0.weight"][:, :64], rtol=0, atol=0)
    torch.testing.assert_close(first[:, -3:], c.teacher.critic.state_dict()["0.weight"][:, -3:], rtol=0, atol=0)


def test_augmented_teacher_survives_dagger_and_student_handoff(chain):
    c = chain
    c.teacher.priv_critic_includes_perception_encoding = True
    c.teacher.critic = base.models.MLP(74, 1, [19, 13], "elu")
    c.cfg.priv_critic_includes_perception_encoding = True
    c.cfg.priv_critic_detached_perception_encoding = False
    runner = SimpleNamespace(
        save=lambda path, infos=None: torch.save({"model_state_dict": c.teacher.state_dict(), "infos": infos}, path)
    )
    pipeline.attach_latent_contract_to_runner(runner, c.cfg)
    runner.save(c.teacher_path)
    # A copied teacher checkpoint has no sidecar; reconstruct its topology and activation.
    env = base.contract_env()
    env.policy_model = "base_imu_student_dagger"
    cfg = SimpleNamespace()
    pipeline.configure_teacher_for_dagger(env, cfg, str(c.teacher_path))
    assert cfg.priv_critic_includes_perception_encoding
    assert not cfg.priv_critic_detached_perception_encoding
    assert env.apply_tanh_to_latent == c.cfg.apply_tanh_to_latent
    rebuilt = base.models.Solo12BaseImuTeacherActorCritic(
        c.obs,
        {"policy": ["policy"], "critic": ["critic"]},
        12,
        teacher_latent_activation=env.teacher_latent_activation,
        **vars(cfg),
    )
    rebuilt.load_state_dict(c.teacher.state_dict(), strict=True)
    torch.testing.assert_close(rebuilt.act_inference(c.obs), c.teacher.act_inference(c.obs), rtol=0, atol=0)
    with pytest.raises(ValueError, match="enable env.priv_critic_includes_perception_encoding"):
        pipeline.configure_base_imu_student(base.contract_env(), SimpleNamespace(), c.checkpoint, str(c.teacher_path))
    cfg = SimpleNamespace()
    pipeline.configure_base_imu_student(c.cfg, cfg, c.checkpoint, str(c.teacher_path))
    obs = TensorDict(
        {"policy": torch.cat((c.history, c.obs["policy"][:, -3:]), -1), "critic": c.obs["critic"]}, batch_size=[6]
    )
    student = base.models.Solo12BaseImuStudentActorCritic(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **vars(cfg)
    )
    path = c.tmp_path / "from-augmented-teacher.pt"
    torch.save(c.checkpoint, path)
    base.train_definitions()["_initialize_student_from_dagger_adapter"](
        SimpleNamespace(alg=SimpleNamespace(policy=student)), str(path)
    )
    for key, value in c.teacher.critic.state_dict().items():
        torch.testing.assert_close(student.critic.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("script", ["train.py", "play_direct_0325.py", "train_solo12_base_imu_dagger.py"])
def test_public_entrypoints_propagate_both_environment_flags(script):
    path = Path(__file__).with_name(script)
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_sync_base_imu_policy_cfg_from_env_cfg"
    )
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    cfg = SimpleNamespace(priv_critic_includes_perception_encoding=True, priv_critic_detached_perception_encoding=False)
    agent = SimpleNamespace(
        policy=SimpleNamespace(
            priv_critic_includes_perception_encoding=False, priv_critic_detached_perception_encoding=True
        )
    )
    namespace[node.name](cfg, agent)
    assert vars(agent.policy) == vars(cfg)


def collect(alg, obs):
    with torch.inference_mode():
        for _ in range(4):
            alg.act(obs)
            obs = TensorDict({k: torch.randn_like(v) for k, v in obs.items()}, batch_size=obs.batch_size)
            alg.process_env_step(obs, torch.randn(8), torch.zeros(8), {})
        alg.compute_returns(obs)
    return obs


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("includes", [False, True])
def test_real_ppo_warmup_freezes_policy_stats_and_adamw_momentum_then_releases(detached, includes):
    policy, obs = make_policy("student", detached=detached, includes=includes)
    alg = ppo.Solo12PPO(
        policy, device="cpu", num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, schedule="fixed"
    )
    alg.optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=0.1)
    alg.init_storage("rl", 8, 4, obs, [12])
    obs = collect(alg, obs)
    alg.update()  # Seed actor optimizer state, including nonzero momentum and stale gradients.
    frozen = {k: v.clone() for k, v in policy.state_dict().items() if not k.startswith("critic")}
    actor_parameters = [p for name, p in policy.named_parameters() if not name.startswith("critic")]
    actor_optim = {p: copy.deepcopy(alg.optimizer.state[p]) for p in actor_parameters}
    critic_before = policy.critic[0].weight.detach().clone()
    expected_actions = policy.act_inference(obs).detach().clone()
    probe = obs.clone()
    alg.start_critic_warmup(2)
    for remaining in [1, 0]:
        obs = collect(alg, obs)
        losses = alg.update()
        assert losses["critic_warmup_active"] == 1
        assert losses["critic_warmup_remaining_iterations"] == remaining
        assert alg.learning_rate == 1e-3
        for key, value in frozen.items():
            torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(policy.act_inference(probe), expected_actions, rtol=0, atol=0)
        for param, before in actor_optim.items():
            for key, value in before.items():
                torch.testing.assert_close(alg.optimizer.state[param][key], value, rtol=0, atol=0)
        assert all(p.requires_grad for p in actor_parameters)
    assert not torch.equal(policy.critic[0].weight, critic_before)
    assert not policy.critic_warmup_active
    obs = collect(alg, obs)
    losses = alg.update()
    assert losses["critic_warmup_active"] == 0
    assert any(not torch.equal(policy.state_dict()[k], v) for k, v in frozen.items() if "normalizer" not in k)


def test_warmup_uses_rollout_rewards_not_intermediate_critic_predictions():
    policy, obs = make_policy("student")
    alg = ppo.Solo12PPO(policy, device="cpu", gamma=0.9)
    alg.start_critic_warmup(1)
    calls = []
    alg.storage = SimpleNamespace(compute_returns=lambda *args, **kwargs: calls.append((args, kwargs)))
    alg.compute_returns(obs)
    assert calls[0][0][1:] == (0.9, 1.0)
    assert calls[0][1] == {"normalize_advantage": False}


def test_warmup_never_runs_symmetry_or_the_adaptive_actor_schedule():
    policy, obs = make_policy("student")

    def forbidden_symmetry(**_kwargs):
        raise AssertionError("A critic-only update must not invoke actor symmetry objectives.")

    alg = ppo.Solo12PPO(
        policy,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=1e-12,
        symmetry_cfg={
            "use_data_augmentation": True,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 100.0,
            "data_augmentation_func": forbidden_symmetry,
        },
    )
    alg.init_storage("rl", 8, 4, obs, [12])
    alg.start_critic_warmup(1)
    collect(alg, obs)
    alg.update()
    assert alg.learning_rate == 1e-3


def test_standalone_checkpoint_restores_topology_and_continues_warmup_once(tmp_path):
    policy, obs = make_policy("student", detached=False)
    alg = ppo.Solo12PPO(policy, device="cpu")
    alg.start_critic_warmup(3, completed=2)
    runner = SimpleNamespace(
        alg=alg,
        save=lambda path, infos=None: torch.save({"infos": infos, "model_state_dict": policy.state_dict()}, path),
    )
    cfg = base.contract_env()
    cfg.priv_critic_includes_perception_encoding = True
    cfg.priv_critic_detached_perception_encoding = False
    pipeline.attach_latent_contract_to_runner(runner, cfg)
    path = tmp_path / "standalone.pt"
    runner.save(path)
    loaded = torch.load(path, weights_only=False)
    restored_env = base.contract_env()
    pipeline.restore_latent_activation_from_checkpoint(restored_env, str(path))
    assert restored_env.priv_critic_includes_perception_encoding
    assert not restored_env.priv_critic_detached_perception_encoding
    restored, _ = make_policy("student", detached=False)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    resumed = SimpleNamespace(alg=ppo.Solo12PPO(restored, device="cpu"))
    pipeline.configure_critic_warmup(resumed, 100, from_adapter=False, checkpoint_infos=loaded["infos"])
    assert resumed.alg.critic_warmup_remaining == 3
    assert resumed.alg.critic_warmup_completed == 2
    loaded["infos"]["solo12_critic_warmup"] = {"remaining": 0, "completed": 5}
    pipeline.configure_critic_warmup(resumed, 100, from_adapter=False, checkpoint_infos=loaded["infos"])
    assert resumed.alg.critic_warmup_remaining == 0


@pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
def test_invalid_warmup_configuration_fails(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        pipeline.configure_critic_warmup(SimpleNamespace(), bad, from_adapter=True)


def test_warmup_requires_a_handoff_or_saved_progress():
    with pytest.raises(ValueError, match="DAgger adapter handoff"):
        pipeline.configure_critic_warmup(SimpleNamespace(), 10, from_adapter=False)


def test_default_configuration_does_not_start_warmup():
    pipeline.configure_critic_warmup(SimpleNamespace(alg=SimpleNamespace()), 0, from_adapter=True)


def test_legacy_separate_critic_encoder_cannot_be_combined_with_shared_perception():
    obs = TensorDict({"policy": torch.randn(8, 408), "critic": torch.randn(8, 472)}, batch_size=[8])
    with pytest.raises(ValueError, match="not both"):
        base.models.Solo12BaseImuStudentActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            12,
            history_len=9,
            history_sample_dim=45,
            teacher_critic_obs_dim=67,
            feed_history_encoding_to_critic=True,
            priv_critic_includes_perception_encoding=True,
        )
