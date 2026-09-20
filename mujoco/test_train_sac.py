from collections import deque
import json
import math
import shlex
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import train_sac
from rsl_rl_sac.algorithms import SAC
from rsl_rl_sac.models import SACActorModel, SACCriticModel
from rsl_rl_sac.modules import LoRALinear, merged_state_dict
from rsl_rl_sac.storage import MixedReplayBuffer, ReplayBuffer
from rsl_rl_sac.utils.logger import Logger
from rsl_rl_sac.utils import wandb_utils


def test_reproducible_command_preserves_shell_sensitive_arguments():
    arguments = [
        "--headless",
        "--num_envs=256",
        "--run-name=[cluster] MuJoCo SAC | fine-tune",
        "env.max_velx_range_curriculum=[0.5, 1.0]",
    ]
    command = train_sac.reproducible_command(arguments)

    assert shlex.split(command) == ["./isaaclab.sh", "-p", "mujoco/train_sac.py", *arguments]


def test_sac_wandb_writer_exposes_command_at_top_level(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

    writer = wandb_utils.WandbSummaryWriter(
        str(tmp_path),
        flush_secs=1,
        cfg={"wandb_project": "test", "command": "./isaaclab.sh -p mujoco/train_sac.py --headless"},
    )
    writer.close()

    assert captured["config"]["command"] == "./isaaclab.sh -p mujoco/train_sac.py --headless"


def test_sac_wandb_writer_stores_every_environment_config_flavour(monkeypatch):
    stored = {}
    monkeypatch.setattr(wandb_utils.wandb, "config", SimpleNamespace(update=stored.update))

    # The MJX adapter exposes a plain namespace; the Isaac environments expose to_dict().
    mjx_cfg = SimpleNamespace(is_finite_horizon=False, episode_length_s=20.0, action_scale=0.3)
    wandb_utils.WandbSummaryWriter.store_config(None, mjx_cfg, {"max_iterations": 1})
    assert stored["env_cfg"] == {"is_finite_horizon": False, "episode_length_s": 20.0, "action_scale": 0.3}
    assert stored["train_cfg"] == {"max_iterations": 1}

    isaac_cfg = SimpleNamespace(to_dict=lambda: {"episode_length_s": 10.0})
    wandb_utils.WandbSummaryWriter.store_config(None, isaac_cfg, {})
    assert stored["env_cfg"] == {"episode_length_s": 10.0}

    wandb_utils.WandbSummaryWriter.store_config(None, {"kp": 9.0}, {})
    assert stored["env_cfg"] == {"kp": 9.0}


def test_sac_logger_counts_individual_environment_steps(tmp_path):
    scalars = []
    logger = Logger.__new__(Logger)
    logger.writer = SimpleNamespace(
        add_scalar=lambda name, value, step, **kwargs: scalars.append((name, value, step)),
    )
    logger.cfg = {"num_steps_per_env": 3, "algorithm": {"rnd_cfg": None}}
    logger.num_envs = 2
    logger.gpu_world_size = 1
    logger.device = "cpu"
    logger.tot_timesteps = 0
    logger.tot_time = 0.0
    logger.ep_extras = []
    logger.rewbuffer = deque()
    logger.lenbuffer = deque()
    logger.logger_type = "tensorboard"
    logger.log_dir = str(tmp_path)

    common = {
        "start_it": 0,
        "total_it": 2,
        "collect_time": 1.0,
        "learn_time": 1.0,
        "loss_dict": {},
        "learning_rate": 1.0e-4,
        "action_std": torch.ones(1),
        "rnd_weight": None,
        "print_minimal": True,
    }
    logger.log(it=0, **common)
    logger.log(it=1, **common)

    assert [item for item in scalars if item[0] == "env_steps"] == [
        ("env_steps", 6, 0),
        ("env_steps", 12, 1),
    ]


def runner_config(args, episode_length_s=20.0):
    """Resolve the derived update schedule the way main() does, then build the config."""
    schedule = train_sac._episodic_schedule(args, {"episode_length_s": episode_length_s})
    return train_sac._runner_config(args, schedule)


def test_runner_config_uses_paper_sac_defaults():
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args)

    assert cfg["class_name"] == "OffPolicyRunner"
    assert cfg["actor"]["init_noise_std"] == pytest.approx(0.15)
    assert cfg["algorithm"]["n_steps"] == 5
    assert cfg["algorithm"]["gamma"] == pytest.approx(0.97)
    assert cfg["algorithm"]["symmetry_cfg"]["use_data_augmentation"] is True
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}


def test_no_symmetry_removes_symmetry_configuration():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--symmetry-mode=none"])
    assert runner_config(args)["algorithm"]["symmetry_cfg"] is None


def test_mjx_action_scaling_uses_environment_bounds():
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_upper_magnitude=__import__("torch").tensor([2.0, 3.0]),
            action_lower_magnitude=__import__("torch").tensor([4.0, 5.0]),
        )
    )
    upper, lower = train_sac._mjx_action_scaling(env, "cpu")
    assert upper.tolist() == [2.0, 3.0]
    assert lower.tolist() == [4.0, 5.0]


def test_fine_tuning_defaults_follow_the_sim_to_online_recipe():
    """arXiv:2602.20220 needs delayed actor updates and a conservative actor learning rate."""
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args)

    assert cfg["algorithm"]["policy_frequency"] == 20
    assert cfg["algorithm"]["actor_learning_rate"] == pytest.approx(1.0e-5)
    assert cfg["algorithm"]["critic_learning_rate"] == pytest.approx(2.0e-4)
    assert cfg["save_replay_buffer"] is False
    assert cfg["save_replay_buffer_every"] == 500


def test_synchronous_actor_updates_remain_available():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--actor-update-every=1"])
    assert runner_config(args)["algorithm"]["policy_frequency"] == 1


def test_defaults_reproduce_the_paper_go1_episodic_schedule():
    """arXiv:2602.20220 runs one robot, updates once per episode, and uses K=1250 at UTD 1.25.

    Our MJX episode is 20 s at 50 Hz, so an episode is the same 1000 steps as their Go1.
    """
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args)

    assert args.num_envs == 1
    assert args.max_episodes == 1500
    assert cfg["max_episodes"] == 1500
    assert "max_iterations" not in cfg
    assert cfg["num_steps_per_env"] == 1000
    assert cfg["algorithm"]["num_mini_batches"] == 1250
    assert cfg["algorithm"]["mini_batch_size"] == 512
    assert cfg["update_schedule"]["utd"] == pytest.approx(1.25)
    # Requested warm start: 5000 new transitions, or five full-length episodes.
    assert cfg["update_schedule"]["transitions_before_updates"] == 5000


def test_episode_limit_has_an_explicit_cli_name():
    args = train_sac.build_parser().parse_args(["--max-episodes=1000"])
    cfg = runner_config(args)

    assert args.max_episodes == 1000
    assert cfg["max_episodes"] == 1000
    assert cfg["update_schedule"]["max_episodes"] == 1000
    with pytest.raises(SystemExit):
        train_sac.build_parser().parse_args(["--max-iterations=1000"])


def test_update_budget_follows_the_episode_length():
    """The schedule is derived, so a shorter episode keeps the ratio instead of the count."""
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args, episode_length_s=10.0)

    assert cfg["num_steps_per_env"] == 500
    assert cfg["algorithm"]["num_mini_batches"] == 625
    assert cfg["update_schedule"]["transitions_before_updates"] == 5000


@pytest.mark.parametrize(
    "flags, updates",
    [(["--utd=5"], 5000), (["--updates-per-iteration=200"], 200)],
)
def test_update_budget_is_settable_as_a_ratio_or_a_count(flags, updates):
    args = train_sac.build_parser().parse_args(["--no-wandb", *flags])
    assert runner_config(args)["algorithm"]["num_mini_batches"] == updates


def test_utd_and_update_count_cannot_both_be_set():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--utd=2", "--updates-per-iteration=10"])
    with pytest.raises(ValueError, match="only one"):
        runner_config(args)


def test_parallel_runs_require_explicit_fixed_rollout_mode():
    args = train_sac.build_parser().parse_args(["--num_envs=256"])
    with pytest.raises(ValueError, match="requires --num-envs=1"):
        runner_config(args)
    args = train_sac.build_parser().parse_args(
        ["--num_envs=256", "--rollout-steps=24", "--updates-per-iteration=200"]
    )
    cfg = runner_config(args)
    assert cfg["num_steps_per_env"] == 24
    assert cfg["algorithm"]["num_mini_batches"] == 200
    assert cfg["update_schedule"]["mode"] == "rollout"


@pytest.mark.parametrize("flag", [
    "--num-transitions-before-weight-updates", "--num_transitions_before_weight_updates",
    "--transitions-before-updates",
])
def test_transition_warmup_aliases(flag):
    args = train_sac.build_parser().parse_args([f"{flag}=4321"])
    assert runner_config(args)["update_schedule"]["transitions_before_updates"] == 4321


def test_legacy_warmup_is_explicitly_converted_to_transitions():
    args = train_sac.build_parser().parse_args(["--start-training=1"])
    assert runner_config(args)["update_schedule"]["transitions_before_updates"] == 1000
    with pytest.raises(SystemExit):
        train_sac.build_parser().parse_args(["--start-training=1", "--transitions-before-updates=5000"])


@pytest.mark.parametrize("flag", [
    "--utd=0", "--utd=-1", "--utd=nan", "--utd=inf", "--num-envs=0", "--batch-size=0",
    "--rollout-steps=0", "--n-steps=0", "--actor-update-every=0", "--transitions-before-updates=-1",
    "--updates-per-iteration=0", "--replay-buffer-size=4", "--start-training=-1",
])
def test_invalid_update_schedule_fails_before_simulator_start(flag):
    args = train_sac.build_parser().parse_args([flag])
    with pytest.raises(ValueError):
        runner_config(args)


def test_subunit_update_budget_is_allowed_and_carried_by_runner():
    args = train_sac.build_parser().parse_args(["--utd=0.0001"])
    assert runner_config(args)["update_schedule"]["utd"] == pytest.approx(0.0001)


def test_mixing_options_require_retained_data():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--offline-fraction=0.3"])

    with pytest.raises(ValueError, match="--offline-replay-buffer"):
        train_sac._validate_offline_arguments(args)


def test_no_mixing_options_needs_no_retained_data():
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    assert train_sac._validate_offline_arguments(args) is None


def write_snapshot(path, num_envs=2, obs_dim=4, action_dim=12):
    """Record a small pretraining snapshot the way an Isaac run would."""
    obs = TensorDict({"policy": torch.zeros(num_envs, obs_dim)}, batch_size=[num_envs])
    buffer = ReplayBuffer(num_envs, 1, obs, (action_dim,), "cpu", buffer_size=num_envs * 4)
    for step in range(4):
        transition = ReplayBuffer.Transition()
        transition.observations = TensorDict(
            {"policy": torch.full((num_envs, obs_dim), float(step))}, batch_size=[num_envs]
        )
        transition.next_observations = transition.observations.clone()
        transition.actions = torch.zeros(num_envs, action_dim)
        transition.rewards = torch.zeros(num_envs)
        transition.dones = torch.zeros(num_envs)
        transition.bootstrap = torch.zeros(num_envs)
        buffer.add_transition(transition)
    buffer.save_snapshot(path)
    return obs


def test_retained_replay_is_installed_with_the_resolved_schedule(tmp_path):
    snapshot = tmp_path / "replay_buffer.pt"
    obs = write_snapshot(snapshot)
    online = ReplayBuffer(2, 1, obs, (12,), "cpu", buffer_size=64)
    runner = SimpleNamespace(alg=SimpleNamespace(replay_buffer=online))
    args = train_sac.build_parser().parse_args(
        ["--no-wandb", f"--offline-replay-buffer={snapshot}", "--device=cpu", "--max-episodes=400"]
    )

    train_sac._install_retained_replay(runner, args)

    mixture = runner.alg.replay_buffer
    assert isinstance(mixture, MixedReplayBuffer)
    assert mixture.online is online
    assert mixture.initial_offline_fraction == pytest.approx(0.5)
    assert mixture.final_offline_fraction == pytest.approx(0.0)
    assert mixture.anneal_iterations == 200
    # The snapshot's own n_steps/gamma are informational; the current run decides both.
    assert mixture.offline.n_steps == args.n_steps
    assert mixture.offline.gamma == pytest.approx(args.gamma)


def test_retained_replay_rejects_a_snapshot_from_another_observation_layout(tmp_path):
    snapshot = tmp_path / "replay_buffer.pt"
    write_snapshot(snapshot, obs_dim=4)
    wider = TensorDict({"policy": torch.zeros(2, 5)}, batch_size=[2])
    runner = SimpleNamespace(
        alg=SimpleNamespace(replay_buffer=ReplayBuffer(2, 1, wider, (12,), "cpu", buffer_size=64))
    )
    args = train_sac.build_parser().parse_args(
        ["--no-wandb", f"--offline-replay-buffer={snapshot}", "--device=cpu"]
    )

    with pytest.raises(ValueError, match="observation layout"):
        train_sac._install_retained_replay(runner, args)


LORA_OBS_DIM = 6
LORA_ACTION_DIM = 3


def build_lora_runner(rank=4, extra_args=(), critic_loss="mse"):
    """A SAC algorithm on a fixed one-batch replay, wrapped the way ``main`` wraps it."""
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", LORA_ACTION_DIM, hidden_dims=[16, 8], obs_normalization=True)
    critic = SACCriticModel(obs, groups, "critic", 1, hidden_dims=[16, 8], num_actions=LORA_ACTION_DIM,
                            obs_normalization=True, distributional_loss=critic_loss, distributional_num_bins=11)
    batch = (
        obs,
        torch.randn(8, LORA_ACTION_DIM),
        torch.randn(8, 1),
        obs,
        torch.zeros(8, 1),
        torch.zeros(8, 1),
        torch.ones(8, 1, dtype=torch.long),
    )
    replay = SimpleNamespace(mini_batch_generator=lambda **kwargs: iter([batch]), add_transition=lambda transition: None)
    algorithm = SAC(actor, critic, replay, device="cpu", policy_frequency=1)
    runner = SimpleNamespace(alg=algorithm)
    args = train_sac.build_parser().parse_args(
        ["--no-wandb", f"--rank={rank}", "--device=cpu", *extra_args]
    )
    return runner, args


def base_weights(actor):
    return {name: p.clone() for name, p in actor.named_parameters() if "lora_" not in name}


def test_lora_run_trains_only_the_adapter():
    runner, args = build_lora_runner()
    train_sac._apply_finetuning(runner, args)
    actor = runner.alg.actor
    frozen_before = base_weights(actor)

    losses = runner.alg.update()

    assert sum(isinstance(m, LoRALinear) for m in actor.modules()) == 3
    assert all(math.isfinite(value) for value in losses.values())
    for name, original in frozen_before.items():
        torch.testing.assert_close(dict(actor.named_parameters())[name], original)
    assert any(
        not torch.equal(p, torch.zeros_like(p)) for n, p in actor.named_parameters() if "lora_b" in n
    )


def test_lora_optimizer_covers_only_trainable_parameters():
    runner, args = build_lora_runner()

    train_sac._apply_finetuning(runner, args)

    optimized = {id(p) for group in runner.alg.actor_optimizer.param_groups for p in group["params"]}
    trainable = {id(p) for p in runner.alg.actor.parameters() if p.requires_grad}
    assert optimized == trainable
    assert runner.alg.actor_optimizer.param_groups[0]["lr"] == pytest.approx(args.actor_learning_rate)


def test_lora_checkpoint_is_a_plain_sac_checkpoint():
    """The saved actor must load into a model that knows nothing about adapters."""
    runner, args = build_lora_runner()
    train_sac._apply_finetuning(runner, args)
    runner.alg.update()
    obs = TensorDict({"policy": torch.randn(4, LORA_OBS_DIM)}, batch_size=[4])
    trained = runner.alg.actor(obs)

    payload = runner.alg.save()

    groups = {"actor": ["policy"], "critic": ["policy"]}
    restored = SACActorModel(obs, groups, "actor", LORA_ACTION_DIM, hidden_dims=[16, 8], obs_normalization=True)
    restored.load_state_dict(payload["actor_state_dict"], strict=True)
    torch.testing.assert_close(restored(obs), trained, atol=1e-6, rtol=1e-5)
    assert payload["mujoco_lora"] == train_sac._finetuning_config(args)


def test_lora_is_off_by_default():
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    assert args.rank == 0
    assert all(c["mode"] == "full" for c in train_sac._finetuning_config(args).values())


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--rank=-1"], "non-negative"),
        (["--lora-alpha=2.0"], "positive rank"),
        (["--rank=4", "--resume"], "merged networks"),
    ],
)
def test_invalid_lora_arguments_are_rejected(arguments, message):
    args = train_sac.build_parser().parse_args(["--no-wandb", *arguments])

    with pytest.raises(ValueError, match=message):
        train_sac._finetuning_config(args)


def test_checkpoint_action_scaling_is_replaced_by_target_environment():
    import torch

    actor = SimpleNamespace(
        action_bias=torch.zeros(2),
        action_range=torch.ones(2),
        log_action_range=torch.zeros(1),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            action_upper_magnitude=torch.tensor([2.0, 4.0]),
            action_lower_magnitude=torch.tensor([6.0, 8.0]),
        )
    )
    train_sac._apply_environment_action_scaling(actor, env)
    assert actor.action_bias.tolist() == [-2.0, -2.0]
    assert actor.action_range.tolist() == [4.0, 6.0]
    assert actor.log_action_range.item() == pytest.approx(__import__("math").log(24.0))


@pytest.mark.parametrize("actor_mode", ["full", "lora", "frozen"])
@pytest.mark.parametrize("critic_mode", ["full", "lora", "frozen"])
@pytest.mark.parametrize("critic_loss", ["mse", "two_hot", "hl_gauss"])
def test_all_tuning_combinations_update_only_requested_state(actor_mode, critic_mode, critic_loss):
    arguments = []
    for name, mode in (("actor", actor_mode), ("critic", critic_mode)):
        arguments.append(f"--{name}-rank={2 if mode == 'lora' else 0}")
        if mode == "frozen":
            arguments.append(f"--freeze-{name}")
    runner, args = build_lora_runner(rank=0, extra_args=arguments, critic_loss=critic_loss)
    alg = runner.alg
    train_sac._apply_finetuning(runner, args)
    before = {name: {k: v.clone() for k, v in getattr(alg, name).state_dict().items()}
              for name in ("actor", "critic")}
    alpha_before = alg.log_alpha.clone()
    alg.train_mode()
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM) + 3}, batch_size=[8])
    alg.act(obs)
    alg.process_env_step(obs, torch.ones(8), torch.zeros(8), {})
    for _ in range(3):
        losses = alg.update()
        assert all(math.isfinite(value) for value in losses.values())

    for name, mode in (("actor", actor_mode), ("critic", critic_mode)):
        model = getattr(alg, name)
        after = model.state_dict()
        changed = {k for k, v in after.items() if not torch.equal(v, before[name][k])}
        if mode == "frozen":
            assert not changed
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            assert getattr(alg, f"{name}_optimizer") is None
        else:
            assert any("weight" in k or "lora_b" in k for k in changed)
            assert "obs_normalizer.count" in changed
            optimizer = getattr(alg, f"{name}_optimizer")
            assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
                id(p) for p in model.parameters() if p.requires_grad
            }
            if mode == "lora":
                assert all("lora_" in k or "obs_normalizer" in k or "_target" in k for k in changed)
                for prefix in (("mlp",) if name == "actor" else ("critic1", "critic2")):
                    assert any(k.startswith(prefix) and "lora_b" in k for k in changed)
    if actor_mode == "frozen":
        assert torch.equal(alg.log_alpha, alpha_before)
    assert all(not p.requires_grad for p in alg.critic.critic1_target.parameters())
    assert all(not p.requires_grad for p in alg.critic.critic2_target.parameters())


def test_frozen_critic_still_provides_action_gradients():
    runner, args = build_lora_runner(rank=1, extra_args=["--freeze-critic"])
    train_sac._apply_finetuning(runner, args)
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM)}, batch_size=[8])
    actions = runner.alg.actor(obs)
    q1, q2 = runner.alg.critic.evaluate_all_q(obs, actions)
    (-torch.min(q1, q2).mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in runner.alg.actor.named_parameters() if "lora_b" in n)
    assert all(p.grad is None for p in runner.alg.critic.parameters())


@pytest.mark.parametrize("layers,count", [("all", 3), ("input", 1), ("output", 1), ("input_and_output", 2)])
def test_critic_adapts_each_twin_independently_and_leaves_targets_dense(layers, count):
    runner, args = build_lora_runner(rank=1, extra_args=["--actor-rank=3", f"--critic-lora-layers={layers}"])
    before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    train_sac._apply_finetuning(runner, args)
    for network in (runner.alg.critic.critic1, runner.alg.critic.critic2):
        adapters = [m for m in network.modules() if isinstance(m, LoRALinear)]
        assert len(adapters) == count
        assert all(m.lora_a.shape[0] == 1 and m.scale == 1 for m in adapters)
    assert all(m.lora_a.shape[0] == 3 and m.scale == 1
               for m in runner.alg.actor.modules() if isinstance(m, LoRALinear))
    assert not any(isinstance(m, LoRALinear) for m in runner.alg.critic.critic1_target.modules())
    for k, v in merged_state_dict(runner.alg.critic).items():
        assert torch.equal(v, before[k])


@pytest.mark.parametrize("critic_loss", ["mse", "two_hot", "hl_gauss"])
def test_lora_critic_targets_average_merged_weights_and_checkpoint_reloads(critic_loss):
    runner, args = build_lora_runner(rank=1, critic_loss=critic_loss)
    train_sac._apply_finetuning(runner, args)
    critic = runner.alg.critic
    for _ in range(2):
        before = {k: v.clone() for k, v in critic.state_dict().items() if "_target" in k}
        runner.alg.update()
        merged = merged_state_dict(critic)
        for k, old in before.items():
            online = merged[k.replace("_target", "")]
            torch.testing.assert_close(merged[k], runner.alg.tau * online + (1-runner.alg.tau) * old)
    payload = runner.alg.save()
    assert not any("lora_" in k or ".base." in k for k in payload["critic_state_dict"])
    restored, _ = build_lora_runner(rank=0, critic_loss=critic_loss)
    restored.alg.load(payload, {"actor": True, "critic": True, "optimizer": False}, strict=True)
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM)}, batch_size=[8])
    actions = torch.randn(8, LORA_ACTION_DIM)
    for method in ("evaluate_all_q", "evaluate_all_target_q"):
        for expected, actual in zip(getattr(critic, method)(obs, actions), getattr(restored.alg.critic, method)(obs, actions)):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="merged networks"):
        restored.alg.load(payload, None, strict=True)


@pytest.mark.parametrize("flags", [[], ["--freeze-critic"], ["--freeze-actor"], ["--freeze-actor", "--freeze-critic"]])
def test_dense_and_frozen_optimizer_resume(flags):
    runner, args = build_lora_runner(rank=0, extra_args=flags)
    train_sac._apply_finetuning(runner, args)
    runner.alg.update()
    payload = runner.alg.save()
    restored, args = build_lora_runner(rank=0, extra_args=[*flags, "--resume"])
    train_sac._apply_finetuning(restored, args)
    assert restored.alg.load(payload, None, strict=True)
    for name in ("actor", "critic"):
        opt = getattr(restored.alg, f"{name}_optimizer")
        if opt is not None:
            assert opt.state_dict()["state"].keys() == payload[f"{name}_optimizer_state_dict"]["state"].keys()
    restored.alg.update()
    if flags:
        incompatible, _ = build_lora_runner(rank=0)
        with pytest.raises(ValueError, match="freeze mode"):
            incompatible.alg.load(payload, None, strict=True)


@pytest.mark.parametrize("flags", [
    ["--critic-rank=-1"], ["--actor-rank=-1"],
    ["--freeze-critic", "--critic-rank=1"], ["--freeze-actor", "--actor-rank=1"],
    ["--critic-lora-alpha=2"], ["--actor-lora-layers=input"],
    ["--rank=1", "--lora-alpha=nan"], ["--rank=1", "--critic-lora-alpha=0"],
    ["--critic-rank=1", "--resume"], ["--actor-rank=1", "--resume"],
])
def test_invalid_per_network_options_fail_early(flags):
    args = train_sac.build_parser().parse_args(flags)
    with pytest.raises(ValueError):
        train_sac._finetuning_config(args)


def test_resolved_overrides_are_recorded_in_runner_config():
    args = train_sac.build_parser().parse_args([
        "--rank=1", "--actor-rank=4", "--lora-alpha=2", "--critic-lora-alpha=3", "--critic-lora-layers=output",
    ])
    config = runner_config(args)["finetuning"]
    assert config == {
        "actor": {"mode": "lora", "rank": 4, "alpha": 2.0, "layers": "all"},
        "critic": {"mode": "lora", "rank": 1, "alpha": 3.0, "layers": "output"},
    }


@pytest.mark.parametrize("critic_loss", ["mse", "two_hot", "hl_gauss"])
@pytest.mark.parametrize("sidecar", ["json", "yaml", "none"])
def test_checkpoint_architecture_and_loss_survive_transfer(tmp_path, critic_loss, sidecar):
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", LORA_ACTION_DIM, hidden_dims=[32, 16], activation="swish",
                          obs_normalization=True, state_dependent_std=False)
    critic = SACCriticModel(obs, groups, "critic", 1, num_actions=LORA_ACTION_DIM, hidden_dims=[24, 12],
                           activation="swish", layer_norm=True, distributional_loss=critic_loss,
                           distributional_num_bins=19, distributional_symlog_limit=5)
    checkpoint = tmp_path / "model.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}, checkpoint)
    saved_cfg = {"actor": {"activation": "swish"}, "critic": {"activation": "swish", "distributional_loss": critic_loss}}
    if sidecar == "json":
        (tmp_path / "run_config.json").write_text(json.dumps({"agent": saved_cfg}))
    elif sidecar == "yaml":
        import yaml
        (tmp_path / "params").mkdir()
        (tmp_path / "params" / "agent.yaml").write_text(yaml.safe_dump(saved_cfg))
    flags = [f"--checkpoint={checkpoint}"]
    if sidecar == "none" and critic_loss == "hl_gauss":
        flags.append("--critic-loss=hl_gauss")
    args = train_sac.build_parser().parse_args(flags)
    config = runner_config(args)
    train_sac._configure_checkpoint_models(config, args)
    assert config["critic"]["distributional_loss"] == critic_loss
    config["actor"].pop("class_name")
    config["critic"].pop("class_name")
    restored_actor = SACActorModel(obs, groups, "actor", LORA_ACTION_DIM, **config["actor"])
    restored_critic = SACCriticModel(obs, groups, "critic", 1, num_actions=LORA_ACTION_DIM, **config["critic"])
    restored_actor.load_state_dict(actor.state_dict(), strict=True)
    restored_critic.load_state_dict(critic.state_dict(), strict=True)
    torch.testing.assert_close(restored_actor(obs), actor(obs), rtol=0, atol=0)
    actions = actor(obs)
    torch.testing.assert_close(restored_critic(obs, actions=actions), critic(obs, actions=actions), rtol=0, atol=0)


def test_freezing_preserves_checkpoint_target_lag():
    runner, args = build_lora_runner(rank=1, extra_args=["--freeze-critic"])
    with torch.no_grad():
        next(runner.alg.critic.critic1_target.parameters()).add_(0.5)
    before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    train_sac._apply_finetuning(runner, args)
    runner.alg.update()
    for k, v in runner.alg.critic.state_dict().items():
        assert torch.equal(v, before[k])
