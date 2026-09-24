from collections import deque
import json
import math
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import train_lora
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


def test_sac_wandb_writer_exposes_filterable_run_config_at_top_level(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

    writer = wandb_utils.WandbSummaryWriter(
        str(tmp_path),
        flush_secs=1,
        cfg={
            "wandb_project": "test",
            "command": "./isaaclab.sh -p mujoco/train_sac.py --headless",
            "offline_replay": {
                "offline_replay_buffer": "/tmp/replay_buffer.pt",
                "offline_fraction": 0.5,
                "offline_fraction_final": 0.0,
                "offline_anneal_iterations": 13,
            },
        },
    )
    writer.close()

    assert captured["config"]["command"] == "./isaaclab.sh -p mujoco/train_sac.py --headless"
    assert captured["config"]["offline_anneal_iterations"] == 13
    assert captured["config"]["offline_fraction"] == pytest.approx(0.5)
    assert captured["config"]["offline_fraction_final"] == pytest.approx(0.0)


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
    schedule = train_sac._interaction_schedule(args, {"episode_length_s": episode_length_s})
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


def test_best_weights_follow_rolling_mean_episode_return(tmp_path):
    calls = []
    saves = []
    logger = SimpleNamespace(
        log=lambda **kwargs: calls.append(kwargs),
        log_dir=str(tmp_path),
        writer=object(),
        rewbuffer=deque(),
        tot_timesteps=0,
        tot_time=0.0,
    )

    def save(path, infos=None):
        Path(path).write_text("checkpoint")
        saves.append((Path(path).name, infos))

    runner = SimpleNamespace(logger=logger, save=save)
    train_sac._install_best_weights_hook(runner, "/source/model_3700.pt")

    for iteration, rewards in [(0, [10.0]), (1, [10.0, 20.0]), (2, [5.0, 6.0])]:
        logger.rewbuffer.clear()
        logger.rewbuffer.extend(rewards)
        logger.tot_timesteps += 1000
        runner.logger.log(it=iteration)

    assert len(calls) == 3
    assert [name for name, _ in saves] == ["best_weights.pt", "best_weights.pt"]
    assert (tmp_path / "best_weights.pt").exists()
    assert saves[-1][1] == {
        "best_model_metric": "Train/mean_reward",
        "best_model_value": 15.0,
        "best_model_iteration": 1,
        "best_model_total_timesteps": 2000,
        "best_model_total_time": 0.0,
        "source_checkpoint": "/source/model_3700.pt",
    }


def test_best_weights_wait_for_a_finished_episode_and_active_writer(tmp_path):
    saves = []
    logger = SimpleNamespace(
        log=lambda **kwargs: None,
        log_dir=str(tmp_path),
        writer=object(),
        rewbuffer=deque(),
        tot_timesteps=0,
        tot_time=0.0,
    )
    runner = SimpleNamespace(logger=logger, save=lambda *args, **kwargs: saves.append((args, kwargs)))
    train_sac._install_best_weights_hook(runner, None)

    runner.logger.log(it=0)
    logger.rewbuffer.append(10.0)
    logger.writer = None
    runner.logger.log(it=1)

    assert saves == []


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


def test_defaults_update_every_1000_interactions_at_the_paper_ratio():
    """arXiv:2602.20220 uses K=1250 updates at UTD 1.25 over a 1000-step Go1 collection.

    We keep that period but count it in environment interactions, so an early termination
    can no longer shrink an update phase.
    """
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args)

    assert args.num_envs == 1
    assert args.max_env_interactions == 50_000
    assert cfg["max_env_interactions"] == 50_000
    assert cfg["update_schedule"]["max_iterations"] == 50
    assert "max_episodes" not in cfg
    assert cfg["num_steps_per_env"] == 1000
    assert cfg["algorithm"]["num_mini_batches"] == 1250
    assert cfg["algorithm"]["mini_batch_size"] == 512
    assert cfg["update_schedule"]["utd"] == pytest.approx(1.25)
    # Requested warm start: 5000 new transitions, or five full update periods.
    assert cfg["update_schedule"]["transitions_before_updates"] == 5000


def test_budget_is_counted_in_interactions_not_episodes():
    args = train_sac.build_parser().parse_args(["--max-env-interactions=12000", "--rollout-steps=500"])
    cfg = runner_config(args)

    assert cfg["max_env_interactions"] == 12000
    assert cfg["update_schedule"]["max_iterations"] == 24
    assert cfg["num_steps_per_env"] == 500
    assert cfg["algorithm"]["num_mini_batches"] == 625
    for retired in ("--max-episodes=1000", "--max-iterations=1000"):
        with pytest.raises(SystemExit):
            train_sac.build_parser().parse_args([retired])


def test_a_partial_final_period_still_gets_its_iteration():
    args = train_sac.build_parser().parse_args(["--max-env-interactions=2500"])
    assert runner_config(args)["update_schedule"]["max_iterations"] == 3


def test_episode_length_does_not_change_the_update_period():
    """Collection spans episode ends, so a shorter episode no longer rescales the budget."""
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = runner_config(args, episode_length_s=10.0)

    assert cfg["num_steps_per_env"] == 1000
    assert cfg["algorithm"]["num_mini_batches"] == 1250
    assert cfg["update_schedule"]["episode_steps"] == 500
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


def test_parallel_runs_split_the_budget_across_environments():
    args = train_sac.build_parser().parse_args(
        ["--num_envs=256", "--rollout-steps=24", "--updates-per-iteration=200"]
    )
    cfg = runner_config(args)
    assert cfg["num_steps_per_env"] == 24
    assert cfg["algorithm"]["num_mini_batches"] == 200
    assert cfg["update_schedule"]["transitions_per_iteration"] == 6144
    # 50,000 interactions at 6,144 per iteration.
    assert cfg["update_schedule"]["max_iterations"] == 9


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


def test_runner_config_records_resolved_retained_replay_schedule():
    args = train_sac.build_parser().parse_args([
        "--offline-replay-buffer=/tmp/replay_buffer.pt",
        "--offline-fraction=0.4",
        "--offline-fraction-final=0.1",
        "--offline-anneal-iterations=13",
    ])

    assert runner_config(args)["offline_replay"] == {
        "offline_replay_buffer": "/tmp/replay_buffer.pt",
        "offline_fraction": pytest.approx(0.4),
        "offline_fraction_final": pytest.approx(0.1),
        "offline_anneal_iterations": 13,
    }


def test_runner_config_records_computed_offline_anneal_default():
    args = train_sac.build_parser().parse_args(["--offline-replay-buffer=/tmp/replay_buffer.pt"])

    assert runner_config(args)["offline_replay"]["offline_anneal_iterations"] == 25


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
        ["--no-wandb", f"--offline-replay-buffer={snapshot}", "--device=cpu", "--max-env-interactions=400000"]
    )

    train_sac._install_retained_replay(runner, args, {"max_iterations": 400})

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
        train_sac._install_retained_replay(runner, args, {"max_iterations": 50})


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


def test_transfer_restores_temperature_and_adam_moments_but_keeps_finetuning_lr():
    source, _ = build_lora_runner(rank=0)
    source.alg.update()
    source.alg.log_alpha.data.fill_(math.log(0.00049))
    payload = source.alg.save()
    target, args = build_lora_runner(rank=0)
    train_sac._apply_finetuning(target, args)
    train_sac._restore_finetuning_state(target.alg, payload, args)
    assert target.alg.log_alpha.exp().item() == pytest.approx(0.00049)
    for name in ("actor", "critic", "alpha"):
        actual = getattr(target.alg, f"{name}_optimizer").state_dict()
        expected = payload[f"{name}_optimizer_state_dict"]
        assert actual["state"].keys() == expected["state"].keys()
        for key in actual["state"]:
            torch.testing.assert_close(actual["state"][key]["exp_avg"], expected["state"][key]["exp_avg"])
        assert actual["param_groups"][0]["lr"] == getattr(args, f"{name}_learning_rate")


def test_mjx_accepts_pretraining_failure_and_joint_limit_settings():
    cfg, unsupported = train_sac.mjx_env.parse_env_overrides([
        "env.base_collision_terminal_penalty=-10.0", "env.soft_qlim_penalty_reward_scale=-0.5",
        "env.joint_physical_limit_hip=[-70,70]", "env.joint_soft_limit_hip_delta=20",
    ])
    assert not unsupported
    assert cfg["base_collision_terminal_penalty"] == -10


def test_lora_transfer_keeps_fresh_adapter_optimizer_and_restores_dense_critic():
    source, _ = build_lora_runner(rank=0)
    source.alg.update()
    payload = source.alg.save()
    target, args = build_lora_runner(rank=0, extra_args=["--actor-rank=1"])
    train_sac._apply_finetuning(target, args)
    train_sac._restore_finetuning_state(target.alg, payload, args)
    assert not target.alg.actor_optimizer.state
    assert target.alg.critic_optimizer.state
    assert target.alg.alpha_optimizer.state
    target.alg.update()


def test_explicit_temperature_and_fresh_optimizer_ablation():
    source, _ = build_lora_runner(rank=0)
    source.alg.update()
    target, args = build_lora_runner(rank=0, extra_args=["--initial-alpha=0.002", "--reset-optimizers"])
    target.alg.log_alpha.data.fill_(math.log(args.initial_alpha))
    train_sac._apply_finetuning(target, args)
    train_sac._restore_finetuning_state(target.alg, source.alg.save(), args)
    assert target.alg.log_alpha.exp().item() == pytest.approx(0.002)
    assert not target.alg.actor_optimizer.state
    assert not target.alg.critic_optimizer.state
    assert not target.alg.alpha_optimizer.state


def test_fixed_temperature_transfer_uses_checkpoint_value_in_bellman_targets():
    source, _ = build_lora_runner(rank=0)
    source.alg.log_alpha.data.fill_(math.log(0.00049))
    target, args = build_lora_runner(rank=0, extra_args=["--freeze-alpha"])
    target.alg.auto_alpha = False
    target.alg.log_alpha.requires_grad_(False)
    target.alg.alpha_optimizer = None
    train_sac._apply_finetuning(target, args)
    train_sac._restore_finetuning_state(target.alg, source.alg.save(), args)
    target.alg.update()
    assert target.alg.log_alpha.exp().item() == pytest.approx(0.00049)
    assert target.alg.save()["alpha"] == pytest.approx(0.00049)


def test_mjx_terminal_and_soft_limit_penalties_match_isaac_units():
    import jax.numpy as jnp
    cfg = {**train_sac.mjx_env.DEFAULT_ENV, "base_collision_terminal_penalty": -10.,
           "soft_qlim_penalty_reward_scale": -0.5}
    terminal, soft = train_sac.mjx_env.additional_reward_terms(
        jnp.array([[0., 0.], [1.2, -1.3]]), jnp.array([[-1., 1.], [-1., 1.]]),
        jnp.array([False, True]), cfg,
    )
    assert terminal.tolist() == [0., -10.]
    assert soft.tolist() == pytest.approx([0., -0.25])


def test_mjx_physical_limits_apply_before_jax_model_creation():
    import mujoco
    import numpy as np
    cfg = {**train_sac.mjx_env.DEFAULT_ENV, "joint_physical_limit_hip": [-70., 70.],
           "use_asymmetric_thigh_limits": True, "joint_physical_limit_front_thigh": [-145., 55.],
           "joint_physical_limit_rear_thigh": [-55., 145.]}
    info = train_sac.mjx_env.build_model(Path(train_sac.__file__).with_name("solo12.xml"), 9., .2, cfg)
    for name, limits in (("FL_hip_joint", [-70., 70.]), ("FL_thigh_joint", [-145., 55.]), ("RL_thigh_joint", [-55., 145.])):
        jid = mujoco.mj_name2id(info.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0
        np.testing.assert_allclose(info.model.jnt_range[jid], np.deg2rad(limits))
        np.testing.assert_allclose(info.mjx_model.jnt_range[jid], np.deg2rad(limits), atol=1e-6)


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


def test_checkpoint_action_scaling_survives_transfer(capsys):
    """The tanh action map is part of the trained policy, so the simulator must not reset it.

    Rescaling it to the raw XML joint ranges multiplied Solo12 hip targets by 3.6x and shifted
    the neutral calf pose by 45 deg, which made a robust checkpoint fall in 0.28 s.
    """
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
    train_sac._report_checkpoint_action_scaling(actor, env)

    assert actor.action_bias.tolist() == [0.0, 0.0]
    assert actor.action_range.tolist() == [1.0, 1.0]
    assert actor.log_action_range.tolist() == [0.0]
    assert "kept from the checkpoint" in capsys.readouterr().out
    assert not hasattr(train_sac, "_apply_environment_action_scaling")


def test_action_bounds_are_measured_from_the_safe_q_action_centre():
    """``target = SAFE_Q + ACTION_SCALE * action``, so q=0 is not the action centre."""
    import numpy as np

    env = train_sac.MjxSolo12VecEnv.__new__(train_sac.MjxSolo12VecEnv)
    env.device = __import__("torch").device("cpu")
    lower, upper = env._action_bounds()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(Path(train_sac.__file__).with_name("solo12.xml")))
    for i, name in enumerate(train_sac.mjx_env.JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[jid]
        centre = float(np.asarray(train_sac.mjx_env.SAFE_Q)[i])
        scale = train_sac.mjx_env.ACTION_SCALE
        assert centre - scale * float(lower[i]) == pytest.approx(low, abs=1e-5)
        assert centre + scale * float(upper[i]) == pytest.approx(high, abs=1e-5)


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


@pytest.mark.parametrize("saved,flag,expected", [
    (None, None, "min"), ("mean", None, "mean"), ("min", None, "min"),
    ("mean", "min", "min"), (None, "mean", "mean"),
])
def test_q_reduction_follows_the_checkpoint_unless_overridden(tmp_path, saved, flag, expected):
    obs = TensorDict({"policy": torch.randn(8, LORA_OBS_DIM)}, batch_size=[8])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SACActorModel(obs, groups, "actor", LORA_ACTION_DIM, hidden_dims=[16, 8])
    critic = SACCriticModel(obs, groups, "critic", 1, num_actions=LORA_ACTION_DIM, hidden_dims=[16, 8])
    payload = {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}
    if saved is not None:
        payload["q_reduction_method"] = saved
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    flags = [f"--checkpoint={checkpoint}"] + ([f"--q-reduction-method={flag}"] if flag else [])
    args = train_sac.build_parser().parse_args(flags)
    config = runner_config(args)
    train_sac._configure_checkpoint_models(config, args)
    assert config["algorithm"]["q_reduction_method"] == expected


def test_q_reduction_without_a_checkpoint_defaults_to_min():
    assert runner_config(train_sac.build_parser().parse_args([]))["algorithm"]["q_reduction_method"] == "min"
    args = train_sac.build_parser().parse_args(["--q-reduction-method=mean"])
    assert runner_config(args)["algorithm"]["q_reduction_method"] == "mean"
    with pytest.raises(SystemExit):
        train_sac.build_parser().parse_args(["--q-reduction-method=max"])


def test_freezing_preserves_checkpoint_target_lag():
    runner, args = build_lora_runner(rank=1, extra_args=["--freeze-critic"])
    with torch.no_grad():
        next(runner.alg.critic.critic1_target.parameters()).add_(0.5)
    before = {k: v.clone() for k, v in runner.alg.critic.state_dict().items()}
    train_sac._apply_finetuning(runner, args)
    runner.alg.update()
    for k, v in runner.alg.critic.state_dict().items():
        assert torch.equal(v, before[k])


def _isaac_env_yaml(tmp_path, **extra):
    import yaml

    config = {
        "episode_length_s": 10,
        "command_resampling_time_s": 5,
        "tracking_std": 0.223,
        "flexed_initial_joint_pos_noise_range": [-0.07, 0.07],
        "joint_physical_limit_hip": [-70, 70],
        "use_asymmetric_thigh_limits": True,
        "joint_soft_limit_calf_delta": 24,
        "max_velx_range_curriculum": [0.4, 0.6],
        "track_lin_vel_xy_reward_scale_curriculum": [1.2, 1.8, 1.5],
        "two_feet_above_height_reward_scale_curriculum": [2.0, 1.2, 1.5],
        "forces_applied_to_base_curriculum_by_phase": [0, 0, 8],
        "include_events_randomization_curriculum": [False, False, True],
        **extra,
    }
    path = tmp_path / "params" / "env.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config))
    return path


def test_source_env_cfg_copies_the_last_curriculum_stage(tmp_path):
    path = _isaac_env_yaml(tmp_path)
    args = train_sac.build_parser().parse_args([f"--source-env-cfg={path}"])
    cfg, unsupported = train_lora.parse_env_overrides([], train_sac._source_env_cfg(args))

    assert unsupported == []
    # Settings the curriculum moved come from the final stage, not from the dumped startup value.
    assert cfg["track_lin_vel_xy_reward_scale"] == 1.5
    assert cfg["two_feet_above_height_reward_scale"] == 1.5
    assert cfg["command_lin_vel_x_range"] == (-0.6, 0.6)
    # Static settings come straight across, including the renamed reset noise.
    assert cfg["episode_length_s"] == 10
    assert cfg["command_resampling_time_s"] == 5
    assert cfg["joint_physical_limit_hip"] == [-70, 70]
    assert cfg["joint_soft_limit_calf_delta"] == 24
    assert cfg["joint_pos_noise_range"] == (-0.07, 0.07)
    # Pushes and startup randomization have no MJX implementation and must stay off.
    assert cfg["forces_applied_to_base_curriculum"] == (0.0,)
    assert cfg["include_events_randomization"] is False


def test_source_env_cfg_can_select_an_earlier_stage(tmp_path):
    path = _isaac_env_yaml(tmp_path)
    args = train_sac.build_parser().parse_args([f"--source-env-cfg={path}", "--curriculum-stage=1"])
    cfg, _ = train_lora.parse_env_overrides([], train_sac._source_env_cfg(args))

    assert cfg["track_lin_vel_xy_reward_scale"] == 1.8
    assert cfg["two_feet_above_height_reward_scale"] == 1.2
    assert cfg["command_lin_vel_x_range"] == (-0.6, 0.6)


def test_explicit_overrides_win_over_the_source_config(tmp_path):
    path = _isaac_env_yaml(tmp_path)
    args = train_sac.build_parser().parse_args([f"--source-env-cfg={path}"])
    cfg, _ = train_lora.parse_env_overrides(
        ["env.tracking_std=0.4"], train_sac._source_env_cfg(args)
    )

    assert cfg["tracking_std"] == 0.4
    assert cfg["episode_length_s"] == 10


def test_source_env_cfg_is_found_next_to_the_checkpoint(tmp_path):
    _isaac_env_yaml(tmp_path)
    checkpoint = tmp_path / "model_3700.pt"
    checkpoint.write_bytes(b"")
    args = train_sac.build_parser().parse_args(["--source-env-cfg=auto", f"--checkpoint={checkpoint}"])

    assert train_sac._source_env_cfg(args)["episode_length_s"] == 10


def test_source_env_cfg_never_swallows_a_following_override(tmp_path):
    path = _isaac_env_yaml(tmp_path)
    args, unknown = train_sac.build_parser().parse_known_args(
        [f"--source-env-cfg={path}", "env.tracking_std=0.4"]
    )

    assert unknown == ["env.tracking_std=0.4"]
    assert args.source_env_cfg == str(path)


def test_isaac_only_settings_are_accepted_and_ignored():
    cfg, unsupported = train_lora.parse_env_overrides(["env.sac_q_offset_init_actions=False"])

    assert unsupported == []
    assert cfg == train_lora.DEFAULT_ENV


def test_curriculum_stage_needs_a_source_config():
    args = train_sac.build_parser().parse_args(["--curriculum-stage=2"])
    with pytest.raises(ValueError, match="--source-env-cfg"):
        train_sac._source_env_cfg(args)
