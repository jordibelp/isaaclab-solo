from collections import deque
import shlex
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import train_sac
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


def test_runner_config_uses_paper_sac_defaults():
    args = train_sac.build_parser().parse_args(["--no-wandb"])
    cfg = train_sac._runner_config(args)

    assert cfg["class_name"] == "OffPolicyRunner"
    assert cfg["actor"]["init_noise_std"] == pytest.approx(0.15)
    assert cfg["algorithm"]["n_steps"] == 5
    assert cfg["algorithm"]["gamma"] == pytest.approx(0.97)
    assert cfg["algorithm"]["symmetry_cfg"]["use_data_augmentation"] is True
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}


def test_no_symmetry_removes_symmetry_configuration():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--symmetry-mode=none"])
    assert train_sac._runner_config(args)["algorithm"]["symmetry_cfg"] is None


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
    cfg = train_sac._runner_config(args)

    assert cfg["algorithm"]["policy_frequency"] == 20
    assert cfg["algorithm"]["actor_learning_rate"] == pytest.approx(1.0e-5)
    assert cfg["algorithm"]["critic_learning_rate"] == pytest.approx(2.0e-4)
    assert cfg["save_replay_buffer"] is False
    assert cfg["save_replay_buffer_every"] == 500


def test_synchronous_actor_updates_remain_available():
    args = train_sac.build_parser().parse_args(["--no-wandb", "--actor-update-every=1"])
    assert train_sac._runner_config(args)["algorithm"]["policy_frequency"] == 1


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
        ["--no-wandb", f"--offline-replay-buffer={snapshot}", "--device=cpu", "--max-iterations=400"]
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
