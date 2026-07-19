from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import train_lora


CHECKPOINT = Path("/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt")


def test_agent_defaults_come_from_config_file():
    args = train_lora.build_parser().parse_args(["--checkpoint", str(CHECKPOINT)])
    assert args.learning_rate == train_lora.DEFAULT_AGENT_CFG.algorithm.learning_rate
    assert args.ppo_epochs == train_lora.DEFAULT_AGENT_CFG.algorithm.num_learning_epochs
    assert args.rollout_steps == train_lora.DEFAULT_AGENT_CFG.num_steps_per_env


def test_hydra_style_agent_overrides():
    args, unknown = train_lora.build_parser().parse_known_args([
        "--checkpoint", str(CHECKPOINT), "--learning-rate=3e-4",
        "agent.algorithm.learning_rate=2e-4",
        "agent.algorithm.num_learning_epochs=7",
        "agent.policy.log_std_range=[-3.0,-0.25]",
        "env.kp=8.0",
    ])
    args, remaining = train_lora.apply_agent_overrides(args, unknown)
    assert args.learning_rate == 2e-4  # agent.* tokens are explicit and applied last
    assert args.ppo_epochs == 7
    assert args.log_std_range == (-3.0, -0.25)
    assert remaining == ["env.kp=8.0"]
    assert train_lora.resolved_agent_config(args)["algorithm"]["learning_rate"] == 2e-4


def test_unknown_agent_override_is_rejected():
    args = train_lora.build_parser().parse_args(["--checkpoint", str(CHECKPOINT)])
    with pytest.raises(ValueError, match="Unsupported agent override"):
        train_lora.apply_agent_overrides(args, ["agent.algorithm.typo=1"])


def test_invalid_hydra_style_choice_is_rejected():
    args = train_lora.build_parser().parse_args(["--checkpoint", str(CHECKPOINT)])
    with pytest.raises(ValueError, match="schedule"):
        train_lora.apply_agent_overrides(args, ["agent.algorithm.schedule=magic"])


def test_layer_selection_contract():
    assert train_lora.selected_layer_indices(4, "all") == (0, 1, 2, 3)
    assert train_lora.selected_layer_indices(4, "input_and_output") == (0, 3)
    assert train_lora.selected_layer_indices(4, "input") == (0,)
    assert train_lora.selected_layer_indices(4, "output") == (3,)


@pytest.mark.parametrize("rank", [1, 2, 4, 8])
def test_lora_starts_at_exact_frozen_network(rank):
    """B=0 at init, so any rank must reproduce the pretrained policy bit-for-bit."""
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, rank, (0, 1, 2, 3), log_std)
    obs = jnp.asarray(np.random.default_rng(0).normal(size=(3, 48)).astype(np.float32))
    scale = 1.0 / rank
    adapted = train_lora.actor_mean(params, obs, actor, norms, scale)
    x = (obs - norms["actor_mean"]) / (norms["actor_std"] + train_lora.OBS_NORM_EPS)
    for index, (weight, bias) in enumerate(actor):
        x = x @ weight.T + bias
        if index != len(actor) - 1:
            x = jax.nn.elu(x)
    np.testing.assert_allclose(adapted, x, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("rank", [1, 3, 8])
def test_adapter_factor_shapes_follow_rank(rank):
    _, actor, critic, _, _, _, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, rank, (0, 3), log_std)
    for index, (weight, _) in enumerate(actor):
        a, b = params["actor"][index]["a"], params["actor"][index]["b"]
        expected = rank if index in (0, 3) else 0
        assert a.shape == (expected, weight.shape[1])
        assert b.shape == (weight.shape[0], expected)
        # b @ a must land back in the frozen weight's shape for checkpoint merging.
        assert (b @ a).shape == weight.shape


def test_nonfinite_gradients_leave_parameters_untouched():
    params = {"w": jnp.ones((3,))}
    state = (jnp.array(0), train_lora.zeros_like_tree(params), train_lora.zeros_like_tree(params))
    bad = {"w": jnp.array([jnp.nan, 1.0, 2.0])}
    updated, _, _, ok = train_lora.adam_update(params, bad, state, 1e-2, 0.5)
    assert not bool(ok)
    np.testing.assert_array_equal(updated["w"], params["w"])


def test_merged_checkpoint_is_ordinary_policy_checkpoint(tmp_path):
    payload, actor, critic, akeys, ckeys, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(1), actor, critic, 1, (0, 3), log_std)
    train_lora.save_checkpoints(tmp_path, 1, payload, params, actor, critic, akeys, ckeys, 1.0, {"test": True})
    merged = torch.load(tmp_path / "model_latest.pt", map_location="cpu", weights_only=False)
    assert merged["infos"]["mujoco_lora"]["iteration"] == 1
    assert merged["model_state_dict"]["actor.0.weight"].shape == (256, 48)
    assert (tmp_path / "adapter_latest.pt").is_file()


def test_reset_merge_preserves_command_countdown_of_running_envs():
    """``reset`` doubles as the post-step merge, so unmasked envs must keep their countdown.

    Rearming it unconditionally silently disabled all in-episode command resampling.
    """
    info = train_lora.build_model(Path(__file__).with_name("solo12.xml"), 9.0, 0.2)
    cfg = dict(train_lora.DEFAULT_ENV)
    reset_fn, _ = train_lora.make_training_functions(info, cfg, num_envs=2)
    state = reset_fn(jax.random.PRNGKey(0))
    running = state._replace(command_steps=jnp.array([7, 3]))
    merged = reset_fn(jax.random.PRNGKey(1), running, jnp.array([True, False]))
    interval = round(cfg["command_resampling_time_s"] / train_lora.STEP_DT)
    assert int(merged.command_steps[0]) == interval  # reset env is rearmed
    assert int(merged.command_steps[1]) == 3  # running env keeps counting down


def test_left_right_mirror_is_an_involution():
    obs = jnp.arange(96, dtype=jnp.float32).reshape(2, 48)
    action = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    mirrored_obs, mirrored_action = train_lora.mirror_lr(obs, action)
    restored_obs, restored_action = train_lora.mirror_lr(mirrored_obs, mirrored_action)
    np.testing.assert_array_equal(restored_obs, obs)
    np.testing.assert_array_equal(restored_action, action)
