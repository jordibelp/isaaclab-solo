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
    assert args.constrain_delta_lora_degrees == 0.0
    assert args.clipping_includes_exploratory_noisy is False
    assert args.frozen_base_weights is True


@pytest.mark.parametrize("token", ["--frozen-base-weights=False", "--frozen-base-weights=false"])
def test_base_weights_can_be_unfrozen(token):
    args = train_lora.build_parser().parse_args(["--checkpoint", str(CHECKPOINT), "--rank=0", token])
    assert args.rank == 0
    assert args.frozen_base_weights is False


@pytest.mark.parametrize("token", ["--clipping-includes-exploratory-noisy", "--clipping-includes-exploratory-noisy=True"])
def test_exploration_clipping_flag_can_be_enabled(token):
    args = train_lora.build_parser().parse_args(["--checkpoint", str(CHECKPOINT), token])
    assert args.clipping_includes_exploratory_noisy is True


def test_exploration_clipping_flag_accepts_explicit_false():
    args = train_lora.build_parser().parse_args([
        "--checkpoint", str(CHECKPOINT), "--clipping-includes-exploratory-noisy=False"
    ])
    assert args.clipping_includes_exploratory_noisy is False


def test_hydra_style_agent_overrides():
    args, unknown = train_lora.build_parser().parse_known_args([
        "--checkpoint", str(CHECKPOINT), "--learning-rate=3e-4",
        "agent.algorithm.learning_rate=2e-4",
        "agent.algorithm.num_learning_epochs=7",
        "agent.policy.log_std_range=[-3.0,-0.25]",
        "agent.policy.constrain_delta_lora_degrees=10.0",
        "env.kp=8.0",
    ])
    args, remaining = train_lora.apply_agent_overrides(args, unknown)
    assert args.learning_rate == 2e-4  # agent.* tokens are explicit and applied last
    assert args.ppo_epochs == 7
    assert args.log_std_range == (-3.0, -0.25)
    assert args.constrain_delta_lora_degrees == 10.0
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


def test_lora_joint_target_delta_is_constrained_in_degrees():
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, 1, (0, 1, 2, 3), log_std)
    params["actor"][3]["b"] = jnp.ones_like(params["actor"][3]["b"]) * 100.0
    obs = jnp.asarray(np.random.default_rng(1).normal(size=(3, 48)).astype(np.float32))

    unconstrained = train_lora.actor_mean(params, obs, actor, norms, 1.0)
    frozen = train_lora.actor_mean(
        train_lora.init_trainable(jax.random.PRNGKey(2), actor, critic, 1, (), log_std),
        obs, actor, norms, 1.0,
    )
    limit = np.deg2rad(10.0)
    constrained = train_lora.actor_mean(params, obs, actor, norms, 1.0, limit)

    assert np.max(np.abs(train_lora.ACTION_SCALE * np.asarray(unconstrained - frozen))) > limit
    np.testing.assert_array_less(
        np.abs(train_lora.ACTION_SCALE * np.asarray(constrained - frozen)), limit + 1e-6
    )


def test_zero_lora_constraint_is_exactly_disabled():
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(3), actor, critic, 1, (3,), log_std)
    params["actor"][3]["b"] = jnp.ones_like(params["actor"][3]["b"])
    obs = jnp.zeros((2, 48))
    np.testing.assert_array_equal(
        train_lora.actor_mean(params, obs, actor, norms, 1.0, 0.0),
        train_lora.actor_mean(params, obs, actor, norms, 1.0),
    )


def test_exploration_and_lora_are_jointly_constrained_in_degrees():
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(4), actor, critic, 1, (3,), log_std)
    params["actor"][3]["b"] = jnp.ones_like(params["actor"][3]["b"]) * 100.0
    obs = jnp.asarray(np.random.default_rng(2).normal(size=(3, 48)).astype(np.float32))
    limit = np.deg2rad(10.0)
    adapted_mean = train_lora.actor_mean(params, obs, actor, norms, 1.0, limit, True)
    noisy_action = adapted_mean + 20.0
    executed = train_lora.clip_exploratory_action(noisy_action, obs, actor, norms, limit)
    frozen_mean = train_lora.network(
        (obs - norms["actor_mean"]) / (norms["actor_std"] + train_lora.OBS_NORM_EPS), actor, (), 0.0
    )
    np.testing.assert_allclose(
        np.abs(train_lora.ACTION_SCALE * np.asarray(executed - frozen_mean)), limit, atol=1e-6
    )


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


def test_rank_zero_unfrozen_initializes_full_dense_actor_and_critic():
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, 0, (), log_std, False)
    assert all(set(layer) == {"weight", "bias"} for layer in params["actor"] + params["critic"])
    obs = jnp.asarray(np.random.default_rng(3).normal(size=(3, 48)).astype(np.float32))
    np.testing.assert_allclose(train_lora.actor_mean(params, obs, actor, norms, 0.0),
                               train_lora.network((obs - norms["actor_mean"]) / (norms["actor_std"] + train_lora.OBS_NORM_EPS), actor, (), 0.0))


def test_rank_zero_frozen_has_no_network_parameters():
    _, actor, critic, _, _, _, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, 0, (), log_std, True)
    assert all(not layer for layer in params["actor"] + params["critic"])


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


def test_reward_metrics_include_scaled_and_cross_run_comparable_values():
    cfg = dict(train_lora.DEFAULT_ENV)
    terms = np.zeros((2, 3, len(train_lora.REWARD_TERM_NAMES)), dtype=np.float32)
    index = train_lora.REWARD_TERM_NAMES.index("two_feet_above_height")
    scale = cfg["two_feet_above_height_reward_scale"]
    terms[..., index] = scale * train_lora.STEP_DT * 0.25
    completed = np.sum(terms, axis=(0, 1))

    metrics = train_lora.reward_metrics(terms, completed, completed_episodes=2, cfg=cfg)

    assert metrics["RewardsPerStep/two_feet_above_height"] == pytest.approx(scale * train_lora.STEP_DT * 0.25)
    assert metrics["PerStepRewardRatio/two_feet_above_height"] == pytest.approx(0.25)
    assert "Episode_Reward/two_feet_above_height" in metrics


def test_reward_metrics_omit_episode_values_when_no_episode_finished():
    terms = np.zeros((2, 3, len(train_lora.REWARD_TERM_NAMES)), dtype=np.float32)
    metrics = train_lora.reward_metrics(
        terms, np.zeros(len(train_lora.REWARD_TERM_NAMES)), completed_episodes=0, cfg=dict(train_lora.DEFAULT_ENV)
    )
    assert "RewardsPerStep/total" in metrics
    assert not any(key.startswith("Episode_Reward/") for key in metrics)


def test_metric_smoother_averages_rollouts_and_pools_completed_episodes():
    cfg = dict(train_lora.DEFAULT_ENV)
    smoother = train_lora.MetricSmoother(window=2, cfg=cfg)
    zeros = np.zeros(len(train_lora.REWARD_TERM_NAMES))

    first = smoother.update(
        {"RewardsPerStep/total": 1.0, "PerStepRewardRatio/track_lin_vel_xy_exp": 0.2},
        episode_return_sum=10.0,
        episode_steps_sum=100.0,
        completed_episodes=1,
        completed_reward_sums=zeros,
    )
    second_rewards = zeros.copy()
    second_rewards[0] = 12.0
    second = smoother.update(
        {"RewardsPerStep/total": 3.0, "PerStepRewardRatio/track_lin_vel_xy_exp": 0.6},
        episode_return_sum=60.0,
        episode_steps_sum=900.0,
        completed_episodes=3,
        completed_reward_sums=second_rewards,
    )

    assert first["SmoothedRewardsPerStep/total"] == pytest.approx(1.0)
    assert second["SmoothedRewardsPerStep/total"] == pytest.approx(2.0)
    assert second["SmoothedPerStepRewardRatio/track_lin_vel_xy_exp"] == pytest.approx(0.4)
    assert second["SmoothedEpisode/completed_count"] == 4
    assert second["SmoothedEpisode_Reward/total"] == pytest.approx(17.5)
    assert "SmoothedEpisode/return" not in second
    assert second["SmoothedEpisode/length_steps"] == pytest.approx(250.0)
    assert second["SmoothedEpisode_Reward/track_lin_vel_xy_exp"] == pytest.approx(
        12.0 / 4 / cfg["episode_length_s"]
    )

    third = smoother.update(
        {"RewardsPerStep/total": 5.0, "PerStepRewardRatio/track_lin_vel_xy_exp": 1.0},
        episode_return_sum=0.0,
        episode_steps_sum=0.0,
        completed_episodes=0,
        completed_reward_sums=zeros,
    )
    assert third["SmoothedRewardsPerStep/total"] == pytest.approx(4.0)
    assert third["SmoothedEpisode/completed_count"] == 3
    assert third["SmoothedEpisode_Reward/total"] == pytest.approx(20.0)


def test_left_right_mirror_is_an_involution():
    obs = jnp.arange(96, dtype=jnp.float32).reshape(2, 48)
    action = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    mirrored_obs, mirrored_action = train_lora.mirror_lr(obs, action)
    restored_obs, restored_action = train_lora.mirror_lr(mirrored_obs, mirrored_action)
    np.testing.assert_array_equal(restored_obs, obs)
    np.testing.assert_array_equal(restored_action, action)


def test_symmetry_augmentation_uses_transformed_behavior_distribution():
    """A mirrored transition must retain the likelihood of the action that generated it."""
    rng = np.random.default_rng(4)
    obs = jnp.asarray(rng.normal(size=(5, 48)).astype(np.float32))
    means = jnp.asarray(rng.normal(size=(5, 12)).astype(np.float32))
    log_std = jnp.asarray(rng.normal(size=(5, 12)).astype(np.float32) * 0.1 - 1.0)
    actions = means + jnp.exp(log_std) * jnp.asarray(rng.normal(size=(5, 12)).astype(np.float32))
    old_logp = train_lora.gaussian_log_prob(actions, means, log_std)
    advantages = jnp.arange(5, dtype=jnp.float32)
    returns = advantages + 10.0

    augmented = train_lora.augment_left_right_batch(
        obs, actions, old_logp, means, log_std, advantages, returns
    )
    aug_obs, aug_actions, aug_logp, aug_means, aug_log_std, aug_advantages, aug_returns = augmented

    np.testing.assert_allclose(
        train_lora.gaussian_log_prob(aug_actions[5:], aug_means[5:], aug_log_std[5:]),
        old_logp,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(aug_logp[5:], old_logp)
    np.testing.assert_allclose(aug_log_std[5:], log_std[:, train_lora.LR_PERM])
    np.testing.assert_allclose(aug_advantages[5:], advantages)
    np.testing.assert_allclose(aug_returns[5:], returns)
    mirrored_obs, _ = train_lora.mirror_lr(obs, actions)
    np.testing.assert_allclose(aug_obs[5:], mirrored_obs)
