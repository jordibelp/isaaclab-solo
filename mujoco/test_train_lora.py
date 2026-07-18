from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

import train_lora


CHECKPOINT = Path("/home/jordibelp/IsaacLab-dirty/logs/skrl/checkpoints/0717_q3a68133_model_15008.pt")


def test_layer_selection_contract():
    assert train_lora.selected_layer_indices(4, "all") == (0, 1, 2, 3)
    assert train_lora.selected_layer_indices(4, "input_and_output") == (0, 3)
    assert train_lora.selected_layer_indices(4, "input") == (0,)
    assert train_lora.selected_layer_indices(4, "output") == (3,)


def test_lora_starts_at_exact_frozen_network():
    _, actor, critic, _, _, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(0), actor, critic, 1, (0, 1, 2, 3), log_std)
    obs = jnp.asarray(np.random.default_rng(0).normal(size=(3, 48)).astype(np.float32))
    scale = 1.0
    adapted = train_lora.actor_mean(params, obs, actor, norms, scale)
    x = (obs - norms["actor_mean"]) / (norms["actor_std"] + train_lora.OBS_NORM_EPS)
    for index, (weight, bias) in enumerate(actor):
        x = x @ weight.T + bias
        if index != len(actor) - 1:
            x = jax.nn.elu(x)
    np.testing.assert_allclose(adapted, x, rtol=1e-6, atol=1e-6)


def test_merged_checkpoint_is_ordinary_policy_checkpoint(tmp_path):
    payload, actor, critic, akeys, ckeys, norms, log_std = train_lora.load_frozen_networks(CHECKPOINT)
    params = train_lora.init_trainable(jax.random.PRNGKey(1), actor, critic, 1, (0, 3), log_std)
    train_lora.save_checkpoints(tmp_path, 1, payload, params, actor, critic, akeys, ckeys, 1.0, {"test": True})
    merged = torch.load(tmp_path / "model_latest.pt", map_location="cpu", weights_only=False)
    assert merged["infos"]["mujoco_lora"]["iteration"] == 1
    assert merged["model_state_dict"]["actor.0.weight"].shape == (256, 48)
    assert (tmp_path / "adapter_latest.pt").is_file()


def test_left_right_mirror_is_an_involution():
    obs = jnp.arange(96, dtype=jnp.float32).reshape(2, 48)
    action = jnp.arange(24, dtype=jnp.float32).reshape(2, 12)
    mirrored_obs, mirrored_action = train_lora.mirror_lr(obs, action)
    restored_obs, restored_action = train_lora.mirror_lr(mirrored_obs, mirrored_action)
    np.testing.assert_array_equal(restored_obs, obs)
    np.testing.assert_array_equal(restored_action, action)
