"""Solo12 Random Network Distillation configuration for RSL-RL."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab_rl.rsl_rl import RslRlRndCfg


@dataclass(frozen=True)
class Solo12RndSetup:
    enabled: bool
    beta: float


def load_checkpoint_with_optional_fresh_rnd(runner, path: str) -> tuple[dict, bool]:
    """Load a runner checkpoint, allowing newly enabled RND to start from fresh weights.

    RSL-RL assumes that an active RND module implies an RND-enabled checkpoint. Older
    checkpoints do not contain those entries, so temporarily hide only the fresh RND
    objects while the standard loader restores the policy, PPO optimizer, and iteration.
    """
    rnd = getattr(runner.alg, "rnd", None)
    if not rnd:
        return runner.load(path), False

    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    has_rnd_model = "rnd_state_dict" in checkpoint
    has_rnd_optimizer = "rnd_optimizer_state_dict" in checkpoint
    del checkpoint

    if has_rnd_model and has_rnd_optimizer:
        return runner.load(path), False
    if has_rnd_model != has_rnd_optimizer:
        raise ValueError(
            "Checkpoint has incomplete RND state: expected both 'rnd_state_dict' and "
            "'rnd_optimizer_state_dict'."
        )

    rnd_optimizer = runner.alg.rnd_optimizer
    runner.alg.rnd = None
    runner.alg.rnd_optimizer = None
    try:
        infos = runner.load(path)
    finally:
        runner.alg.rnd = rnd
        runner.alg.rnd_optimizer = rnd_optimizer
    return infos, True


def configure_solo12_rnd(env_cfg, agent_cfg) -> Solo12RndSetup:
    """Configure RSL-RL RND from the opt-in fields in a Solo12 environment config."""

    if not hasattr(env_cfg, "rnd_network"):
        return Solo12RndSetup(enabled=False, beta=0.0)

    enabled = bool(env_cfg.rnd_network)
    beta = float(env_cfg.beta_curiosity)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError(f"env.beta_curiosity must be finite and non-negative, got {beta}.")
    if not enabled:
        if beta != 0.0:
            raise ValueError("env.beta_curiosity requires env.rnd_network=True.")
        return Solo12RndSetup(enabled=False, beta=0.0)
    if beta == 0.0:
        raise ValueError("env.rnd_network=True requires env.beta_curiosity > 0.0.")
    if getattr(agent_cfg, "class_name", None) not in ("OnPolicyRunner", "OffPolicyRunner"):
        raise ValueError("Solo12 RND requires an RSL-RL PPO or SAC runner.")

    # Follow the robotics-focused RND setup from Schwarke et al. (CoRL 2023):
    # normalize the curiosity state (not the intrinsic reward), use a one-hidden-layer
    # target, and give the predictor one extra hidden layer.
    agent_cfg.algorithm.rnd_cfg = RslRlRndCfg(
        weight=beta,
        reward_normalization=False,
        state_normalization=True,
        learning_rate=1.0e-3,
        num_outputs=1,
        predictor_hidden_dims=[5, 5],
        target_hidden_dims=[5],
    )
    agent_cfg.obs_groups = dict(agent_cfg.obs_groups)
    agent_cfg.obs_groups["rnd_state"] = ["rnd_state"]
    return Solo12RndSetup(enabled=True, beta=beta)
