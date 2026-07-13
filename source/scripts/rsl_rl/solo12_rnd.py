"""Solo12 Random Network Distillation configuration for RSL-RL."""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab_rl.rsl_rl import RslRlRndCfg


@dataclass(frozen=True)
class Solo12RndSetup:
    enabled: bool
    beta: float


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
    if getattr(agent_cfg, "class_name", None) != "OnPolicyRunner":
        raise ValueError("Solo12 RND requires the RSL-RL OnPolicyRunner / PPO path.")

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
