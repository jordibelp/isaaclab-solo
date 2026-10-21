# SPDX-License-Identifier: BSD-3-Clause
"""Startup physics randomization shared by both Borinot goalkeeper tasks.

All four terms are uniform samples, independent per environment, fixed across resets.
Mass scaling also scales each selected body's inertia (the IsaacLab helper's default).
Use ``env.event_randomization.racket_mass=False`` for a single-term ablation, or
``env.include_events_randomization=False`` to disable all four terms. The dict keys
already exist in Hydra, so these overrides do not need a leading ``+``.

Joint friction uses the same helper and range as Solo12. In Isaac Sim 5.x this sets
static and dynamic friction efforts and viscous friction; in 4.5 only the static
load-dependent friction coefficient is available. See the shared helper for units.
"""
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass


def randomize_base_com(env, env_ids, asset_cfg, com_range):
    # Keep the PI mixer nominal. Only the physical rotor moments use the sampled CoM.
    robot = env.scene[asset_cfg.name]
    env._nominal_base_com = robot.root_physx_view.get_coms()[0, asset_cfg.body_ids[0], :3].clone()
    mdp.randomize_rigid_body_com(env, env_ids, com_range=com_range, asset_cfg=asset_cfg)


@configclass
class BorinotEventCfg:
    joint_friction = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 1.0),
            "operation": "abs",  # scaling nominal zero friction would have no effect
        },
    )
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
        },
    )
    racket_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="paddle"),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTermCfg(
        func=randomize_base_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        },
    )


def configure_randomization(cfg):
    """Apply toggles after Hydra, before DirectRLEnv builds its event manager."""
    tags = ("joint_friction", "add_base_mass", "racket_mass", "base_com")
    if set(cfg.event_randomization) != set(tags):
        raise ValueError(f"event_randomization must contain exactly {tags}.")
    if any(not isinstance(value, bool) for value in cfg.event_randomization.values()):
        raise ValueError("event_randomization values must be booleans.")
    if not cfg.include_events_randomization:
        cfg.events = None
    elif cfg.events is not None:
        for tag, enabled in cfg.event_randomization.items():
            if not enabled:
                setattr(cfg.events, tag, None)
