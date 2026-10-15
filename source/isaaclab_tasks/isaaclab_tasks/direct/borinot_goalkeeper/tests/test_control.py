# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the actual config resolver and action method without starting Isaac Sim."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

FOLDER = Path(__file__).resolve().parents[1]
MODES = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control", "PI_rotors_PD_joint_pos_control")


def source_class(filename):
    return next(n for n in ast.parse((FOLDER / filename).read_text()).body if isinstance(n, ast.ClassDef))


def method(filename, name):
    node = next(n for n in source_class(filename).body if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), scope)
    return scope[name]


def config():
    values = {}
    for n in source_class("borinot_goalkeeper_env_cfg.py").body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    values["sim"] = SimpleNamespace(dt=1/240, render_interval=4)
    values["km_over_kf"] = values["km"] / values["kf"]
    values["robot"] = SimpleNamespace(spawn=SimpleNamespace(articulation_props=SimpleNamespace(enabled_self_collisions=False)), actuators={"arm": SimpleNamespace(
        stiffness=35.0, damping=0.6, effort_limit_sim=2.7, velocity_limit_sim=15.0,
        friction=0.0, armature=0.00036207, effort_limit=None, velocity_limit=None)})
    return SimpleNamespace(**values)


def test_requested_default_selector_exists():
    assert getattr(config(), "control_mode", None) == MODES[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_self_collision_selector_overrides_spawn_settings_after_hydra(enabled):
    cfg = config()
    cfg.enabled_self_collisions = enabled
    cfg.robot.spawn.articulation_props.enabled_self_collisions = not enabled
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    assert cfg.robot.spawn.articulation_props.enabled_self_collisions is enabled


@pytest.mark.parametrize("mode,actions,observations,kp,kd", [
    (MODES[0], 8, 65, 0.0, 0.0), (MODES[1], 8, 65, 35.0, 0.6), (MODES[2], 6, 63, 35.0, 0.6),
])
def test_resolver_updates_spaces_timing_and_existing_actuator(mode, actions, observations, kp, kd):
    cfg = config()
    cfg.control_mode = mode
    cfg.effort_lim_sim = 1.8
    cfg.velocity_lim_sim = 72.0
    actuator = cfg.robot.actuators["arm"]
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (actions, observations, observations)
    assert (cfg.sim.dt, cfg.decimation, cfg.sim.render_interval) == (0.005, 2, 2)
    assert cfg.robot.actuators["arm"] is actuator
    assert (actuator.stiffness, actuator.damping) == (kp, kd)
    assert (actuator.effort_limit_sim, actuator.velocity_limit_sim) == (1.8, 72.0)
    assert (actuator.friction, actuator.armature) == (0.0, 0.00036207)


@pytest.mark.parametrize("physics,policy,decimation", [(200, 100, 2), (240, 60, 4), (400, 100, 4), (100, 100, 1)])
def test_frequency_overrides(physics, policy, decimation):
    cfg = config()
    cfg.physics_hz, cfg.policy_hz = physics, policy
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    assert cfg.decimation == decimation
    assert math.isclose(cfg.sim.dt * cfg.decimation, 1 / policy)
    assert math.ceil(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation)) == 20 * policy


@pytest.mark.parametrize("field,value", [
    ("control_mode", "typo"), ("physics_hz", 0), ("policy_hz", -1), ("policy_hz", 300),
    ("policy_hz", 60), ("effort_lim_sim", -1.0), ("velocity_lim_sim", 0.0),
])
def test_invalid_control_settings_fail_before_simulation(field, value):
    cfg = config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)


def action_env(mode, hold_arm=False):
    cfg = config()
    cfg.control_mode, cfg.hold_arm = mode, hold_arm
    method("borinot_goalkeeper_env_cfg.py", "configure_control")(cfg)
    calls = {}

    def record(name):
        def setter(value=None, **kwargs):
            calls[name] = value.clone() if value is not None else kwargs
        return setter

    positions = torch.tensor(cfg.rotor_pos_b, dtype=torch.float64)
    allocation = torch.stack((torch.ones(6, dtype=torch.float64), positions[:, 1], -positions[:, 0],
                              torch.tensor(cfg.rotor_spin, dtype=torch.float64) * cfg.km_over_kf))
    robot = SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=torch.zeros(3, 3, dtype=torch.float64)),
        permanent_wrench_composer=SimpleNamespace(set_forces_and_torques=record("wrench")),
        set_joint_position_target=record("position"), set_joint_velocity_target=record("velocity"),
        set_joint_effort_target=record("effort"))
    return SimpleNamespace(cfg=cfg, robot=robot, num_envs=3, device="cpu", joint_ids=[0, 1], base_id=0,
        actions=torch.zeros(3, cfg.action_space, dtype=torch.float64), _physics_samples=0,
        _sample_power=lambda: None, _sample_physics_events=lambda: None, physics_dt=cfg.sim.dt,
        _rate_scale=torch.tensor(cfg.body_rate_scale), _rate_integral=torch.zeros(3, 3, dtype=torch.float64),
        _kp=torch.tensor(cfg.rate_kp), _ki=torch.tensor(cfg.rate_ki), _hover_total=30.0, _hover_per_rotor=5.0,
        _allocation=allocation, _allocation_inverse=torch.linalg.pinv(allocation),
        _rotor_thrust=torch.full((3, 6), 5.0, dtype=torch.float64),
        _motor_alpha=1-math.exp(-cfg.sim.dt/cfg.motor_time_constant_s),
        _joint_scale=torch.tensor(cfg.joint_target_scale)), calls


@pytest.mark.parametrize("mode", MODES[:2])
def test_direct_rotors_have_no_rate_feedback_and_keep_motor_lag(mode):
    env, calls = action_env(mode)
    env.actions[:, :6] = torch.tensor([-1., -.5, 0., .5, 1., 0.])
    env.robot.data.root_ang_vel_b[:] = torch.tensor([5., -4., 3.])
    requested = torch.tensor([0., 2.5, 5., 7.5, 10., 5.], dtype=torch.float64)
    expected = 5.0 + env._motor_alpha * (requested - 5.0)
    apply = method("borinot_goalkeeper_env.py", "_apply_action")
    apply(env)
    torch.testing.assert_close(env._rotor_thrust, expected.expand(3, -1))
    assert not env._rate_integral.any()
    wrench = env._rotor_thrust @ env._allocation.T
    torch.testing.assert_close(calls["wrench"]["forces"][:, 0, 2], wrench[:, 0].float())
    torch.testing.assert_close(calls["wrench"]["torques"][:, 0], wrench[:, 1:])
    apply(env)
    torch.testing.assert_close(env._rotor_thrust, (expected + env._motor_alpha * (requested-expected)).expand(3, -1))


def test_torque_actions_have_signed_effort_limits_without_joint_pd():
    env, calls = action_env(MODES[0])
    env.actions[:, 6:] = torch.tensor([[-1., 1.], [.5, -.5], [0., 0.]])
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(calls["effort"], env.actions[:, 6:] * env.cfg.effort_lim_sim)
    assert "position" not in calls and "velocity" not in calls


@pytest.mark.parametrize("mode", MODES[1:])
def test_pd_modes_use_two_scalar_angles_and_clear_feedforward_effort(mode):
    env, calls = action_env(mode)
    env.actions[:, -2:] = torch.tensor([[-1., 1.], [.5, -.5], [0., 0.]])
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(calls["position"], env.actions[:, -2:] * env._joint_scale)
    assert not calls["velocity"].any() and not calls["effort"].any()


def test_pi_mode_retains_the_original_rate_controller():
    env, _ = action_env(MODES[2])
    env.actions[:, :4] = torch.tensor([.2, .5, -.25, .1])
    env.robot.data.root_ang_vel_b[:] = torch.tensor([.2, -.1, .05])
    error = env.actions[:, 1:4] * env._rate_scale - env.robot.data.root_ang_vel_b
    integral = error * env.physics_dt
    torque = env._kp * error + env._ki * integral
    total_thrust = env._hover_total * (1 + .7 * env.actions[:, :1])
    requested = (torch.cat((total_thrust, torque), -1) @ env._allocation_inverse.T).clamp(0., 10.)
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    torch.testing.assert_close(env._rate_integral, integral)
    torch.testing.assert_close(env._rotor_thrust, 5.0 + env._motor_alpha * (requested-5.0))


@pytest.mark.parametrize("mode", MODES)
def test_hold_arm_is_neutral_pd_in_every_mode(mode):
    env, calls = action_env(mode, hold_arm=True)
    env.actions[:, -2:] = 1.0
    method("borinot_goalkeeper_env.py", "_apply_action")(env)
    assert not calls["position"].any() and not calls["effort"].any()
    assert env.cfg.robot.actuators["arm"].stiffness == 35.0
