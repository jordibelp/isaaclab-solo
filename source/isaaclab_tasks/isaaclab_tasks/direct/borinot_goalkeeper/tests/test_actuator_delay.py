# SPDX-License-Identifier: BSD-3-Clause
"""Policy-command delay: whole physics steps, neutral after resets, before every controller."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_control import MODES, action_env, actuator_delay, method

ActuatorDelay = actuator_delay.ActuatorDelay
DIRECT = Path(__file__).resolve().parents[2]
TASKS = ("borinot_goalkeeper", "borinot_goalkeeper_simple", "borinot_cmd_vel")


def delay_cfg(delay=0.0, delay_range=None, physics_hz=200):
    return SimpleNamespace(actuator_delay=delay, actuator_delay_range=delay_range, physics_hz=physics_hz)


@pytest.mark.parametrize("seconds,hz,steps", [(0, 200, 0), (0.0, 200, 0), (0.005, 200, 1), (0.02, 200, 4),
                                              (0.04, 200, 8), (0.01, 300, 3), (1 / 300, 300, 1), (0.06, 100, 6)])
def test_delay_is_a_whole_number_of_physics_steps(seconds, hz, steps):
    assert actuator_delay.delay_steps(seconds, hz) == steps


@pytest.mark.parametrize("seconds", [0.0125, 0.001, -0.005, math.nan, math.inf, True, "0.02", None])
def test_off_grid_or_invalid_delays_fail_instead_of_rounding(seconds):
    with pytest.raises(ValueError):
        actuator_delay.delay_steps(seconds, 200)


@pytest.mark.parametrize("delay,delay_range,bounds", [
    (0.0, None, (0, 0)), (0.02, None, (4, 4)), (0.0, (0.0, 0.04), (0, 8)), (0.0, [0, 0.04], (0, 8)),
    (0.0, (0.01, 0.01), (2, 2)), (0.0, (0.015, 0.03), (3, 6)),
])
def test_bounds_come_from_the_fixed_delay_or_the_hydra_range(delay, delay_range, bounds):
    assert actuator_delay.actuator_delay_bounds(delay_cfg(delay, delay_range)) == bounds


@pytest.mark.parametrize("delay,delay_range", [
    (0.02, (0.0, 0.04)), (0.0, (0.04, 0.0)), (0.0, (0.01,)), (0.0, 0.02), (0.0, (0.0, 0.0125)), (0.0, (-0.005, 0.0)),
])
def test_conflicting_or_malformed_settings_fail(delay, delay_range):
    with pytest.raises(ValueError):
        actuator_delay.actuator_delay_bounds(delay_cfg(delay, delay_range))


def test_zero_delay_returns_the_policy_command_itself():
    delay = ActuatorDelay(delay_cfg(), 3, 8, "cpu")
    actions = torch.randn(3, 8)
    assert delay(actions) is actions and delay._queue is None
    delay.reset(torch.tensor([1]))
    assert delay(actions) is actions and not delay.seconds.any()


@pytest.mark.parametrize("steps", [1, 3, 8])
def test_fixed_delay_returns_the_command_sent_steps_earlier_after_a_neutral_start(steps):
    delay = ActuatorDelay(delay_cfg(steps / 200), 3, 2, "cpu", torch.float64)
    delay.reset(torch.arange(3))
    scale = torch.tensor([[1.], [10.], [100.]], dtype=torch.float64)
    sent = [torch.full((3, 2), k + 1.0, dtype=torch.float64) * scale for k in range(20)]
    for k, command in enumerate(sent):
        expected = sent[k - steps] if k >= steps else torch.zeros_like(command)
        torch.testing.assert_close(delay(command), expected, rtol=0, atol=0)
    torch.testing.assert_close(delay.seconds, torch.full((3,), steps / 200))


def test_reset_refills_only_the_selected_environment_with_neutral_commands():
    delay = ActuatorDelay(delay_cfg(0.01), 3, 1, "cpu", torch.float64)  # 2 physics steps
    for k in range(5):
        delay(torch.full((3, 1), k + 1.0, dtype=torch.float64))
    delay.reset(torch.tensor([1]))
    received = [delay(torch.full((3, 1), k + 6.0, dtype=torch.float64)).flatten().tolist() for k in range(3)]
    # Environment 1 receives two neutral commands, then its first post-reset command.
    assert received == [[4., 0., 4.], [5., 0., 5.], [6., 6., 6.]]


def test_each_environment_reads_its_own_delay():
    delay = ActuatorDelay(delay_cfg(delay_range=(0.0, 0.04)), 9, 1, "cpu", torch.float64)
    delay.steps.copy_(torch.arange(9))
    for k in range(1, 12):
        expected = (k - torch.arange(9)).clamp_min(0).double()  # zero until the first command arrives
        torch.testing.assert_close(delay(torch.full((9, 1), float(k), dtype=torch.float64)).flatten(), expected,
                                   rtol=0, atol=0)


def test_legacy_task_uses_its_actual_sim_dt_without_a_second_frequency_setting():
    cfg = SimpleNamespace(actuator_delay=0.025, actuator_delay_range=None, sim=SimpleNamespace(dt=1 / 120))
    delay = ActuatorDelay(cfg, 2, 8, "cpu")
    assert (delay.low, delay.high) == (3, 3)
    for k in range(6):
        torch.testing.assert_close(delay(torch.full((2, 8), k + 1.)), torch.full((2, 8), max(k - 2., 0.)))


@pytest.mark.parametrize("angles", [False, True])
def test_legacy_mapping_receives_the_delayed_command_and_preserves_newest_actions(angles):
    methods = class_methods("borinot")
    scope = {"torch": torch}
    names = ["_map_thrust", "_map_actions_angle_des" if angles else "_map_actions"]
    for name in names:
        exec(compile(ast.Module(body=[methods[name]], type_ignores=[]), "legacy", "exec"), scope)
    from types import MethodType
    cfg = SimpleNamespace(thrust_mode="delta", hover_thrust_per_rotor=4., thrust_min=0., thrust_max=12.,
                          thrust_max_scale=3., delta_thrust_scale=1., torque_min=(-2., -2.),
                          torque_max=(2., 2.), torque_lim=2., bias_torque_j2=.1)
    calls = {}
    env = SimpleNamespace(cfg=cfg, device="cpu", actions=torch.ones(2, 10 if angles else 8), _leg_dof_ids=[0, 1],
        robot=SimpleNamespace(set_joint_position_target=lambda x, **kw: calls.update(position=x),
                              set_joint_velocity_target=lambda x, **kw: calls.update(velocity=x)))
    env._map_thrust = MethodType(scope["_map_thrust"], env)
    delay = ActuatorDelay(delay_cfg(.01), 2, env.actions.shape[1], "cpu")
    mapping = scope[names[-1]]
    for k in range(5):
        arrived = delay(env.actions)
        thrust = mapping(env, arrived)
        if not angles:
            thrust, torque = thrust
            torch.testing.assert_close(torque, torch.tensor([[0., .1], [0., .1]]) if k < 2
                                       else torch.full((2, 2), 2.))
        else:
            torch.testing.assert_close(calls["position"], torch.full((2, 2), 0. if k < 2 else math.pi / 4))
        torch.testing.assert_close(thrust, torch.full((2, 6), 4. if k < 2 else 8.))
        assert env.actions.eq(1.).all()


def test_range_draws_uniform_whole_steps_at_each_reset_and_holds_them_between_resets():
    torch.manual_seed(3)
    n = 9000
    delay = ActuatorDelay(delay_cfg(delay_range=(0.0, 0.04)), n, 1, "cpu")
    ids = torch.arange(n)
    delay.reset(ids)
    counts = torch.bincount(delay.steps, minlength=9)
    assert len(counts) == 9 and counts.min() > 850 and counts.max() < 1150  # 1000 expected per value
    before = delay.steps.clone()
    for _ in range(4):
        delay(torch.zeros(n, 1))
    assert torch.equal(delay.steps, before)
    delay.reset(ids[:50])
    assert torch.equal(delay.steps[50:], before[50:]) and not torch.equal(delay.steps[:50], before[:50])
    assert torch.equal((delay.seconds * 200).round().long(), delay.steps) and delay.seconds.max() <= .04 + 1e-7


@pytest.mark.parametrize("steps,expected", [(0, [1, 1, 2, 2, 3, 3]), (1, [0, 1, 1, 2, 2, 3]), (2, [0, 0, 1, 1, 2, 2])])
def test_policy_commands_held_over_decimation_arrive_after_the_physics_step_delay(steps, expected):
    delay = ActuatorDelay(delay_cfg(steps / 200), 1, 1, "cpu", torch.float64)
    received = []
    for policy_action in (1., 2., 3.):
        actions = torch.full((1, 1), policy_action, dtype=torch.float64)
        for _ in range(2):  # decimation 2: 200 Hz physics, 100 Hz policy
            received.append(delay(actions).item())
    assert received == expected


def snapshot(env, calls):
    values = {"thrust": env._rotor_thrust.clone(), "commands": env._actuator_commands.clone(),
              "integral": env._rate_integral.clone(), "costs": env._action_limit_costs.clone(),
              "forces": calls["wrench"]["forces"].clone(), "torques": calls["wrench"]["torques"].clone()}
    values.update({name: calls[name].clone() for name in ("effort", "position", "velocity") if name in calls})
    return values


def assert_same(left, right):
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)


def delayed_pair(make, steps):
    """A delayed environment and an undelayed twin that is fed the shifted command sequence."""
    delayed, delayed_calls = make()
    delayed.cfg.actuator_delay = steps / delayed.cfg.physics_hz
    delayed._actuator_delay = ActuatorDelay(delayed.cfg, delayed.num_envs, delayed.cfg.action_space, "cpu", torch.float64)
    reference, reference_calls = make()
    return (delayed, delayed_calls), (reference, reference_calls)


def run_shifted(pair, policy_actions, steps):
    (delayed, delayed_calls), (reference, reference_calls) = pair
    apply = method("borinot_goalkeeper_env.py", "_apply_action")
    sent = [actions for actions in policy_actions for _ in range(delayed.cfg.decimation)]  # held per physics step
    arrived = [torch.zeros_like(sent[0])] * steps + sent[:len(sent) - steps]
    for command, late in zip(sent, arrived):
        delayed.actions.copy_(command)
        reference.actions.copy_(late)
        apply(delayed)
        apply(reference)
        torch.testing.assert_close(delayed.actions, command, rtol=0, atol=0)  # observation keeps the newest output
        assert_same(snapshot(delayed, delayed_calls), snapshot(reference, reference_calls))


@pytest.mark.parametrize("mode,sac", [(mode, False) for mode in MODES] + [(mode, True) for mode in MODES[:2]])
@pytest.mark.parametrize("steps", [1, 3, 8])
def test_every_direct_and_pi_mode_controls_with_the_delayed_command_only(mode, sac, steps):
    def make():
        env, calls = action_env(mode)
        env.cfg.sac_action_mapping = sac
        env._sac_joint_offset = torch.tensor([.1, -.2], dtype=torch.float64)
        return env, calls

    torch.manual_seed(steps)
    policy_actions = [torch.randn(3, make()[0].cfg.action_space, dtype=torch.float64) for _ in range(9)]
    run_shifted(delayed_pair(make, steps), policy_actions, steps)


def indi_env():
    from test_indi import fixture, home_quat

    ctrl = fixture()  # four environments at 300 Hz
    env, calls = action_env("thrust_and_torque_control")
    env.cfg, env._indi, env.num_envs = ctrl.task, ctrl, 4
    env.actions = torch.zeros(4, 9, dtype=torch.float64)
    env._actuator_delay = ActuatorDelay(env.cfg, 4, 9, "cpu", torch.float64)
    env._rotor_thrust = torch.full((4, 6), ctrl.hover, dtype=torch.float64)
    env._actuator_commands = torch.zeros(4, 8, dtype=torch.float64)
    env._base_com_offset = torch.zeros(4, 3, dtype=torch.float64)
    env._action_limit_costs = torch.zeros(4, 3, dtype=torch.float64)
    env._rate_integral = torch.zeros(4, 3, dtype=torch.float64)
    env.robot.data.root_quat_w = home_quat(ctrl)
    env.robot.data.root_ang_vel_b = torch.zeros(4, 3, dtype=torch.float64)
    env.robot.data.joint_pos = torch.zeros(4, 2, dtype=torch.float64)
    env.robot.data.joint_vel = torch.zeros(4, 2, dtype=torch.float64)
    return env, calls


@pytest.mark.parametrize("steps", [1, 4])
def test_indi_receives_delayed_references(steps):
    torch.manual_seed(11)
    scale = torch.tensor([2., 2., 2., 1., 1., 1., .3, .5, .5], dtype=torch.float64)
    policy_actions = [torch.randn(4, 9, dtype=torch.float64) * scale for _ in range(6)]
    run_shifted(delayed_pair(indi_env, steps), policy_actions, steps)


@pytest.mark.parametrize("make", [indi_env, lambda: action_env(MODES[2])], ids=["indi", "pi"])
def test_inner_loops_react_to_measurements_without_the_command_delay(make):
    (delayed, delayed_calls), (reference, reference_calls) = delayed_pair(make, 6)
    apply = method("borinot_goalkeeper_env.py", "_apply_action")
    previous = None
    for substep in range(10):
        rates = torch.tensor([.4, -.3, .2], dtype=torch.float64) if substep >= 3 else torch.zeros(3, dtype=torch.float64)
        for env in (delayed, reference):
            env.robot.data.root_ang_vel_b = rates.expand(env.num_envs, -1).clone()
            apply(env)  # neutral policy commands throughout
        current = snapshot(delayed, delayed_calls)
        assert_same(current, snapshot(reference, reference_calls))
        if substep == 3:  # the rate disturbance changes rotor thrust in the same physics step
            assert not torch.equal(current["torques"], previous["torques"])
        previous = current


def class_methods(task):
    tree = ast.parse((DIRECT / task / f"{task}_env.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("task", TASKS)
def test_every_task_validates_before_the_simulator_resets_after_logging_and_logs_the_delay(task):
    methods = class_methods(task)
    init = ast.unparse(methods["__init__"])
    assert (init.index("actuator_delay_bounds(cfg)") < init.index("super().__init__(")
            < init.index("ActuatorDelay(cfg, self.num_envs, cfg.action_space, self.device)"))
    reset = ast.unparse(methods["_reset_idx"])
    assert reset.index("self._update_curriculum(env_ids)") < reset.index("self._actuator_delay.reset(env_ids)")
    assert "Episode/actuator_delay_s" in ast.unparse(methods["_update_curriculum"])


def test_all_tasks_read_the_policy_output_only_through_the_shared_delay():
    text = ast.unparse(class_methods("borinot_goalkeeper")["_apply_action"])
    assert text.count("self.actions") == 1 and "actions = self._actuator_delay(self.actions)" in text
    assert "BorinotGoalkeeperEnv._apply_action(self)" in ast.unparse(class_methods("borinot_cmd_vel")["_apply_action"])
    simple = next(n for n in ast.parse((DIRECT / "borinot_goalkeeper_simple/borinot_goalkeeper_simple_env.py")
                                       .read_text()).body if isinstance(n, ast.ClassDef))
    assert any(isinstance(n, ast.Assign) and ast.unparse(n) == "_apply_action = BorinotGoalkeeperEnv._apply_action"
               for n in simple.body)
    for task in ("borinot_goalkeeper", "borinot_cmd_vel"):
        assert "ActuatorDelay" in ast.unparse(class_methods(task)["get_code_snapshot_files"])


def literal_fields(path):
    cls = next(n for n in ast.parse((DIRECT / path).read_text()).body if isinstance(n, ast.ClassDef))
    return {n.targets[0].id: n.value for n in cls.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}


@pytest.mark.parametrize("path", ["borinot_goalkeeper/borinot_goalkeeper_env_cfg.py",
                                  "borinot_goalkeeper_simple/borinot_goalkeeper_simple_env_cfg.py"])
def test_defaults_add_no_delay_and_cmd_vel_inherits_them(path):
    fields = literal_fields(path)
    delay, delay_range = (ast.literal_eval(fields[name]) for name in ("actuator_delay", "actuator_delay_range"))
    assert type(delay) is float and delay == 0.0 and delay_range is None
    assert not {"actuator_delay", "actuator_delay_range"} & literal_fields("borinot_cmd_vel/borinot_cmd_vel_env_cfg.py").keys()


def test_episode_log_reports_the_mean_delay_of_completed_episodes_only():
    from test_curriculum_checkpoint import make_env, update_curriculum

    env = make_env(n=3)
    env._actuator_delay.seconds = torch.tensor([.01, .5, .03])
    env._episode_steps[1] = 0  # initial/non-episode reset
    update_curriculum(env, torch.arange(3))
    torch.testing.assert_close(env.extras["log"]["Episode/actuator_delay_s"], torch.tensor(.02))
