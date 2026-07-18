from pathlib import Path

import mujoco
import numpy as np

import play_direct_mujoco as sim2sim


def test_model_contract():
    env = sim2sim.Solo12Mujoco(Path(__file__).with_name("solo12.xml"))
    assert env.model.opt.timestep == sim2sim.PHYSICS_DT
    assert env.model.nu == 12
    assert np.isclose(env.model.body_mass.sum(), 3.028572, atol=1e-7)
    assert env.observation((0.5, 0.0, 0.0)).shape == (48,)
    assert np.allclose(env.data.qpos[env.joint_qpos], sim2sim.SAFE_Q)


def test_action_pd_and_step_are_finite():
    env = sim2sim.Solo12Mujoco(Path(__file__).with_name("solo12.xml"))
    env.step(np.zeros(12))
    assert np.isfinite(env.data.qpos).all()
    assert np.max(np.abs(env.data.ctrl)) <= sim2sim.EFFORT_LIMIT


def test_command_parser():
    assert sim2sim.parse_commands("0.5 0 0; 0 .3 0;") == [(0.5, 0.0, 0.0), (0.0, 0.3, 0.0)]
