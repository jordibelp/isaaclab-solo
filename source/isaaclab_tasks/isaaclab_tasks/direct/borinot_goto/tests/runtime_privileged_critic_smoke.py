# SPDX-License-Identifier: BSD-3-Clause
"""Local Isaac Sim 5 smoke for privileged plant provenance and partial reset."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
try:
    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    cfg = load_cfg_from_registry("Borinot-goto-minTime", "env_cfg_entry_point")
    cfg.control_mode = "dsam_indi"
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.flight_randomization.wind_enabled = True
    cfg.flight_randomization.thrust_rotor_gain_delta = .2
    cfg.flight_randomization.motor_time_constant_range_s = (.01, .08)
    env = gym.make("Borinot-goto-minTime", cfg=cfg).unwrapped
    env.reset()
    assert env.cfg.use_asym_actor_critic is True
    assert env._critic_static_plant.shape == (4, 30)
    mass = env.robot.root_physx_view.get_masses().to(env.device)
    inertia = env.robot.root_physx_view.get_inertias().to(env.device)[:, env.base_id]
    torch.testing.assert_close(env._critic_static_plant[:, 0], mass.sum(-1))
    torch.testing.assert_close(env._critic_static_plant[:, 2:5], inertia[:, [0,4,8]])
    friction = env.robot.root_physx_view.get_dof_friction_properties().to(env.device)[:, env.joint_ids]
    torch.testing.assert_close(env._critic_static_plant[:, 11:17], friction.transpose(1, 2).flatten(1))
    env._base_push_force_z_w.fill_(2)
    env._base_push_points_b[:, 0] = .1
    env._apply_action()
    data = env.robot.data
    q = data.root_quat_w
    vertical = torch.zeros_like(data.root_lin_vel_w)
    vertical[:, 2] = 2.
    push = quat_apply_inverse(q, vertical)
    wind, wind_moment = env._wind.wrench_world(data.root_lin_vel_w)
    expected = torch.cat((push + quat_apply_inverse(q, wind),
        torch.linalg.cross(env._base_push_points_b-env._base_com_b, push)
        + quat_apply_inverse(q, wind_moment)), -1)
    torch.testing.assert_close(env._critic_disturbance_b, expected)
    applied = env._rotor_plant.kf * env._rotor_plant.speed.square() / env._hover_per_rotor
    torch.testing.assert_close(env._critic_plant_observation()[:, -12:-6], applied)
    assert not torch.allclose(applied, env._indi.observation())
    actor = env._get_observations()["policy"].clone()
    before = env._critic_disturbance_b.clone()
    env._critic_disturbance_b.add_(3)
    torch.testing.assert_close(actor, env._get_observations()["policy"], atol=0, rtol=0)
    env._critic_disturbance_b.copy_(before)
    env._reset_idx(torch.tensor([0], device=env.device))
    assert not env._critic_disturbance_b[0].any()
    torch.testing.assert_close(env._critic_disturbance_b[1:], before[1:])
    print("PRIVILEGED_PLANT_PASS", flush=True)
    env.close()
except Exception:
    import traceback
    traceback.print_exc()
finally:
    app.close()
