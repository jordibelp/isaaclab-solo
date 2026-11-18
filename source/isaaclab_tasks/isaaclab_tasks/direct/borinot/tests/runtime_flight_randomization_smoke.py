# SPDX-License-Identifier: BSD-3-Clause
"""Run one registered Borinot task per headless Isaac Sim process."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
sys.argv = [sys.argv[0], *overrides]
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, _agent):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 8, args.device, 18
    if hasattr(cfg, "forces_applied_to_base_curriculum"):
        cfg.forces_applied_to_base_curriculum = (0.0, 2.0)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset()
        rotor, wind = env._rotor_plant, env._wind
        assert rotor is not None and wind is not None and wind.cfg.wind_enabled
        delay = env._actuator_delay
        assert cfg.actuator_delay_range == "auto" and (delay.low, delay.high) == (0, int(.02 / env.physics_dt + 1e-6))
        assert ((delay.steps >= delay.low) & (delay.steps <= delay.high)).all()
        assert rotor.kf.min() >= .87 * cfg.kf and rotor.kf.max() <= 1.13 * cfg.kf
        assert rotor.km.min() >= .90 * cfg.km and rotor.km.max() <= 1.10 * cfg.km
        actual_tau = -env.physics_dt / torch.log1p(-rotor.alpha)
        assert actual_tau.min() >= .035 - 1e-7 and actual_tau.max() <= .045 + 1e-7

        if hasattr(env, "_wrench"):
            assert env._wrench.task.kf == cfg.kf
            assert isinstance(env._wrench.motor_alpha, float)
        ids = torch.arange(env.num_envs, device=env.device)
        composer = env.robot.permanent_wrench_composer
        base = env.base_id if hasattr(env, "base_id") else env._base_body_id[0]
        wind.steady[:] = torch.tensor([2., 0., 0.], device=env.device)
        wind.gust[:] = torch.tensor([0., 1., 0.], device=env.device)
        wind.moment[:] = torch.tensor([0., 0., .04], device=env.device)
        wind.until_next[:] = 1000
        wind.total[:] = wind.left[:] = 3
        has_push = hasattr(env, "_base_push_force_z_w")
        if has_push:
            env._base_push_steps_until_next[:] = 1000
            env._base_push_steps_left[:] = 3
            env._base_push_force_z_w[:] = 1.5
            env._base_push_points_b.copy_(env._base_com_b)
            env._base_push_points_b[:, 0] += .1

        original = env._apply_action
        checked = 0

        def checked_action():
            nonlocal checked
            data = env.robot.data
            quat, velocity = data.root_quat_w.clone(), data.root_lin_vel_w.clone()
            speed_before = rotor.speed.clone()
            original()
            if checked >= env.cfg.decimation:
                return
            thrust = rotor.kf * rotor.speed.square()
            force = torch.zeros_like(velocity)
            force[:, 2] = thrust.sum(-1)
            air = wind.steady + wind.gust * wind.weight - velocity
            wind_world = .08 * air.norm(dim=-1, keepdim=True) * air
            force += quat_apply_inverse(quat, wind_world)
            spin = rotor.spin
            yaw = (rotor.km * rotor.speed.square() * spin).sum(-1)
            if getattr(cfg, "control_mode", None) in ("dsam_indi", "model_wrench_indi", "model_wrench_yawless_indi"):
                yaw += ((rotor.speed - speed_before) * spin).sum(-1) * cfg.indi.rotor_spin_inertia_kg_m2 / env.physics_dt
            yaw += quat_apply_inverse(quat, wind.moment * wind.weight)[:, 2]
            if has_push:
                push_w = torch.zeros_like(force)
                push_w[:, 2] = 1.5
                push_b = quat_apply_inverse(quat, push_w)
                force += push_b
                yaw += torch.linalg.cross(env._base_push_points_b - env._base_com_b, push_b)[:, 2]
            torch.testing.assert_close(composer.composed_force_as_torch[:, base], force, atol=2e-5, rtol=0)
            torch.testing.assert_close(composer.composed_torque_as_torch[:, base, 2], yaw, atol=2e-5, rtol=0)
            checked += 1

        env._apply_action = checked_action
        with torch.inference_mode():
            for _ in range(8):
                obs, reward, *_ = env.step(torch.full_like(env.actions, .1))
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
            assert checked == cfg.decimation
            speed, kf, km, steady, left = (t.clone() for t in (rotor.speed, rotor.kf, rotor.km, wind.steady, wind.left))
            env._reset_idx(ids[:1])
            assert (rotor.speed[0] == rotor.reset_speed).all() and rotor.speed[1:].equal(speed[1:])
            assert rotor.kf.equal(kf) and rotor.km.equal(km)
            assert wind.steady[1:].equal(steady[1:]) and wind.left[1:].equal(left[1:])
            assert not wind.weight[0].any() and not wind.moment[0].any()
            assert env._actuator_delay._queue[:, 0].count_nonzero() == 0
        print(f"FLIGHT_RANDOMIZATION_OK {args.task} {getattr(cfg, 'control_mode', 'legacy')}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
