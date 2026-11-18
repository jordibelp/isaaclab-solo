"""Run with isaaclab.sh -p <this file> --headless. Fresh Kit process; no training."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("goto", "cmd-vel"), default="goto")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse
from isaaclab_tasks.direct.borinot_adaptive.cfg import GoToStudentCfg, VelocityStudentCfg


def check():
    goto = args.task == "goto"
    cfg = GoToStudentCfg() if goto else VelocityStudentCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 43
    env = gym.make(f"Borinot-{args.task}-student-rl", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        h = env._history
        assert obs["policy"].shape == (4, 901 if goto else 900)
        assert obs["critic"].shape == (4, 99 if goto else 98)
        assert h.shape == (4, 31, 28)
        assert not any(cfg.sensor_std)
        data = env.robot.data
        quat = data.root_quat_w
        expected = torch.cat((
            data.root_ang_vel_b,
            quat_apply_inverse(quat, data.body_lin_acc_w[:, env.base_id]),
            data.joint_pos[:, env.joint_ids],
            data.joint_vel[:, env.joint_ids],
            matrix_from_quat(quat).flatten(1),
            data.root_pos_w[:, 2:3],
            env.actions,
        ), -1)
        torch.testing.assert_close(h[:, -1], expected, atol=0, rtol=0)
        assert torch.equal(obs["critic"][:, 66:94], h[:, -1])
        if goto:
            expected_goal = quat_apply_inverse(
                quat, env._goal_position_w - data.root_pos_w)
            torch.testing.assert_close(obs["policy"][:, -5:-2], expected_goal)
        assert h[:, :-1].count_nonzero() == 0 and h[:, -1, -8:].count_nonzero() == 0
        assert torch.equal(obs["policy"], env._get_observations()["policy"])
        a = torch.linspace(-0.2, 0.2, 8, device=env.device).expand(4, -1)
        obs, _, terminated, truncated, _ = env.step(a)
        assert not (terminated | truncated).any()
        torch.testing.assert_close(h[:, -2:, -8:], a[:, None].expand(-1, 2, -1))
        assert h[:, :-3].count_nonzero() == 0
        torch.testing.assert_close(obs["policy"][:, 868:896], h[:, -1])
        # Modify only an oracle input, without advancing the plant.
        env._static[:, 0] += 1
        other = env._get_observations()
        assert torch.equal(obs["policy"], other["policy"]) and not torch.equal(obs["critic"], other["critic"])
        # A new command cannot alter history or direct current measurements.
        if goto:
            env._goal_position_w[:, 0] += 1
        else:
            env._commands[:, 0] += 1
        changed = env._get_observations()["policy"]
        command_width = 5 if goto else 4
        assert torch.equal(changed[:, :-command_width], other["policy"][:, :-command_width])
        assert not torch.equal(changed[:, -command_width:], other["policy"][:, -command_width:])
        keep = h[1:].clone()
        env._reset_idx(torch.tensor([0], device=env.device))
        env._get_observations()
        assert h[0, :-1].count_nonzero() == 0 and torch.equal(h[1:], keep)
        print("PASS: sizes, endpoint padding, read stability, issued-action timing, bypass identity, "
              "privilege isolation, command isolation, partial reset.")
    finally:
        env.close()


if __name__ == "__main__":
    check()
    app.close()
