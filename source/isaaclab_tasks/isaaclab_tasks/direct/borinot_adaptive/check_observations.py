"""Run with isaaclab.sh -p <this file> --headless. Fresh Kit process; no training."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args()).app

import gymnasium as gym
import torch
from isaaclab_tasks.direct.borinot_adaptive.cfg import GoToStudentCfg


def check():
    cfg = GoToStudentCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 43
    env = gym.make("Borinot-goto-student-rl", cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        h = env._history
        assert obs["policy"].shape == (4, 869) and obs["critic"].shape == (4, 97)
        assert h[:, :-1].count_nonzero() == 0 and h[:, -1, -8:].count_nonzero() == 0
        assert torch.equal(obs["policy"], env._get_observations()["policy"])
        a = torch.linspace(-0.2, 0.2, 8, device=env.device).expand(4, -1)
        obs, _, terminated, truncated, _ = env.step(a)
        assert not (terminated | truncated).any()
        torch.testing.assert_close(h[:, -2:, -8:], a[:, None].expand(-1, 2, -1))
        assert h[:, :-3].count_nonzero() == 0
        torch.testing.assert_close(obs["policy"][:, 837:864], h[:, -1])
        # Modify only an oracle input, without advancing the plant.
        env._static[:, 0] += 1
        other = env._get_observations()
        assert torch.equal(obs["policy"], other["policy"]) and not torch.equal(obs["critic"], other["critic"])
        # A new command cannot alter history or direct current measurements.
        env._goal_position_w[:, 0] += 1
        changed = env._get_observations()["policy"]
        assert torch.equal(changed[:, :-5], other["policy"][:, :-5])
        assert not torch.equal(changed[:, -5:-2], other["policy"][:, -5:-2])
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
