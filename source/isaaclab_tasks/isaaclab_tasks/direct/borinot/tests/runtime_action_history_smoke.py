# SPDX-License-Identifier: BSD-3-Clause
"""Check policy-rate command history in a live Borinot task (one task per process)."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
AppLauncher.add_app_launcher_args(parser)
args, extra = parser.parse_known_args()
sys.argv = [sys.argv[0], *extra]
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent):
    steps = cfg.include_prev_action_steps
    assert steps > 1
    cfg.scene.num_envs = 3
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = 42
    cfg.actuator_delay = 0.03
    cfg.actuator_delay_range = None
    cfg.curriculum = False
    cfg.goal_debug_vis = False
    cfg.reset_robot_on_tilt_threshold = False
    baseline = cfg.copy()
    baseline.include_prev_action_steps = 0
    baseline.configure_control()
    one = baseline.copy()
    one.include_prev_action_steps = 1
    one.configure_control()
    assert (one.observation_space, one.state_space) == (baseline.observation_space, baseline.state_space)

    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset()
        dim = env.cfg.action_space
        extra_dim = (steps - 1) * dim
        assert obs["policy"].shape == (3, baseline.observation_space + extra_dim)
        assert env.single_observation_space["policy"].shape == obs["policy"].shape[1:]
        shares_history = hasattr(cfg, "use_asym_actor_critic") and not cfg.use_asym_actor_critic
        assert obs["critic"].shape == (3, baseline.state_space + (extra_dim if shares_history else 0))
        assert env.single_observation_space["critic"].shape == obs["critic"].shape[1:]
        expected = env.actions.new_zeros(3, steps, dim)
        assert not obs["policy"][:, -extra_dim:].any()
        assert env._actuator_delay._queue is not None
        assert env.cfg.decimation > 1
        # Capture the actual input consumed by the plant, not the issued command.
        delayed = []
        apply_delay = env._actuator_delay

        class CaptureDelay:
            def __getattr__(self, name):
                return getattr(apply_delay, name)

            def __call__(self, action):
                result = apply_delay(action)
                delayed.append(result.clone())
                return result

        env._actuator_delay = CaptureDelay()
        for tick in range(2 * steps + 1):
            delayed.clear()
            # Distinct command per env, tick and action index. Deliberately stay
            # inside all scalar clamps: the known sent vector is the expected one.
            submitted = (torch.arange(3 * dim, device=env.device).reshape(3, dim) + 1 + 3 * tick) * 0.002
            expected = torch.cat((submitted[:, None], expected[:, :-1]), dim=1)
            obs, _, died, timeout, _ = env.step(submitted)
            assert not (died | timeout).any()
            assert len(delayed) == env.cfg.decimation
            if tick == 0:
                assert not torch.equal(submitted, delayed[-1])
            torch.testing.assert_close(env.actions, submitted, rtol=0, atol=0)
            for current in (obs, env._get_observations(), env._get_observations()):
                torch.testing.assert_close(current["policy"][:, -extra_dim:], expected[:, 1:].flatten(1), rtol=0, atol=0)
                if shares_history:
                    torch.testing.assert_close(current["critic"][:, -extra_dim:], expected[:, 1:].flatten(1), rtol=0, atol=0)
            # The feature appends to the baseline view without disturbing existing
            # fields (including GoTo's optional short actor layout).
            at_state = env._get_observations()
            for legacy_steps in (0, 1):
                cfg.include_prev_action_steps = legacy_steps
                without_history = env._get_observations()
                torch.testing.assert_close(at_state["policy"][:, :-extra_dim], without_history["policy"], rtol=0, atol=0)
                critic_core = at_state["critic"][:, :-extra_dim] if shares_history else at_state["critic"]
                torch.testing.assert_close(critic_core, without_history["critic"], rtol=0, atol=0)
            cfg.include_prev_action_steps = steps
            if tick == steps:
                ids = torch.tensor([1], device=env.device)
                if hasattr(env, "_launch_ball"):
                    env._launch_ball(ids)
                    torch.testing.assert_close(env._get_observations()["policy"][:, -extra_dim:],
                                               expected[:, 1:].flatten(1), rtol=0, atol=0)
                env._reset_idx(ids)
                expected[ids] = 0
                current = env._get_observations()
                torch.testing.assert_close(env.actions, expected[:, 0], rtol=0, atol=0)
                torch.testing.assert_close(current["policy"][:, -extra_dim:], expected[:, 1:].flatten(1), rtol=0, atol=0)
        print(f"ACTION_HISTORY_PASS {args.task} {cfg.control_mode}: policy={cfg.observation_space}, critic={cfg.state_space}", flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
