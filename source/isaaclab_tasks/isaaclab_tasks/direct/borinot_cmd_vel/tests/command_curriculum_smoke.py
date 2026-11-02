# SPDX-License-Identifier: BSD-3-Clause
"""Live command/force stage transitions using controlled completed-episode scores."""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg


def main():
    cfg = BorinotCmdVelEnvCfg()
    cfg.scene.num_envs, cfg.sim.device = 4, args.device
    cfg.seed = 42
    cfg.include_events_randomization, cfg.standing_env_prob = False, 0.
    cfg.command_lin_vel_lim_curriculum = [1., 3., 5.]
    cfg.forces_applied_to_base_curriculum = (0., 1., 3., 6., 9.)
    cfg.curriculum_min_episodes, cfg.curriculum_window_episodes, cfg.curriculum_min_steps = 1, 4, 2
    env = gym.make("Borinot-cmd-vel", cfg=cfg).unwrapped
    report = {"checks": [], "transitions": []}

    def check(name, passed):
        report["checks"].append({"name": name, "passed": bool(passed)})
        assert passed, name

    try:
        env.reset()
        device = env.device
        ids = torch.tensor([0], device=device)
        all_ids = torch.arange(4, device=device)
        check("combined_stage_count", env.get_curriculum_max_global_idx() == 6)
        check("initial_command_and_force_limits", env._command_ranges[0, :2].tolist() == [[-1., 1.]] * 2
              and env._force_limits.tolist() == [0., 0., 0., 1., 3., 6., 9.])
        for selected in range(1, 7):
            # Exercise real physics/observation path before injecting a known finish.
            obs, reward, _, _, _ = env.step(torch.zeros(4, cfg.action_space, device=device))
            check(f"stage_{selected}_finite_physics", torch.isfinite(reward).all()
                  and all(torch.isfinite(value).all() for value in obs.values()))
            env.common_step_counter = env._stage_start_step + cfg.curriculum_min_steps
            env._episode_steps[0] = env.max_episode_length
            env._episode_xy_integral[0] = cfg.episode_length_s
            env._reset_idx(ids)
            check(f"stage_{selected}_reset_and_clear", env.force_level == selected
                  and env._episode_stage[0] == selected and not env._stage_episodes
                  and not env._curriculum_window.any())
            check(f"stage_{selected}_old_episodes_keep_stage", (env._episode_stage[1:] == 0).all())
            # Same uniform draws, but each episode uses its own stage's range.
            torch.manual_seed(100 + selected)
            samples = torch.rand(4, 4, device=device)
            ranges = env._command_ranges[env._episode_stage]
            expected = ranges[:, :, 0] + samples * (ranges[:, :, 1] - ranges[:, :, 0])
            torch.manual_seed(100 + selected)
            env._resample_commands(all_ids)
            check(f"stage_{selected}_stage_local_sampling", torch.equal(env._commands, expected))
            env._base_push_steps_until_next.zero_()
            env._update_pushes()
            force_limit = [0., 0., 0., 1., 3., 6., 9.][selected]
            check(f"stage_{selected}_force_order", env._base_push_force_z_w[0].abs() <= force_limit
                  and (force_limit == 0 or env._base_push_force_z_w[0].abs() > 0)
                  and not env._base_push_force_z_w[1:].any())
            report["transitions"].append({"evaluated": selected - 1, "selected": selected,
                "xy_limit_m_s": float(env._command_ranges[selected, 0, 1]), "force_limit_n": force_limit})
            env._episode_steps[1] = env.max_episode_length
            env._episode_xy_integral[1] = cfg.episode_length_s
            env._update_curriculum(torch.tensor([1], device=device))
            check(f"stage_{selected}_old_success_excluded", env.force_level == selected
                  and env._stage_episodes == 0)
            # The injected finish is for this check only, not the next physics timeout.
            env._episode_steps[1], env._episode_xy_integral[1] = 1, 0.
        env._episode_steps[0], env._episode_xy_integral[0] = env.max_episode_length, cfg.episode_length_s
        env._reset_idx(ids)
        check("final_stage_stable", env.force_level == 6)
        cfg.curriculum, cfg.skip_curriculum = False, True
        env._init_command_curriculum()
        check("skip_selects_final_command_and_force", env.force_level == 6
              and (env._episode_stage == 6).all() and env._force_limits[6] == 9)
        cfg.skip_curriculum, cfg.curriculum_start_level = False, 4
        env._init_command_curriculum()
        check("fixed_combined_start", env.force_level == 4 and env._force_limits[4] == 3)
        report["passed"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASSED {len(report['checks'])} command curriculum checks", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    app.close()
