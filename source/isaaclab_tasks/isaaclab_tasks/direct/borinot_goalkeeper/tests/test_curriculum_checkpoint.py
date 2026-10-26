# SPDX-License-Identifier: BSD-3-Clause
"""Run the real curriculum update on controlled tensors, without Isaac Sim.

train.py keeps one best checkpoint per shot level. It only scores a level once the
newest completed episodes, at least as many as its reward buffer holds, all started at
that level. These tests pin that count.
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

folder = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("goalkeeper_curriculum_math", folder / "task_math.py")
task_math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_math)
tree = ast.parse((folder / "borinot_goalkeeper_env.py").read_text())
methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
           and n.name in ("_update_curriculum", "_record_shots", "get_curriculum_global_idx")]
namespace = {"torch": torch, "promotion_ready": task_math.promotion_ready}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(folder), "exec"), namespace)
update_curriculum = namespace["_update_curriculum"]
record_shots = namespace["_record_shots"]
get_curriculum_global_idx = namespace["get_curriculum_global_idx"]


def make_env(n=8, promote=False):
    cfg = SimpleNamespace(
        shot_half_extents_yz=((0.045, 0.045), (0.20, 0.12), (0.40, 0.22), (0.65, 0.32)),
        curriculum=promote, curriculum_min_steps=1, curriculum_min_shots=1, curriculum_success_threshold=0.0,
    )
    return SimpleNamespace(
        cfg=cfg, device="cpu", extras={"power_mean_w": {}}, _desired_actuation=None, common_step_counter=10,
        shot_level=0, _stage_start_step=0, _stage_shots=0, _window_cursor=0,step_dt=.01,
        _curriculum_window=torch.zeros(16), outcome_counts=torch.zeros(10, dtype=torch.long),
        _shot_steps=torch.ones(n, dtype=torch.long), _outcome=torch.full((n,), 7),
        _episode_steps=torch.ones(n,dtype=torch.long), _episode_shots=torch.zeros(n,dtype=torch.long),
        _episode_saves=torch.zeros(n,dtype=torch.long), _episode_paddle_hits=torch.zeros(n,dtype=torch.long),
        _shot_stage=torch.zeros(n,dtype=torch.long),
        _episode_stage=torch.zeros(n, dtype=torch.long), _episode_rewards=torch.zeros(n),
        _hit=torch.ones(n, dtype=torch.bool), _clear=torch.ones(n, dtype=torch.bool),
        _episode_terms={"outcome": torch.zeros(n)}, curriculum_stage_pure_episodes=0,
        _actuator_delay=SimpleNamespace(seconds=torch.zeros(n)),
    )


def test_current_level_endings_accumulate_across_batches():
    env = make_env()
    update_curriculum(env, torch.arange(3))
    update_curriculum(env, torch.arange(5))
    assert env.curriculum_stage_pure_episodes == 8
    assert get_curriculum_global_idx(env) == 0


def test_promotion_restarts_count_and_reports_new_level():
    env = make_env(promote=True)
    record_shots(env, torch.arange(4))
    assert env.shot_level == 1
    assert get_curriculum_global_idx(env) == 1
    assert env.curriculum_stage_pure_episodes == 0
    assert env._episode_shots.tolist() == [1,1,1,1,0,0,0,0]


def test_only_newest_endings_after_last_older_level_episode_count():
    env = make_env()
    env.shot_level = 1
    env.curriculum_stage_pure_episodes = 50
    # Completion order follows ascending env ids, as in RSL-RL's reward buffer.
    env._episode_stage[:6] = torch.tensor([1, 1, 0, 1, 1, 1])
    update_curriculum(env, torch.arange(6))
    assert env.curriculum_stage_pure_episodes == 3
    env._episode_stage[:] = 1
    update_curriculum(env, torch.arange(4))
    assert env.curriculum_stage_pure_episodes == 7


def test_older_level_ending_last_gives_zero():
    env = make_env()
    env.shot_level = 1
    env._episode_stage[:4] = torch.tensor([1, 1, 1, 0])
    update_curriculum(env, torch.arange(4))
    assert env.curriculum_stage_pure_episodes == 0


def test_resets_without_an_ending_do_not_change_count():
    env = make_env()
    env.curriculum_stage_pure_episodes = 5
    env._episode_steps[:] = 0
    update_curriculum(env, torch.arange(4))
    assert env.curriculum_stage_pure_episodes == 5


def test_old_stage_shots_do_not_fill_new_stage_window_or_change_episode_purity():
    env = make_env()
    env.shot_level = 1
    env.curriculum_stage_pure_episodes = 12
    record_shots(env, torch.arange(8))
    assert env._stage_shots == 0
    assert env.curriculum_stage_pure_episodes == 12
    assert env.outcome_counts[7] == 8


def test_shot_counts_cannot_unlock_episode_return_checkpoint_gate():
    env = make_env()
    for _ in range(25):
        record_shots(env, torch.arange(8))
    assert env._stage_shots == 200
    assert env.curriculum_stage_pure_episodes == 0
