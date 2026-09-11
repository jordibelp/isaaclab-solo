"""CPU-only regression checks for the race curriculum and its per-rollout runner hook."""

import ast
import math
import random
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[2]
ENV_PATH = SOURCE / "isaaclab_tasks/isaaclab_tasks/direct/solo12_race/solo12_race_env.py"
CFG_PATH = ENV_PATH.with_name("solo12_race_env_cfg.py")
TRAIN_PATH = Path(__file__).with_name("train.py")
SUCCESS = "Curriculum/backward_force_eligible_success_count"
COMPLETED = "Curriculum/backward_force_eligible_completed_count"


def _load_curriculum():
    # Execute the actual methods without importing/launching Isaac Sim, as in the other script tests.
    tree = ast.parse(ENV_PATH.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12RaceEnv")
    names = {
        "_configure_backward_force_curriculum", "_backward_force_episode_statistics",
        "update_backward_force_curriculum", "current_backward_force",
    }
    env_class.bases = []
    env_class.body = [node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    train_tree = ast.parse(TRAIN_PATH.read_text())
    functions = [
        node for node in train_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_aggregate_race_episode_statistics", "_update_backward_force_curriculum"}
    ]
    namespace = {"math": math, "torch": torch, "deque": deque}
    exec(compile(ast.Module(body=[env_class, *functions], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    return namespace


class TestBackwardForceCurriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        namespace = _load_curriculum()
        cls.env_type = namespace["Solo12RaceEnv"]
        cls.update = staticmethod(namespace["_update_backward_force_curriculum"])
        cls.aggregate = staticmethod(namespace["_aggregate_race_episode_statistics"])

    def make_env(self, minimum=1, window=1, episodes=1, stages=(14.0, 15.0, 16.0), threshold=0.7):
        env = self.env_type()
        env.cfg = SimpleNamespace(
            backward_force=13.0,
            backward_force_curriculum=stages,
            backward_force_curriculum_sr_threshold=threshold,
            min_iterations_with_curriculum_stage=minimum,
            backward_force_curriculum_window_iterations=window,
            backward_force_curriculum_min_episodes=episodes,
            race_scene="straightSimple",
        )
        env._configure_backward_force_curriculum()
        env._backward_force_episode_stage = torch.zeros(4, dtype=torch.long)
        env.episode_length_buf = torch.ones(4, dtype=torch.long)
        env.reset_terminated = torch.zeros(4, dtype=torch.bool)
        env.reset_time_outs = torch.zeros(4, dtype=torch.bool)
        return env

    def tick(self, env, successes=0, completed=0):
        runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env), current_learning_iteration=6300)
        self.update(runner, [{SUCCESS: successes, COMPLETED: completed}] if completed else [])

    def metric(self, env, suffix):
        return env._backward_force_curriculum_metrics[f"Curriculum/backward_force_{suffix}"]

    def test_dwell_counts_complete_rollouts_and_resets_at_every_stage(self):
        env = self.make_env(minimum=5)
        for old_force, new_force in [(13.0, 14.0), (14.0, 15.0), (15.0, 16.0)]:
            for _ in range(4):
                self.tick(env, 9, 10)
                self.assertEqual(env.current_backward_force, old_force)
            self.tick(env, 9, 10)
            self.assertEqual(env.current_backward_force, new_force)
            self.assertEqual(env._backward_force_curriculum_stage_iterations, 0)
            self.assertEqual(len(env._backward_force_curriculum_history), 0)
            self.assertEqual(self.metric(env, "rollout_stage_iterations"), 5)
            self.assertEqual(self.metric(env, "promoted"), 1)

    def test_dwell_does_not_require_consecutive_successes(self):
        env = self.make_env(minimum=5)
        for success in [90, 20, 90, 20, 69]:
            self.tick(env, success, 100)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 71, 100)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_empty_rollouts_are_no_data_but_count_dwell(self):
        env = self.make_env(minimum=5)
        for _ in range(5):
            self.tick(env)
        self.assertEqual(env._backward_force_curriculum_stage_iterations, 5)
        self.assertEqual(self.metric(env, "window_episodes"), 0)
        self.assertNotIn("Curriculum/backward_force_success_rate", env._backward_force_curriculum_metrics)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 9, 10)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_all_failures_are_zero_not_missing_and_threshold_is_strict(self):
        env = self.make_env()
        self.tick(env, 0, 10)
        self.assertEqual(self.metric(env, "success_rate"), 0.0)
        self.tick(env, 70, 100)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 71, 100)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_unequal_reset_batches_are_weighted_by_episode_count(self):
        env = self.make_env()
        infos = [{"Episode/successRate": s / n, SUCCESS: s, COMPLETED: n} for s, n in [(1, 1), (60, 100)]]
        self.update(SimpleNamespace(env=SimpleNamespace(unwrapped=env)), infos)
        self.assertAlmostEqual(self.metric(env, "success_rate"), 61 / 101)
        self.assertEqual(env.current_backward_force, 13.0)

    def test_unequal_rollouts_are_pooled_not_averaged(self):
        env = self.make_env(minimum=2, window=10, episodes=500)
        self.tick(env, 1, 1)
        self.tick(env, 300, 500)
        self.assertAlmostEqual(self.metric(env, "success_rate"), 301 / 501)
        self.assertEqual(env.current_backward_force, 13.0)

    def test_sample_floor_extends_then_shrinks_to_recent_window(self):
        env = self.make_env(window=10, episodes=500, threshold=1.0)
        for _ in range(24):
            self.tick(env, 20, 20)
        self.assertEqual(self.metric(env, "window_episodes"), 480)
        self.assertEqual(self.metric(env, "window_iterations"), 24)
        self.tick(env, 20, 20)
        self.assertEqual(self.metric(env, "window_episodes"), 500)
        self.assertEqual(self.metric(env, "window_iterations"), 25)
        self.tick(env, 20, 20)
        self.assertEqual(self.metric(env, "window_episodes"), 500)
        self.assertEqual(self.metric(env, "window_iterations"), 25)
        for _ in range(10):
            self.tick(env, 500, 500)
        self.assertEqual(self.metric(env, "window_episodes"), 5000)
        self.assertEqual(self.metric(env, "window_iterations"), 10)

    def test_insufficient_samples_wait_and_new_stage_cannot_reuse_old_evidence(self):
        env = self.make_env(window=10, episodes=500)
        for _ in range(4):
            self.tick(env, 100, 100)
            self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 100, 100)
        self.assertEqual(env.current_backward_force, 14.0)
        self.tick(env, 1, 1)
        self.assertEqual(env.current_backward_force, 14.0)
        self.assertEqual(self.metric(env, "window_episodes"), 1)

    def test_stored_success_cannot_promote_when_dwell_expires_without_new_endings(self):
        env = self.make_env(minimum=5, window=10, episodes=500)
        self.tick(env, 500, 500)
        for _ in range(20):
            self.tick(env)
            self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 1, 1)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_window_matches_full_history_reference_with_sparse_random_counts(self):
        env = self.make_env(window=10, episodes=500, threshold=1.0)
        rng = random.Random(23)
        history = []
        for iteration in range(1, 1001):
            count = rng.choice([0, 0, 1, 20, 100, 600])
            successes = rng.randrange(count + 1)
            history.append((successes, count))
            self.tick(env, successes, count)
            start = max(0, iteration - 10)
            while start > 0 and sum(n for _, n in history[start:]) < 500:
                start -= 1
            expected_success = sum(s for s, _ in history[start:])
            expected_total = sum(n for _, n in history[start:])
            self.assertEqual(self.metric(env, "window_successes"), expected_success)
            self.assertEqual(self.metric(env, "window_episodes"), expected_total)
            self.assertLessEqual(len(env._backward_force_curriculum_history), 500 + 10)

    def test_expiring_old_failures_cannot_promote_without_new_endings(self):
        env = self.make_env(window=3, episodes=500)
        self.tick(env, 0, 1000)
        self.tick(env, 500, 500)
        self.tick(env)
        self.tick(env)  # The 1,000 failures leave the recent window; the sample floor still holds.
        self.assertEqual(self.metric(env, "success_rate"), 1.0)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 1, 1)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_reset_statistics_exclude_mixed_episodes_but_not_their_ppo_metrics(self):
        env = self.make_env()
        env._backward_force_curriculum_stage = 1
        env._backward_force_episode_stage[:] = torch.tensor([0, 1, 1, 1])
        env.reset_terminated[:] = torch.tensor([True, True, False, False])
        env.reset_time_outs[:] = torch.tensor([True, False, True, False])  # First ending must not count twice.
        stats = env._backward_force_episode_statistics(torch.arange(4), torch.tensor([True, True, False, False]))
        self.assertEqual(stats, {"Episode/success_count": 2, "Episode/completed_count": 3, SUCCESS: 1, COMPLETED: 2})
        self.assertTrue(torch.all(env._backward_force_episode_stage == 1))
        env.reset_terminated[:] = True
        env.reset_time_outs[:] = False
        stats = env._backward_force_episode_statistics(torch.arange(4), torch.ones(4, dtype=torch.bool))
        self.assertEqual(stats[COMPLETED], 4)

    def test_zero_length_reset_is_not_an_episode_even_with_stale_done_flags(self):
        env = self.make_env()
        env.episode_length_buf[:] = 0
        env.reset_terminated[:] = True
        stats = env._backward_force_episode_statistics(torch.arange(4), torch.ones(4, dtype=torch.bool))
        self.assertTrue(all(value == 0 for value in stats.values()))

    def test_promotion_retags_only_episodes_with_no_steps_at_previous_force(self):
        env = self.make_env()
        env.episode_length_buf[:] = torch.tensor([0, 4, 0, 10])
        self.tick(env, 10, 10)
        self.assertEqual(env._backward_force_episode_stage.tolist(), [1, 0, 1, 0])
        env.episode_length_buf += 1
        env.reset_terminated[:] = True
        stats = env._backward_force_episode_statistics(torch.arange(4), torch.ones(4, dtype=torch.bool))
        self.assertEqual(stats[COMPLETED], 2)
        self.assertEqual(stats["Episode/completed_count"], 4)

    def test_console_and_wandb_success_rate_and_counts_are_pooled_once(self):
        infos = [{"RewardsPerStep/total": 2.0}]
        for successes, count in [(1, 1), (60, 100)]:
            infos.append({
                "Episode/success_count": torch.tensor(successes), "Episode/completed_count": count,
                "Episode/successRate": successes / count, "Episode/finishRatio": successes / count,
                SUCCESS: successes, COMPLETED: count, "Unrelated": 3,
            })
        self.aggregate(infos)
        self.assertAlmostEqual(infos[0]["Episode/successRate"], 61 / 101)
        self.assertEqual(infos[0]["Episode/finishRatio"], infos[0]["Episode/successRate"])
        self.assertEqual(infos[0][COMPLETED], 101)
        self.assertEqual(sum("Episode/successRate" in info for info in infos), 1)
        self.assertEqual(sum(COMPLETED in info for info in infos), 1)
        self.assertEqual(infos[2]["Unrelated"], 3)
        env = self.make_env()
        self.update(SimpleNamespace(env=SimpleNamespace(unwrapped=env)), infos)
        self.assertEqual(env.current_backward_force, 13.0)

    def test_zero_counts_log_no_fabricated_success_rate(self):
        infos = [{"Episode/completed_count": 0, "Episode/success_count": 0, "Episode/successRate": 0.0}]
        self.aggregate(infos)
        self.assertNotIn("Episode/successRate", infos[0])
        self.assertEqual(infos[0][COMPLETED], 0)

    def test_legacy_ratios_without_stage_pure_counts_cannot_promote(self):
        env = self.make_env()
        self.update(SimpleNamespace(env=SimpleNamespace(unwrapped=env)), [{"Episode/successRate": 1.0}])
        self.assertEqual(env.current_backward_force, 13.0)

    def test_disabled_and_final_curriculum_keep_force(self):
        env = self.make_env(stages=())
        self.tick(env, 1, 1)
        self.assertEqual(env.current_backward_force, 13.0)
        env = self.make_env(stages=(14.0,))
        self.tick(env, 1, 1)
        self.tick(env, 1, 1)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_invalid_positive_integer_settings_rejected(self):
        for name in ["minimum", "window", "episodes"]:
            for value in [0, -1, 1.5, True, "5", None]:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                    self.make_env(**{name: value})

    def test_invalid_counts_rejected_without_counting_rollout(self):
        env = self.make_env()
        for counts in [(1, 0), (-1, 2), (0, -1), (1.5, 2), (True, 2), (0, float("nan"))]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                env.update_backward_force_curriculum(*counts)
        self.assertEqual(env._backward_force_curriculum_stage_iterations, 0)

    def test_non_race_runner_and_its_metrics_unchanged(self):
        self.update(SimpleNamespace(env=SimpleNamespace()), [])
        self.update(SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace())), [])
        infos = [{"Episode/successRate": 0.25}]
        self.aggregate(infos)
        self.assertEqual(infos, [{"Episode/successRate": 0.25}])

    def test_production_defaults_for_large_parallel_training(self):
        tree = ast.parse(CFG_PATH.read_text())
        cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12RaceEnvCfg")
        expected = {
            "min_iterations_with_curriculum_stage": 64,
            "backward_force_curriculum_window_iterations": 10,
            "backward_force_curriculum_min_episodes": 500,
            "backward_force_curriculum_sr_threshold": 0.7,
        }
        actual = {
            node.target.id: ast.literal_eval(node.value) for node in cfg.body
            if isinstance(node, ast.AnnAssign) and node.target.id in expected
        }
        self.assertEqual(actual, expected)
        env = self.make_env(minimum=actual["min_iterations_with_curriculum_stage"], window=10, episodes=500)
        for _ in range(63):
            self.tick(env, 400, 500)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 400, 500)
        self.assertEqual(env.current_backward_force, 14.0)
        self.assertEqual(self.metric(env, "window_episodes"), 5000)


if __name__ == "__main__":
    unittest.main()
