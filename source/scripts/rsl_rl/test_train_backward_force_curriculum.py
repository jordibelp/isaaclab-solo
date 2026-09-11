"""CPU-only regression checks for the race curriculum and its per-rollout runner hook."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path(__file__).resolve().parents[2]
ENV_PATH = SOURCE / "isaaclab_tasks/isaaclab_tasks/direct/solo12_race/solo12_race_env.py"
TRAIN_PATH = Path(__file__).with_name("train.py")


def _load_curriculum():
    # The environment and train.py launch/import Isaac Sim; execute only the real
    # curriculum methods, following the other training-script unit tests.
    tree = ast.parse(ENV_PATH.read_text())
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Solo12RaceEnv"
    )
    names = {"_configure_backward_force_curriculum", "update_backward_force_curriculum", "current_backward_force"}
    env_class.bases = []
    env_class.body = [node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in names]
    train_tree = ast.parse(TRAIN_PATH.read_text())
    functions = [
        node
        for node in train_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_mean_episode_info", "_update_backward_force_curriculum"}
    ]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=[env_class, *functions], type_ignores=[]), str(ENV_PATH), "exec"), namespace)
    return namespace["Solo12RaceEnv"], namespace["_update_backward_force_curriculum"]


class TestBackwardForceCurriculum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env_type, update = _load_curriculum()
        cls.update = staticmethod(update)

    def make_env(self, minimum=1, stages=(14.0, 15.0, 16.0)):
        env = self.env_type()
        env.cfg = SimpleNamespace(
            backward_force=13.0,
            backward_force_curriculum=stages,
            backward_force_curriculum_sr_threshold=0.7,
            min_iterations_with_curriculum_stage=minimum,
            race_scene="straightSimple",
        )
        env._configure_backward_force_curriculum()
        return env

    def tick(self, env, success_rate):
        runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env), current_learning_iteration=6300)
        ep_infos = [] if success_rate is None else [{"Episode/successRate": success_rate}]
        self.update(runner, ep_infos)

    def test_five_complete_rollouts_required_at_initial_and_each_subsequent_stage(self):
        env = self.make_env(minimum=5)
        for old_force, new_force in [(13.0, 14.0), (14.0, 15.0), (15.0, 16.0)]:
            for _ in range(4):
                self.tick(env, 0.9)
                self.assertEqual(env.current_backward_force, old_force)
            self.tick(env, 0.9)
            self.assertEqual(env.current_backward_force, new_force)
            self.assertEqual(env._backward_force_curriculum_stage_iterations, 0)

    def test_elapsed_rollouts_are_not_consecutive_successes_or_stale_success(self):
        env = self.make_env(minimum=5)
        for success in [0.9, 0.2, 0.9, 0.2, 0.69]:
            self.tick(env, success)
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 0.71)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_no_episode_rollouts_count_but_cannot_trigger_a_transition(self):
        env = self.make_env(minimum=5)
        for _ in range(5):
            self.tick(env, None)
        self.assertEqual(env.current_backward_force, 13.0)
        self.assertEqual(env._backward_force_curriculum_stage_iterations, 5)
        self.tick(env, 0.9)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_default_one_preserves_current_iteration_aggregate_and_strict_threshold(self):
        env = self.make_env()
        self.assertFalse(env.update_backward_force_curriculum(0.7))
        runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env))
        self.update(runner, [{"Episode/successRate": 0.9}, {"Episode/successRate": 0.1}])
        self.assertEqual(env.current_backward_force, 13.0)
        self.tick(env, 0.9)
        self.tick(env, 0.9)
        self.assertEqual(env.current_backward_force, 15.0)

    def test_disabled_and_final_curriculum_keep_force(self):
        env = self.make_env(stages=())
        self.tick(env, 1.0)
        self.assertEqual(env.current_backward_force, 13.0)
        env = self.make_env(stages=(14.0,))
        self.tick(env, 1.0)
        self.tick(env, 1.0)
        self.assertEqual(env.current_backward_force, 14.0)

    def test_invalid_minimum_rejected(self):
        for minimum in [0, -1, 1.5, True, "5", None]:
            with self.subTest(minimum=minimum), self.assertRaisesRegex(ValueError, "min_iterations_with_curriculum_stage"):
                self.make_env(minimum=minimum)

    def test_invalid_success_rejected_without_counting_an_iteration(self):
        env = self.make_env()
        for success in [float("nan"), float("inf"), -0.1, 1.1]:
            with self.subTest(success=success), self.assertRaises(ValueError):
                env.update_backward_force_curriculum(success)
        self.assertEqual(env._backward_force_curriculum_stage_iterations, 0)

    def test_non_race_runner_ignored(self):
        self.update(SimpleNamespace(env=SimpleNamespace()), [])
        self.update(SimpleNamespace(env=SimpleNamespace(unwrapped=SimpleNamespace())), [])


if __name__ == "__main__":
    unittest.main()
