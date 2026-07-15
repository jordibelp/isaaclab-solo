import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


TRAIN_PATH = Path(__file__).with_name("train.py")


def _load_functions(*names):
    tree = ast.parse(TRAIN_PATH.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


class TestCurriculumCheckpointStages(unittest.TestCase):
    def test_two_feet_phase_is_part_of_curriculum_state(self):
        get_state, = _load_functions("_get_curriculum_state_from_runner")
        cfg = SimpleNamespace(
            curriculum_two_feet=True,
            command_lin_vel_x_range=(-0.6, 0.6),
            base_push_force_xy_range=(0.0, 0.0),
        )
        env = SimpleNamespace(
            cfg=cfg,
            _two_feet_curriculum_phase=2,
            _max_velx_range_curriculum_idx=0,
            _base_push_force_curriculum_idx=0,
            get_curriculum_global_idx=lambda: 0,
        )
        runner = SimpleNamespace(env=SimpleNamespace(unwrapped=env))

        state = get_state(runner)

        self.assertEqual(state["global_idx"], 0)
        self.assertEqual(state["two_feet_phase"], 2)

    def test_each_two_feet_phase_has_a_distinct_checkpoint(self):
        checkpoint_stage, = _load_functions("_curriculum_checkpoint_stage")

        stage_1, filename_1 = checkpoint_stage({"global_idx": 0, "two_feet_phase": 1})
        stage_2, filename_2 = checkpoint_stage({"global_idx": 0, "two_feet_phase": 2})

        self.assertNotEqual(stage_1, stage_2)
        self.assertEqual(filename_1, "best_model_curriculum_idx_0_two_feet_phase_1.pt")
        self.assertEqual(filename_2, "best_model_curriculum_idx_0_two_feet_phase_2.pt")

    def test_legacy_curriculum_filename_is_preserved(self):
        checkpoint_stage, = _load_functions("_curriculum_checkpoint_stage")

        stage, filename = checkpoint_stage({"global_idx": 3, "two_feet_phase": None})

        self.assertEqual(stage, (3, None))
        self.assertEqual(filename, "best_model_curriculum_idx_3.pt")


if __name__ == "__main__":
    unittest.main()
