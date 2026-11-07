"""Checkpoint naming/selection regressions without starting Isaac Sim."""

import ast
import os
import re
import shlex
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

TRAIN_PATH = Path(__file__).with_name("train.py")
PARSE_CFG_PATH = TRAIN_PATH.parents[2] / "isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py"


def _load_helpers():
    namespace = {
        "os": os,
        "re": re,
        "Path": Path,
        "statistics": statistics,
        "shlex": shlex,
        "RslRlBaseRunnerCfg": object,
    }
    tree = ast.parse(TRAIN_PATH.read_text())
    names = {
        "_checkpoint_path",
        "_patch_runner_save_with_checkpoint_names",
        "_patch_runner_save_with_checkpoint_retention",
        "_save_best_model_checkpoints",
        "_curriculum_checkpoint_stage",
        "_install_sac_best_model_hook",
        "_infer_input_checkpoint_name",
        "_resolve_resume_path",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    nodes += [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_PERIODIC_CHECKPOINT_PATTERN" for target in node.targets)
    ]
    launcher = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PeriodicEvalVideoLauncher"
    )
    nodes += [node for node in launcher.body if isinstance(node, ast.FunctionDef) and node.name == "_launch"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    tree = ast.parse(PARSE_CFG_PATH.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_checkpoint_path"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARSE_CFG_PATH), "exec"), namespace)
    return namespace


class TestCheckpointNames(unittest.TestCase):
    def setUp(self):
        self.helpers = _load_helpers()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.saved = []

        def save(path, infos=None):
            Path(path).write_text("checkpoint")
            self.saved.append((Path(path), infos))

        self.runner = SimpleNamespace(save=save, tot_timesteps=1200, tot_time=3.0)

    def install_names(self, prefix="1107_1ec752ct"):
        self.helpers["_patch_runner_save_with_checkpoint_names"](self.runner, prefix)

    def test_periodic_and_best_saves_use_same_prefix_without_double_prefixing(self):
        self.install_names()
        infos = {"curriculum_global_idx": 9}
        for name in (
            "model_10.pt",
            "best_model.pt",
            "best_model_curriculum_idx_9.pt",
            "1107_1ec752ct_best_model_curriculum_idx_9.pt",
        ):
            self.runner.save(str(self.directory / name), infos=infos)
        self.assertEqual(
            [path.name for path, _ in self.saved],
            [
                "1107_1ec752ct_model_10.pt",
                "1107_1ec752ct_best_model.pt",
                "1107_1ec752ct_best_model_curriculum_idx_9.pt",
                "1107_1ec752ct_best_model_curriculum_idx_9.pt",
            ],
        )
        self.assertTrue(all(saved_infos is infos for _, saved_infos in self.saved))
        self.assertTrue(all(path.exists() for path, _ in self.saved))

    def test_non_wandb_run_preserves_legacy_names(self):
        self.install_names(prefix="")
        self.runner.save(self.directory / "model_10.pt")
        self.assertEqual(self.saved[0][0].name, "model_10.pt")

    def test_best_save_returns_real_path_and_keeps_stage_metadata(self):
        self.install_names()
        state = dict(
            global_idx=9,
            two_feet_phase=2,
            max_velx_range_idx=4,
            base_push_force_idx=3,
            command_lin_vel_x_abs=1.0,
            base_push_force_xy_abs=2.0,
        )
        _, name = self.helpers["_curriculum_checkpoint_stage"](state)
        result = self.helpers["_save_best_model_checkpoints"](
            self.runner, str(self.directory), state, name, 42.0, 10, "/parent/model_1.pt"
        )
        self.assertEqual(Path(result).name, "1107_1ec752ct_best_model_curriculum_idx_9_two_feet_phase_2.pt")
        self.assertTrue(Path(result).is_file())
        self.assertEqual(self.saved[-1][0].name, "1107_1ec752ct_best_model.pt")
        self.assertEqual(self.saved[0][1]["best_model_curriculum_idx"], 9)
        self.assertEqual(self.saved[0][1]["best_model_two_feet_phase"], 2)
        self.assertEqual(self.saved[0][1]["source_checkpoint"], "/parent/model_1.pt")

    def test_retention_sorts_iterations_and_never_prunes_best_or_eval_files(self):
        self.helpers["_patch_runner_save_with_checkpoint_retention"](self.runner, 2)
        self.install_names()
        (self.directory / "model_1.pt").touch()  # Legacy files in a mixed directory.
        for name in (
            "best_model.pt",
            "best_model_curriculum_idx_9.pt",
            "periodic_eval_it000001_t000001s.pt",
            "model_9.pt",
            "model_10.pt",
            "model_100.pt",
        ):
            self.runner.save(str(self.directory / name))
        self.assertEqual(
            sorted(path.name for path in self.directory.iterdir()),
            [
                "1107_1ec752ct_best_model.pt",
                "1107_1ec752ct_best_model_curriculum_idx_9.pt",
                "1107_1ec752ct_model_10.pt",
                "1107_1ec752ct_model_100.pt",
                "1107_1ec752ct_periodic_eval_it000001_t000001s.pt",
            ],
        )

    def test_loader_accepts_legacy_selectors_and_sorts_without_prefix(self):
        run = self.directory / "run"
        run.mkdir()
        for name in (
            "model_9.pt",
            "1107_1ec752ct_model_10.pt",
            "1107_1ec752ct_model_100.pt",
            "1107_1ec752ct_best_model_curriculum_idx_9.pt",
        ):
            (run / name).touch()
        get_path = self.helpers["get_checkpoint_path"]
        for selector in (r"model_.*.pt", r"^model_\d+\.pt$"):
            self.assertEqual(
                Path(get_path(str(self.directory), checkpoint=selector)).name, "1107_1ec752ct_model_100.pt"
            )
        self.assertEqual(
            Path(get_path(str(self.directory), checkpoint=r"^best_model_curriculum_idx_9\.pt$")).name,
            "1107_1ec752ct_best_model_curriculum_idx_9.pt",
        )
        self.assertEqual(
            Path(get_path(str(self.directory), checkpoint=r"1107_1ec752ct_model_10\.pt$")).name,
            "1107_1ec752ct_model_10.pt",
        )

    def test_loader_still_resolves_old_runs_and_nested_checkpoint_dirs(self):
        run = self.directory / "run" / "checkpoints"
        run.mkdir(parents=True)
        for name in ("model_9.pt", "model_10.pt"):
            (run / name).touch()
        result = self.helpers["get_checkpoint_path"](
            str(self.directory), checkpoint="model_.*.pt", other_dirs=["checkpoints"]
        )
        self.assertEqual(Path(result), run / "model_10.pt")

    def test_resume_accepts_automatic_selector_and_explicit_new_path(self):
        run = self.directory / "run"
        run.mkdir()
        path = run / "1107_1ec752ct_model_10.pt"
        path.touch()
        cfg = SimpleNamespace(load_run=".*", load_checkpoint="model_.*.pt")
        resolve = self.helpers["_resolve_resume_path"]
        self.assertEqual(resolve(str(self.directory), cfg, None), str(path))
        self.assertEqual(resolve(str(self.directory), cfg, str(path)), str(path))

    def test_input_checkpoint_label_does_not_repeat_run_id(self):
        infer = self.helpers["_infer_input_checkpoint_name"]
        run = "2026-11-07_11-39-58_name_1ec752ct"
        self.assertEqual(
            infer(f"/{run}/1107_1ec752ct_best_model_curriculum_idx_9.pt"), "1107_1ec752ct_best_model_curriculum_idx_9"
        )
        self.assertEqual(infer(f"/{run}/best_model.pt"), "1ec752ct_best_model")

    def test_sac_best_hook_saves_prefixed_stage_and_generic_best(self):
        self.install_names()
        self.runner.logger = SimpleNamespace(
            log=Mock(), log_dir=str(self.directory), writer=object(), rewbuffer=[1.0, 3.0]
        )
        state = dict(
            global_idx=9,
            two_feet_phase=None,
            max_velx_range_idx=None,
            base_push_force_idx=None,
            command_lin_vel_x_abs=None,
            base_push_force_xy_abs=None,
        )
        self.helpers["_get_curriculum_state_from_runner"] = lambda runner: state
        self.helpers["_install_sac_best_model_hook"](self.runner, str(self.directory), None, float("-inf"))
        self.runner.logger.log(it=10)
        self.assertEqual([path.name for path, _ in self.saved], ["1107_1ec752ct_best_model_curriculum_idx_9.pt"])
        state = None
        self.runner.logger.log(it=11)
        self.assertEqual(self.saved[-1][0].name, "1107_1ec752ct_best_model.pt")

    def test_periodic_eval_child_receives_the_actual_saved_filename(self):
        self.install_names()
        build_command = Mock(return_value=["python", "play.py"])
        launcher = SimpleNamespace(
            runner=self.runner,
            checkpoint_dir=self.directory,
            video_dir=self.directory,
            child_log_dir=self.directory,
            max_steps=10,
            episodes=1,
            wandb_run_id="1ec752ct",
            children=[],
            _episode_length_s=lambda: 1.0,
            _build_command=build_command,
        )
        self.helpers["_THIS_DIR"] = TRAIN_PATH.parent
        self.helpers["subprocess"] = SimpleNamespace(Popen=Mock(), STDOUT=-2)
        self.helpers["_launch"](launcher, {"it": 10})
        path = build_command.call_args.kwargs["checkpoint_path"]
        self.assertEqual(path.name, "1107_1ec752ct_periodic_eval_it000010_t000003s.pt")
        self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
