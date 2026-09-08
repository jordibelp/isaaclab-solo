import ast
import copy
import math
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


TRAIN_PATH = Path(__file__).with_name("train_race_env_params_tcn_dagger.py")


def _load_definitions(*names, extra_namespace=None):
    """Load import-unsafe script definitions without launching Isaac Sim."""

    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    future_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {
        "copy": copy,
        "F": F,
        "nn": nn,
        "os": os,
        "torch": torch,
    }
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    exec(compile(ast.Module(body=future_imports + definitions, type_ignores=[]), str(TRAIN_PATH), "exec"), namespace)
    return [namespace[name] for name in names], namespace


def _argument_keywords(flag):
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        positional_strings = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
        if flag in positional_strings:
            values = {}
            for keyword in node.keywords:
                try:
                    values[keyword.arg] = ast.literal_eval(keyword.value)
                except ValueError:
                    values[keyword.arg] = keyword.value
            return values
    raise AssertionError(f"Could not find parser argument {flag}.")


class TestRaceDaggerActionSupervision(unittest.TestCase):
    def test_reproducible_command_round_trips_shell_sensitive_arguments(self):
        arguments = [
            "--run-name=JointState DAgger teacher",
            "--teacher-checkpoint=/tmp/teacher run/model.pt",
            "env.friction_static_range=[0.5, 1.5]",
        ]
        (_,), namespace = _load_definitions(
            "_maybe_init_wandb",
            extra_namespace={
                "Any": object,
                "_REPRODUCIBLE_COMMAND": shlex.join(
                    [
                        "./isaaclab.sh",
                        "-p",
                        "source/scripts/rsl_rl/train_race_env_params_tcn_dagger.py",
                        *arguments,
                    ]
                ),
                "_snapshot_wandb_run_files": lambda *_args: [],
                "args_cli": SimpleNamespace(
                    disable_wandb=False,
                    log_project_name=None,
                    wandb_entity=None,
                    wandb_name=None,
                ),
            },
        )
        maybe_init_wandb = namespace["_maybe_init_wandb"]
        captured = {}
        fake_run = SimpleNamespace(config=SimpleNamespace(update=lambda *_args, **_kwargs: None))
        fake_wandb = SimpleNamespace(
            init=lambda **kwargs: captured.update(kwargs) or fake_run,
            define_metric=lambda *_args, **_kwargs: None,
        )

        with mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            maybe_init_wandb("/tmp/log", "test-run", {"task": "test-task"})

        expected_tokens = [
            "./isaaclab.sh",
            "-p",
            "source/scripts/rsl_rl/train_race_env_params_tcn_dagger.py",
            *arguments,
        ]
        self.assertEqual(shlex.split(captured["config"]["command"]), expected_tokens)
        self.assertEqual(captured["config"]["task"], "test-task")

    def test_actor_finetuning_is_opt_in_and_action_weight_defaults_to_half(self):
        finetune_keywords = _argument_keywords("--finetune-student-actor")
        weight_keywords = _argument_keywords("--action-loss-weight")

        self.assertEqual(finetune_keywords["action"], "store_true")
        self.assertFalse(finetune_keywords["default"])
        self.assertEqual(weight_keywords["default"], 0.5)

    def test_action_loss_weight_must_be_finite_and_nonnegative(self):
        (validate_weight,), _ = _load_definitions(
            "_validate_action_loss_weight", extra_namespace={"math": math}
        )

        self.assertEqual(validate_weight(0.5), 0.5)
        self.assertEqual(validate_weight(0.0), 0.0)
        for invalid in (-0.1, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and >= 0"):
                validate_weight(invalid)

    def test_adapter_only_replay_buffer_keeps_original_two_tensor_interface(self):
        (replay_buffer,), _ = _load_definitions("ReplayBuffer")
        buffer = replay_buffer(capacity=4, history_dim=2, latent_dim=1)
        buffer.add(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]]))

        self.assertFalse(buffer.stores_action_targets)
        self.assertIsNone(buffer.current_obs)
        self.assertIsNone(buffer.target_action)
        self.assertEqual(len(buffer.sample(1, torch.device("cpu"))), 2)

    def test_action_supervision_replay_buffer_stores_all_four_fields(self):
        (replay_buffer,), _ = _load_definitions("ReplayBuffer")
        buffer = replay_buffer(capacity=3, history_dim=1, latent_dim=1, current_obs_dim=1, action_dim=1)
        buffer.add(
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[11.0], [12.0], [13.0]]),
            current_obs=torch.tensor([[21.0], [22.0], [23.0]]),
            target_action=torch.tensor([[31.0], [32.0], [33.0]]),
        )

        self.assertTrue(buffer.stores_action_targets)
        self.assertEqual(len(buffer.sample(2, torch.device("cpu"))), 4)
        self.assertEqual(sorted(buffer.current_obs.squeeze(-1).tolist()), [21.0, 22.0, 23.0])
        self.assertEqual(sorted(buffer.target_action.squeeze(-1).tolist()), [31.0, 32.0, 33.0])

    def test_joint_loss_has_requested_weight_and_updates_latent_and_actor_paths(self):
        (compute_losses,), _ = _load_definitions("_compute_dagger_losses")
        pred_z = torch.tensor([[1.0]], requires_grad=True)
        pred_action = torch.tensor([[2.0]], requires_grad=True)
        target_z = torch.tensor([[0.0]])
        target_action = torch.tensor([[0.0]])

        total, latent, action = compute_losses(
            pred_z,
            target_z,
            pred_action=pred_action,
            target_action=target_action,
            action_loss_weight=0.5,
        )
        total.backward()

        self.assertEqual(latent.item(), 1.0)
        self.assertEqual(action.item(), 4.0)
        self.assertEqual(total.item(), 3.0)
        self.assertEqual(pred_z.grad.item(), 2.0)
        self.assertEqual(pred_action.grad.item(), 2.0)

    def test_action_loss_updates_adapter_and_student_actor_but_not_teacher(self):
        (compute_losses,), _ = _load_definitions("_compute_dagger_losses")
        adapter = nn.Linear(2, 1, bias=False)
        student_actor = nn.Linear(2, 1, bias=False)
        teacher_actor = copy.deepcopy(student_actor).requires_grad_(False)
        teacher_before = copy.deepcopy(teacher_actor.state_dict())
        optimizer = torch.optim.AdamW(
            list(adapter.parameters()) + list(student_actor.parameters()),
            lr=1.0e-2,
            weight_decay=0.0,
        )

        history = torch.tensor([[1.0, -2.0]])
        current_obs = torch.tensor([[0.5]])
        pred_z = adapter(history)
        target_z = pred_z.detach().clone()
        pred_action = student_actor(torch.cat((current_obs, pred_z), dim=-1))
        target_action = pred_action.detach() + 1.0
        adapter_before = copy.deepcopy(adapter.state_dict())
        actor_before = copy.deepcopy(student_actor.state_dict())

        total, latent, action = compute_losses(
            pred_z,
            target_z,
            pred_action=pred_action,
            target_action=target_action,
            action_loss_weight=0.5,
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        self.assertEqual(latent.item(), 0.0)
        self.assertGreater(action.item(), 0.0)
        self.assertFalse(torch.equal(adapter.weight, adapter_before["weight"]))
        self.assertFalse(torch.equal(student_actor.weight, actor_before["weight"]))
        self.assertTrue(torch.equal(teacher_actor.weight, teacher_before["weight"]))

    def test_student_actor_is_an_independent_trainable_teacher_copy(self):
        (make_student_actor,), _ = _load_definitions("_make_student_actor")
        teacher = SimpleNamespace(actor=nn.Linear(2, 1, bias=False))
        teacher.actor.weight.data.copy_(torch.tensor([[2.0, 3.0]]))
        teacher.actor.requires_grad_(False)

        student_actor = make_student_actor(teacher)

        self.assertTrue(torch.equal(student_actor.weight, teacher.actor.weight))
        self.assertTrue(student_actor.weight.requires_grad)
        student_actor.weight.data.add_(1.0)
        self.assertFalse(torch.equal(student_actor.weight, teacher.actor.weight))
        self.assertFalse(teacher.actor.weight.requires_grad)

    def test_teacher_action_target_uses_teacher_latent_and_student_uses_predicted_latent(self):
        (actor_mean_and_std, build_targets), _ = _load_definitions(
            "_actor_mean_and_std", "_teacher_targets_and_trainable_student_action"
        )
        del actor_mean_and_std  # Loaded because the target builder calls it through its module globals.

        teacher = SimpleNamespace(
            actor_obs_normalizer=nn.Identity(),
            current_obs_dim=1,
            env_params_dim=1,
            actor_env_params_encoder=nn.Linear(1, 1, bias=False),
            actor=nn.Linear(2, 1, bias=False),
            state_dependent_std=False,
            noise_std_type="scalar",
            std=torch.tensor([1.0]),
        )
        teacher.actor_env_params_encoder.weight.data.fill_(2.0)
        teacher.actor.weight.data.copy_(torch.tensor([[3.0, 5.0]]))
        teacher.actor_env_params_encoder.requires_grad_(False)
        teacher.actor.requires_grad_(False)
        student_actor = nn.Linear(2, 1, bias=False)
        student_actor.weight.data.copy_(torch.tensor([[7.0, 11.0]]))

        z_teacher, teacher_action, rollout_action, student_action, current_obs = build_targets(
            teacher,
            student_actor,
            torch.tensor([[2.0, 4.0]]),
            torch.tensor([[6.0]]),
            stochastic_actions=False,
        )

        self.assertEqual(current_obs.item(), 2.0)
        self.assertEqual(z_teacher.item(), 8.0)
        self.assertEqual(teacher_action.item(), 46.0)  # 3 * current + 5 * z_teacher
        self.assertEqual(student_action.item(), 80.0)  # 7 * current + 11 * z_hat
        self.assertEqual(rollout_action.item(), 80.0)

    def test_checkpoint_round_trip_includes_actor_and_objective_metadata(self):
        args_cli = SimpleNamespace(learning_rate=3.0e-4, weight_decay=0.0, action_loss_weight=0.5)
        (save_checkpoint, load_checkpoint), _ = _load_definitions(
            "_save_checkpoint", "_load_adapter_checkpoint", extra_namespace={"args_cli": args_cli}
        )
        adapter = nn.Linear(2, 1)
        student_actor = nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(list(adapter.parameters()) + list(student_actor.parameters()), lr=3.0e-4)
        optimizer.zero_grad()
        (adapter(torch.ones(1, 2)).sum() + student_actor(torch.ones(1, 3)).sum()).backward()
        optimizer.step()
        layout = {"kind": "joint_state", "history_len": 20, "history_dim": 24, "flat_dim": 480}
        dims = {"teacher_obs_dim": 79, "history_flat_dim": 480, "latent_dim": 8, "action_dim": 12}

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "adapter.pt")
            save_checkpoint(
                path=path,
                adapter=adapter,
                student_actor=student_actor,
                history_normalizer=None,
                optimizer=optimizer,
                iteration=7,
                samples=123,
                best_loss=0.2,
                best_latent_mse=0.1,
                teacher_checkpoint="teacher.pt",
                layout=layout,
                dims=dims,
                action_loss_weight=0.5,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)

            self.assertIsInstance(payload["student_actor_state_dict"], dict)
            self.assertEqual(payload["objective"]["name"], "latent_mse_plus_action_mse")
            self.assertTrue(payload["objective"]["finetune_student_actor"])
            self.assertEqual(payload["objective"]["action_loss_weight"], 0.5)

            restored_adapter = nn.Linear(2, 1)
            restored_actor = nn.Linear(3, 1)
            restored_optimizer = torch.optim.AdamW(
                list(restored_adapter.parameters()) + list(restored_actor.parameters()), lr=1.0e-2
            )
            loaded = load_checkpoint(
                checkpoint_path=path,
                adapter=restored_adapter,
                student_actor=restored_actor,
                history_normalizer=None,
                optimizer=restored_optimizer,
                layout=layout,
                dims=dims,
                load_optimizer=True,
                device=torch.device("cpu"),
            )

            self.assertEqual(loaded["iteration"], 7)
            self.assertTrue(torch.equal(restored_adapter.weight, adapter.weight))
            self.assertTrue(torch.equal(restored_actor.weight, student_actor.weight))
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], args_cli.learning_rate)
            self.assertEqual(len(restored_optimizer.state), len(optimizer.state))

            adapter_only_optimizer = torch.optim.AdamW(restored_adapter.parameters(), lr=3.0e-4)
            with self.assertRaisesRegex(ValueError, "fine-tuned student actor"):
                load_checkpoint(
                    checkpoint_path=path,
                    adapter=restored_adapter,
                    student_actor=None,
                    history_normalizer=None,
                    optimizer=adapter_only_optimizer,
                    layout=layout,
                    dims=dims,
                    load_optimizer=True,
                    device=torch.device("cpu"),
                )

            legacy_path = os.path.join(tmp_dir, "legacy_adapter.pt")
            torch.save(
                {
                    "adapter_state_dict": adapter.state_dict(),
                    "optimizer_state_dict": torch.optim.AdamW(adapter.parameters()).state_dict(),
                    "layout": layout,
                    "dims": dims,
                },
                legacy_path,
            )
            legacy_actor = nn.Linear(3, 1)
            legacy_actor_before = copy.deepcopy(legacy_actor.state_dict())
            load_checkpoint(
                checkpoint_path=legacy_path,
                adapter=restored_adapter,
                student_actor=legacy_actor,
                history_normalizer=None,
                optimizer=restored_optimizer,
                layout=layout,
                dims=dims,
                load_optimizer=False,
                device=torch.device("cpu"),
            )
            self.assertTrue(torch.equal(legacy_actor.weight, legacy_actor_before["weight"]))
            self.assertTrue(torch.equal(legacy_actor.bias, legacy_actor_before["bias"]))


if __name__ == "__main__":
    unittest.main()
