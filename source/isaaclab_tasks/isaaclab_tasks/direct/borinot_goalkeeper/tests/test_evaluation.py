# SPDX-License-Identifier: BSD-3-Clause
"""Test the public evaluator's actual counting and statistics without launching Isaac Sim."""
import ast
import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[5] / "scripts/rsl_rl/eval_borinot_goalkeeper.py"
TREE = ast.parse(SCRIPT.read_text())
SCOPE = {"np": np, "torch": torch, "csv": csv, "Path": Path, "hashlib": hashlib}
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("OUTCOMES", "SAVE", "CSV_FIELDS")
                                           for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), SCOPE)
    elif isinstance(node, ast.FunctionDef) and node.name in ("evaluate", "interval", "summarize", "append_summary", "file_hash"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), SCOPE)


class Episodes:
    """Env 0 fails every step without a shot; env 1 times out every 3 steps with two shots."""
    def __init__(self, simple):
        self.unwrapped = self
        self.num_envs, self.device, self.max_episode_length = 2, "cpu", 3
        self.step_dt, self.tick, self.simple = .01, 0, simple
        self.resets = []

    def get_observations(self):
        return {"policy": torch.zeros(2, 1)}

    def step(self, actions):
        self.tick += 1
        done = torch.tensor([True, self.tick % 3 == 0])
        finished = torch.tensor([False, self.tick % 3 != 1])
        # One contact on the first step, then a save on the second, then a goal on the third.
        self.reset_terminated = torch.tensor([True, False])
        self.unresolved_at_reset = torch.tensor([True, False])
        self.delay_before_reset = torch.tensor([.01, .02])
        return self.get_observations(), torch.tensor([1., 2.]), done, {
            "goalkeeper_shot_finished": finished,
            "goalkeeper_outcome": torch.tensor([0, 7 if self.tick % 3 == 2 else 3]),
            "goalkeeper_had_paddle_hit": torch.tensor([False, self.tick % 3 == 2]),
            "goalkeeper_events": {"hit_racket": torch.tensor([False, self.tick % 3 == 1])}}


@pytest.mark.parametrize("simple", [False, True])
def test_exact_unequal_quotas_ignore_extra_fast_failures_and_keep_final_step(simple):
    env = Episodes(simple)
    model = SimpleNamespace(reset=lambda dones: env.resets.append(dones.clone()))
    rows, outcomes = SCOPE["evaluate"](env, lambda obs: torch.zeros(2, 1), model, torch.tensor([2, 1]), simple)
    assert rows["env_index"] == [0, 0, 1] and rows["episode_index"] == [0, 1, 0]
    assert rows["steps"] == [1, 1, 3] and rows["return"] == [1., 1., 6.]
    assert rows["shots"] == [0, 0, 2] and rows["saves"] == [0, 0, 1]
    assert rows["contacts"] == [0, 0, 1]
    assert rows["unresolved_shot"] == [True, True, False]
    assert sum(outcomes) == 2 and outcomes[7] == 1 and outcomes[3] == 1
    assert len(env.resets) == 3  # recurrent state reset for every done, even after quota
    result = SCOPE["summarize"](rows, outcomes, .01, 7)
    assert result["save_fraction"] == .5
    assert result["save_fraction_counting_reset_drops"] == .25
    assert result["robot_failure_fraction"] == pytest.approx(2 / 3)
    assert result["mean_episode_duration_s"] == pytest.approx(.05 / 3)


def test_no_resolved_shots_are_unavailable_not_a_measured_zero_success_rate():
    rows = {key: [0.] * 2 for key in ("shots", "saves", "contacts", "return", "unresolved_shot", "actuator_delay_s")}
    rows.update(steps=[2, 3], robot_failure=[1, 0])
    result = SCOPE["summarize"](rows, [0] * 10, .01, 4)
    assert result["save_fraction"] is None and result["save_fraction_ci95"] == [None, None]
    assert result["racket_contact_fraction"] is None
    assert result["save_fraction_counting_reset_drops"] is None


def test_bootstrap_resamples_episodes_and_is_seeded():
    numerator, denominator = np.array([0., 1., 2.]), np.array([0., 1., 2.])
    assert SCOPE["interval"](numerator, denominator, 7) == [1., 1.]
    assert SCOPE["interval"](numerator, denominator, 7) == SCOPE["interval"](numerator, denominator, 7)


def test_nonfinite_actions_abort_instead_of_recording_a_score():
    env = Episodes(False)
    with pytest.raises(RuntimeError, match="non-finite actions"):
        SCOPE["evaluate"](env, lambda obs: torch.full((2, 1), float("nan")), None, torch.ones(2), False)


def test_csv_appends_rows_and_rejects_an_incompatible_header(tmp_path):
    path = tmp_path / "summary.csv"
    for seed in (1, 2):
        SCOPE["append_summary"](path, [{"label": "policy", "seed": seed}])
    rows = list(csv.DictReader(path.open()))
    assert [row["seed"] for row in rows] == ["1", "2"]
    path.write_text("old,columns\n")
    with pytest.raises(RuntimeError, match="different columns"):
        SCOPE["append_summary"](path, [])


def test_checkpoint_hash_matches_a_known_digest(tmp_path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"known checkpoint bytes")
    assert SCOPE["file_hash"](path) == hashlib.sha256(path.read_bytes()).hexdigest()
