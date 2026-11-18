# SPDX-License-Identifier: BSD-3-Clause
"""Offscreen GTK regression; run with system Python in a desktop session."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("gi")
from borinot_eval_gui import EvalWindow, Gtk


@pytest.fixture
def ui(tmp_path):
    if not Gtk.init_check()[0]:
        pytest.skip("No GTK display.")
    window = Gtk.OffscreenWindow()
    view = EvalWindow(window)
    path = tmp_path / "test.pt"
    path.touch()
    view.policies[0].path.set_text(str(path))
    yield view
    window.destroy()


def test_add_copies_condition_without_overwriting_original_or_colliding_with_existing_labels(ui):
    ui.condition_label.set_text("Condition 3")  # skip a pre-existing name
    ui.delay.set_text("5")
    ui.checks["gust"].set_active(True)
    ui.add_condition()
    assert ui.current == 2
    assert ui.condition_label.get_text() == "Condition 4"
    assert ui.delay.get_text() == "5" and ui.checks["gust"].get_active()
    # Same delay and checkboxes are valid as long as labels differ.
    conditions = ui.checked_recipe()["conditions"]
    assert len(conditions) == 3
    assert conditions[0]["delay_ms"] == conditions[2]["delay_ms"] == 5
    ui.delay.set_text("10:20")
    ui.checks["gust"].set_active(False)
    ui.condition_select.set_active(0)
    assert ui.delay.get_text() == "5" and ui.checks["gust"].get_active()
    assert ui.checked_recipe()["conditions"][2]["delay_range_ms"] == [10., 20.]
    ui.condition_select.set_active(2)
    ui.remove_condition()
    assert len(ui.checked_recipe()["conditions"]) == 2
    assert ui.checked_recipe()["conditions"][0]["randomizations"] == ["gust"]
