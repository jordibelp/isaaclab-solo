# SPDX-License-Identifier: BSD-3-Clause
"""Local GTK launcher for Borinot Eval. It does not start Isaac Sim until you press Run."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gio, GLib, Gtk

from borinot_eval.spec import AGENTS, DEFAULT_COMMANDS, RANDOMIZATIONS, TASKS, read_json, validate, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ("thrust_and_torque_control", "thrust_and_PD_joint_pos_control", "PI_rotors_PD_joint_pos_control",
         "dsam_indi", "wrench", "model_wrench", "model_wrench_indi", "model_wrench_yawless",
         "model_wrench_yawless_indi", "cbrt", "cbrt_indi")


def combo(values):
    widget = Gtk.ComboBoxText()
    for value in values:
        widget.append(value, value)
    widget.set_active(0)
    return widget


def entry(value="", width=12):
    return Gtk.Entry(text=str(value), width_chars=width)


def csv_text(values):
    return ", ".join(f"{x:g}" for x in values)


def numbers(widget):
    return [float(x) for x in widget.get_text().replace(",", " ").split()]


def button(label, action):
    widget = Gtk.Button(label=label)
    widget.connect("clicked", lambda *_: action())
    return widget


def chooser(parent, title, save=False, extension=None):
    action = Gtk.FileChooserAction.SAVE if save else Gtk.FileChooserAction.OPEN
    dialog = Gtk.FileChooserDialog(title=title, transient_for=parent, action=action)
    dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Choose", Gtk.ResponseType.OK)
    if extension:
        file_filter = Gtk.FileFilter()
        file_filter.set_name(f"{extension.upper()} files")
        file_filter.add_pattern(f"*.{extension}")
        dialog.add_filter(file_filter)
    if save:
        dialog.set_do_overwrite_confirmation(True)
        dialog.set_current_name("borinot-evaluation.json")
    chosen = dialog.get_filename() if dialog.run() == Gtk.ResponseType.OK else None
    dialog.destroy()
    return Path(chosen) if chosen else None


def json_editor(value):
    widget = Gtk.TextView(monospace=True)
    widget.get_buffer().set_text(json.dumps(value, indent=2))
    return widget


def json_value(widget):
    buf = widget.get_buffer()
    return json.loads(buf.get_text(*buf.get_bounds(), False))


class PolicyRow:
    def __init__(self, ui, policy):
        self.ui, self.other = ui, policy
        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        top = Gtk.Box(spacing=6)
        self.path = entry(policy.get("checkpoint", ""), 20)
        self.label = entry(policy.get("label", ""), 20)
        self.label.set_placeholder_text("Policy label")
        self.path.set_placeholder_text("Local .pt file")
        top.pack_start(self.label, False, False, 0)
        top.pack_start(self.path, True, True, 0)
        top.pack_start(button("Browse…", self.browse), False, False, 0)
        top.pack_start(button("Remove", lambda: ui.remove_policy(self)), False, False, 0)
        self.widget.pack_start(top, False, False, 0)
        self.settings = Gtk.Box(spacing=6)
        self.mode = combo(MODES)
        self.mode.set_active_id(policy.get("control_mode", ui.shared.get("control_mode", MODES[0])))
        self.hz = entry(policy.get("policy_hz", ui.shared.get("policy_hz", 100)), 5)
        self.agent = combo(("PPO", "SAC", "SAC MSE"))
        self.agent.set_active(AGENTS.index(policy.get("agent", AGENTS[0])))
        for item in (self.mode, Gtk.Label(label="Policy Hz"), self.hz, self.agent,
                     button("Policy config…", self.edit_config)):
            self.settings.pack_start(item, False, False, 0)
        self.widget.pack_start(self.settings, False, False, 0)

    def browse(self):
        path = chooser(self.ui.window, "Choose a trusted checkpoint", extension="pt")
        if path:
            self.path.set_text(str(path))
            if not self.label.get_text():
                self.label.set_text(path.stem)

    def edit_config(self):
        value = self.ui.edit_json("Per-policy environment / runner settings",
                                  {k: self.other[k] for k in ("env", "runner") if k in self.other})
        if value is not None:
            self.other = {k: v for k, v in self.other.items() if k not in ("env", "runner")} | value

    def values(self, kind):
        value = {k: v for k, v in self.other.items() if k not in ("checkpoint", "label", "agent", "policy_hz", "control_mode")}
        value.update(checkpoint=self.path.get_text().strip(), label=self.label.get_text().strip(),
                     agent=AGENTS[self.agent.get_active()])
        if kind != "legacy":
            value.update(control_mode=self.mode.get_active_id(), policy_hz=int(self.hz.get_text()))
        return value


class EvalWindow:
    def __init__(self, window=None):
        self.window = window or Gtk.Window()
        self.window.set_title("Borinot Eval")
        self.window.set_default_size(1000, 820)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=14)
        self.window.add(outer)
        top = Gtk.Box(spacing=8)
        title = Gtk.Label(label="Borinot Eval", xalign=0)
        title.get_style_context().add_class("title")
        top.pack_start(title, True, True, 0)
        top.pack_start(button("Load recipe…", self.open_recipe), False, False, 0)
        top.pack_start(button("Save recipe…", self.save_recipe), False, False, 0)
        outer.pack_start(top, False, False, 0)
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        outer.pack_start(scroll, True, True, 0)
        self.body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=11)
        scroll.add(self.body)

        self.task = combo(TASKS)
        self.task.set_active_id("Borinot-goto-minTime")
        self.stage, self.trials, self.batch, self.seed, self.replay = [entry(x, 6) for x in (0, 64, 64, 42, 4)]
        self.section("1 · Task and shared scenario")
        self.line(("Task", self.task), ("Stage", self.stage), ("Trials", self.trials), ("Parallel envs", self.batch),
                  ("Seed", self.seed), ("Replay trials", self.replay))
        self.command_label = Gtk.Label(xalign=0)
        self.command, self.deadline = entry("", 22), entry(2.4, 6)
        self.deadline_label = Gtk.Label(label="Deadline (s)")
        self.body.pack_start(self.command_label, False, False, 0)
        self.line(("", self.command), ("", self.deadline_label), ("", self.deadline))
        self.sample = Gtk.CheckButton(label="Sample additional GoTo goals from a box (first trial uses the entered goal)")
        self.body.pack_start(self.sample, False, False, 0)
        self.extent, self.time_range = entry("7, 1, 0.5"), entry("2.2, 2.6")
        self.box_line = self.line(("Half-extents (m)", self.extent), ("Deadline range (s)", self.time_range))
        self.shared, self.box_center = {}, [0., 0., 1.5]
        self.body.pack_start(button("Shared environment config…", self.edit_shared), False, False, 0)
        self.hydra_overrides = entry(width=50)
        self.hydra_overrides.set_placeholder_text("env.reset_robot_on_tilt_threshold=False")
        self.hydra_overrides.set_tooltip_text(
            "Space-separated env.path=value settings for every policy and condition. Dedicated GUI controls take priority."
        )
        row = Gtk.Box(spacing=8)
        row.pack_start(Gtk.Label(label="Hydra env overrides"), False, False, 0)
        row.pack_start(self.hydra_overrides, True, True, 0)
        self.body.pack_start(row, False, False, 0)

        self.section("2 · Checkpoints")
        self.policy_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.body.pack_start(self.policy_box, False, False, 0)
        self.policies = []
        self.body.pack_start(button("+ Add policy", self.add_policy), False, False, 0)

        self.section("3 · Delay and randomizations")
        self.conditions = []
        self.condition_select = Gtk.ComboBoxText()
        self.condition_label, self.delay = entry("", 14), entry(0, 8)
        self.line(("Condition", self.condition_select), ("Label", self.condition_label),
                  ("Delay ms (fixed or min:max)", self.delay), ("", button("+ Condition", self.add_condition)),
                  ("", button("Remove", self.remove_condition)))
        rand_grid = Gtk.Grid(column_spacing=16, row_spacing=4)
        self.checks = {}
        for i, (key, label) in enumerate(RANDOMIZATIONS.items()):
            check = Gtk.CheckButton(label=label)
            self.checks[key] = check
            rand_grid.attach(check, i % 4, i // 4, 1, 1)
        self.body.pack_start(rand_grid, False, False, 0)
        self.current = -1
        self.condition_select.connect("changed", self.switch_condition)
        self.set_conditions([{"delay_ms": 0}, {"delay_ms": 20}])

        bars = Gtk.Box(spacing=8)
        outer.pack_start(bars, False, False, 0)
        self.wandb = Gtk.CheckButton(label="Log to private W&B")
        self.launch = button("Run evaluation", self.start)
        self.launch.get_style_context().add_class("suggested-action")
        self.stop = button("Stop", self.cancel)
        self.open = button("Open report", self.show_report)
        self.stop.set_sensitive(False)
        self.open.set_sensitive(False)
        for control in (self.wandb, self.launch, self.stop, self.open):
            bars.pack_start(control, False, False, 0)
        self.status = Gtk.Label(label="Choose checkpoints or load a recipe.", xalign=0, wrap=True, selectable=True)
        outer.pack_start(self.status, False, False, 0)
        self.process, self.output = None, None
        self.add_policy()
        self.task.connect("changed", lambda *_: self.update_task(reset_command=True))
        self.sample.connect("toggled", lambda *_: self.update_task())
        self.update_task(reset_command=True)
        self.window.connect("delete-event", self.on_close)

    def section(self, text):
        label = Gtk.Label(xalign=0)
        label.set_markup("<b>" + text + "</b>")
        self.body.pack_start(label, False, False, 0)

    def line(self, *fields):
        row = Gtk.Box(spacing=8)
        for label, widget in fields:
            if label:
                row.pack_start(Gtk.Label(label=label), False, False, 0)
            row.pack_start(widget, False, False, 0)
        self.body.pack_start(row, False, False, 0)
        return row

    def add_policy(self, policy=None):
        row = PolicyRow(self, policy or {})
        self.policies.append(row)
        self.policy_box.pack_start(row.widget, False, False, 0)
        row.widget.show_all()
        self.update_task()

    def remove_policy(self, row):
        self.policies.remove(row)
        row.widget.destroy()

    def update_task(self, reset_command=False):
        kind = TASKS[self.task.get_active_id()]
        self.command_label.set_text({"goal": "Goal XYZ (m, in world axes)",
                                     "velocity": "Command vx, vy, vz, yaw rate · m/s and rad/s",
                                     "legacy": "Command vx, vy, yaw rate · m/s and rad/s",
                                     "keeper": "Training-style shot sampling at the selected stage"}[kind])
        if reset_command and kind in DEFAULT_COMMANDS:
            self.command.set_text(csv_text(DEFAULT_COMMANDS[kind]))
        self.command.set_sensitive(kind != "keeper")
        for widget in (self.deadline, self.deadline_label, self.sample):
            widget.set_sensitive(kind == "goal")
        self.box_line.set_sensitive(kind == "goal" and self.sample.get_active())
        self.stage.set_sensitive(kind != "legacy")
        for row in self.policies:
            row.mode.set_sensitive(kind != "legacy")
            row.hz.set_sensitive(kind != "legacy")
        for key, widget in self.checks.items():
            disabled = ((kind == "legacy" and key in ("joint_friction", "add_base_mass", "base_com", "racket_mass",
                                                      "vertical_push")) or
                        (kind == "keeper" and key == "vertical_push"))
            widget.set_sensitive(not disabled)

    def stash_condition(self):
        if self.current >= 0:
            self.conditions[self.current] = (self.condition_label.get_text(), self.delay.get_text(),
                                             [k for k, w in self.checks.items() if w.get_active()])

    def condition_menu(self, active):
        self.condition_select.handler_block_by_func(self.switch_condition)
        self.condition_select.remove_all()
        for i, (label, delay, _) in enumerate(self.conditions, 1):
            self.condition_select.append_text(label or (f"{delay} ms" if delay else f"Condition {i}"))
        self.condition_select.set_active(active)
        self.condition_select.handler_unblock_by_func(self.switch_condition)
        self.current = -1
        self.switch_condition()

    def set_conditions(self, conditions):
        self.conditions = [(c.get("label", ""), ":".join(str(x) for x in c["delay_range_ms"])
                            if "delay_range_ms" in c else str(c.get("delay_ms", 0)), c.get("randomizations", []))
                           for c in conditions]
        self.condition_menu(0)

    def switch_condition(self, *_):
        self.stash_condition()
        self.current = self.condition_select.get_active()
        if self.current >= 0:
            label, delay, flags = self.conditions[self.current]
            self.condition_label.set_text(label)
            self.delay.set_text(delay)
            for key, check in self.checks.items():
                check.set_active(key in flags)

    def add_condition(self):
        self.stash_condition()
        _, delay, flags = self.conditions[self.current]
        names = {label.strip() for label, _, _ in self.conditions}
        index = len(self.conditions) + 1
        while f"Condition {index}" in names:
            index += 1
        self.conditions.append((f"Condition {index}", delay, flags.copy()))
        self.condition_menu(len(self.conditions) - 1)

    def remove_condition(self):
        if len(self.conditions) > 1:
            self.conditions.pop(self.current)
            self.condition_menu(max(0, self.current - 1))

    def edit_json(self, title, value):
        dialog = Gtk.Dialog(title=title, transient_for=self.window, modal=True)
        dialog.set_default_size(670, 520)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL, "_Apply", Gtk.ResponseType.OK)
        frame = Gtk.ScrolledWindow()
        widget = json_editor(value)
        frame.add(widget)
        dialog.get_content_area().pack_start(frame, True, True, 0)
        message = Gtk.Label(xalign=0)
        dialog.get_content_area().pack_start(message, False, False, 0)
        dialog.show_all()
        chosen = None
        while dialog.run() == Gtk.ResponseType.OK:
            try:
                chosen = json_value(widget)
                if not isinstance(chosen, dict):
                    raise ValueError("Enter a JSON object.")
                break
            except ValueError as exc:
                chosen = None
                message.set_text(str(exc))
        dialog.destroy()
        return chosen

    def edit_shared(self):
        value = self.edit_json("Shared environment config", self.shared)
        if value is not None:
            self.shared = value

    def recipe(self):
        self.stash_condition()
        task = self.task.get_active_id()
        kind = TASKS[task]
        scenario = {} if kind == "keeper" else {"command": numbers(self.command)}
        if kind == "goal":
            scenario["deadline_s"] = float(self.deadline.get_text())
            if self.sample.get_active():
                scenario["sample_box"] = {"center": self.box_center, "half_extents": numbers(self.extent),
                                          "deadline_s": numbers(self.time_range)}
        conditions = []
        for label, delay, flags in self.conditions:
            values = [float(x) for x in delay.split(":")]
            condition = {"delay_ms": values[0]} if len(values) == 1 else {"delay_range_ms": values}
            conditions.append(condition | {"label": label, "randomizations": flags})
        return {"task": task, "episodes": int(self.trials.get_text()), "num_envs": int(self.batch.get_text()),
                "seed": int(self.seed.get_text()), "stage": int(self.stage.get_text()), "replay_count": int(self.replay.get_text()),
                "scenario": scenario, "env": self.shared, "conditions": conditions,
                "hydra_overrides": self.hydra_overrides.get_text(),
                "policies": [row.values(kind) for row in self.policies]}

    def checked_recipe(self):
        return {k: v for k, v in validate(self.recipe(), Path.cwd()).items() if k != "kind"}

    def load(self, path):
        spec = validate(read_json(path), path.resolve().parent)
        for row in list(self.policies):
            self.remove_policy(row)
        self.shared = spec["env"]
        self.hydra_overrides.set_text(spec["hydra_overrides"])
        self.task.set_active_id(spec["task"])
        for widget, key in ((self.stage, "stage"), (self.seed, "seed"), (self.trials, "episodes"),
                            (self.batch, "num_envs"), (self.replay, "replay_count")):
            widget.set_text(str(spec[key]))
        scenario = spec["scenario"]
        self.command.set_text(csv_text(scenario.get("command", [])))
        self.deadline.set_text(str(scenario.get("deadline_s", 2.4)))
        box = scenario.get("sample_box")
        self.sample.set_active(box is not None)
        if box:
            self.extent.set_text(csv_text(box["half_extents"]))
            self.time_range.set_text(csv_text(box["deadline_s"]))
            self.box_center = box["center"]
        else:
            self.box_center = [0., 0., 1.5]
        for policy in spec["policies"]:
            self.add_policy(policy)
        self.set_conditions(spec["conditions"])
        self.status.set_text(f"Loaded {path}. Check the stage and training observation settings before running.")

    def open_recipe(self):
        path = chooser(self.window, "Open evaluation recipe", extension="json")
        if path:
            try:
                self.load(path)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                self.status.set_text(str(exc))

    def save_recipe(self):
        try:
            recipe = self.checked_recipe()
            path = chooser(self.window, "Save evaluation recipe", save=True, extension="json")
            if path:
                write_json(path, recipe)
                self.status.set_text(f"Saved {path}")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.status.set_text(str(exc))

    def start(self):
        if self.process:
            return
        try:
            recipe = self.checked_recipe()
            out = ROOT / "logs/borinot_eval" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="gui-recipe-", dir=out.parent, delete=False) as f:
                json.dump(recipe, f)
            cmd = [str(ROOT / "isaaclab.sh"), "-p", str(HERE / "eval_borinot.py"), "--recipe", f.name, "--output", str(out)]
            if self.wandb.get_active():
                cmd += ["--wandb", "online"]
            self.process = subprocess.Popen(cmd, cwd=ROOT, text=True, start_new_session=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.status.set_text(str(exc))
            return
        self.output = out
        self.launch.set_sensitive(False)
        self.stop.set_sensitive(True)
        self.open.set_sensitive(False)
        self.status.set_text("Starting Isaac Sim…")
        threading.Thread(target=self.watch, args=(self.process, Path(f.name), out), daemon=True).start()

    def watch(self, process, recipe, output):
        last = ""
        for line in process.stdout:
            if line.strip():
                last = line.strip()
                GLib.idle_add(self.status.set_text, last)
        code = process.wait()
        recipe.unlink(missing_ok=True)
        GLib.idle_add(self.finished, output, code, last)

    def finished(self, output, code, last):
        self.process = None
        self.launch.set_sensitive(True)
        self.stop.set_sensitive(False)
        report = (output / "report.html").is_file()
        self.open.set_sensitive(report)
        if code == 0 and report:
            self.status.set_text(f"Finished: {output}")
        elif report:
            self.status.set_text(f"Evaluation complete. W&B logging did not finish (exit {code}). Local report: {output}")
        else:
            self.status.set_text(f"Stopped or failed (exit {code}): {last}. Output: {output}")

    def show_report(self):
        if self.output:
            Gio.AppInfo.launch_default_for_uri((self.output / "report.html").as_uri(), None)

    def cancel(self):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            self.status.set_text("Stopping this evaluation…")

    def on_close(self, *_):
        if self.process:
            self.status.set_text("Stop the running evaluation before closing this window.")
            return True
        Gtk.main_quit()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    ui = EvalWindow()
    if args.recipe:
        ui.load(args.recipe)
    ui.window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
