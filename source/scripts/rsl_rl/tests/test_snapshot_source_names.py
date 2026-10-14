# SPDX-License-Identifier: BSD-3-Clause
"""Snapshots must retain different source files that share a basename."""
import ast
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def test_inherited_agent_config_sources_do_not_collide(tmp_path, monkeypatch):
    sources = []
    for folder in ("goalkeeper", "borinot"):
        p = tmp_path / folder / "agents/rsl_rl_ppo_cfg.py"
        p.parent.mkdir(parents=True)
        p.write_text(folder)
        sources.append(p)
    env_cfg = tmp_path / "env_cfg.py"
    env_cfg.write_text("env config")
    env = tmp_path / "env.py"
    env.write_text("env")
    uploaded = {}

    class Artifact:
        def __init__(self, **kwargs):
            pass

        def add_file(self, filename, name=None):
            name = name or Path(filename).name
            assert name not in uploaded, f"Snapshot file would overwrite {name}"
            uploaded[name] = Path(filename).read_text()

    run = SimpleNamespace(config=SimpleNamespace(update=lambda *args, **kwargs: None),
                          log_artifact=lambda artifact: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=run, Artifact=Artifact))
    filename = Path(__file__).resolve().parents[2] / "skrl/helpers.py"
    tree = ast.parse(filename.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_wandb_snapshot")
    namespace = {"Path": Path, "shutil": shutil, "_jsonify": lambda value: {}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(filename), "exec"), namespace)
    run_dir = tmp_path / "run"
    namespace["_wandb_snapshot"](str(run_dir), None, None, SimpleNamespace(), str(env_cfg), str(env),
                                 extra_files=[str(sources[0]), str(sources[1]), str(sources[0])])
    assert len(uploaded) == 4
    assert uploaded["rsl_rl_ppo_cfg.py"] == "goalkeeper"  # keep the primary config's legacy filename
    copied = {p.name: p.read_text() for p in (run_dir / "code_snapshot").iterdir()}
    assert copied == uploaded
