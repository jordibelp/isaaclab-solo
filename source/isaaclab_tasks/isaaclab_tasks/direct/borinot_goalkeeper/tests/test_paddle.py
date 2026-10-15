# SPDX-License-Identifier: BSD-3-Clause
"""The actual centered-hit kernel must retain its meaning after a diameter change."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

tree = ast.parse((Path(__file__).parents[1] / "borinot_goalkeeper_env.py").read_text())
method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_sample_physics_events")
kernel = [n for n in method.body if isinstance(n, ast.Assign)
          and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("center_sigma", "center_quality")]


@pytest.mark.parametrize("diameter", [.12, .16, .24])
def test_centered_hit_uses_fraction_of_round_head(diameter):
    radius = diameter / 2
    relative = torch.tensor([[.1, 0., 0.], [0., radius/2, 0.], [0., 0., radius], [0., -radius, 0.]])
    scope = {"self": SimpleNamespace(cfg=SimpleNamespace(paddle_diameter_m=diameter)),
             "relative": relative, "torch": torch}
    exec(compile(ast.Module(body=kernel, type_ignores=[]), "actual_centered_hit", "exec"), scope)
    torch.testing.assert_close(scope["center_quality"], torch.tensor([1., math.exp(-4/9), math.exp(-16/9), math.exp(-16/9)]))
