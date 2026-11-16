"""Check browser output against MuJoCo FK and the actual task limit method.

Run with an existing Python containing numpy, torch and mujoco:
python parity.py <browser evidence directory>
No simulator startup, physics stepping, policy or hardware access.
"""
import ast
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
evidence = Path(sys.argv[1])
model = mujoco.MjModel.from_xml_path(str(REPO / 'mujoco/solo12.xml'))
data = mujoco.MjData(model)
source = ast.parse((REPO / 'source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_env.py').read_text())
functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in (
    '_compute_joint_soft_pos_limits', '_validate_joint_limit_pair')]
cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Solo12Env')
functions += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_configure_joint_position_limits']
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=functions, type_ignores=[]), '<production limit resolver>', 'exec'), namespace)
worst_fk = worst_limits = worst_ghost = worst_path = 0.

def pose(state, q=None):
    data.qpos[:3] = state['rootPosition']
    data.qpos[3:7] = state['rootQuaternion']
    for name, value in zip(state['joint_names'], q if q is not None else state['joint_positions_rad']):
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    mujoco.mj_forward(model, data)

cases = json.loads((evidence / 'parity-cases.json').read_text())
for case in cases:
    state = case['state']
    pose(state)
    for name, position in case['bodies'].items():
        error = float(np.max(np.abs(data.xpos[model.body(name).id] - position)))
        worst_fk = max(worst_fk, error)
        assert error < 1e-10, (case['label'], name, error)
    buffer = SimpleNamespace(joint_pos_limits=torch.zeros(1, 12, 2, dtype=torch.float64))
    def write(limits, joint_ids):
        buffer.joint_pos_limits[:, joint_ids] = limits
    env = SimpleNamespace(cfg=SimpleNamespace(joint_names=state['joint_names'], **state['limits']),
                          _joint_ids=list(range(12)), _robot=SimpleNamespace(data=buffer, write_joint_position_limit_to_sim=write))
    with redirect_stdout(io.StringIO()):
        namespace['_configure_joint_position_limits'](env)
    for key, actual in [('hard', buffer.joint_pos_limits), ('soft', env._joint_soft_pos_limits)]:
        expected = torch.deg2rad(torch.tensor([b[key] for b in case['bounds']], dtype=torch.float64))
        error = (actual[0] - expected).abs().max().item()
        worst_limits = max(worst_limits, error)
        assert error < 1e-12, (case['label'], key, error)

envelopes = json.loads((evidence / 'envelopes.json').read_text())
for sample in envelopes:
    state, selected = sample['state'], sample['selected']
    index = state['joint_names'].index(selected)
    leg = selected.split('_')[0]
    for endpoint, ghost in enumerate(sample['ghosts']):
        q = state['joint_positions_rad'].copy()
        q[index] = math.radians(sample['soft'][endpoint])
        pose(state, q)
        for k, name in enumerate(('FL', 'FR', 'RL', 'RR')):
            error = float(np.max(np.abs(data.xpos[model.body(name+'_foot').id] - ghost['feet'][k])))
            worst_ghost = max(worst_ghost, error)
            assert error < 1e-10, (selected, endpoint, name, error)
    # Infer each sampled joint angle independently from the circular foot locus.
    # The line's vertices must match mj_forward at precisely the renderer's sample angles.
    hard = next(c for c in cases if c['label']=='default safe')['bounds'][index]['hard']
    expected, colors = [], []
    lower_soft, upper_soft = sample['soft']
    for lower, upper, color in ((hard[0], lower_soft, [.39,.46,.49]), (lower_soft, upper_soft, [.38,.86,.78]), (upper_soft, hard[1], [.39,.46,.49])):
        if lower == upper:
            continue
        previous = None
        count = max(2, math.ceil((upper-lower)/3))
        for k in range(count+1):
            q = state['joint_positions_rad'].copy()
            q[index] = math.radians(lower+(upper-lower)*k/count)
            pose(state, q)
            position = data.xpos[model.body(leg+'_foot').id].copy()
            if previous is not None:
                expected.extend([*previous, *position]); colors.extend(color*2)
            previous = position
    error = float(np.max(np.abs(np.array(expected)-sample['positions'])))
    worst_path = max(worst_path, error)
    assert error < 1e-7, (selected, error)  # Browser line buffers are float32.
    np.testing.assert_allclose(colors, sample['colors'], atol=1e-7)

result = dict(poses=len(cases), envelopes=len(envelopes), max_link_error_m=worst_fk,
              max_limit_error_rad=worst_limits, max_ghost_error_m=worst_ghost, max_path_error_m=worst_path)
(evidence / 'parity-result.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS:', json.dumps(result))
