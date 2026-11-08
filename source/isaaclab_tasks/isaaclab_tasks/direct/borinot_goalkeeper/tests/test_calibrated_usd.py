# SPDX-License-Identifier: BSD-3-Clause
"""Asset identity, independent inertia tensors, and double-counting regressions."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr.Usd")
from pxr import Gf, Sdf, Usd, UsdPhysics

REPO = Path(__file__).resolve().parents[6]
ASSETS = REPO / "source/isaaclab_assets/data/Robots/Borinot"
DIRECT = REPO / "source/isaaclab_tasks/isaaclab_tasks/direct"
OLD = ASSETS / "borinotFlat.usd"
NEW = ASSETS / "borinotFlat_6b3e09ce.usd"
# Own-CoM tensors in link axes, independently frozen from Jan's short URDF.
EXPECTED = {
    "base": (2.71746155, [0, .00234265, .02261913],
             [[.0273092143, 0, 0], [0, .0274698016, .0000166803], [0, .0000166803, .0397976770]]),
    "flying_arm_2__link1": (.14853845, [.00001377, -.078707, -.01935853],
             [[.00041107, 0, 0], [0, .00003024, -.00004671], [0, -.00004671, .00041193]]),
    "flying_arm_2__link2": (.127, [0, -.111706, -.015478],
             [[.00018274, 0, 0], [0, .000019135, -.000018119], [0, -.000018119, .00019053]]),
}


def config():
    tree = ast.parse((DIRECT / "borinot/inertial_properties_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    return SimpleNamespace(**{n.target.id: ast.literal_eval(n.value) for n in cls.body
                              if isinstance(n, ast.AnnAssign)})


def properties(stage, name):
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath('/Borinot/' + name))
    # Gf uses row vectors; transpose its matrix into column-vector convention.
    rotation = np.array(Gf.Matrix3d(Gf.Quatd(api.GetPrincipalAxesAttr().Get()).GetNormalized())).T
    tensor = rotation @ np.diag(api.GetDiagonalInertiaAttr().Get()) @ rotation.T
    return float(api.GetMassAttr().Get()), np.array(api.GetCenterOfMassAttr().Get()), tensor


@pytest.mark.parametrize('name', EXPECTED)
def test_new_asset_matches_short_urdf_mass_com_and_full_tensor(name):
    assert NEW.is_file(), 'Create the versioned calibrated asset; do not overwrite the original.'
    stage = Usd.Stage.Open(str(NEW))
    actual = properties(stage, name)
    for value, expected in zip(actual, EXPECTED[name]):
        np.testing.assert_allclose(value, expected, atol=2e-9, rtol=2e-7)


def test_copy_preserves_every_non_inertial_attribute_and_prim():
    assert NEW.is_file()
    before, after = Usd.Stage.Open(str(OLD)), Usd.Stage.Open(str(NEW))
    # The only extra subtree is the requested visual-only tip-weight marker.
    marker = Sdf.Path('/Borinot/flying_arm_2__link2/tip_mass_visual')
    assert [str(p.GetPath()) for p in before.TraverseAll()] == [
        str(p.GetPath()) for p in after.TraverseAll() if not p.GetPath().HasPrefix(marker)]
    allowed = {'physics:mass', 'physics:centerOfMass', 'physics:diagonalInertia', 'physics:principalAxes'}
    for a in before.TraverseAll():
        b = after.GetPrimAtPath(a.GetPath())
        assert a.GetTypeName() == b.GetTypeName()
        assert a.GetAppliedSchemas() == b.GetAppliedSchemas()
        assert [r.GetName() for r in a.GetRelationships()] == [r.GetName() for r in b.GetRelationships()]
        for r in a.GetRelationships():
            assert r.GetTargets() == b.GetRelationship(r.GetName()).GetTargets()
        assert [x.GetName() for x in a.GetAttributes()] == [x.GetName() for x in b.GetAttributes()]
        for attr in a.GetAttributes():
            assert attr.GetConnections() == b.GetAttribute(attr.GetName()).GetConnections()
            if a.GetName() in EXPECTED and attr.GetName() in allowed:
                continue
            assert attr.Get() == b.GetAttribute(attr.GetName()).Get(), f'{a.GetPath()}.{attr.GetName()}'


def test_defaults_do_not_add_motor_mass_or_restore_the_old_115g_link():
    cfg = config()
    assert cfg.include_motor_masses is False
    assert cfg.use_inference_leg_inertias is True
    for i, name in enumerate(('flying_arm_2__link1', 'flying_arm_2__link2')):
        assert cfg.leg_masses_kg[i] == EXPECTED[name][0]
        np.testing.assert_allclose(cfg.leg_com_positions_l[i], EXPECTED[name][1], atol=0, rtol=0)
        np.testing.assert_allclose(cfg.leg_inertias_l_kg_m2[i], EXPECTED[name][2], atol=0, rtol=0)


def test_stale_motor_mass_flag_cannot_double_count_the_new_base():
    assert NEW.is_file()
    spec = importlib.util.spec_from_file_location('audit_inertia', DIRECT / 'borinot/inertial_properties.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/Robot', 'Xform').GetReferences().AddReference(str(NEW), '/Borinot')
    cfg = config()
    cfg.include_motor_masses = True
    with pytest.raises(ValueError, match='already includes.*rotor'):
        helper.apply_inertial_properties(stage, '/Robot', cfg)


def test_all_borinot_default_configs_select_the_new_asset():
    for file in ('borinot/borinot_env_cfg.py', 'borinot_goalkeeper/borinot_goalkeeper_env_cfg.py',
                 'borinot_goalkeeper_simple/borinot_goalkeeper_simple_env_cfg.py'):
        text = (DIRECT / file).read_text()
        assert 'Borinot/borinotFlat_6b3e09ce.usd' in text
