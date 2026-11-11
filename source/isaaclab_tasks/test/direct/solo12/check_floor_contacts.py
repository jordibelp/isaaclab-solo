# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the registered task and actual per-tile PhysX spring contacts."""

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="solo12-two-feet-IMU-teacher")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.solo12.solo12_env import Solo12Env
from isaaclab_tasks.utils.hydra import hydra_task_config

original_setup = Solo12Env._setup_scene
fixture_rows = []


def with_fixtures(self):
    original_setup(self)
    info = self._floor_stiffness_randomization
    if info is None:
        return
    nominal_mass = info["reference_mass_kg"]
    delta = np.asarray(info["delta_x_mm"])
    # Off the flat tile's diagonal: a sphere on that seam can engage two parallel springs.
    for row in range(delta.shape[0]):
        origin = self._terrain.terrain_origins[row, 2].cpu().numpy()
        cfg = RigidObjectCfg(
            prim_path=f"/World/FloorProbe/ball_{row}",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=nominal_mass / 2),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(restitution=0.0, restitution_combine_mode="max"),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(float(origin[0] + 2.123), float(origin[1] - 1.456), 0.07)),
        )
        self.scene.rigid_objects[f"floor_probe_{row}"] = RigidObject(cfg)
        fixture_rows.append(row)


Solo12Env._setup_scene = with_fixtures


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(cfg, agent_cfg):
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 230902
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    info = u._floor_stiffness_randomization
    assert info is not None
    delta, k = np.asarray(info["delta_x_mm"]), np.asarray(info["stiffness_n_per_m"])
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Get(stage, "/World/ground/terrain/mesh")
    face_count = len(mesh.GetFaceVertexCountsAttr().Get())
    collision_root = stage.GetPrimAtPath("/World/ground/terrain/floor_stiffness_colliders")
    colliders = [UsdGeom.Mesh.Get(stage, str(prim.GetPath()) + "/mesh") for prim in collision_root.GetChildren()]
    assigned_count = 0
    for collider in colliders:
        assigned_count += len(collider.GetFaceVertexCountsAttr().Get())
        material, _ = UsdShade.MaterialBindingAPI(collider.GetPrim()).ComputeBoundMaterial("physics")
        label = collider.GetPrim().GetParent().GetName()
        api = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
        stiffness = api.GetCompliantContactStiffnessAttr().Get()
        if label != "border":
            r, c = (int(part[1:]) for part in label.split("_"))
            assert np.isclose(stiffness, k[r, c], rtol=1e-6)
        assert api.GetCompliantContactDampingAttr().Get() == 60.0
    assert assigned_count == face_count
    assert not UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Get()
    assert len(colliders) == delta.size + 1
    assert np.isfinite(k).all() and k.min() > 0
    assert not np.allclose(k, 3000)
    env.reset()
    spawn_rows = sorted(set(u._terrain.env_origins[:, 0].cpu().tolist()))
    assert len(spawn_rows) > 2
    steady_z = []
    for step in range(600):
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
        if step >= 550:
            steady_z.append(
                [float(u.scene.rigid_objects[f"floor_probe_{row}"].data.root_pos_w[0, 2]) for row in fixture_rows]
            )
    steady_variation_mm = float(np.ptp(np.asarray(steady_z), axis=0).max() * 1000)
    assert steady_variation_mm < 0.01, steady_variation_mm
    physics = []
    for row in fixture_rows:
        probe = u.scene.rigid_objects[f"floor_probe_{row}"]
        z = float(probe.data.root_pos_w[0, 2])
        vz = float(probe.data.root_lin_vel_w[0, 2])
        physics.append(
            {
                "row": row,
                "requested_delta_mm": float(delta[row, 2]),
                "measured_sink_mm": (0.02 - z) * 1000,
                "vertical_speed_m_s": vz,
            }
        )
    expected = np.array([x["requested_delta_mm"] for x in physics])
    actual = np.array([x["measured_sink_mm"] for x in physics])
    print("PHYSICS", json.dumps(physics), flush=True)
    # Contact position offsets can contribute a common shift; the softness slope must be real.
    slope, intercept = np.polyfit(expected, actual, 1)
    assert 0.8 < slope < 1.2, (slope, intercept)
    assert max(abs(expected - actual)) < 0.15, (expected, actual)
    before = json.dumps(info, sort_keys=True)
    u.event_manager.apply(mode="curriculum_startup")
    u._reset_idx(torch.tensor([0, 2], device=u.device))
    assert json.dumps(u._floor_stiffness_randomization, sort_keys=True) == before
    after = []
    for _ in range(200):
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
    for row in fixture_rows:
        z = float(u.scene.rigid_objects[f"floor_probe_{row}"].data.root_pos_w[0, 2])
        after.append((0.02 - z) * 1000)
    np.testing.assert_allclose(after, actual, atol=0.02, rtol=0)
    result = {
        "passed": True,
        "metadata": info,
        "num_tile_colliders": len(colliders),
        "num_faces": face_count,
        "flat_spawn_x_values": spawn_rows,
        "physics": physics,
        "steady_position_variation_mm": steady_variation_mm,
        "physics_sink_slope": float(slope),
        "physics_sink_offset_mm": float(intercept),
        "partial_reset_and_curriculum_materials_preserved": True,
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print("FLOOR_LIVE_PASSED", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
