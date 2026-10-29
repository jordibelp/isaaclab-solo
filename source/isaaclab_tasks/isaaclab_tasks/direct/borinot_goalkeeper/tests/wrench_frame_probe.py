# SPDX-License-Identifier: BSD-3-Clause
"""Known-load, rotated real-Borinot fixture for the installed incoming-wrench API."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import UsdPhysics
from scipy.spatial.transform import Rotation
import isaaclab.sim as sim_utils
import isaaclab_tasks
from isaaclab_tasks.direct.borinot_cmd_vel.borinot_cmd_vel_env_cfg import BorinotCmdVelEnvCfg
from isaaclab.utils.math import matrix_from_quat

cfg = BorinotCmdVelEnvCfg()
cfg.scene.num_envs = 1
cfg.seed = 42
cfg.sim.device = args.device
cfg.control_mode = 'thrust_and_PD_joint_pos_control'
cfg.include_events_randomization = False
cfg.robot.spawn.articulation_props.fix_root_link = True
cfg.robot_home = (0., 0., 3.)
cfg.robot_home_quat = tuple(Rotation.from_euler('xyz', [27., -33., 42.], degrees=True).as_quat(scalar_first=True))
cfg.kp, cfg.kd, cfg.effort_lim_sim = 60., 5., 30.
env = gym.make('Borinot-cmd-vel', cfg=cfg).unwrapped
env.reset()
robot = env.robot
thigh = robot.find_bodies('flying_arm_2__link1')[0][0]
calf = robot.find_bodies('flying_arm_2__link2')[0][0]
base = env.base_id
joint = next(UsdPhysics.Joint(p) for p in sim_utils.get_current_stage().Traverse()
             if p.IsA(UsdPhysics.Joint) and any(str(t).endswith('/flying_arm_2__link1')
             for t in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()))
q = joint.GetLocalRot1Attr().Get()
rot1 = torch.tensor([q.GetReal(), *q.GetImaginary()], device=env.device)[None]
pos1 = torch.tensor(tuple(joint.GetLocalPos1Attr().Get()), device=env.device)
q0 = joint.GetLocalRot0Attr().Get()
rot0 = torch.tensor([q0.GetReal(), *q0.GetImaginary()], device=env.device)[None]
pos0 = torch.tensor(tuple(joint.GetLocalPos0Attr().Get()), device=env.device)
target = torch.tensor([[.35, -.55]], device=env.device)
robot.write_joint_position_limit_to_sim(torch.stack((target-1e-6, target+1e-6), -1), joint_ids=env.joint_ids)
robot.write_joint_state_to_sim(target, torch.zeros_like(target), joint_ids=env.joint_ids)
report = {'body_names':robot.body_names, 'hip_joint':str(joint.GetPath()),
          'local_pos0':pos0.tolist(), 'local_pos1':pos1.tolist(),
          'local_rot0':rot0.tolist(), 'local_rot1':rot1.tolist(), 'cases':[]}
try:
    for load in ([0., 0., 0.], [2., -1., 3.]):
        extra = torch.tensor(load, device=env.device)[None]
        robot.permanent_wrench_composer.reset()
        for _ in range(1500):
            robot.set_joint_position_target(target, joint_ids=env.joint_ids)
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=extra[:,None], torques=torch.zeros_like(extra[:,None]), body_ids=[calf], is_global=True)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(env.physics_dt)
        raw = robot.data.body_incoming_joint_wrench_b[0, thigh].clone()
        rotations = matrix_from_quat(robot.data.body_link_quat_w[0])
        hip = robot.data.body_link_pos_w[0,thigh]+rotations[thigh]@pos1
        masses = robot.root_physx_view.get_masses()[0].to(env.device)
        forces = masses[:,None]*torch.tensor([0.,0.,-9.81],device=env.device)
        forces[base]=0
        forces[calf]+=extra[0]
        expected_force = -forces.sum(0)
        expected_moment = -torch.linalg.cross(robot.data.body_com_pos_w[0]-hip,forces).sum(0)
        frames = {'world':torch.eye(3,device=env.device), 'base':rotations[base],
                  'child':rotations[thigh], 'child_joint':rotations[thigh]@matrix_from_quat(rot1)[0],
                  'parent_joint':rotations[base]@matrix_from_quat(rot0)[0]}
        points = {'hip':hip, 'child_com':robot.data.body_com_pos_w[0,thigh],
                  'child_origin':robot.data.body_link_pos_w[0,thigh], 'base_com':robot.data.body_com_pos_w[0,base]}
        errors={}
        for fname,R in frames.items():
            for pname,p in points.items():
                predicted=torch.cat((R.T@expected_force,R.T@(expected_moment+torch.linalg.cross(hip-p,expected_force))))
                errors[fname+'/'+pname] = float((predicted-raw).abs().max())
        record={'load_world_n':load,'raw_wrench':raw.tolist(),'joint_pos':robot.data.joint_pos[0].tolist(),
                'max_joint_speed':float(robot.data.joint_vel.abs().max()),
                'expected_force_world':expected_force.tolist(),'expected_moment_hip_world':expected_moment.tolist(),
                'candidate_max_errors':dict(sorted(errors.items(),key=lambda item:item[1]))}
        report['cases'].append(record)
        print(json.dumps(record),flush=True)
        assert errors['child_joint/hip'] < 2e-4, record
    report['passed'] = True
finally:
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    env.close()
    app.close()
