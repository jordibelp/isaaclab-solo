# SPDX-License-Identifier: BSD-3-Clause
"""Live optional-racket and point-payload mass, inertia, clone and reset checks."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--image', type=Path)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
if args.image:
    args.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import UsdGeom, UsdPhysics
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config('Borinot-cmd-vel', 'rsl_rl_cfg_entry_point')
def main(cfg, agent_cfg):
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 4, args.device, 42
    cfg.curriculum = False
    cfg.reset_rpy_noise_degs = (0., 0., 0.)
    cfg.reset_xy_noise_m = 0.
    if args.image:
        cfg.viewer.origin_type = 'env'
        cfg.viewer.eye, cfg.viewer.lookat = (1.4, -1.8, 2.2), (0., 0., 1.3)
    env = gym.make('Borinot-cmd-vel', cfg=cfg, render_mode='rgb_array' if args.image else None).unwrapped
    report = {'arm_holds_racket':cfg.arm_holds_racket, 'payload_kg':cfg.ee_includes_puntual_mass,
              'control_mode':cfg.control_mode, 'checks':[]}

    def check(name, passed):
        report['checks'].append({'name':name, 'passed':bool(passed)})
        print('CHECK',name,bool(passed),flush=True)
        assert passed, name

    try:
        obs, _ = env.reset()
        view = env.robot.root_physx_view
        calf = env.robot.find_bodies('flying_arm_2__link2')[0][0]
        m = cfg.ee_includes_puntual_mass
        shaft = cfg.shaft_mass_kg
        shaft_com = torch.tensor([0., -.08, -.021], dtype=torch.float64)
        tip = torch.tensor([0., -.136, -.00495], dtype=torch.float64)
        identity = torch.eye(3, dtype=torch.float64)
        expected_com = (shaft*shaft_com + m*tip)/(shaft+m)
        # Independent oracle: assemble inertia about link origin, then shift
        # back once to composite CoM (production shifts the components directly).
        expected_inertia = torch.diag(torch.tensor([shaft*.16**2/12,1e-6,shaft*.16**2/12],dtype=torch.float64))
        expected_inertia += shaft*(shaft_com.dot(shaft_com)*identity-torch.outer(shaft_com,shaft_com))
        expected_inertia += m*(tip.dot(tip)*identity-torch.outer(tip,tip))
        expected_inertia -= (shaft+m)*(expected_com.dot(expected_com)*identity-torch.outer(expected_com,expected_com))
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        check('calf_mass_in_all_clones', torch.allclose(masses[:,calf],torch.full((4,),shaft+m),atol=1e-7,rtol=0))
        check('calf_com_in_all_clones', torch.allclose(coms[:,calf,:3].double(),expected_com.expand(4,-1),atol=1e-7,rtol=0))
        check('full_inertia_in_calf_axes', torch.allclose(inertias[:,calf].reshape(4,3,3).double(),
                                                        expected_inertia.expand(4,-1,-1),atol=2e-9,rtol=1e-5))
        check('positive_inertia', torch.linalg.eigvalsh(inertias[:,calf].reshape(4,3,3)).min()>0)
        check('paddle_body_exists_only_when_selected', ('paddle' in env.robot.body_names)==cfg.arm_holds_racket)
        check('paddle_joint_exists_only_when_selected', bool(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/paddle_fixed_joint'))==cfg.arm_holds_racket)
        check('no_payload_body_or_extra_dof', env.robot.num_bodies==3+int(cfg.arm_holds_racket) and env.robot.num_joints==2)
        expected_count = 3+int(cfg.arm_holds_racket)
        check('contact_filters_follow_body_set', len(env._body_contact_sensors)==expected_count and
              all(sensor.contact_physx_view.filter_count==expected_count-1 for sensor in env._body_contact_sensors))
        check('no_missing_racket_randomizer', cfg.arm_holds_racket or cfg.events is None or cfg.events.racket_mass is None)
        check('spaces_preserved', obs['policy'].shape==(4,cfg.observation_space))
        nominal_total = 2.6470000743865967 + .1485379934310913 + shaft+m + (cfg.paddle_mass_kg if cfg.arm_holds_racket else 0.)
        check('hover_includes_selected_payload_and_racket', abs(env.robot_mass_kg-nominal_total)<1e-6 and
              abs(env._hover_per_rotor-nominal_total*9.81/6)<1e-6)
        mesh=UsdGeom.Mesh(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/flying_arm_2__link2/visuals/flying_arm_2__link2/flying_arm_2__link2'))
        points=torch.tensor([tuple(p) for p in mesh.GetPointsAttr().Get()],dtype=torch.float64)
        check('payload_is_at_actual_distal_mesh_tip', (points-tip).abs().amax(-1).min()<1e-7 and abs(float(points[:,1].min())-float(tip[1]))<1e-7)
        # Save the physical joint-space inertia and independently constructed
        # tip Jacobian for cross-variant point-mass contribution checks.
        tip_w = env.robot.data.body_link_pos_w[0,calf]+quat_apply(env.robot.data.body_link_quat_w[0,calf],tip.to(env.device,dtype=torch.float32))
        jcolumns=[]
        for parent, joint_name in [('base',cfg.joint_names[0]),('flying_arm_2__link1',cfg.joint_names[1])]:
            parent_id=env.robot.find_bodies(parent)[0][0]
            joint=UsdPhysics.RevoluteJoint(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/'+parent+'/'+joint_name))
            p0=torch.tensor(tuple(joint.GetLocalPos0Attr().Get()),device=env.device)
            q0=joint.GetLocalRot0Attr().Get()
            q0=torch.tensor([q0.GetReal(),*q0.GetImaginary()],device=env.device)
            pq=env.robot.data.body_link_quat_w[0,parent_id]
            origin=env.robot.data.body_link_pos_w[0,parent_id]+quat_apply(pq,p0)
            axis=quat_apply(quat_mul(pq,q0),torch.tensor([1.,0.,0.],device=env.device))
            jcolumns.append(torch.linalg.cross(axis,tip_w-origin))
        jacobian=torch.stack(jcolumns,dim=1).cpu()
        report['point_joint_inertia_per_kg']=(jacobian.T@jacobian).tolist()
        matrix=view.get_generalized_mass_matrices()[0].cpu()
        ids=torch.tensor(env.joint_ids)+6
        report['joint_mass_matrix']=matrix[ids][:,ids].tolist()
        report['calf_mass_kg']=float(masses[0,calf])
        report['calf_com_m']=coms[0,calf,:3].tolist()
        report['calf_inertia_kg_m2']=inertias[0,calf].reshape(3,3).tolist()
        report['nominal_robot_mass_kg']=env.robot_mass_kg
        report['contact_body_names']=list(env.contact_body_names)
        env._reset_idx(torch.tensor([1],device=env.device))
        env._episode_steps.fill_(env.max_episode_length-1)
        action=torch.zeros(4,cfg.action_space,device=env.device)
        obs,reward,terminated,truncated,_=env.step(action)
        check('automatic_reset_exercised',truncated.all())
        for _ in range(10):
            obs,reward,_,_,_=env.step(action)
            check('finite_rollout',torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values()))
        check('mass_properties_survive_partial_and_auto_resets',torch.equal(masses,view.get_masses()) and
              torch.equal(coms,view.get_coms()) and torch.equal(inertias,view.get_inertias()))
        for field,value in [('ee_includes_puntual_mass',-.1),('ee_includes_puntual_mass',float('nan')),
                            ('ee_includes_puntual_mass',float('inf')),('arm_holds_racket','False')]:
            invalid=cfg.copy()
            setattr(invalid,field,value)
            try:
                invalid.configure_control()
            except ValueError:
                check('invalid_payload_configuration_rejected',True)
            else:
                check('invalid_payload_configuration_rejected',False)
        if args.image:
            from PIL import Image
            env.reset()
            for _ in range(15):
                env.sim.render()
                frame = env.render(recompute=True)
            check('rendered_payload_scene_has_content', frame is not None and frame.shape == (720,1280,3)
                  and frame.std()>5)
            args.image.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(frame).save(args.image)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2))
        print('PASSED',len(report['checks']),'checks',flush=True)
    finally:
        env.close()

if __name__=='__main__':
    try:
        main()
    finally:
        app.close()
