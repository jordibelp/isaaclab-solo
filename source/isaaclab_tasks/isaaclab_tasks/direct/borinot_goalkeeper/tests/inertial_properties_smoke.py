# SPDX-License-Identifier: BSD-3-Clause
"""Read actual PhysX properties against an independent inference URDF and energy oracle.

Run headless with --urdf /path/to/borinot_flying_arm_2.urdf --output /scratch/result.json.
No source USD, training log, checkpoint, robot, or online W&B record is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--urdf', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tasks', default='all', help='all, or one registered task ID')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from pxr import Usd, UsdPhysics
import isaaclab_tasks  # noqa: F401
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

root = ET.parse(args.urdf).getroot()
report = {'cases': [], 'urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest()}
tasks = ['Borinot-Direct-v0', 'Borinot-Goalkeeper-Direct-v0',
         'Borinot-Goalkeeper-Simple-Direct-v0', 'Borinot-cmd-vel'] if args.tasks == 'all' else [args.tasks]


def urdf_properties(name):
    item = root.find(f"link[@name='{name}']/inertial")
    a = item.find('inertia').attrib
    I = torch.tensor([[float(a['ixx']), float(a['ixy']), float(a['ixz'])],
                      [float(a['ixy']), float(a['iyy']), float(a['iyz'])],
                      [float(a['ixz']), float(a['iyz']), float(a['izz'])]], dtype=torch.float64)
    return float(item.find('mass').get('value')), torch.tensor([float(x) for x in item.find('origin').get('xyz').split()], dtype=torch.float64), I


def source_properties(stage, name):
    api = UsdPhysics.MassAPI(stage.GetPrimAtPath('/Borinot/' + name))
    q = api.GetPrincipalAxesAttr().Get()
    q = torch.tensor([q.GetReal(), *q.GetImaginary()], dtype=torch.float64)
    R = matrix_from_quat((q / q.norm())[None])[0] if q.norm() else torch.eye(3, dtype=torch.float64)
    I = R @ torch.diag(torch.tensor(tuple(api.GetDiagonalInertiaAttr().Get()), dtype=torch.float64)) @ R.T
    return float(api.GetMassAttr().Get()), torch.tensor(tuple(api.GetCenterOfMassAttr().Get()), dtype=torch.float64), I


def add_about_origin(mass, com, inertia):
    return inertia + mass * (com.dot(com) * torch.eye(3, dtype=torch.float64) - torch.outer(com, com))


try:
    for task in tasks:
        cases = [('default', True, True, False), ('legacy', False, False, False)]
        if task == 'Borinot-cmd-vel':
            cases += [('motor-only', True, False, False), ('leg-only', False, True, False),
                      ('randomized', True, True, True), ('racket-and-old-payload', True, True, False)]
        for label, motors, leg, randomized in cases:
            cfg = parse_env_cfg(task, device=args.device, num_envs=4)
            cfg.seed = 42
            if hasattr(cfg, 'include_events_randomization'): cfg.include_events_randomization = randomized
            if hasattr(cfg, 'curriculum'): cfg.curriculum = False
            if hasattr(cfg, 'reset_rpy_noise_degs'): cfg.reset_rpy_noise_degs = (0., 0., 0.)
            if task == 'Borinot-cmd-vel':
                # A nonzero old payload must NOT be added on top of the inference calf.
                cfg.ee_includes_puntual_mass = .15 if leg else .1
                cfg.arm_holds_racket = label == 'racket-and-old-payload'
            cfg.inertial_properties.include_motor_masses = motors
            cfg.inertial_properties.use_inference_leg_inertias = leg
            robot_cfg = cfg.robot_cfg if task == 'Borinot-Direct-v0' else cfg.robot
            source = Path(robot_cfg.spawn.usd_path)
            before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            source_stage = Usd.Stage.Open(str(source))
            expected = {name: source_properties(source_stage, name) for name in ['base', 'flying_arm_2__link1', 'flying_arm_2__link2']}
            if leg:
                for name in ['flying_arm_2__link1', 'flying_arm_2__link2']: expected[name] = urdf_properties(name)
            elif task != 'Borinot-Direct-v0':
                shaft = cfg.shaft_mass_kg
                payload = cfg.ee_includes_puntual_mass if task == 'Borinot-cmd-vel' else 0.
                sc, tip = torch.tensor([0., -.08, -.021], dtype=torch.float64), torch.tensor([0., -.136, -.00495], dtype=torch.float64)
                mass = shaft + payload; com = (shaft*sc + payload*tip)/mass
                I = add_about_origin(shaft, sc, torch.diag(torch.tensor([shaft*.16**2/12, 1e-6, shaft*.16**2/12], dtype=torch.float64)))
                I += add_about_origin(payload, tip, torch.zeros(3,3,dtype=torch.float64))
                I -= add_about_origin(mass, com, torch.zeros(3,3,dtype=torch.float64))
                expected['flying_arm_2__link2'] = mass, com, I
            if motors:
                mass, com, I = expected['base']; moment = mass*com; I0 = add_about_origin(mass, com, I)
                for i in range(6):
                    pos = torch.tensor([float(x) for x in root.find(f"joint[@name='borinot__j_rotor{i}']/origin").get('xyz').split()], dtype=torch.float64)
                    pm = float(root.find(f"link[@name='borinot__rotor{i}']/inertial/mass").get('value'))
                    mass += pm; moment += pm*pos
                    I0 += add_about_origin(pm, pos, torch.zeros(3,3,dtype=torch.float64))
                com = moment/mass
                expected['base'] = mass, com, I0-add_about_origin(mass, com, torch.zeros(3,3,dtype=torch.float64))
            env = gym.make(task, cfg=cfg).unwrapped
            case = {'task': task, 'variant': label, 'checks': [], 'properties': {}}
            report['cases'].append(case)

            def check(name, passed):
                case['checks'].append({'name': name, 'passed': bool(passed)})
                assert bool(passed), f'{task}/{label}: {name}'

            try:
                obs, _ = env.reset()
                view = env.robot.root_physx_view
                masses, coms, inertias = view.get_masses().clone(), view.get_coms().clone(), view.get_inertias().clone()
                nominal_total = sum(p[0] for p in expected.values())
                if 'paddle' in env.robot.body_names: nominal_total += cfg.paddle_mass_kg
                check('nominal_total_mass', abs(float(env.robot.data.default_mass[0].sum())-nominal_total)<1e-6)
                for name, (mass, com, I) in expected.items():
                    index = env.robot.find_bodies(name)[0][0]
                    actualI = inertias[:, index].reshape(4,3,3).double()
                    scale = masses[:,index].double()/mass if randomized and name == 'base' else torch.ones(4,dtype=torch.float64)
                    check(name+'_mass', torch.allclose(masses[:,index].double(), mass*scale, atol=2e-7,rtol=1e-6))
                    check(name+'_inertia', torch.allclose(actualI, I[None]*scale[:,None,None],atol=1e-8,rtol=2e-5))
                    if randomized and name == 'base':
                        check('base_random_mass_range', ((scale>=.9)&(scale<=1.2)).all())
                        check('base_random_com_range', ((coms[:,index,:3].double()-com).abs()<=.0100001).all())
                    else: check(name+'_com', torch.allclose(coms[:,index,:3].double(),com.expand(4,-1),atol=1e-7,rtol=0))
                    case['properties'][name] = {'mass':masses[:,index].tolist(),'com':coms[:,index,:3].tolist(),'inertia':actualI.tolist()}
                check('no_extra_bodies_or_dofs',env.robot.num_bodies==3+int('paddle' in env.robot.body_names) and env.robot.num_joints==2)
                hover = cfg.hover_thrust_per_rotor if task == 'Borinot-Direct-v0' else env._hover_per_rotor
                if motors or leg or task != 'Borinot-Direct-v0': check('nominal_hover',abs(hover-nominal_total*9.81/6)<1e-6)
                snapshots = env.get_code_snapshot_files()
                check('override_sources_in_snapshot',all(any(Path(p).name==n for p in snapshots) for n in ['inertial_properties.py','inertial_properties_cfg.py']))
                env._reset_idx(torch.tensor([0],device=env.device))
                env.reset()
                if hasattr(env,'_episode_steps'): env._episode_steps.fill_(env.max_episode_length-1)
                env.episode_length_buf.fill_(env.max_episode_length-1)
                action=torch.zeros((4,cfg.action_space),device=env.device)
                for _ in range(3):
                    obs,reward,_,_,_=env.step(action)
                    check('finite_step',torch.isfinite(reward).all() and all(torch.isfinite(x).all() for x in obs.values()))
                check('properties_persist_through_resets',torch.equal(masses,view.get_masses()) and torch.equal(coms,view.get_coms()) and torch.equal(inertias,view.get_inertias()))
                check('source_usd_unchanged',before_hash==hashlib.sha256(source.read_bytes()).hexdigest())
                case['nominal_total_mass_kg']=nominal_total
                print('CASE PASSED',task,label,len(case['checks']),flush=True)
            finally:
                env.close()
                sim_utils.create_new_stage()
                torch.cuda.empty_cache()
    report['passed']=True
finally:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    app.close()
