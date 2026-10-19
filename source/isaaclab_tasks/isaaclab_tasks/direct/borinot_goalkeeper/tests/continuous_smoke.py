# SPDX-License-Identifier: BSD-3-Clause
"""Real contacts, one-ball scheduling, nonterminal saves and a full 20 s episode."""
import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs',type=int,default=16)
parser.add_argument('--output',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args,overrides=parser.parse_known_args();sys.argv=[sys.argv[0]]+overrides
app=AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config('Borinot-Goalkeeper-Direct-v0','rsl_rl_cfg_entry_point')
def main(cfg,agent_cfg):
    cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=53
    cfg.curriculum=False;cfg.reset_grid_enabled=False
    cfg.reset_xy_noise_m=cfg.reset_z_noise_m=0.
    cfg.hold_arm=True
    env=gym.make('Borinot-Goalkeeper-Direct-v0',cfg=cfg).unwrapped
    actions=torch.zeros(env.num_envs,cfg.action_space,device=env.device)
    report={'checks':[],'mode':cfg.control_mode,'episode_length_s':cfg.episode_length_s,'observations':cfg.observation_space}
    ids=env.robot._ALL_INDICES
    def check(name,value):
        passed=bool(value);report['checks'].append({'name':name,'passed':passed})
        print(f'CHECK {name}: {passed}',flush=True);assert passed,name
    def shoot(position,velocity):
        state=env.ball.data.default_root_state.clone();state[:,:3]=position+env.scene.env_origins
        state[:,7:10]=torch.tensor(velocity,device=env.device);state[:,10:]=0.
        env.ball.write_root_state_to_sim(state);env.ball_contact.reset()
        env._previous_ball.copy_(position)
    def constant_position(x,y,z):return torch.tensor([x,y,z],device=env.device).expand(env.num_envs,-1)
    try:
        obs,_=env.reset()
        check('new_observation_contract',obs['policy'].shape==(env.num_envs,cfg.observation_space)
              and obs['critic'].shape==(env.num_envs,cfg.state_space))
        check('single_physical_ball_per_environment',env.ball.root_physx_view.count==env.num_envs)
        cfg.shot_resampling_time_range_s=(10.,10.)

        # A real centre racket impact must score without any robot write/reset.
        env.reset();env.step(actions)
        shoot(env._paddle_pose()[0]+constant_position(.045,0.,0.),[-3.,0.,0.])
        writes=[];original_write=env.robot.write_root_state_to_sim
        def tracked_write(state,env_ids=None):
            writes.append(True);return original_write(state,env_ids)
        env.robot.write_root_state_to_sim=tracked_write
        saved=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        contact=torch.zeros_like(saved);kept_momentum=False
        for _ in range(50):
            obs,reward,term,trunc,info=env.step(actions)
            contact|=info['goalkeeper_paddle_hit']
            selected=info['goalkeeper_outcome']==7
            saved|=selected
            if selected.any():
                check('physical_save_does_not_terminate',not term[selected].any() and not trunc[selected].any())
                check('physical_save_reward',torch.all(info['goalkeeper_reward_terms']['outcome'][selected]==cfg.rwd_scale_save))
                check('saved_ball_is_parked_and_inactive',not env._ball_active[selected].any() and (env.ball.data.root_pos_w[selected,2]<-1.).all())
                kept_momentum|=bool((env.robot.data.root_vel_w[selected].abs().sum(-1)>1e-5).any())
        env.robot.write_root_state_to_sim=original_write
        check('real_racket_contacts_and_cleared_saves',contact.any() and saved.any())
        check('save_never_teleports_robot_or_clears_episode',not writes and (env._episode_steps>=50).all())
        check('post_impact_momentum_is_retained',kept_momentum)
        check('save_count_is_per_shot_not_per_waiting_tick',env.outcome_counts[7]==saved.sum())

        # All ball failures are nonterminal and use their signed scales exactly once.
        arm=env.robot.find_bodies('flying_arm_2__link1')[0][0]
        cases=[('base',1,cfg.rwd_scale_ball_base_collision,lambda:env.robot.data.root_pos_w-env.scene.env_origins+constant_position(.38,0.,0.),[-3.,0.,0.]),
               ('arm',2,cfg.rwd_scale_ball_arm_collision,lambda:env.robot.data.body_com_pos_w[:,arm]-env.scene.env_origins+constant_position(.22,0.,-.065),[-3.,0.,0.]),
               ('goal',3,cfg.rwd_scale_goal,lambda:constant_position(-.2,.85,1.),[-3.,0.,0.]),
               ('floor',4,cfg.rwd_scale_ball_floor,lambda:constant_position(1.,.8,.04),[0.,0.,0.]),
               ('miss',9,cfg.rwd_scale_miss,lambda:constant_position(3.1,.8,1.),[0.,0.,0.])]
        for name,code,scale,target,velocity in cases:
            env.reset();env.step(actions);shoot(target(),velocity)
            found=torch.zeros_like(saved);total=torch.zeros(env.num_envs,device=env.device)
            for _ in range(35):
                _,_,term,trunc,info=env.step(actions)
                selected=info['goalkeeper_outcome']==code;found|=selected
                check_result=not term[selected].any() and not trunc[selected].any()
                assert check_result,(name,'terminated')
                total+=torch.where(selected,info['goalkeeper_reward_terms']['outcome'],0.)
            check(name+'_real_result_is_nonterminal',found.any())
            check(name+'_penalty_is_once',torch.allclose(total[found],torch.full_like(total[found],scale)))

        # A planned launch cannot replace an unresolved ball, even if overdue.
        cfg.shot_resampling_time_range_s=(.05,.05);env.reset()
        shoot(constant_position(1.,.8,2.),[0.,0.,0.])
        for _ in range(20):env.step(actions)
        check('overdue_schedule_waits_for_active_ball',env._ball_active.all() and (env._shot_steps==20).all())

        cfg.shot_resampling_mode='resolution';cfg.shot_resampling_time_range_s=(.1,.1)
        env.reset();shoot(constant_position(1.,.8,.025),[0.,0.,0.])
        env.step(actions)
        check('resolution_mode_waits_after_result',not env._ball_active.any() and (env._next_launch_steps==10).all())
        for _ in range(9):env.step(actions)
        check('no_early_resolution_relaunch',not env._ball_active.any())
        env.step(actions)
        check('resolution_relaunch_at_configured_tick',env._ball_active.all() and (env._shot_steps==1).all())

        # Artificial 20 s collection boundary. Hold only the base for this clock test;
        # contact/impact preservation above uses the unheld robot and real dynamics.
        cfg.shot_resampling_mode='launch';cfg.shot_resampling_time_range_s=(2.,3.)
        env.reset();env.episode_length_buf[:]=50000
        home=env.robot.data.default_root_state.clone()
        home[:,:3]=torch.tensor(cfg.robot_home,device=env.device)+env.scene.env_origins
        home[:,3:7]=torch.tensor(cfg.robot_home_quat,device=env.device);home[:,7:]=0.
        launches=[];launch=env._launch_ball
        def record_launch(env_ids):
            assert ((~env._ball_active[env_ids]) | (env._episode_steps[env_ids]==0)).all()
            launches.append((env._episode_steps[env_ids].cpu().tolist(),env_ids.cpu().tolist()))
            return launch(env_ids)
        env._launch_ball=record_launch
        for index in range(env.max_episode_length):
            env.robot.write_root_state_to_sim(home)
            obs,reward,term,trunc,info=env.step(actions)
            assert not term.any(),('clock_fixture_safety_failure',index)
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            if index+1<env.max_episode_length:assert not trunc.any(),index
        check('full_twenty_seconds_then_time_limit',trunc.all() and (env._episode_steps==0).all())
        check('multiple_async_shots_with_one_ball',len(launches)>6)
        check('timeouts_keep_pre_reset_state_for_bootstrap','time_outs_obs' in info)
        check('episode_stats_count_multiple_shots',float(info['log']['Episode/shots'])>=6)
        report['launch_batches']=launches
        env._launch_ball=launch

        # Only base crash, tilt and area violation are default safety terminations.
        env.reset();outside=home.clone();outside[:,0]=env.scene.env_origins[:,0]+cfg.goal_area_x_m[1]+.2
        env.robot.write_root_state_to_sim(outside)
        _,_,term,trunc,_=env.step(actions)
        check('leaving_area_terminates',term.all() and not trunc.any())
        env.reset();tilted=home.clone();tilted[:,3:7]=torch.tensor([2**-.5,2**-.5,0.,0.],device=env.device)
        env.robot.write_root_state_to_sim(tilted)
        _,_,term,trunc,info=env.step(actions)
        check('excessive_tilt_terminates',term.all() and info['goalkeeper_tilt_threshold'].all())
        env.reset();env._area_min[2]=-.2
        crashed=home.clone();crashed[:,2]=env.scene.env_origins[:,2]+.025
        env.robot.write_root_state_to_sim(crashed)
        _,_,term,trunc,info=env.step(actions)
        check('real_base_floor_crash_terminates',term.all() and info['goalkeeper_robot_contact_crash'].all())
        report['passed']=True
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n');env.close()


try:main()
finally:app.close()
