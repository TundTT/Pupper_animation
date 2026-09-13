"""Matched direction-by-direction evaluation, including 250 Hz foot traces."""
import argparse
import json
import hashlib
from pathlib import Path
import jax
from jax import numpy as jp
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks
from brax.training.acme import running_statistics
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv

COMMANDS={
    'stand':(0.,0.,0.),'forward15':(.15,0.,0.),'forward20':(.2,0.,0.),'forward35':(.35,0.,0.),
    'reverse15':(-.15,0.,0.),'reverse20':(-.2,0.,0.),'reverse35':(-.35,0.,0.),
    'turn_left':(0.,0.,.5),'turn_right':(0.,0.,-.5),'forward_turn':(.2,0.,.4),
    'reverse_turn':(-.2,0.,.4),'left':(0.,.1,0.),'right':(0.,-.1,0.)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',action='append',required=True,help='LABEL=checkpoint path (repeatable)')
    p.add_argument('--output_dir',required=True)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--floor_friction',type=float,default=0.,help='Override floor sliding friction; zero keeps XML value')
    p.add_argument('--foot_model',choices=('rigid_flush','legacy_soft'),help='Explicit physics transfer test; default uses each saved model')
    p.add_argument('--commands',nargs='+',choices=list(COMMANDS),help='Subset of command names')
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    c=get_config();c.sensor_noise=0.;c.latency_probability=0.;c.reset_joint_noise=0.;c.push_probability=0.;c.command_hold_steps=args.steps+1
    c.floor_friction=args.floor_friction
    selected_commands={name:COMMANDS[name] for name in (args.commands or COMMANDS)}
    commands=jp.array([v for v in selected_commands.values() for _ in range(args.seeds)])
    seeds=jp.array([s for _ in selected_commands for s in range(args.seeds)])
    keys=jax.vmap(jax.random.PRNGKey)(seeds)
    net=networks.make_ppo_networks(36*c.observation_history,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    make=networks.make_inference_fn(net)
    reports={};interfaces={}
    for specification in args.params:
        label,path=specification.split('=',1)
        metadata=json.loads((Path(path).parent/'run.json').read_text())
        # Each saved policy must run with its own physical action conversion.
        # Reward weights stay common so cost diagnostics remain comparable.
        c.action_scale=tuple(metadata['config']['action_scale'])
        c.foot_model=args.foot_model or metadata['config'].get('foot_model','legacy_soft')
        c.moving_height_offset=metadata['config'].get('moving_height_offset',0.)
        c.planned_swing_duration=metadata['config'].get('planned_swing_duration',.16)
        c.planned_swing_height=metadata['config'].get('planned_swing_height',.003)
        c.planned_swing_bonus=metadata['config'].get('planned_swing_bonus',0.)
        c.planned_swing_foot_weights=tuple(metadata['config'].get('planned_swing_foot_weights',(1.,1.,1.,1.)))
        interfaces[label]=dict(foot_model=c.foot_model,action_scale=c.action_scale,moving_height_offset=c.moving_height_offset,
                               planned_swing_duration=c.planned_swing_duration,planned_swing_height=c.planned_swing_height,
                               planned_swing_foot_weights=c.planned_swing_foot_weights)
        env=PupperWalkEnv(c)
        reset=jax.jit(jax.vmap(env.reset))
        @jax.jit
        def step(params,state,key):
            action,_=jax.vmap(make(params,deterministic=True))(state.obs,key)
            state,trace=jax.vmap(env.step_with_diagnostics)(state,action)
            return state,trace,action
        if hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest()!=metadata['model_sha256']:
            raise ValueError('Checkpoint model differs from the evaluation model')
        params=model.load_params(path)
        state=reset(keys)
        state.info.update(command=commands,initial_command=commands)
        state=state.replace(obs=state.obs.reshape(-1,c.observation_history,36).at[:,:,6:9].set(commands[:,None,:]).reshape(-1,env.observation_size))
        records=[];actions=[];metrics=[];alive=[];dead=np.zeros(len(commands),dtype=bool)
        for i in range(args.steps):
            alive.append(~dead)
            state,trace,action=step(params,state,keys)
            records.append(jax.device_get(trace));actions.append(np.asarray(action));metrics.append(jax.device_get(state.metrics))
            dead|=np.asarray(state.done,dtype=bool)
            if (i+1)%100==0:print(label,'controls',i+1,'fallen',int(dead.sum()),flush=True)
        data={k:np.stack([r[k] for r in records]) for k in records[0]}
        data.update(action=np.array(actions),action_scale=np.array(c.action_scale),alive=np.array(alive),commands=np.asarray(commands))
        data.update({'metric_'+k:np.stack([m[k] for m in metrics]) for k in metrics[0]})
        np.savez_compressed(out/f'{label}.npz',**data)
        report={}
        for ci,name in enumerate(selected_commands):
            indices=range(ci*args.seeds,(ci+1)*args.seeds)
            active=data['alive'][50:,list(indices)]
            mask=active
            average=lambda k:float(data['metric_'+k][50:,list(indices)][mask].mean()) if mask.any() else None
            durations=[];peaks=[];touchdowns=[];periods=[];major=0;major_durations=[];major_peaks=[]
            foot_events=[dict(durations=[],peaks=[],major_durations=[],major_peaks=[],stride_distances=[],stride_periods=[]) for _ in range(4)]
            for index in indices:
                count=int(data['alive'][:,index].sum())
                co=data['contact'][50:count,index].reshape(-1,4)
                z=data['z'][50:count,index].reshape(-1,4)
                v=data['normal_velocity'][50:count,index].reshape(-1,4)
                foot_positions=data['foot_position'][50:count,index].reshape(-1,4,3)
                for foot in range(4):
                    starts=np.flatnonzero(co[:-1,foot]&~co[1:,foot])+1
                    ends=np.flatnonzero(~co[:-1,foot]&co[1:,foot])+1
                    major_starts=[];major_positions=[]
                    for start in starts:
                        end=ends[ends>start]
                        if not len(end):continue
                        end=end[0];duration=(end-start)*.004;peak=z[start:end,foot].max()
                        durations.append(duration*1000);peaks.append(peak*1000);touchdowns.append(max(-v[end-1,foot],0.))
                        foot_events[foot]['durations'].append(duration*1000);foot_events[foot]['peaks'].append(peak*1000)
                        if duration>.04 and peak>.002:
                            major+=1;major_starts.append(start*.004)
                            major_positions.append(foot_positions[start,foot,:2])
                            major_durations.append(duration*1000);major_peaks.append(peak*1000)
                            foot_events[foot]['major_durations'].append(duration*1000);foot_events[foot]['major_peaks'].append(peak*1000)
                    periods.extend(np.diff(major_starts).tolist())
                    foot_events[foot]['stride_periods'].extend((np.diff(major_starts)*1000).tolist())
                    if len(major_positions)>1:
                        foot_events[foot]['stride_distances'].extend((np.linalg.norm(np.diff(major_positions,axis=0),axis=1)*1000).tolist())
            stat=lambda values,fn=np.median:float(fn(values)) if len(values) else None
            angle=np.degrees(np.arccos(np.clip(data['tip_alignment'][50:,list(indices)],-1,1)))
            contact=data['contact'][50:,list(indices)]&active[:,:,None,None]
            a=data['action'][50:,list(indices)]*np.array(c.action_scale)/np.array(c.smoothness_reference_scale)
            action_second=np.diff(a,n=2,axis=0)
            second_mask=active[2:]&active[1:-1]&active[:-2]
            # Full-rollout spectra are meaningful only for surviving episodes.
            centered=a-a.mean(axis=0,keepdims=True)
            spectrum=abs(np.fft.rfft(centered,axis=0))**2
            frequencies=np.fft.rfftfreq(len(a),c.ctrl_dt)
            angular=data['angular_velocity'][50:,list(indices)].transpose(1,0,2,3).reshape(args.seeds,-1,3)
            joint=data['joint_velocity'][50:,list(indices)].transpose(1,0,2,3).reshape(args.seeds,-1,12)
            positions=data['torso_position'][50:,list(indices)].transpose(1,0,2,3).reshape(args.seeds,-1,3)
            full_survival=bool((~dead[list(indices)]).all())
            middle=data['planned_swing'][50:,list(indices)] & (data['planned_height'][50:,list(indices)]>.5*c.planned_swing_height) & active[:,:,None,None]
            report[name]=dict(survived=int((~dead[list(indices)]).sum()),seeds=args.seeds,
                velocity_error=average('velocity_error'),tracking_yaw=average('tracking_yaw'),
                yaw_error=average('yaw_error'),
                action_second_difference_rms=float(np.sqrt(np.mean(action_second[second_mask]**2))) if second_mask.any() else None,
                action_power_above_5hz=float(spectrum[frequencies>=5].sum()/max(spectrum[1:].sum(),1e-12)) if full_survival else None,
                action_dominant_hz=float(frequencies[1+np.argmax(spectrum[1:].sum(axis=(1,2)))]) if full_survival and spectrum[1:].sum()>1e-8 else None,
                body_angular_acceleration_rms=float(np.sqrt(np.mean((np.diff(angular,axis=1)/.004)**2))) if full_survival else None,
                joint_acceleration_rms=float(np.sqrt(np.mean((np.diff(joint,axis=1)/.004)**2))) if full_survival else None,
                middle_swing_contact_fraction=float(contact[middle].mean()) if middle.any() else None,
                net_xy_displacement_m=float(np.linalg.norm(positions[:,-1,:2]-positions[:,0,:2],axis=1).mean()) if full_survival else None,
                yaw_rate_mean=float(angular[:,:,2].mean()) if full_survival else None,
                yaw_rate_std=float(angular[:,:,2].std()) if full_survival else None,
                action_delta_rms=average('action_delta_rms'),slip_cost=-average('foot_slip') if mask.any() else None,
                ring_side_fraction=average('ring_side_fraction'),tip_lean_fraction=average('tip_lean_fraction'),
                ring_side_cost=-average('ring_side') if mask.any() else None,
                ring_bottom_cost=-average('ring_bottom') if mask.any() else None,
                ring_rub_cost=-average('ring_rub') if mask.any() else None,
                tilt_deg=average('tilt_deg'),tip_tilt_p95_deg=stat(angle[contact],lambda x:np.percentile(x,95)),
                airborne_median_ms=stat(durations),swing_peak_median_mm=stat(peaks),
                major_airborne_median_ms=stat(major_durations),major_swing_peak_median_mm=stat(major_peaks),
                touchdown_speed_median=stat(touchdowns),touchdown_speed_p95=stat(touchdowns,lambda x:np.percentile(x,95)),
                complete_events=len(durations),major_swing_fraction=major/max(len(durations),1),
                cadence_hz=1/stat(periods) if periods else None)
            per_foot={}
            for foot,foot_name in enumerate(('front_r','front_l','back_r','back_l')):
                z_all=data['z'][50:,list(indices)]
                valid=np.broadcast_to(active[:,:,None],z_all[:,:,:,foot].shape)
                fc=data['contact'][50:,list(indices)][:,:,:,foot][valid]
                zf=z_all[:,:,:,foot][valid]
                speed=data['tangential_speed'][50:,list(indices)][:,:,:,foot][valid]
                events=foot_events[foot]
                per_foot[foot_name]=dict(contact_fraction=stat(fc,np.mean),
                    contact_slide_speed_rms=stat(speed[fc],lambda x:np.sqrt(np.mean(x*x))),
                    contact_slide_distance_m=float(np.sum(speed*fc)*.004/args.seeds),
                    low_clearance_motion_fraction=stat((zf<.002)&(speed>.05),np.mean),
                    fraction_above_4mm=stat(zf>.004,np.mean),height_p95_mm=stat(zf,lambda x:1000*np.percentile(x,95)),
                    fraction_above_10mm=stat(zf>.010,np.mean),
                    major_swings_reaching_10mm=stat(np.asarray(events['major_peaks'])>=10.,np.mean),
                    major_swings_reaching_12mm=stat(np.asarray(events['major_peaks'])>=12.,np.mean),
                    stride_distance_median_mm=stat(events['stride_distances']),
                    stride_period_median_ms=stat(events['stride_periods']),
                    complete_swings=len(events['durations']),major_swings=len(events['major_durations']),
                    airborne_median_ms=stat(events['durations']),swing_peak_median_mm=stat(events['peaks']),
                    major_airborne_median_ms=stat(events['major_durations']),major_swing_peak_median_mm=stat(events['major_peaks']))
            report[name]['per_foot']=per_foot
        reports[label]=report
        (out/'report.json').write_text(json.dumps(reports,indent=2,allow_nan=False))
        print(label,json.dumps(report,indent=2),flush=True)
    (out/'protocol.json').write_text(json.dumps(dict(config=c.to_dict(),policy_interfaces=interfaces,commands=selected_commands,seeds=args.seeds,steps=args.steps,discard_controls=50,params=args.params,
        floor_sliding_friction=float(env.mj_model.geom_friction[env.floor,0]),
        model_sha256=hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
