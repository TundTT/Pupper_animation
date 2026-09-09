"""Full-sequence audit without autoreset: all four legs, randomized encoder phases.

Each external command gets 20 seconds, with 2 seconds stand between commands.
Success uses the historical final thresholds: 0.1 rad and 0.25 rad/s,
all four completed and no termination. A stricter clearance-qualified result
is reported separately.
Drift and held-error distributions include all samples up to first termination.
"""
import argparse
import functools
import json
from pathlib import Path
import jax
from jax import numpy as jp
import numpy as np
from brax.envs import training
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from workspace.hybrid_env import HybridAlignEnv, WHEEL, COMMAND_LEG, LIFT, VERIFY, wrap
from workspace.hybrid_train import network_factory, randomization


def build_policy(env,params):
    net=network_factory()(observation_size=env.observation_size,action_size=8,preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(params,deterministic=True)


def evaluate(params_path, n=64, seed=20260908, nominal=False, slew=.25, abduction_guard=None):
    if abduction_guard is None:
        config=json.loads((Path(params_path).parent/'config.json').read_text())
        abduction_guard=config.get('abduction_guard',.25)  # legacy attempt-one config
    env=HybridAlignEnv(slew_rate=slew,abduction_guard=abduction_guard)
    policy=build_policy(env,model.load_params(str(params_path)))
    if nominal:
        wrapped=training.VmapWrapper(env)
    else:
        wrapped=training.DomainRandomizationVmapWrapper(env,functools.partial(randomization(),rng=jax.random.split(jax.random.PRNGKey(seed+1),n)))
    state=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(seed),n))
    select=jax.vmap(env.select_command,in_axes=(0,0))
    # Alternating front-first orders exercise both fronts as first and second lift.
    orders=jp.array([[1,2,3,4],[2,1,4,3]])[jp.arange(n)%2]
    def chunk(state,commands):
        def tick(s,command):
            old=s
            s=select(s,command)
            a,_=policy(s.obs,jax.random.PRNGKey(0))
            s=wrapped.step(s,a)
            s=jax.tree.map(lambda before,after:jp.where((old.done>0).reshape((n,)+(1,)*(after.ndim-1)),before,after),old,s)
            leg=jp.maximum(COMMAND_LEG[s.info['active_command']],0)
            up=(s.info['phase']>=LIFT)&(s.info['phase']<=VERIFY)
            held=(jp.arange(4)[None,:]!=leg[:,None])|~up[:,None]
            e=jp.abs(wrap(s.info['hold']-s.pipeline_state.q[:,7+WHEEL]))
            target_e=jp.abs(wrap(s.info['target']-s.pipeline_state.q[:,7+WHEEL]))
            data=dict(done=s.done,fall=s.metrics['fall'],drift=s.metrics['drift'],tilt=s.metrics['tilt'],
                      held_error=e,held_mask=held,completed=s.info['completed'],phase=s.info['phase'],
                      clearance=s.metrics['clearance'],unsafe=s.metrics['unsafe_rotation'],target_error=target_e,
                      wheel_speed=jp.abs(s.pipeline_state.qd[:,6+WHEEL]),qpos=s.pipeline_state.q,action=a,
                      rotating=s.info['was_rotating'])
            return s,data
        return jax.lax.scan(tick,state,commands)
    rollout=jax.jit(chunk)
    history=[]
    for stage in range(9):
        cmd=jp.zeros(n,dtype=jp.int32) if stage%2==0 else orders[:,stage//2]
        length=100 if stage%2==0 else 1000
        # Fixed 100-step chunks bound host/device memory and expose progress.
        for _ in range(length//100):
            state,d=rollout(state,jp.tile(cmd,(100,1)))
            history.append(jax.tree.map(np.asarray,d))
        print('audit stage',stage,'completed',np.asarray(state.info['completed']).sum(axis=0).tolist(),flush=True)
    d=jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*history)
    alive=~np.maximum.accumulate(d['done'].astype(bool),axis=0)
    # Include the first terminating sample; exclude all subsequent invalid physics.
    valid=np.concatenate([np.ones((1,n),bool),alive[:-1]],axis=0)
    survived=alive[-1]
    falls=np.any(d['fall'].astype(bool)&valid,axis=0)
    unsafe=np.any(d['unsafe'].astype(bool)&valid,axis=0)
    ever_completed=np.any(d['completed']&valid[:,:,None],axis=0)
    final_error=d['target_error'][-1]
    success=survived&ever_completed.all(axis=1)&(final_error<.1).all(axis=1)&(d['wheel_speed'][-1]<.25).all(axis=1)
    strict=success&(final_error<.06).all(axis=1)&~unsafe
    previous=np.concatenate([np.zeros((1,n,4),bool),d['completed'][:-1]],axis=0)
    completion_event=d['completed']&~previous&valid[:,:,None]
    touchdown=np.any(completion_event&(d['clearance'][:,:,None]<.006),axis=0)
    strict &= touchdown.all(axis=1)
    snapshot_values=d['held_error'][(d['held_mask']&valid[:,:,None])]
    held_values=d['target_error'][(d['completed']&valid[:,:,None])]
    drifts=d['drift'][valid]
    rotating_clearance=d['clearance'][d['rotating']&valid]
    result=dict(params=str(params_path),num_envs=n,seed=seed,physics='nominal' if nominal else 'randomized',
                abduction_guard=abduction_guard,
                slew_rate=slew,sequence_seconds=90,all_four_success_count=int(success.sum()),
                all_four_success_fraction=float(success.mean()),survival_fraction=float(survived.mean()),
                strict_success_count=int(strict.sum()),
                touchdown_verified_success_count=int((success&touchdown.all(axis=1)).sum()),
                fall_count=int(falls.sum()),unsafe_rotation_count=int(unsafe.sum()),
                rotating_samples=int(rotating_clearance.size),
                min_rotating_clearance_m=float(rotating_clearance.min()) if rotating_clearance.size else None,
                per_leg_completed=dict(zip(['front_r','front_l','back_r','back_l'],ever_completed.sum(axis=0).tolist())),
                drift_p95=float(np.percentile(drifts,95)),drift_max=float(drifts.max()),
                held_error_p95=float(np.percentile(held_values,95)) if held_values.size else None,
                held_error_mean=float(held_values.mean()) if held_values.size else None,
                snapshot_hold_error_p95=float(np.percentile(snapshot_values,95)),
                held_samples=int(held_values.size),final_calibrated_error_p95=float(np.percentile(final_error,95)),
                completed_only_success_count=int((survived&ever_completed.all(axis=1)).sum()),
                thresholds=dict(calibrated_error_rad=.1,final_speed_rad_s=.25,strict_error_rad=.06,settle_error_rad=.035,settle_speed_rad_s=.08,settle_dwell_s=.5,min_rotate_clearance_m=.005))
    return result,d


def main():
    p=argparse.ArgumentParser();p.add_argument('--params',required=True);p.add_argument('--num_envs',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260908);p.add_argument('--nominal',action='store_true')
    p.add_argument('--abduction_guard',type=float,default=None,help='default: saved training config; old checkpoints use 0.25')
    p.add_argument('--slew',type=float,default=.25);p.add_argument('--out',required=True);args=p.parse_args()
    result,data=evaluate(args.params,args.num_envs,args.seed,args.nominal,args.slew,args.abduction_guard)
    Path(args.out).write_text(json.dumps(result,indent=2))
    np.savez_compressed(str(Path(args.out).with_suffix('.npz')),**data)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
