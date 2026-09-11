"""PC audit: full sequences, interrupted sequences and physical clearance metrics."""
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
from .env import AlignEnv
from .train import network_factory,source_hashes
from .randomize import domain_randomize_wheeled
from . import configs as c,contract as ct,curriculum

def load_policy(params,env):
    config=json.loads(Path(params).parent.joinpath('config.json').read_text())
    if config['motion_contract_version']!=c.MOTION_VERSION or config['source_hashes']!=source_hashes():
        raise ValueError('Training sources/geometry differ from this checkout. Evaluate in the exact training checkout.')
    net=network_factory()(82,8,preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(model.load_params(str(params)),deterministic=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',type=Path,required=True);p.add_argument('--envs',type=int,default=64)
    p.add_argument('--stage',choices=['foundation','single','sequence'],default='sequence')
    p.add_argument('--seed',type=int,default=20260910);p.add_argument('--nominal',action='store_true')
    p.add_argument('--interrupt',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();env=AlignEnv(noise=False,automatic=True,stage=args.stage);policy=load_policy(args.params,env);n=args.envs
    randomization=functools.partial(curriculum.randomization(args.stage) or domain_randomize_wheeled,rng=jax.random.split(jax.random.PRNGKey(args.seed+1),n))
    wrapped=training.VmapWrapper(env) if args.nominal else training.DomainRandomizationVmapWrapper(env,randomization)
    state=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(args.seed),n))
    orders=np.array([[1,2,3,4],[2,1,4,3],[3,1,4,2],[4,2,3,1]])[np.arange(n)%4]
    state.info['order']=jp.asarray(orders)
    state.info['early_interrupt']=jp.full((n,),args.interrupt)
    @jax.jit
    def chunk(state):
        def tick(s,_):
            previous=s;action,_=policy(s.obs,jax.random.PRNGKey(0));s=wrapped.step(s,action)
            s=jax.tree.map(lambda a,b:jp.where((previous.done>0).reshape((n,)+(1,)*(b.ndim-1)),a,b),previous,s)
            return s,(s.metrics,s.done,s.info['motion']['completed'])
        return jax.lax.scan(tick,state,None,length=104)
    history=[]
    budget=c.SEQUENCE_STEPS if args.stage=='sequence' else c.SINGLE_STEPS
    count=4 if args.stage=='sequence' else 1
    for block in range(budget//104):
        state,record=chunk(state);history.append(jax.tree.map(np.asarray,record))
        if block%30==29:print('Audited',2*(block+1),'seconds',flush=True)
        finished=(np.asarray(state.info['sequence']['index'])>=count)&(np.asarray(state.info['sequence']['age'])>=104)
        if np.all(finished|(np.asarray(state.done)>0)):break
    metrics,done,completed=jax.tree.map(lambda *x:np.concatenate(x,axis=0),*history)
    valid=np.concatenate([np.ones((1,n),bool),~np.maximum.accumulate(done.astype(bool),axis=0)[:-1]])
    survived=done[-1]==0
    final_error=np.abs(np.asarray(ct.wrap(state.info['motion']['target']-state.pipeline_state.q[:,7+ct.WHEEL],jp)))
    final_speed=np.abs(np.asarray(state.pipeline_state.qd[:,6+ct.WHEEL]))
    requested=np.ones((n,4),bool) if count==4 else np.arange(4)[None,:]==ct.COMMAND_LEG[orders[:,0]][:,None]
    settled=((final_error<.08)|~requested).all(axis=1)&((final_speed<.08)|~requested).all(axis=1)
    requested_done=(completed[-1]|~requested).all(axis=1)
    result=dict(stage=args.stage,requested_completed=int((survived&requested_done).sum()),simulated_seconds=len(done)*c.CONTROL_DT,timeouts=int(np.asarray(state.info['sequence']['timeouts']).sum()),params=str(args.params),seed=args.seed,envs=n,nominal=args.nominal,interrupt=args.interrupt,
        survived=int(survived.sum()),all_four_completed=int((survived&completed[-1].all(axis=1)).sum()),
        final_aligned_and_settled=int((survived&settled).sum()),
        max_final_angle_error_rad=float(final_error.max()),max_final_wheel_speed_rad_s=float(final_speed.max()),
        falls=int(np.any((metrics['fall']>0)&valid,axis=0).sum()),
        unsafe_rotations=int(np.any((metrics['unsafe_rotation']>0)&valid,axis=0).sum()),
        min_wheel_gap_m=float(metrics['wheel_gap'][valid].min()),min_body_gap_m=float(metrics['body_gap'][valid].min()),
        max_impact_speed_m_s=float(metrics['impact_speed'][valid].max()),
        drift_p95_m=float(np.percentile(metrics['drift'][valid],95)),
        max_command_speed_rad_s=float(metrics['command_speed'][valid].max()),
        max_command_accel_rad_s2=float(metrics['command_accel'][valid].max()),
        status='Simulation audit only; hardware validation still required')
    result['passes_simulation_gate']=(result['requested_completed']==n and result['final_aligned_and_settled']==n
        and result['unsafe_rotations']==0 and result['min_wheel_gap_m']>.005
        and result['min_body_gap_m']>0 and result['max_impact_speed_m_s']<.10)
    # Distinguish a gate stall from genuine rotation towards the wrong target.
    result['phase_seconds_mean']={name:float((metrics['phase_'+name]*valid).sum()/n*c.CONTROL_DT)
        for name in ('idle','lift','rotate','verify','lower','hold')}
    result['blocked_gate_seconds_mean']={name:float((metrics[name+'_gate_blocked']*valid).sum()/n*c.CONTROL_DT)
        for name in ('floor','wheel','body','stability')}
    result['rotation_enabled_seconds_mean']=float((metrics['rotation_enabled']*valid).sum()/n*c.CONTROL_DT)
    result['per_wheel']={name:dict(attempted=int(requested[:,i].sum()),completed=int(completed[-1,:,i].sum()),
        final_angle_error_median_rad=float(np.median(final_error[requested[:,i],i])) if requested[:,i].any() else None,
        final_angle_error_p95_rad=float(np.percentile(final_error[requested[:,i],i],95)) if requested[:,i].any() else None)
        for i,name in enumerate(c.LEGS)}
    result['max_impact_by_phase_m_s']={name:float(np.max(np.where(valid&(metrics['phase_'+name]>0),metrics['impact_speed'],0)))
        for name in ('idle','lift','rotate','verify','lower','hold')}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    identity=args.params.parent/'wandb_run.json'
    if identity.exists():
        from training.wandb_logging import ExperimentLogger
        saved=json.loads(identity.read_text())
        if saved['mode']=='offline':
            print('Audit saved locally. Offline W&B sessions cannot resume; sync the saved training session and upload the audit files separately.')
            return
        config=json.loads(args.params.parent.joinpath('config.json').read_text())
        logger=ExperimentLogger(args.params.parent,config,entity=saved['entity'],project=saved['project'],mode=saved['mode'])
        try:logger.audits();logger.artifacts()
        finally:logger.finish()

if __name__=='__main__':main()
