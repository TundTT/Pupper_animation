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
from . import configs as c,contract as ct

def load_policy(params,env):
    config=json.loads(Path(params).parent.joinpath('config.json').read_text())
    if config['motion_contract_version']!=2 or config['source_hashes']!=source_hashes():
        raise ValueError('Training sources/geometry differ from this checkout. Evaluate in the exact training checkout.')
    net=network_factory()(82,8,preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(net)(model.load_params(str(params)),deterministic=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',type=Path,required=True);p.add_argument('--envs',type=int,default=64)
    p.add_argument('--seed',type=int,default=20260910);p.add_argument('--nominal',action='store_true')
    p.add_argument('--interrupt',action='store_true');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();env=AlignEnv(noise=False);policy=load_policy(args.params,env);n=args.envs
    randomization=functools.partial(domain_randomize_wheeled,rng=jax.random.split(jax.random.PRNGKey(args.seed+1),n))
    wrapped=training.VmapWrapper(env) if args.nominal else training.DomainRandomizationVmapWrapper(env,randomization)
    state=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(args.seed),n))
    orders=np.array([[1,2,3,4],[2,1,4,3],[3,1,4,2],[4,2,3,1]])[np.arange(n)%4]
    select=jax.vmap(env.select_command,in_axes=(0,0))
    @jax.jit
    def chunk(state,commands):
        def tick(s,command):
            previous=s;s=select(s,command);action,_=policy(s.obs,jax.random.PRNGKey(0));s=wrapped.step(s,action)
            # Freeze terminated environments; report each first terminal sample.
            s=jax.tree.map(lambda a,b:jp.where((previous.done>0).reshape((n,)+(1,)*(b.ndim-1)),a,b),previous,s)
            return s,(s.metrics,s.done,s.info['motion']['completed'])
        return jax.lax.scan(tick,state,commands)
    history=[]
    for leg_index in range(4):
        for block in range(16):  # 32 s/leg at 52 Hz; chunks keep memory bounded.
            commands=np.tile(orders[:,leg_index],(104,1))
            if block==0:commands[:]=0
            if args.interrupt and block==2:commands[:]=0
            state,record=chunk(state,jp.asarray(commands));history.append(jax.tree.map(np.asarray,record))
        print('Audited leg',leg_index+1,flush=True)
    metrics,done,completed=jax.tree.map(lambda *x:np.concatenate(x,axis=0),*history)
    valid=np.concatenate([np.ones((1,n),bool),~np.maximum.accumulate(done.astype(bool),axis=0)[:-1]])
    survived=done[-1]==0
    final_error=np.abs(np.asarray(ct.wrap(state.info['motion']['target']-state.pipeline_state.q[:,7+ct.WHEEL],jp)))
    final_speed=np.abs(np.asarray(state.pipeline_state.qd[:,6+ct.WHEEL]))
    settled=(final_error<.08).all(axis=1)&(final_speed<.08).all(axis=1)
    result=dict(params=str(args.params),seed=args.seed,envs=n,nominal=args.nominal,interrupt=args.interrupt,
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
    result['passes_simulation_gate']=(result['all_four_completed']==n and result['final_aligned_and_settled']==n
        and result['unsafe_rotations']==0 and result['min_wheel_gap_m']>.005
        and result['min_body_gap_m']>0 and result['max_impact_speed_m_s']<.10)
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
