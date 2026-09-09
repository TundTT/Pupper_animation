"""One from-scratch PPO attempt for the eight-position-joint hybrid policy."""
import argparse
from datetime import datetime
import functools
import json
from pathlib import Path
import time
import jax
from brax.io import model
from brax.training.agents.ppo import networks, train
from workspace.hybrid_env import TrainingEnv
from workspace.randomize import domain_randomize_wheeled
from workspace.hybrid_wrappers import wrap_for_training


def network_factory():
    return functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=(128,128,128),
                             value_hidden_layer_sizes=(256,256,256),activation=jax.nn.elu)


def randomization():
    # Broad established wheel-physics distribution; mixed actuator semantics are
    # asserted in the smoke test. No teacher or checkpoint from another behavior.
    return domain_randomize_wheeled


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=50_000_000)
    p.add_argument('--envs',type=int,default=4096);p.add_argument('--wandb',action='store_true')
    p.add_argument('--out',default=None);args=p.parse_args()
    name=f'wheel-align-hybrid_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    out=Path(args.out or Path(__file__).parent/'hybrid_results'/name);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        name=json.loads((out/'config.json').read_text())['name']
    cfg=dict(num_timesteps=args.steps,num_envs=args.envs,episode_length=1200,num_evals=11,
             num_eval_envs=64,unroll_length=20,num_minibatches=16,batch_size=256,num_updates_per_batch=4,
             learning_rate=3e-4,discounting=.97,entropy_cost=.01,normalize_observations=True,
             seed=0,deterministic_eval=True,action_repeat=1,max_devices_per_host=1)
    metadata=dict(ppo=cfg,policy_layers=[128,128,128],value_layers=[256,256,256],activation='elu',
                  slew_rate=.25,wheel_kp=2.,wheel_kd=.35,action_size=8,observation_size=51,
                  position_rows=[0,1,3,4,6,7,9,10],wheel_rows=[2,5,8,11],name=name,
                  initialization='random; no checkpoint',physics_randomization='domain_randomize_wheeled defaults')
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    run=None
    if args.wandb:
        import wandb
        resume_id=json.loads((out/'wandb.json').read_text())['id'] if (out/'wandb.json').exists() else None
        run=wandb.init(entity='QuadMorph',project='wheel-leg lift and align triangle base',name=name,config=metadata,id=resume_id,resume='allow',
                       tags=['align-hybrid','first-attempt','8-position-4-PD'])
        (out/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url)))
    start=time.time()
    def progress(step,metrics):
        d={k:float(v) for k,v in metrics.items()};d.update(step=step,wall_seconds=time.time()-start)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
        n=d.get('eval/avg_episode_length',1)
        keys=['lift','progress','stance','drift','held_error','completed','fall','unsafe_rotation']
        print(json.dumps(dict(step=step,elapsed=round(time.time()-start),eplen=n,
                             **{k:round(d.get('eval/episode_'+k,0)/max(n,1),5) for k in keys})),flush=True)
        if run:run.log(d,step=step)
    def save(step,make_policy,params):
        model.save_params(str(out/f'params_{step}'),params)
        (out/'latest.json').write_text(json.dumps(dict(step=step,path=str(out/f'params_{step}'))))
    print('OUTPUT',out,'DEVICES',jax.devices(),flush=True)
    _,params,_=train.train(TrainingEnv(),**cfg,network_factory=network_factory(),randomization_fn=randomization(),wrap_env_fn=wrap_for_training,
                            progress_fn=progress,policy_params_fn=save)
    model.save_params(str(out/'mjx_params'),params)
    if run:run.finish()
    print('FINISHED',out,flush=True)

if __name__=='__main__':main()
