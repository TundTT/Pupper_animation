"""PC-only PPO entry. Refuses CPU training and never overwrites an existing run."""
import argparse
from datetime import datetime
import functools
import hashlib
import json
import os
if os.name!='nt':os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import subprocess
import time
import jax
from brax.io import model
from brax.training.agents.ppo import networks,train
from . import configs as c, curriculum
from .env import AlignEnv
from .randomize import domain_randomize_wheeled
from .hybrid_wrappers import wrap_for_training
from training.wandb_logging import ExperimentLogger, ENTITY, PROJECT

ROOT=Path(__file__).resolve().parents[2]

def network_factory():
    return functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=(128,128,128),
        value_hidden_layer_sizes=(256,256,256),activation=jax.nn.elu)

def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.json'))+[
        c.MODEL_PATH,Path(__file__).with_name('uv.lock'),ROOT/'training/wandb_logging.py']
    paths+=list(c.MODEL_PATH.with_name('meshes').glob('*.stl'))
    paths+=list((ROOT/'ros2_ws/src/neural_controller/include/neural_controller').glob('wheel_align_*.hpp'))
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=None)
    p.add_argument('--stage',choices=list(curriculum.STEPS),default='sequence')
    p.add_argument('--init-from',type=Path,help='Transfer compatible v5 actor/normalizer; resets optimizer, critic and step count')
    p.add_argument('--envs',type=int,default=2048)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--wandb-entity',default=ENTITY)
    p.add_argument('--wandb-project',default=PROJECT)
    p.add_argument('--wandb-mode',choices=['online','offline'],default='online')
    p.add_argument('--no-wandb',action='store_true',help='Explicit local-only run; online logging is the default')
    p.add_argument('--video-every-steps',type=int,default=25_000_000,help='Periodic video interval; 0 means final only')
    args=p.parse_args()
    args.steps=curriculum.STEPS[args.stage] if args.steps is None else args.steps
    if not any(d.platform=='gpu' for d in jax.devices()):
        raise SystemExit('GPU training only. Use preflight.py for CPU validation; install the cuda extra on Linux/WSL2 on the PC.')
    if args.envs not in (256,512,1024,2048,4096) or args.steps<=0:
        raise SystemExit('--envs must be 256, 512, 1024, 2048 or 4096 (divides the PPO batch); --steps must be positive.')
    if args.video_every_steps<0:
        raise SystemExit('--video-every-steps must be nonnegative')
    out=args.out or ROOT/'runs'/datetime.now().strftime(f'align-motion-v5-{args.stage}_%Y%m%d_%H%M%S')
    hashes=source_hashes()
    initial,parent=curriculum.initialization(args.init_from,args.stage,hashes)
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(num_timesteps=args.steps,num_envs=args.envs,episode_length=c.SEQUENCE_STEPS if args.stage=='sequence' else c.SINGLE_STEPS,num_evals=11,
        num_eval_envs=64,unroll_length=20,num_minibatches=16,batch_size=256,num_updates_per_batch=4,
        learning_rate=3e-4,discounting=.999,entropy_cost=.01,normalize_observations=True,
        seed=args.seed,deterministic_eval=True,action_repeat=1,max_devices_per_host=1)
    metadata=dict(motion_contract_version=c.MOTION_VERSION,motion_contract_id=c.MOTION_ID,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=hashes,curriculum_stage=args.stage,parent_checkpoint=parent,ppo=cfg,observation_size=83,action_size=8,
        ctrl_dt=c.CONTROL_DT,physics_dt=c.PHYSICS_DT,
        policy_layers=[128,128,128],activation='elu',devices=[str(d) for d in jax.devices()],
        status='training; not evaluated or approved for hardware')
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    logger=None if args.no_wandb else ExperimentLogger(out,metadata,entity=args.wandb_entity,
        project=args.wandb_project,mode=args.wandb_mode)
    if logger:
        logger.run.summary['training_status']='running'
        logger.run.summary['simulation_audit_status']='pending'
        logger.run.summary['policy_video_status']='pending'
        print('W&B:',logger.state['url'] or 'offline; upload with wandb sync',flush=True)
    start=time.monotonic()
    last_step=0;last_video_step=0;last_video=None
    def progress(step,metrics):
        data={k:float(v) for k,v in metrics.items()}
        data.update(step=int(step),seconds=time.monotonic()-start)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        if logger:logger.metrics(step,data)
        print(json.dumps(data),flush=True)
    def video(step,make_policy,params):
        nonlocal last_video_step,last_video
        from .policy_video import record_policy
        path=record_policy(make_policy(params,deterministic=True),out/'videos'/f'policy-{int(step)}.mp4',training_step=step,stage=args.stage)
        if logger:logger.video(path,step)
        last_video_step=int(step);last_video=path
    def save(step,make_policy,params):
        nonlocal last_step
        last_step=int(step)
        path=out/f'params_{step}'
        model.save_params(str(path),params)
        (out/'latest.json').write_text(json.dumps(dict(step=int(step),path=path.name)))
        if args.video_every_steps and int(step)-last_video_step>=args.video_every_steps:
            try:video(step,make_policy,params)
            except Exception as error:
                # Preserve training/checkpoints; retry at the next callback and at completion.
                (out/'video-error.txt').write_text(str(error)+'\n')
                if logger:logger.run.summary['policy_video_status']='render failed; retry pending'
                print('Policy video failed:',error,flush=True)
    print('Training output:',out,flush=True)
    exit_code=1
    try:
        make_policy,params,_=train.train(AlignEnv(training=True,stage=args.stage,noise=args.stage!='foundation'),**cfg,network_factory=network_factory(),
            randomization_fn=curriculum.randomization(args.stage),restore_params=initial,restore_value_fn=False,wrap_env_fn=wrap_for_training,
            progress_fn=progress,policy_params_fn=save)
        model.save_params(str(out/'mjx_params'),params)
        if logger:logger.run.summary['training_status']='completed'
        if last_video is None or last_video_step!=last_step:video(last_step,make_policy,params)
        if args.stage!='sequence':
            from .policy_video import record_policy
            diagnostic=record_policy(make_policy(params,deterministic=True),out/'videos'/f'full-sequence-{last_step}.mp4',training_step=last_step)
            if logger:logger.video(diagnostic,last_step,key='policy/full_sequence_diagnostic',caption='Full-angle diagnostic; intermediate stage checkpoint')
        if logger:
            logger.video(last_video,last_step,key='policy/final',caption=f'Final {args.stage} nominal simulation; checkpoint selection pending')
            logger.audits();logger.artifacts()
            logger.run.summary['handoff_status']='training and video complete; audits pending'
        exit_code=0
        print('Training finished; policy video recorded. Run evaluation before deployment.',flush=True)
    except BaseException:
        if logger:
            logger.run.summary['training_status']='completed; postprocessing incomplete' if (out/'mjx_params').exists() else 'interrupted or failed'
            logger.run.summary['handoff_status']='incomplete; inspect local logs and saved checkpoints'
        raise
    finally:
        if logger:logger.finish(exit_code=exit_code)

if __name__=='__main__':main()
