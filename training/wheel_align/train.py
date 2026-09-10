"""PC-only PPO entry. Refuses CPU training and never overwrites an existing run."""
import argparse
from datetime import datetime
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import time
import jax
from brax.io import model
from brax.training.agents.ppo import networks,train
from . import configs as c
from .env import AlignEnv
from .randomize import domain_randomize_wheeled
from .hybrid_wrappers import wrap_for_training

ROOT=Path(__file__).resolve().parents[2]

def network_factory():
    return functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=(128,128,128),
        value_hidden_layer_sizes=(256,256,256),activation=jax.nn.elu)

def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.json'))+[
        c.MODEL_PATH,Path(__file__).with_name('uv.lock')]
    paths+=list(c.MODEL_PATH.with_name('meshes').glob('*.stl'))
    paths+=list((ROOT/'ros2_ws/src/neural_controller/include/neural_controller').glob('wheel_align_*.hpp'))
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=50_000_000)
    p.add_argument('--envs',type=int,default=2048)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--out',type=Path,default=None)
    args=p.parse_args()
    if not any(d.platform=='gpu' for d in jax.devices()):
        raise SystemExit('GPU training only. Use preflight.py for CPU validation; install the cuda extra on Linux/WSL2 on the PC.')
    if args.envs not in (256,512,1024,2048,4096) or args.steps<=0:
        raise SystemExit('--envs must be 256, 512, 1024, 2048 or 4096 (divides the PPO batch); --steps must be positive.')
    out=args.out or ROOT/'runs'/datetime.now().strftime('align-motion-v2_%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    cfg=dict(num_timesteps=args.steps,num_envs=args.envs,episode_length=6656,num_evals=11,
        num_eval_envs=16,unroll_length=20,num_minibatches=16,batch_size=256,num_updates_per_batch=4,
        learning_rate=3e-4,discounting=.99,entropy_cost=.01,normalize_observations=True,
        seed=args.seed,deterministic_eval=True,action_repeat=1,max_devices_per_host=1)
    metadata=dict(motion_contract_version=2,motion_contract_id='quadmorph-align-motion-v2',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes=source_hashes(),ppo=cfg,observation_size=82,action_size=8,
        ctrl_dt=c.CONTROL_DT,physics_dt=c.PHYSICS_DT,
        policy_layers=[128,128,128],activation='elu',devices=[str(d) for d in jax.devices()],
        status='training; not evaluated or approved for hardware')
    (out/'config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    start=time.monotonic()
    def progress(step,metrics):
        data={k:float(v) for k,v in metrics.items()}
        data.update(step=int(step),seconds=time.monotonic()-start)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
        print(json.dumps(data),flush=True)
    def save(step,make_policy,params):
        path=out/f'params_{step}'
        model.save_params(str(path),params)
        (out/'latest.json').write_text(json.dumps(dict(step=int(step),path=path.name)))
    print('Training output:',out,flush=True)
    _,params,_=train.train(AlignEnv(training=True),**cfg,network_factory=network_factory(),
        randomization_fn=domain_randomize_wheeled,wrap_env_fn=wrap_for_training,
        progress_fn=progress,policy_params_fn=save)
    model.save_params(str(out/'mjx_params'),params)
    print('Training finished. Run evaluation before export or deployment.',flush=True)

if __name__=='__main__':main()
