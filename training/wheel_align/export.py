"""Export a trained v4 checkpoint and RTNeural parity fixtures. Never overwrite v1."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
# The PC agent found TF32 made the float32 export parity check fail on its GPU.
# Keep this confined to the exporter process; training precision is unchanged.
jax.config.update('jax_default_matmul_precision', 'highest')
from jax import numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from . import configs as c,contract as ct
from .train import source_hashes,network_factory

BLOCKS=[('body_angular_velocity',3),('projected_gravity',3),('effective_command_one_hot',5),
    ('joint_position',12),('joint_velocity',12),('last_action',8),('target_error_sin',4),('target_error_cos',4),
    ('phase',6),('progress',1),('motion_reference',8),('applied_position',8),('applied_velocity',8)]

def payload_from_params(params,config):
    mean,std=(np.asarray(x,dtype=np.float64) for x in (params[0].mean,params[0].std))
    if mean.shape!=(82,) or std.shape!=(82,) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or not np.all(std>0):
        raise ValueError('Invalid observation normalization')
    actor=params[1]['params'];layers=[]
    for i,shape in enumerate(((82,128),(128,128),(128,128),(128,16))):
        layer=actor[f'hidden_{i}'];kernel=np.asarray(layer['kernel'],dtype=np.float64);bias=np.asarray(layer['bias'],dtype=np.float64)
        if kernel.shape!=shape or bias.shape!=(shape[1],):raise ValueError('Unexpected actor architecture')
        if i==0:bias=bias-(mean/std)@kernel;kernel=kernel/std[:,None]
        if i==3:kernel,bias=kernel[:,:8],bias[:8]
        layers.append(dict(type='dense',activation='tanh' if i==3 else 'elu',shape=[None,len(bias)],
            weights=[kernel.astype(np.float32).tolist(),bias.astype(np.float32).tolist()]))
    layout=[];offset=0
    for name,size in BLOCKS:layout.append(dict(name=name,offset=offset,size=size));offset+=size
    low=np.full(12,-2.);high=np.full(12,2.);low[ct.POS]=ct.LOW;high[ct.POS]=ct.HIGH
    return dict(in_shape=[None,82],layers=layers,behavior='wheel_align_hybrid',motion_contract_version=c.MOTION_VERSION,
        motion_contract_id=c.MOTION_ID,observation_history=1,single_observation_size=82,
        observation_layout=layout,observation_clip=None,policy_action_size=8,
        position_joint_rows=c.POSITION_ACTUATOR_ROWS,wheel_joint_rows=c.WHEEL_ACTUATOR_ROWS,
        command_states=c.COMMAND_STATES,command_leg=ct.COMMAND_LEG.tolist(),joint_names=list(c.JOINT_NAMES),
        action_types=['velocity' if i in ct.WHEEL else 'position' for i in range(12)],
        default_joint_pos=c.DEFAULT_POSE.tolist(),action_scale=[.2,.3,0.]*4,
        action_semantics='bounded residual about phase reference; see motion contract v4',
        joint_lower_limits=low.tolist(),joint_upper_limits=high.tolist(),ctrl_dt=c.CONTROL_DT,
        kps=[5.,5.,0.]*4,kds=[.25,.25,.35]*4,
        source_commit=config['source_commit'],source_hashes=config['source_hashes'],
        status='Exported checkpoint; hardware validation required')

def numpy_inference(payload,x):
    for layer in payload['layers']:
        w,b=(np.asarray(v,dtype=np.float32) for v in layer['weights']);x=x@w+b
        x=np.tanh(x) if layer['activation']=='tanh' else np.where(x>0,x,np.expm1(np.minimum(x,0)))
    return x

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--params',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise SystemExit('Refusing to overwrite existing output.')
    config=json.loads(args.params.parent.joinpath('config.json').read_text())
    if config['motion_contract_version']!=c.MOTION_VERSION or config['source_hashes']!=source_hashes():raise SystemExit('Checkout does not match training provenance.')
    params=model.load_params(str(args.params));payload=payload_from_params(params,config)
    payload['checkpoint_sha256']=hashlib.sha256(args.params.read_bytes()).hexdigest()
    net=network_factory()(82,8,preprocess_observations_fn=running_statistics.normalize)
    infer=networks.make_inference_fn(net)(params,deterministic=True)
    x=np.random.default_rng(42).normal(0,.3,(128,82)).astype(np.float32)
    expected=np.asarray(infer(jp.asarray(x),jax.random.PRNGKey(0))[0])
    actual=numpy_inference(payload,x);error=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(error) or error>3e-5:raise ValueError(f'Export parity failed: {error}')
    payload['numpy_brax_max_action_error']=error
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    np.savetxt(args.out.with_suffix('.reference.csv'),np.c_[x,expected],delimiter=',',fmt='%.9g')
    print(json.dumps(dict(output=str(args.out),sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),max_action_error=error)))

if __name__=='__main__':main()
