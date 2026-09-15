#!/usr/bin/env python3
"""Export the saved leg-to-wheel actor to RTNeural; verify against Brax."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fixtures',required=True)
    args=parser.parse_args()
    path=Path(args.params); run=json.loads(path.with_name('run.json').read_text()); c=run['config']
    if c['command_states']!=['stand','FL','FR','BR','BL'] or c['observation_history']!=4:
        raise ValueError('Unexpected command/history contract')
    if c['hidden_layer_sizes']!=[128,128,128] or c['activation']!='elu' or not c['ppo']['normalize_observations']:
        raise ValueError('Unexpected network contract')
    params=model.load_params(str(path)); mean=np.asarray(params[0].mean,dtype=np.float64);std=np.asarray(params[0].std,dtype=np.float64)
    if mean.shape!=(140,) or std.shape!=(140,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0):
        raise ValueError('Invalid normalization')
    entries=params[1]['params']; layers=[]
    for i,shape in enumerate(((140,128),(128,128),(128,128),(128,24))):
        entry=entries[f'hidden_{i}'];w=np.asarray(entry['kernel'],dtype=np.float64);b=np.asarray(entry['bias'],dtype=np.float64)
        if w.shape!=shape or b.shape!=(shape[1],):raise ValueError('Unexpected actor shape')
        if i==0:b=b-(mean/std)@w;w=w/std[:,None]
        if i==3:w=w[:,:12];b=b[:12]
        layers.append(dict(type='dense',activation='tanh' if i==3 else 'elu',shape=[None,len(b)],weights=[w.astype(np.float32).tolist(),b.astype(np.float32).tolist()]))
    net=networks.make_ppo_networks(140,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=(128,128,128),activation=jax.nn.elu)
    policy=jax.jit(networks.make_inference_fn(net)(params,deterministic=True))
    rng=np.random.default_rng(915); obs=(mean+std*rng.normal(size=(128,140))).astype(np.float32)
    for row in obs:
        for offset in range(0,140,35):row[offset+6:offset+11]=np.eye(5)[rng.integers(5)]
    expected=np.asarray(jax.vmap(lambda x:policy(x,jax.random.PRNGKey(0))[0])(obs))
    actual=obs.copy()
    for layer in layers:
        w,b=map(lambda x:np.asarray(x,dtype=np.float32),layer['weights']);actual=actual@w+b
        actual=np.tanh(actual) if layer['activation']=='tanh' else np.where(actual>=0,actual,np.expm1(np.minimum(actual,0)))
    error=float(np.max(np.abs(actual-expected)))
    if error>3e-5:raise ValueError(f'Export parity failed: {error}')
    np.savetxt(args.fixtures,np.concatenate([obs,expected],axis=1),delimiter=',',fmt='%.9g')
    data=dict(in_shape=[None,140],layers=layers,behavior='leg_to_wheel',deployment_status='source_bundle_requires_hardware_adapter',
        observation_history=4,single_observation_size=35,observation_history_order='newest_first',observation_history_reset='repeat_first_measured_frame',
        observation_normalization='checkpoint mean/std folded into first layer; no additional normalization or clipping',
        observation_layout=['body_angular_velocity[3]','projected_gravity[3]','command_one_hot[5]','joint_position_minus_home[12]','last_raw_policy_action[12]'],
        command_states=c['command_states'],command_to_foot=[-1,1,0,2,3],leg_order=['FR','FL','BR','BL'],
        joint_names=[f'leg_{leg}_{joint}' for leg in ('front_r','front_l','back_r','back_l') for joint in (1,2,3)],
        action_types=['position']*12,action_scale=c['action_scale'],default_joint_pos=run['home_joint_pos'],kps=run['kp'],kds=run['kd'],
        joint_lower_limits=run['joint_lower_limits'],joint_upper_limits=run['joint_upper_limits'],ctrl_dt=c['ctrl_dt'],
        lowering=dict(filter='hip_descent_v1',initial_hip_speed_rad_s=8.,ease_seconds=.2,clearance_fade_m=[.02,.04],last_action='raw policy output; previous actuator target stored separately'),
        sequencing=dict(clearance_gate_m=.015,settle_seconds=.3,heating_confirmation='explicit operator input',hold_timeout=None,converted_bits=['FR','FL','BR','BL']),
        provenance=dict(checkpoint_sha256=sha(path),run_sha256=sha(path.with_name('run.json')),model_sha256=run['model_sha256'],ring_sha256=run['ring_sha256'],exporter_sha256=sha(__file__)),
        parity=dict(fixtures=128,numpy_max_abs_action_error=error,tolerance=3e-5))
    Path(args.output).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(export_sha256=sha(args.output),**data['parity'])))

if __name__=='__main__':main()
