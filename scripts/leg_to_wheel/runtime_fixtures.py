#!/usr/bin/env python3
"""Generate C++ runtime references using the JAX actor and original Python limiter."""
import argparse
import importlib.util
import json
from pathlib import Path
import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',required=True);p.add_argument('--motion-source',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();run=json.loads(Path(args.params).with_name('run.json').read_text());c=run['config']
    spec=importlib.util.spec_from_file_location('original_motion',args.motion_source);motion=importlib.util.module_from_spec(spec);spec.loader.exec_module(motion)
    net=networks.make_ppo_networks(140,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=(128,128,128),activation=jax.nn.elu)
    policy=jax.jit(networks.make_inference_fn(net)(model.load_params(args.params),deterministic=True))
    home=np.asarray(run['home_joint_pos']);scale=np.asarray(c['action_scale']);lower=np.asarray(run['joint_lower_limits']);upper=np.asarray(run['joint_upper_limits'])
    history=None;previous=np.zeros(12);previous_target=np.zeros(12);previous_command=0;foot=-1;elapsed=0.;rows=[]
    commands=[0]*16+[cmd for leg in (1,2,3,4) for cmd in ([leg]*20+[0]*25)]
    for step,command in enumerate(commands):
        if command:foot=-1;elapsed=0.
        elif previous_command:foot=(-1,1,0,2,3)[previous_command];elapsed=0.
        omega=.1*np.sin(step*.13+np.arange(3));gravity=np.array([.04*np.sin(step*.03),.03*np.cos(step*.05),-1.]);gravity/=np.linalg.norm(gravity)
        q=home+.12*np.sin(step*.09+np.arange(12)*.3)
        clearance=np.full(4,.055)
        if foot>=0:clearance[foot]=max(-.002,.06-.3*elapsed)
        frame=np.concatenate([omega,gravity,np.eye(5)[command],q-home,previous]).astype(np.float32)
        history=np.tile(frame,4) if history is None else np.concatenate([frame,history[:-35]])
        action=np.asarray(policy(history,jax.random.PRNGKey(0))[0]);phase=min(elapsed/.2,1.)
        strength=(1-3*phase**2+2*phase**3)*(np.clip((clearance[foot]-.02)/.02,0,1) if foot>=0 else 1.)
        applied=motion.limit_lowering_targets(action,previous_target,foot,.02,scale,8.,strength)
        target=np.clip(home+applied*scale,lower,upper)
        rows.append(np.concatenate([omega,gravity,q,[command],clearance,history,action,applied,target]))
        previous=action;previous_target=applied;previous_command=command
        if foot>=0:elapsed+=.02
    np.savetxt(args.out,rows,delimiter=',',fmt='%.17g')
    print('Saved',len(rows),'sequential JAX/Python runtime fixtures')

if __name__=='__main__':main()
