"""Pinned float32 walking inference and runtime history; simulation only.

No ROS/controller activation. Matches the selected export and runtime's initial
2 s measured-pose ramp, 2 s action fade and 520/10 Hz scheduling.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .core import HERE

POLICY=HERE.parents[1]/'ros2_ws/src/neural_controller/launch/policy_walk_v2.json'
EXPORT_SHA256='854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8'

class WalkingPolicy:
    def __init__(self,initial_q,init_seconds=2.,fade_seconds=2.):
        if hashlib.sha256(POLICY.read_bytes()).hexdigest()!=EXPORT_SHA256:raise ValueError('Walking export mismatch')
        self.spec=json.loads(POLICY.read_text());self.layers=[]
        for layer in self.spec['layers']:
            if layer['type']!='dense':raise ValueError('Unsupported export layer')
            w,b=layer['weights'];self.layers.append((np.array(w,dtype=np.float32),np.array(b,dtype=np.float32),layer['activation']))
        self.home=np.array(self.spec['default_joint_pos']);self.scale=np.array(self.spec['action_scale'])
        self.initial_q=np.array(initial_q);self.init_seconds=init_seconds;self.fade_seconds=fade_seconds
        self.history=None;self.last_action=np.zeros(12,dtype=np.float32);self.last_target=self.initial_q.copy();self.last_observation=None
    def infer(self,observation):
        x=np.asarray(observation,dtype=np.float32)
        for w,b,activation in self.layers:
            x=x@w+b
            if activation=='elu':x=np.where(x>0,x,np.expm1(np.minimum(x,0)))
            elif activation=='tanh':x=np.tanh(x)
            else:raise ValueError('Unsupported activation')
        return x
    def target(self,elapsed,body_omega,gravity,q,command):
        if elapsed<self.init_seconds:
            u=elapsed/self.init_seconds;return self.initial_q*(1-u)+self.home*u
        frame=np.r_[body_omega,gravity,np.clip(command,self.spec['command_low'],self.spec['command_high']),[0,0,1],np.asarray(q)-self.home,self.last_action].astype(np.float32)
        frame=np.clip(frame,-100,100)
        if self.history is None:self.history=np.tile(frame,(4,1))
        else:self.history=np.vstack([frame,self.history[:3]])
        self.last_observation=self.history.reshape(-1).copy()
        action=self.infer(self.last_observation)
        fade=min((elapsed-self.init_seconds)/self.fade_seconds,1.) if self.fade_seconds else 1.
        self.last_action=fade*action
        self.last_target=np.clip(self.home+self.last_action*self.scale,self.spec['joint_lower_limits'],self.spec['joint_upper_limits'])
        return self.last_target.copy()
