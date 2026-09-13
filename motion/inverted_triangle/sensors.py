"""Explicit joint/IMU sensor sensitivity; no contacts or shape parameters output."""
from collections import deque
import mujoco as mj
import numpy as np

class JointImuSensors:
    def __init__(self,spec=None,seed=0):
        self.spec=spec or {};self.rng=np.random.default_rng(seed)
        allowed={'joint_bias_rad','joint_noise_rad','velocity_noise_rad_s','orientation_bias_rad','gyro_bias_rad_s','delay_steps'}
        if set(self.spec)-allowed:raise ValueError('Unknown sensor field')
        delay=self.spec.get('delay_steps',0)
        if not isinstance(delay,int) or delay<0:raise ValueError('Invalid sensor delay')
        self.delay=delay;self.queue=deque()
    def read(self,r):
        R=r.d.xmat[r.base].reshape(3,3)
        sample=dict(quaternion=r.d.xquat[r.base].copy(),q=r.d.qpos[7:].copy(),qd=r.d.qvel[6:].copy(),omega=R.T@r.d.cvel[r.base,:3])
        self.queue.append(sample)
        while len(self.queue)>self.delay+1:self.queue.popleft()
        s={k:v.copy() for k,v in self.queue[0].items()}
        s['q']+=np.asarray(self.spec.get('joint_bias_rad',np.zeros(12)))+self.rng.normal(0,self.spec.get('joint_noise_rad',0),12)
        s['qd']+=self.rng.normal(0,self.spec.get('velocity_noise_rad_s',0),12)
        s['omega']+=self.spec.get('gyro_bias_rad_s',[0,0,0])
        roll,pitch=self.spec.get('orientation_bias_rad',[0,0]);bias=np.array([np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)])
        q=np.empty(4);mj.mju_mulQuat(q,s['quaternion'],bias);s['quaternion']=q
        mat=np.empty(9);mj.mju_quat2Mat(mat,q);s['gravity']=mat.reshape(3,3).T@np.array([0,0,-1.])
        if not all(np.isfinite(v).all() for v in s.values()):raise ValueError('Nonfinite sensor sample')
        return s
