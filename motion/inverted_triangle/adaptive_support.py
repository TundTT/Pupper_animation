"""Bounded experimental support adaptation from nominal CAD and joint/IMU data.

No actual formation, contacts, base position or privileged dynamics enter this
controller. PD/Jacobian force estimates are model estimates, not foot sensors.
"""
import numpy as np
import mujoco as mj
from .core import Robot,KP,KD,PROX

class AdaptiveSupport:
    def __init__(self,goal,config=None):
        self.config=dict(reference_cap=.09,command_cap=.20,force_target_N=4.,force_gain=.008,integral_gain=.35,estimate_time_s=.25,regularization=1e-6)
        self.config.update(config or {});self.nominal=Robot();self.goal=np.array(goal)
        self.reference=self.goal.copy();self.integral=np.zeros(12);self.estimated_load_N=np.zeros(4)
        self.max_offset_rad=0.;self.extension=np.zeros(4)
    def update(self,quaternion_wxyz,q,qd,previous_command,dt):
        r=self.nominal;r.d.qpos[:3]=[0,0,1];r.d.qpos[3:7]=quaternion_wxyz;r.d.qpos[7:]=q;r.d.qvel[:]=0;mj.mj_forward(r.m,r.d)
        tau=KP*(previous_command-q)-KD*qd
        for i in range(4):
            points=r.points(i)[r.tip_masks[i]];point=points[np.argmin(points[:,2])]
            jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac);mj.mj_jac(r.m,r.d,jac,rot,point,r.bodies[i])
            joints=np.arange(3*i,3*i+3);J=jac[:,6+joints]
            external=r.d.qfrc_bias[6+joints]-tau[joints]
            force=np.linalg.solve(J@J.T+np.eye(3)*self.config['regularization'],J@external)
            alpha=1-np.exp(-dt/self.config['estimate_time_s']);self.estimated_load_N[i]+=alpha*(force[2]-self.estimated_load_N[i])
            error=np.clip(self.config['force_target_N']-self.estimated_load_N[i],-6.,6.)
            z=J[2,:2];norm=max(np.linalg.norm(z),.01)
            self.reference[joints[:2]]-=self.config['force_gain']*error*z/norm*dt
        cap=self.config['reference_cap'];self.reference=np.clip(self.reference,self.goal-cap,self.goal+cap)
        self.reference[[2,5,8,11]]=self.goal[[2,5,8,11]]
        self.integral+=self.config['integral_gain']*(self.reference-q)*dt
        self.integral=np.clip(self.integral,-.15,.15)
        command=self.reference+self.integral
        offset=np.clip(command-self.goal,-self.config['command_cap'],self.config['command_cap'])
        self.max_offset_rad=max(self.max_offset_rad,float(np.abs(offset).max()))
        return offset
