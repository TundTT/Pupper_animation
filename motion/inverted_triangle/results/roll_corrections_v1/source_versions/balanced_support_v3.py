"""Bounded relative support adaptation from nominal geometry and joint/IMU data.

The vertical PD/Jacobian projection is only an estimated load. Equalizing these
estimates avoids requiring an accurate payload mass or calibrated absolute force.
No simulator contacts, actual shape, base position or dynamics enter update().
"""
import numpy as np
import mujoco as mj
from .adaptive_support import AdaptiveSupport
from .core import KP,KD

class BalancedSupport(AdaptiveSupport):
    def __init__(self,goal,config=None):
        settings=dict(reference_cap=.085,command_cap=.2,force_gain=.008,integral_gain=.5,estimate_time_s=.4,relative_load=True)
        settings.update(config or {});super().__init__(goal,settings)
    def update(self,quaternion_wxyz,q,qd,previous_command,dt):
        r=self.nominal;r.d.qpos[:3]=[0,0,1];r.d.qpos[3:7]=quaternion_wxyz;r.d.qpos[7:]=q;r.d.qvel[:]=0;mj.mj_forward(r.m,r.d)
        tau=KP*(previous_command-q)-KD*qd;jacobians=[]
        for i in range(4):
            points=r.points(i)[r.tip_masks[i]];point=points[np.argmin(points[:,2])]
            jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac);mj.mj_jac(r.m,r.d,jac,rot,point,r.bodies[i])
            joints=np.arange(3*i,3*i+3);z=jac[2,6+joints];jacobians.append(z)
            external=r.d.qfrc_bias[6+joints]-tau[joints]
            load=z@external/(z@z+1e-8)
            self.estimated_load_N[i]+=(1-np.exp(-dt/self.config['estimate_time_s']))*(load-self.estimated_load_N[i])
        desired=float(np.clip(self.estimated_load_N,0,30).mean()) if self.config['relative_load'] else self.config['force_target_N']
        for i,z in enumerate(jacobians):
            error=np.clip(desired-self.estimated_load_N[i],-6.,6.)
            self.reference[3*i:3*i+3]-=self.config['force_gain']*error*z/max(np.linalg.norm(z),.01)*dt
        cap=self.config['reference_cap'];self.reference=np.clip(self.reference,self.goal-cap,self.goal+cap)
        self.integral=np.clip(self.integral+self.config['integral_gain']*(self.reference-q)*dt,-.15,.15)
        offset=np.clip(self.reference+self.integral-self.goal,-self.config['command_cap'],self.config['command_cap'])
        self.max_offset_rad=max(self.max_offset_rad,float(abs(offset).max()))
        return offset
