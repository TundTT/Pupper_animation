"""REJECTED experimental settling controller with only ideal joint/IMU inputs.

No true contact force, actual leg geometry or ground-truth base position is read.
The gravity-corrected PD/Jacobian load estimate is NOT a foot-force measurement.
This prototype uses nominal MuJoCo kinematics; it is not a deployed Pi executor.
"""
import mujoco as mj
import numpy as np
from .core import Robot, KP, KD, PROX, LEGS

class SettlingFeedback:
    def __init__(self):
        self.nominal=Robot()
        self.extension=np.zeros(4)
        self.offset=np.zeros(12)
        self.estimated_load_N=np.zeros(4)
        self.max_offset_rad=0.

    def update(self, quaternion_wxyz, q, qd, previous_command, dt):
        r=self.nominal
        r.d.qpos[:3]=[0,0,1.]  # Arbitrary kinematic origin, away from the floor.
        r.d.qpos[3:7]=quaternion_wxyz;r.d.qpos[7:]=q;r.d.qvel[:]=0
        mj.mj_forward(r.m,r.d)
        R=r.d.xmat[r.base].reshape(3,3)
        roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arcsin(np.clip(-R[2,0],-1,1))
        tau=KP*(previous_command-q)-KD*qd
        for i,leg in enumerate(LEGS):
            sid=r.m.site(f'leg_{leg}_3_foot_site').id
            jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac)
            mj.mj_jacSite(r.m,r.d,jac,rot,sid)
            joints=np.array([3*i,3*i+1]);J=jac[2,6+joints]
            denominator=J@J
            if denominator<1e-5:continue
            external=r.d.qfrc_bias[6+joints]-tau[joints]
            estimate=float(J@external/denominator)
            self.estimated_load_N[i]=estimate
            # Reach down only when the model-based load estimate is small.
            deficit=np.clip(3.-estimate,0.,6.)
            self.extension[i]=np.clip(self.extension[i]-.0005*deficit*dt,-.012,0.)
            foot_body=R.T@(r.d.site_xpos[sid]-r.d.xpos[r.base])
            levelling=.7*(roll*foot_body[1]-pitch*foot_body[0])
            vertical=self.extension[i]+np.clip(levelling,-.006,.006)
            desired=J*vertical/denominator
            # Small departures from the walking target; never move the hubs.
            desired=np.clip(desired,-.08,.08)
            self.offset[joints]+=(1-np.exp(-dt/.25))*(desired-self.offset[joints])
        self.max_offset_rad=max(self.max_offset_rad,float(abs(self.offset).max()))
        return self.offset.copy()
