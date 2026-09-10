"""CPU-only fixed-action feasibility probe; no actor, optimizer or training.

This is one nominal front-left sequence, not a policy or a randomized audit.
It reproduces the passive LIFT stall and checks that v3 permits a full motion.
"""
import json
import numpy as np
import mujoco
from . import configs as c, contract as ct, geometry

# Bounded static pose found by a nominal native-MuJoCo feasibility search.
# This is a diagnostic input, not an open-loop command for the physical robot.
FEASIBLE_FL_ACTION=np.array([.7713135583,-.8988899122,-.9262398848,-.5229962635,
                             .4911208589,.4832556511,-.7127956861,-.0712221591])

def probe(action,seconds=25):
    m=mujoco.MjModel.from_xml_path(str(c.MODEL_PATH));d=mujoco.MjData(m)
    m.actuator_gainprm[ct.POS,0]=5;m.actuator_biasprm[ct.POS,1]=-5
    m.actuator_biasprm[ct.POS,2]=-.25
    mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[ct.POS]=c.DEFAULT_POSE[ct.POS]
    for _ in range(1560):mujoco.mj_step(m,d)
    s=ct.reset(d.qpos[7:].copy(),d.qpos[7+ct.WHEEL].copy())
    phases=np.zeros(6,int);min_rotating=np.full(3,np.inf)
    for _ in range(round(seconds/c.CONTROL_DT)):
        q=d.qpos[7:].copy();qd=d.qvel[6:].copy();rotation=np.empty(9)
        mujoco.mju_quat2Mat(rotation,d.qpos[3:7])
        gravity=rotation.reshape(3,3).T@np.array([0.,0.,-1.])
        s=ct.finish(s,q,qd);s=ct.select(s,1,q);s=ct.prepare(s)
        s,w=ct.begin(s,q,qd,d.qvel[3:6],gravity,action)
        phases[int(s['phase'])]+=1
        if s['was_rotating']:min_rotating=np.minimum(min_rotating,geometry.margins(q,gravity,1))
        for _ in range(10):
            s=ct.integrate(s);d.ctrl[ct.POS]=s['applied'];d.ctrl[ct.WHEEL]=w;mujoco.mj_step(m,d)
    return dict(motion_contract_version=c.MOTION_VERSION,optimizer_steps=0,
        phase_seconds=dict(zip(('idle','lift','rotate','verify','lower','hold'),(phases*c.CONTROL_DT).tolist())),
        completed=s['completed'].tolist(),
        min_rotation_gate_margins_m=min_rotating.tolist() if np.isfinite(min_rotating).all() else None,
        status='Fixed-action nominal CPU diagnostic; not a trained policy or full safety audit')

def main():
    print(json.dumps(dict(zero_action=probe(np.zeros(8)),feasible_fl=probe(FEASIBLE_FL_ACTION)),indent=2))

if __name__=='__main__':main()
