"""Native CPU motion audit with zero policy residual. No training or optimizer."""
import json
import numpy as np
import mujoco
from . import configs as c, contract as ct, geometry

def probe(commands=(1,2,3,4),interrupt=False):
    m=mujoco.MjModel.from_xml_path(str(c.MODEL_PATH));d=mujoco.MjData(m)
    m.actuator_gainprm[ct.POS,0]=5;m.actuator_biasprm[ct.POS,1]=-5;m.actuator_biasprm[ct.POS,2]=-.25
    ids=[m.geom(n).id for n in c.WHEEL_COLLISION_GEOM_NAMES]
    def clearance():
        z=d.geom_xmat[ids].reshape(4,3,3)[:,2,2]
        return d.geom_xpos[ids,2]-.048*np.sqrt(np.maximum(1-z*z,0))-.01675*np.abs(z)
    mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[ct.POS]=ct.NEUTRAL
    for _ in range(1560):mujoco.mj_step(m,d)
    s=ct.reset(d.qpos[7:].copy(),d.qpos[7+ct.WHEEL].copy());reports=[]
    for requested in commands:
        k=int(ct.COMMAND_LEG[requested]);phases=np.zeros(6,int)
        gap_min=1.;body_min=1.;peak=0.;rot_min=np.ones(3);peak_phase=0;elapsed=0.
        for n in range(c.SINGLE_STEPS):
            q=d.qpos[7:].copy();qd=d.qvel[6:].copy();rotation=np.empty(9)
            mujoco.mju_quat2Mat(rotation,d.qpos[3:7]);g=rotation.reshape(3,3).T@np.array([0.,0.,-1.])
            s=ct.finish(s,q,qd)
            command=0 if n<104 or (interrupt and 520<=n<624) else requested
            s=ct.select(s,command,q);s=ct.prepare(s)
            s,w=ct.begin(s,q,qd,d.qvel[3:6],g,np.zeros(8));phases[int(s['phase'])]+=1
            if s['was_rotating']:rot_min=np.minimum(rot_min,geometry.margins(q,g,k))
            for _ in range(10):
                previous=clearance();s=ct.integrate(s);d.ctrl[ct.POS]=s['applied'];d.ctrl[ct.WHEEL]=w
                mujoco.mj_step(m,d);clear=clearance()
                impact=float(np.max(np.where(clear<.008,np.maximum(-(clear-previous)/c.PHYSICS_DT,0),0)))
                if impact>peak:peak=impact;peak_phase=int(s['phase'])
                mujoco.mju_quat2Mat(rotation,d.qpos[3:7]);g=rotation.reshape(3,3).T@np.array([0.,0.,-1.])
                _,gap,body=geometry.margins(d.qpos[7:],g,k);gap_min=min(gap_min,gap);body_min=min(body_min,body)
            elapsed=(n+1)*c.CONTROL_DT
            if s['completed'][k]:break
        reports.append(dict(leg=c.LEGS[k],completed=bool(s['completed'][k]),seconds=elapsed,
            phase_seconds=(phases*c.CONTROL_DT).tolist(),min_wheel_gap=float(gap_min),min_body_gap=float(body_min),
            peak_impact=peak,peak_impact_phase=peak_phase,min_rotation_margins=rot_min.tolist()))
    return dict(motion_contract_version=c.MOTION_VERSION,policy_optimizer_updates=0,interrupt=interrupt,legs=reports,
        completed=s['completed'].tolist(),status='Zero-residual nominal CPU diagnostic; not a randomized or hardware audit')

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interrupt',action='store_true');args=p.parse_args()
    print(json.dumps(probe(interrupt=args.interrupt),indent=2))

if __name__=='__main__':main()
