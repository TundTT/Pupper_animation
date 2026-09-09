"""Choose a wider encoder guard from recorded poses and stress physical clearance.

No trained weights or runtime privileged inputs are changed. Model geometry is
used offline to test the clearance of newly admitted poses. This empirical test
is deliberately separate from the encoder/IMU-only runtime guard.
"""
import json
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
MODEL = ROOT.parent.parent/'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'


def collect():
    rows=[];positions=[];legs=[]
    for name in ['audit64','nominal64','slew05_audit64']:
        with np.load(ROOT/'hybrid_results/attempt1_corrected'/(name+'.npz')) as z:
            d={k:z[k] for k in ['qpos','phase','done','clearance','tilt']}
        for stage in range(4):
            lo=100+1100*stage;hi=lo+1000
            for r in range(64):
                leg=([1,0,2,3] if r%2==0 else [0,1,3,2])[stage]
                q=d['qpos'][lo:hi,r];phase=d['phase'][lo:hi,r]
                alive=~np.maximum.accumulate(d['done'][:hi,r].astype(bool))[lo:hi]
                mask=(phase>=1)&(phase<=3)&alive
                hip=q[:,8+3*leg]*(1 if leg%2==0 else -1)
                abd=np.abs(q[:,7+3*leg]-(1 if leg%2==0 else -1))
                rows.append(np.stack([hip,abd,d['tilt'][lo:hi,r],d['clearance'][lo:hi,r],mask],axis=1))
                positions.append(q);legs.append(np.full(len(q),leg))
    return np.concatenate(rows),np.concatenate(positions),np.concatenate(legs)


def main():
    a,q,legs=collect();base=(a[:,0]>1.1)&(a[:,2]<.12)&(a[:,4]>0)
    candidates=[]
    for margin in [.25,.30,.35,.40,.45,.50]:
        sel=base&(a[:,1]<margin);new=sel&(a[:,1]>=.25)
        clear=a[sel,3];new_clear=a[new,3]
        candidates.append(dict(abduction_guard_rad=margin,admitted_samples=int(sel.sum()),new_samples=int(new.sum()),
                               unsafe_samples=int((clear<.005).sum()),min_clearance_m=float(clear.min()),
                               new_min_clearance_m=float(new_clear.min()) if len(new_clear) else None))
    chosen=.35
    new_idx=np.flatnonzero(base&(a[:,1]>=.25)&(a[:,1]<chosen))
    rng=np.random.default_rng(20260909)
    # Cover every newly admitted sample, including the minimum-clearance tail.
    m=mujoco.MjModel.from_xml_path(str(MODEL));d=mujoco.MjData(m)
    gids=np.array([m.geom('leg_'+leg+'_3_wheel_collision').id for leg in ['front_r','front_l','back_r','back_l']])
    nominal_radius=m.geom_size[gids,0].copy()
    m.geom_size[gids,0]=nominal_radius+.0025
    quaternions=np.empty((len(new_idx),4));temp=np.empty(4)
    minima=np.full(4,np.inf);unsafe=0
    for index in new_idx:
        pose=q[index].astype(float).copy()
        # Lower torso by a full 10 mm; use the largest wheel radius. Encoder
        # perturbations cover ±5 mrad; gravity-direction error ±10 mrad/axis.
        pose[2]-=.010
        pose[7:]+=rng.uniform(-.005,.005,12)
        roll,pitch=rng.uniform(-.010,.010,2)
        qr=np.array([np.cos(roll/2),np.sin(roll/2),0.,0.])
        qp=np.array([np.cos(pitch/2),0.,np.sin(pitch/2),0.])
        mujoco.mju_mulQuat(temp,qr,qp)
        perturbed=np.empty(4);mujoco.mju_mulQuat(perturbed,temp,pose[3:7]);pose[3:7]=perturbed
        d.qpos[:]=pose;mujoco.mj_forward(m,d)
        leg=legs[index];gid=gids[leg];az=d.geom_xmat[gid].reshape(3,3)[2,2]
        extent=m.geom_size[gid,0]*np.sqrt(max(1-az*az,0))+m.geom_size[gid,1]*abs(az)
        clear=float(d.geom_xpos[gid,2]-extent)
        minima[leg]=min(minima[leg],clear);unsafe+=int(clear<.005)
    result=dict(chosen_guard_rad=chosen,previous_guard_rad=.25,physical_margin_m=.005,
                candidates=candidates,stress_samples=len(new_idx),stress_unsafe_samples=unsafe,
                stress_min_clearance_by_leg_m=dict(zip(['front_r','front_l','back_r','back_l'],minima.tolist())),
                perturbations=dict(torso_drop_m=.010,wheel_radius_increase_m=.0025,joint_angle_noise_rad=.005,roll_pitch_noise_rad=.010),
                scope='Empirical offline validation of newly admitted recorded poses; all four legs. Not a guarantee outside the tested distribution. Dynamic rollout must also pass.')
    out=ROOT/'hybrid_results/attempt2_guard/guard_validation.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    assert unsafe==0,'Wider guard rejected: physical clearance margin violated under stress.'

if __name__=='__main__':main()
