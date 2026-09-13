"""Record integrated dynamics, then independently replay commands for dense CAD.

CAD transforms are evaluated on reproduced physics states, never on a planned
pose path. Rendering/snapshot transforms do not constitute dynamic acceptance.
"""
import json
from pathlib import Path
import numpy as np
import mujoco as mj
from .core import Robot
from .clearance import CADClearance

class RollRecorder:
    def __init__(self,r,command):
        self.start=r.snapshot(command);self.commands=[];self.qpos=[];self.qvel=[];self.loads=[]
        self.max_slip=np.zeros(4);self.max_normal_speed=np.zeros(4);self.impulse=np.zeros(4)
        self.contact_impulse_peak=np.zeros(4);self.floor_peak=0.;self.tilt_peak=0.
    def add(self,r,command):
        self.commands.append(np.array(command));self.qpos.append(r.d.qpos.copy());self.qvel.append(r.d.qvel.copy())
        loads=r.contacts_precise();self.loads.append(loads);self.impulse+=loads/520
        self.contact_impulse_peak=np.maximum(self.contact_impulse_peak,loads/520)
        self.floor_peak=max(self.floor_peak,r.unintended_floor_force());self.tilt_peak=max(self.tilt_peak,float(np.rad2deg(r.tilt())))
        for k in range(r.d.ncon):
            c=r.d.contact[k]
            if r.floor not in (c.geom1,c.geom2):continue
            g=c.geom2 if c.geom1==r.floor else c.geom1;ids=np.flatnonzero(r.geoms==g)
            if not ids.size:continue
            i=ids[0];jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac)
            mj.mj_jac(r.m,r.d,jac,rot,c.pos,r.bodies[i]);v=jac@r.d.qvel
            self.max_slip[i]=max(self.max_slip[i],np.linalg.norm(v[:2]));self.max_normal_speed[i]=max(self.max_normal_speed[i],abs(v[2]))
    def finish(self,r,output,cad=True,kp=None,kd=None):
        output=Path(output);np.savez_compressed(output/'integration.npz',commands=self.commands,qpos=self.qpos,qvel=self.qvel,loads=self.loads)
        (output/'initial_integration_state.json').write_text(json.dumps(self.start))
        result=dict(physics_hz=520,contact_kinematics='MuJoCo step contact points and post-step velocities',normal_impulse_Ns=self.impulse.tolist(),peak_step_normal_impulse_Ns=self.contact_impulse_peak.tolist(),max_contact_slip_m_s=self.max_slip.tolist(),max_contact_normal_speed_m_s=self.max_normal_speed.tolist(),max_floor_force_N=self.floor_peak,max_tilt_deg=self.tilt_peak)
        if not cad:return result
        clone=Robot(r.friction,r.dynamics,r.manifest.get('formation'),contact_model='rigid_flush' if r.manifest.get('contact_reference') else 'cad');clone.restore(self.start)
        geometry=Robot(r.friction,r.dynamics,r.manifest.get('formation'),contact_model='rigid_flush' if r.manifest.get('contact_reference') else 'cad')
        checker=CADClearance(geometry);loads=np.asarray(self.loads);changes=np.flatnonzero(np.any(np.diff(loads>1,axis=0),axis=1))+1
        chosen=set(range(0,len(self.commands),10))
        for k in changes:chosen.update(range(max(0,k-26),min(len(self.commands),k+27)))
        records=[];max_error=0.;options={} if kp is None else dict(kp=kp,kd=kd)
        # Reintegrate ALL steps, inspect at 52 Hz and 520 Hz around contact changes.
        for k,command in enumerate(self.commands):
            clone.tick(command,**options)
            error=max(float(abs(clone.d.qpos-self.qpos[k]).max()),float(abs(clone.d.qvel-self.qvel[k]).max()));max_error=max(error,max_error)
            if k in chosen:
                geometry.d.qpos[:]=clone.d.qpos;geometry.d.qvel[:]=clone.d.qvel;mj.mj_forward(geometry.m,geometry.d)
                records.append(dict(step=k,**checker.measure()))
        if max_error>1e-9:raise ValueError(f'Dynamic command replay diverged: {max_error}')
        # Refine the ten smallest gaps at every physics step +/-50 ms. This is
        # an independent geometric evaluation of already verified replay states.
        dense=set()
        for row in sorted(records,key=lambda x:x['minimum_m'])[:10]:dense.update(range(max(0,row['step']-26),min(len(self.commands),row['step']+27)))
        for k in sorted(dense-chosen):
            geometry.d.qpos[:]=self.qpos[k];geometry.d.qvel[:]=self.qvel[k];mj.mj_forward(geometry.m,geometry.d)
            records.append(dict(step=k,**checker.measure()))
        minimum=min(records,key=lambda x:x['minimum_m']);hits=[x for x in records if x['intersections']]
        result.update(replay_max_state_error=max_error,cad_samples=len(records),cad_regular_period_s=10/520,contact_changes=changes.tolist(),cad_minimum=minimum,first_cad_intersection=hits[0] if hits else None,cad_pass=minimum['minimum_m']>=.001 and not hits,coverage='All shin/shin, shin/motor, shin/body and shin/backpack CAD pairs; adjacent fixed assembly contacts excluded')
        (output/'dense_cad_samples.json').write_text(json.dumps(sorted(records,key=lambda x:x['step'])));return result
