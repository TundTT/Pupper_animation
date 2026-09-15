#!/usr/bin/env python3
"""CPU/native MuJoCo diagnostic of actual trained actor and C++ alignment core.
No hardware/ROS. Reports every failed case without changing acceptance thresholds.
"""
import argparse,ctypes,json,hashlib
from pathlib import Path
import mujoco,numpy as np

def main():
 assert mujoco.__version__=='3.6.0','Use pinned native MuJoCo3.6.0 for these fixtures/diagnostics'
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 lib=ctypes.CDLL(str(a.library.resolve()));ptr=ctypes.c_void_p;arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
 lib.align_create.argtypes=[ctypes.c_char_p];lib.align_create.restype=ptr;lib.align_destroy.argtypes=[ptr];lib.align_error.argtypes=[ptr];lib.align_error.restype=ctypes.c_char_p
 lib.align_reset.argtypes=[ptr,arr,arr,arr,arr,arr];lib.align_tick.argtypes=[ptr,arr,arr,arr,arr,ctypes.c_int,arr];lib.align_status.argtypes=[ptr,arr]
 m=mujoco.MjModel.from_xml_path(str(a.model));assert abs(m.opt.timestep-1/520)<1e-12
 assert hashlib.sha256(a.model.read_bytes()).hexdigest()=='4275f813ad1b6b1763ec42f70f23e871114724b3a51baa238c0b4efaafa588cb'
 wheels=[m.geom('leg_'+k+'_3_wheel_collision').id for k in ['front_r','front_l','back_r','back_l']];floor=m.geom('floor').id
 cases=[]
 for k in range(4):
  d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
  for _ in range(1560):mujoco.mj_step(m,d)
  def sense():return np.ascontiguousarray(d.qpos[7:]),np.ascontiguousarray(d.qvel[6:]),np.ascontiguousarray(-d.xmat[1].reshape(3,3)[2,:]),np.ascontiguousarray(d.qvel[3:6])
  q,qd,g,w=sense();homes=np.ascontiguousarray(q[2::3]);n=lib.align_create(str(a.policy).encode());assert n
  status=np.zeros(51);command=np.zeros(12);trace=[];failure=[];rotated=False;moving_steps=unsafe_steps=0;max_gate=-1e9;verified_time=None;lower_sent=False;reason='duration_limit';maxangle=0
  if lib.align_reset(n,q,qd,g,w,homes):reason=lib.align_error(n).decode();failure.append(reason)
  else:
   for step in range(75*520):
    t=step/520;event=-1
    if step==520:event=[2,1,3,4][k]
    if step==6*520:event=5
    if step%10==0 and verified_time is not None and not lower_sent and t-verified_time>.5:event=0;lower_sent=True
    q,qd,g,w=sense()
    if lib.align_tick(n,q,qd,g,w,event,command):reason=lib.align_error(n).decode();failure.append(reason);break
    lib.align_status(n,status);max_gate=max(max_gate,status[8]);maxangle=max(maxangle,abs(q[3*k+2]-homes[k]));rotated|=maxangle>.1
    if status[4] and verified_time is None:verified_time=t
    d.ctrl[:]=command;mujoco.mj_step(m,d)
    axis=d.geom_xmat[wheels].reshape(4,3,3)[:,2,2];bottom=d.geom_xpos[wheels,2]-.048*np.sqrt(np.maximum(1-axis*axis,0))-.01675*np.abs(axis)
    moving=np.abs(d.qvel[8::3])>.08;moving_steps+=int(moving.any());unsafe_steps+=int(np.any(moving&(bottom<=.005)))
    if -g[2]<np.cos(.6):reason='tilt_stop';failure.append(reason);break
    bad=False
    for contact in d.contact:
     if contact.dist<-.0001 and not (floor in contact.geom and any(x in wheels for x in contact.geom)):bad=True
    if bad:reason='prohibited_contact';failure.append(reason);break
    if step%10==0:trace.append({'time':t,'event':event,'status':status.tolist(),'q':q.tolist(),'qd':qd.tolist(),'command':command.tolist(),'true_bottom_m':bottom.tolist()})
    if t>8 and status[0]==0:
     reason='aligned_supported_recovery' if int(status[5])&(1<<k) else 'incomplete_supported_recovery';break
  if unsafe_steps:failure.append('true_clearance_le_5mm_during_hub_motion')
  if not int(status[5])&(1<<k):failure.append('alignment_incomplete')
  result={'leg':k,'reason':reason,'seconds':len(trace)*10/520,'verified':bool(status[4]),'completed_mask':int(status[5]),'rotation_occurred':bool(rotated),'max_hub_displacement_rad':maxangle,'max_conservative_floor_m':max_gate,'moving_substeps':moving_steps,'unsafe_clearance_substeps':unsafe_steps,'failures':failure,'pass':not failure}
  (a.out/f'leg{k}_trace.json').write_text(json.dumps(trace));cases.append(result);lib.align_destroy(n);print(json.dumps(result),flush=True)
 (a.out/'result.json').write_text(json.dumps({'scope':'Native MuJoCo actual trained actor + new C++ core; four nominal single-leg diagnostics, not acceptance or MJX parity','policy_sha256':hashlib.sha256(a.policy.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'cases':cases},indent=2)+'\n')
if __name__=='__main__':main()
