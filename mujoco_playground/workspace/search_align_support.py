"""Find mechanically feasible coordinated stance shifts, not reward tuning."""
import json
import mujoco
import numpy as np
from workspace import configs
from workspace.wheel_align_env import PupperWheelAlignEnv

def main():
 e=PupperWheelAlignEnv(configs.get_wheel_align_config());m=e.sys.mj_model
 maps=np.load('workspace/output/align_diagnosis/support_map.npz')['joint_shift_map']
 ids=e._wheel_geom_id; candidates=[]
 for sx in np.arange(-.04,.021,.01):
  for sy in np.arange(-.03,.041,.01):
   for lift in (.5,.7,.9,1.1):
    x=mujoco.MjData(m);x.qpos[:]=e._init_q
    off=np.zeros(12);off[e._pos_rows]=(maps@np.array([sx,sy])).ravel()
    for t in range(600):
     x.ctrl[:]=np.asarray(e._default_pose)+off*min(t/100,1)
     x.ctrl[e._hip_rows[0]]+=lift*max(min((t-100)/100,1),0)
     mujoco.mj_step(m,x)
    az=np.abs(x.geom_xmat[ids].reshape(4,3,3)[:,2,2]);z=x.geom_xpos[ids,2]-m.geom_size[ids,0]*np.sqrt(1-az**2)-m.geom_size[ids,1]*az
    if .003<=z[0]<=.012 and np.max(z[1:])<.001:
     candidates.append(dict(shift=[sx,sy],lift=lift,clearance_mm=(z*1000).tolist(),drift_mm=float(np.linalg.norm(x.qpos[:2])*1000),torso_z=float(x.qpos[2])))
 candidates.sort(key=lambda r:r['drift_mm'])
 open('workspace/output/align_diagnosis/support_candidates.json','w').write(json.dumps(candidates,indent=2))
 print('FEASIBLE',len(candidates));print(json.dumps(candidates[:15],indent=2),flush=True)

if __name__=='__main__':main()
