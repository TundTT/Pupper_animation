"""Native physics probe for a coordinated home-stance weight shift."""
import json
import numpy as np
import mujoco
from workspace import configs
from workspace.wheel_align_env import PupperWheelAlignEnv


def main():
 e=PupperWheelAlignEnv(configs.get_wheel_align_config());m=e.sys.mj_model
 d=mujoco.MjData(m);d.qpos[:]=e._init_q;d.ctrl[:]=e._default_pose;mujoco.mj_forward(m,d)
 maps=[]
 for row in range(4):
  jac=np.zeros((3,m.nv));rot=np.zeros((3,m.nv));mujoco.mj_jacGeom(m,d,jac,rot,int(e._wheel_geom_id[row]))
  jr=[configs.JOINT_NAMES.index(f'leg_{list(configs.FOOT_ROW_BY_LEG)[row]}_{j}') for j in (1,2)]
  maps.append(-np.linalg.pinv(jac[:,e._joint_dof[jr]])[:,:2])
 maps=np.array(maps)
 print('MAP',maps.tolist(),flush=True)
 results=[]
 for row,leg in enumerate(configs.FOOT_ROW_BY_LEG):
  pts=d.geom_xpos[e._wheel_geom_id,:2][np.arange(4)!=row].copy();center=pts.mean(0)
  pts=pts[np.argsort(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))];edges=np.roll(pts,-1,axis=0)-pts
  inward=np.stack((-edges[:,1],edges[:,0]),axis=-1)/np.linalg.norm(edges,axis=1)[:,None]
  margins=np.sum(inward*(d.subtree_com[e._torso_idx,:2]-pts),axis=1)
  nominal=inward[margins.argmin()]*max(.005-margins.min(),0)
  for mult in (0.,1.,1.5,2.):
   for lift in (.5,.6,.7):
    x=mujoco.MjData(m);x.qpos[:]=e._init_q;x.ctrl[:]=e._default_pose
    off=np.zeros(12);off[e._pos_rows]=(maps@(nominal*mult)).ravel()
    for t in range(1000):
     x.ctrl[:]=np.asarray(e._default_pose)+off*min(t/125,1)
     x.ctrl[e._hip_rows[row]]+=float(e._hip_sign[row])*lift*max(min((t-125)/125,1),0)
     mujoco.mj_step(m,x)
    az=np.abs(x.geom_xmat[e._wheel_geom_id].reshape(4,3,3)[:,2,2]);z=x.geom_xpos[e._wheel_geom_id,2]-m.geom_size[e._wheel_geom_id,0]*np.sqrt(1-az**2)-m.geom_size[e._wheel_geom_id,1]*az
    item={'leg':leg,'shift_mm':(nominal*mult*1000).tolist(),'lift':lift,'clearance_mm':(z*1000).tolist(),'body_xy_mm':(x.qpos[:2]*1000).tolist(),'offsets':off[e._pos_rows].tolist()}
    results.append(item)
    print(leg,mult,lift,'clear',np.round(z*1000,2),'xy',np.round(x.qpos[:2]*1000,2),flush=True)
 np.savez('workspace/output/align_diagnosis/support_map.npz',joint_shift_map=maps)
 open('workspace/output/align_diagnosis/support_probe.json','w').write(json.dumps(results,indent=2))

if __name__=='__main__':main()
