#!/usr/bin/env python3
"""Independent native-MuJoCo geometry fixtures, not learned balance evidence."""
import argparse,hashlib,json
from pathlib import Path
import mujoco,numpy as np

def main():
 assert mujoco.__version__=='3.6.0','Use pinned native MuJoCo3.6.0 for these fixtures/diagnostics'
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 m=mujoco.MjModel.from_xml_path(str(a.model));d=mujoco.MjData(m)
 assert hashlib.sha256(a.model.read_bytes()).hexdigest()=='4275f813ad1b6b1763ec42f70f23e871114724b3a51baa238c0b4efaafa588cb'
 wheels=[m.geom('leg_'+k+'_3_wheel_collision').id for k in ['front_r','front_l','back_r','back_l']]
 box=next(i for i in range(m.ngeom) if m.geom_bodyid[i]==1 and m.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX)
 pos=np.array([0,1,3,4,6,7,9,10]);nom=np.array([1,0,-1,0,1,0,-1,0]);low=np.array([-1.12,-.32,-2.41,-3.04]*2)+.06;high=np.array([2.41,3.04,1.12,.32]*2)-.06
 rng=np.random.default_rng(3811);rows=[];feasible={};sphere=np.hypot(.0505,.01675)
 for trial in range(100000):
  q=np.zeros(12);q[pos]=np.clip(nom+rng.uniform(-.75,.75,8),low,high);q[2::3]=rng.uniform(-100,100,4)
  g=np.array([0.,0.,-1.]) if trial>=1024 else np.array([*rng.normal(0,.08,2),-1.]);g/=np.linalg.norm(g)
  d.qpos[:7]=[0,0,0,1,0,0,0];d.qpos[7:]=q;mujoco.mj_forward(m,d)
  centers=d.geom_xpos[wheels];axis=d.geom_xmat[wheels].reshape(4,3,3)[:,:,2];az=-axis@g;radial=np.sqrt(np.maximum(1-az*az,0));z=-centers@g
  lower=z-.0505*radial-.01675*np.abs(az);upper=z-.0455*radial-.01675*np.abs(az)
  wheel=min(np.linalg.norm(centers[i]-centers[j])-2*sphere for i in range(4) for j in range(i+1,4))
  local=(centers-d.geom_xpos[box])@d.geom_xmat[box].reshape(3,3);body=np.min(np.linalg.norm(np.maximum(np.abs(local)-m.geom_size[box],0),axis=1))-sphere
  for k in range(4):
   margins=np.array([lower[k]-np.min(np.delete(upper,k)),wheel,body]);row=np.r_[q,g,k,margins]
   if trial<1024 and k==trial%4:rows.append(row)
   if trial>=1024 and np.all(margins>[.010,.010,.005]) and k not in feasible:feasible[k]=row
  if len(feasible)==4:break
 assert len(feasible)==4,'No static feasible pose for each leg within trained action bounds'
 np.savetxt(a.out,rows+[feasible[k] for k in range(4)],delimiter=',',fmt='%.17g')
 print(json.dumps({'native_geometry_rows':len(rows)+4,'static_feasible_legs':list(feasible),'scope':'kinematic fixtures, not learned lifting or dynamic support'}))
if __name__=='__main__':main()
