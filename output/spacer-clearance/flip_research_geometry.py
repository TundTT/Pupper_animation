"""Kinematic floor-envelope screening, not a motion plan or dynamics validation."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'point-up-clearance'))
from geometry_review import Review,LEGS
import mujoco as mj
import numpy as np

OUT=Path(__file__).resolve().parent
r=Review()
# The pinned original mesh is unchanged by the 9 mm-gap commit; add its exact offset.
for leg in LEGS:r.visuals['leg_'+leg+'_3'][0].set('pos','0 0 0.0226')
q=r.model.qpos0.copy();q[2]=.2;q[3:7]=[1,0,0,0];q[9::3]+=np.pi
r.data.qpos[:]=q;mj.mj_forward(r.model,r.data)
lowest=min(r.world_mesh('leg_'+leg+'_3').vertices[:,2].min() for leg in LEGS)
q[2]-=lowest
r.data.qpos[:]=q;mj.mj_forward(r.model,r.data)
start=q.copy()
result=dict(description='Original full-length shin CAD plus 9 mm gap; level body, neutral proximal pose, all four hub angles at source reference + pi. Body translated vertically until lowest full-CAD vertex touches floor. No load equilibrium or thermal deformation modeled.',body_z_m=float(q[2]),limbs=[])
for i,leg in enumerate(LEGS):
    body='leg_'+leg+'_3';bid=r.model.body(body).id
    r.data.qpos[:]=start;mj.mj_forward(r.model,r.data)
    hub=float(r.data.xpos[bid,2]);up=float(r.world_mesh(body).vertices[:,2].min())
    vals=[]
    for deg in np.arange(0,360.01,.5):
        r.data.qpos[:]=start;r.data.qpos[9+3*i]=r.model.qpos0[9+3*i]+np.deg2rad(deg)
        mj.mj_forward(r.model,r.data)
        vals.append((float(deg),float(r.world_mesh(body).vertices[:,2].min())))
    worst=min(vals,key=lambda x:x[1])
    down=vals[0][1]
    result['limbs'].append(dict(leg=leg,initial_hub_height_mm=hub*1000,initial_lowest_mm=up*1000,point_down_lowest_without_lift_mm=down*1000,worst_rotation_deg=worst[0],worst_lowest_without_lift_mm=worst[1]*1000,rigid_vertical_lift_for_5mm_margin_mm=5-worst[1]*1000))
(OUT/'flip-research-geometry.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
