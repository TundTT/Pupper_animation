"""Check intermediate proximal poses and verify FCL transform implementation."""
import json
import numpy as np
from scipy.optimize import brentq
from analyze import Spacer, OUT, LEGS, collision_object, mesh_distance

s=Spacer();apex=s.a.apex.copy();result={'intermediate_pose_sweeps':[]}
result['point_up_001mm_gap_spacer_mm']=brentq(lambda t:s.measure(t)['gap_mm']-.01,0,10)
for mm in [6,10,12,16,20]:
    _,low=s.sweep(mm,0,'neutral',step=.5)
    result.setdefault('neutral_comparison',[]).append(low)
for leg in range(4):
    for fraction in [.1,.2,.3,.4,.5,.6,.7,.8,.9]:
        s.a.apex[leg,2*leg]=(1 if leg%2==0 else -1)*(1-fraction)+apex[leg,2*leg]*fraction
        s.a.apex[leg,2*leg+1]=apex[leg,2*leg+1]*fraction
        s.motors.pop((leg,'apex'),None)
        for mm in [5.7,16]:
            _,low=s.sweep(mm,leg,'apex',step=2)
            result['intermediate_pose_sweeps'].append(dict(lift_fraction=fraction,**low))
    print('INTERMEDIATE_DONE',LEGS[leg],flush=True)
    s.a.apex[:]=apex
    s.motors.clear()
# Cross-check reusable local-frame BVH transforms against world-space geometry.
result['world_transform_crosscheck']=[]
for leg in range(4):
    for spin in [180,206]:
        v=s.measure(16,spin,leg)
        body='leg_'+LEGS[leg]+'_3';bid=s.r.model.body(body).id
        shin=s.r.world_mesh(body)
        shin.vertices+=s.r.data.xmat[bid].reshape(3,3)[:,2]*.016
        gap,hit,_=mesh_distance(collision_object(shin),collision_object(s.r.world_mesh('leg_'+LEGS[leg]+'_1')))
        delta=abs(gap*1000-v['gap_mm'])
        assert delta<1e-7 and hit==v['collision']
        result['world_transform_crosscheck'].append(dict(leg=LEGS[leg],spin_deg=spin,gap_error_mm=delta))
(OUT/'validation.json').write_text(json.dumps(result,indent=2))
print('STATIC CONTACT APPROX',result['point_up_001mm_gap_spacer_mm'])
print('NEUTRAL COMPARISON',result['neutral_comparison'])
for mm in [5.7,16]:
    rows=[x for x in result['intermediate_pose_sweeps'] if x['spacer_mm']==mm]
    print('INTERMEDIATE MINIMUM',min(rows,key=lambda x:x['gap_mm']))
