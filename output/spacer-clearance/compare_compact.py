"""Compare 8 and 9 mm spacers using the original CAD and existing lift poses."""
import json
from analyze import Spacer, OUT, LEGS, LEG_REF, ROBOT_REF

s=Spacer();apex=s.a.apex.copy();rows=[]
for leg in range(4):
    for tenth in range(11):
        fraction=tenth/10
        s.a.apex[leg,2*leg]=(1 if leg%2==0 else -1)*(1-fraction)+apex[leg,2*leg]*fraction
        s.a.apex[leg,2*leg+1]=apex[leg,2*leg+1]*fraction
        s.motors.pop((leg,'apex'),None)
        for mm in [8,9]:
            _,low=s.sweep(mm,leg,'apex',step=2)
            rows.append(dict(lift_fraction=fraction,**low))
    s.a.apex[:]=apex;s.motors.clear()
    print('DONE',LEGS[leg],flush=True)
minima={str(mm):min((x for x in rows if x['spacer_mm']==mm),key=lambda x:x['gap_mm']) for mm in [8,9]}
result=dict(leg_commit=LEG_REF,pose_commit=ROBOT_REF,method='Original full-length CAD, all four limbs, 11 proximal poses from neutral to existing apex, full revolution at 2-degree spacing with local-minimum refinement',minima=minima,sweeps=rows)
(OUT/'compact-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(minima,indent=2),flush=True)
