"""Offline CAD clearance for an axial spacer; no hardware connection."""
from pathlib import Path
import sys, json, csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'point-up-clearance'))
import numpy as np
import fcl
from scipy.optimize import minimize_scalar, brentq
from geometry_review import LEGS, LEG_REF, rotation, vector
from run_analysis import Analysis, collision_object, mesh_distance, ROBOT_REF

OUT=Path(__file__).resolve().parent

class Spacer:
    def __init__(self):
        self.a=Analysis();self.r=self.a.r;self.shins={};self.motors={}
        for leg in range(4):
            body='leg_'+LEGS[leg]+'_3';g=self.r.visuals[body][0]
            mesh=self.a.raw.copy()
            mesh.vertices=mesh.vertices@rotation(g.get('quat')).T+vector(g.get('pos'),[0,0,0])
            self.shins[leg]=collision_object(mesh)

    def measure(self,mm,spin=180.,leg=0,pose='neutral'):
        self.a.pose(leg,pose,np.deg2rad(spin))
        body=self.r.model.body('leg_'+LEGS[leg]+'_3').id
        R=self.r.data.xmat[body].reshape(3,3).copy()
        t=self.r.data.xpos[body].copy()+R[:,2]*mm/1000
        self.shins[leg].setTransform(fcl.Transform(R,t))
        key=(leg,pose)
        if key not in self.motors:
            self.motors[key]=collision_object(self.r.world_mesh('leg_'+LEGS[leg]+'_1'))
        gap,hit,near=mesh_distance(self.shins[leg],self.motors[key])
        return dict(gap_mm=gap*1000,collision=hit,nearest_world_m=near)

    def sweep(self,mm,leg=0,pose='neutral',step=2):
        rows=[]
        for spin in np.arange(0,360+step/2,step):
            v=self.measure(mm,float(spin),leg,pose)
            rows.append(dict(spacer_mm=mm,leg=LEGS[leg],pose=pose,spin_deg=float(spin),gap_mm=v['gap_mm'],collision=v['collision']))
        low=min(rows,key=lambda x:x['gap_mm'])
        # Refine every sampled local minimum, retaining any sampled intersections.
        if not any(x['collision'] for x in rows):
            for i in range(1,len(rows)-1):
                if rows[i]['gap_mm']<=min(rows[i-1]['gap_mm'],rows[i+1]['gap_mm']):
                    res=minimize_scalar(lambda x:self.measure(mm,x,leg,pose)['gap_mm'],bounds=(rows[i-1]['spin_deg'],rows[i+1]['spin_deg']),method='bounded',options={'xatol':.0001})
                    if res.fun<low['gap_mm']:low={**rows[i],'spin_deg':float(res.x),'gap_mm':float(res.fun)}
        return rows,dict(**low,intersecting_samples=sum(x['collision'] for x in rows))

def main():
    s=Spacer()
    result=dict(leg_commit=LEG_REF,pose_commit=ROBOT_REF,spacer_definition='Additional translation of entire original shin along positive local Z of joint-3 body, beyond original geom pos z=13.6 mm. No shortening.',method='FCL non-convex CAD triangle meshes; all four limbs, neutral and existing alignment apex; 2 degree rotation sweep with local-minimum refinement',static=[],sweeps=[])
    for mm in [0,1,2,3,4,5,8,10,12,16,20]:
        v=s.measure(mm);result['static'].append(dict(spacer_mm=mm,**v));print('STATIC',mm,v,flush=True)
    allrows=[]
    for leg in range(4):
        for pose in ['neutral','apex']:
            for mm in [0,16,20]:
                rows,low=s.sweep(mm,leg,pose);allrows.extend(rows);result['sweeps'].append(low);print('SWEEP',low,flush=True)
    # Solve the worst full-turn clearance across each of the eight configurations.
    result['thresholds']=[]
    for leg in range(4):
        for pose in ['neutral','apex']:
            def worst(mm):return s.sweep(mm,leg,pose,step=3)[1]['gap_mm']
            vals={}
            # 0.01 mm gap approximates first clearance while avoiding ambiguous tangency.
            for margin in [.01,3,5]:
                mm=brentq(lambda t:worst(t)-margin,0,35,xtol=.0005)
                rows,low=s.sweep(mm,leg,pose,step=.5)
                vals[str(margin)]=dict(spacer_mm=float(mm),verified_minimum=low)
            result['thresholds'].append(dict(leg=LEGS[leg],pose=pose,targets_mm=vals))
            print('THRESHOLD',result['thresholds'][-1],flush=True)
            (OUT/'measurements.json').write_text(json.dumps(result,indent=2))
    with (OUT/'rotation-clearance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    (OUT/'measurements.json').write_text(json.dumps(result,indent=2))
    print('DONE',flush=True)

if __name__=='__main__':main()
