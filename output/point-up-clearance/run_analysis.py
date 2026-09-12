"""CAD triangle-mesh distances and intersections using FCL. Kinematic only."""
import csv
import hashlib
import json
import re
import subprocess
import time

import fcl
import numpy as np
from scipy.optimize import brentq, minimize_scalar

from geometry_review import Review, OUT, REPO, LEGS, LEG_REF
from surface_check import TIP, shortened, intersection

ROBOT_REF='e3e1d9737c51f904ff7d0f4cae5692ac4703f659'
REFERENCE_PATH='ros2_ws/src/neural_controller/include/neural_controller/wheel_align_reference_data.hpp'

def collision_object(mesh):
    model=fcl.BVHModel()
    model.beginModel()
    model.addSubModel(np.asarray(mesh.vertices,dtype=np.float64),np.asarray(mesh.faces,dtype=np.int32))
    model.endModel()
    return fcl.CollisionObject(model)

def mesh_distance(a,b):
    collision=fcl.CollisionResult()
    count=fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=1),collision)
    if count:return 0.,True,None
    result=fcl.DistanceResult()
    gap=fcl.distance(a,b,fcl.DistanceRequest(enable_nearest_points=True),result)
    return float(gap),False,np.asarray(result.nearest_points).tolist()

class Analysis:
    def __init__(self):
        self.r=Review()
        self.raw=self.r.meshes['CustomLegFoot']
        text=subprocess.check_output(['git','show',ROBOT_REF+':'+REFERENCE_PATH],cwd=REPO).decode()
        array=text.split('poses{{',1)[1].split('}};',1)[0]
        self.apex=np.array([[float(v) for v in row.split(',') if v.strip()] for row in re.findall(r'\{\{([^}]+)\}\}',array)])
        assert self.apex.shape==(4,8)
        self.mesh_cache={}
        self.fixed_motor={}

    def pose(self,leg,pose,spin):
        hip=0. if pose=='neutral' else self.apex[leg,2*leg+1]
        abd=(1. if leg%2==0 else -1.) if pose=='neutral' else self.apex[leg,2*leg]
        self.r.set_pose(leg=leg,hip=hip,abd=abd,spin=spin)

    def measure(self,length,spin=np.pi,leg=0,pose='neutral'):
        self.pose(leg,pose,spin)
        key=round(length,10)
        if key not in self.mesh_cache:self.mesh_cache[key]=shortened(self.raw,length)
        shin=self.r.world_mesh('leg_'+LEGS[leg]+'_3',custom=self.mesh_cache[key])
        a=collision_object(shin)
        motor_key=(leg,pose)
        if motor_key not in self.fixed_motor:
            self.fixed_motor[motor_key]=collision_object(self.r.world_mesh('leg_'+LEGS[leg]+'_1'))
        return mesh_distance(a,self.fixed_motor[motor_key])

    def tangency(self,leg=0,pose='neutral',spin=np.pi):
        lo,hi=.040,TIP
        assert not self.measure(lo,spin,leg,pose)[1]
        for _ in range(19):
            mid=(lo+hi)/2
            if self.measure(mid,spin,leg,pose)[1]:hi=mid
            else:lo=mid
        return (lo+hi)/2

    def sweep(self,length,leg=0,pose='neutral',step=2.):
        rows=[]
        for degrees in np.arange(0.,360.+step/2,step):
            gap,hit,_=self.measure(length,np.deg2rad(degrees),leg,pose)
            rows.append(dict(length_mm=length*1000,leg=LEGS[leg],pose=pose,spin_deg=float(degrees),gap_mm=gap*1000,collision=hit))
        minima=[]
        for side in [0,1]:
            section=[x for x in rows if side*180<=x['spin_deg']<=(side+1)*180]
            lowest=min(section,key=lambda x:x['gap_mm'])
            if not any(x['collision'] for x in section):
                def value(degrees):return self.measure(length,np.deg2rad(degrees),leg,pose)[0]*1000
                result=minimize_scalar(value,bounds=(max(side*180,lowest['spin_deg']-step),min((side+1)*180,lowest['spin_deg']+step)),method='bounded',options={'xatol':.002})
                if result.fun<lowest['gap_mm']:lowest={**lowest,'spin_deg':float(result.x),'gap_mm':float(result.fun)}
            minima.append(dict(direction='180_to_0' if side==0 else '180_to_360',minimum=lowest,intersecting_sample_count=sum(x['collision'] for x in section)))
        return rows,minima

def main():
    a=Analysis();r=a.r
    result=dict(leg_source_commit=LEG_REF,robot_source_commit=ROBOT_REF,mujoco_version=__import__('mujoco').__version__,fcl_version=fcl.__version__,method='FCL triangle-surface intersection and BVH mesh distance, not convex hulls or training capsules',tip_axis_extent_mm=TIP*1000,tip_max_radial_extent_mm=float(np.linalg.norm(a.raw.vertices[:,:2],axis=1).max()*1000),shortening='Piecewise compression of local y below -26 mm; hub, thickness, side petals and mounting unchanged',static={},lifts=[],sweeps=[])
    tangency=a.tangency()
    result['point_up_contact_length_mm']=tangency*1000
    print('TANGENCY',tangency*1000,flush=True)
    for margin in [.001,.003,.005]:
        L=brentq(lambda L:a.measure(L)[0]-margin,.045,.060,xtol=1e-7)
        result['static'][str(int(margin*1000))+'mm_margin_length_mm']=L*1000
    for L in [TIP,.060,.059,.058,.057,.055,.054,.050,.048]:
        gap,hit,nearest=a.measure(L)
        result['static'][str(round(L*1000,4))]=dict(gap_mm=gap*1000,collision=hit,nearest_world_m=nearest)
    for leg in range(4):
        for pose in ['neutral','apex']:
            a.pose(leg,pose,np.pi)
            body='leg_'+LEGS[leg]+'_3';hub=r.data.xpos[r.model.body(body).id].copy()
            for L in [TIP,.054]:
                shin=r.world_mesh(body,custom=shortened(a.raw,L))
                result['lifts'].append(dict(leg=LEGS[leg],pose=pose,length_mm=L*1000,hub_z_relative_body_mm=(hub[2]-.1313)*1000,hub_z_assuming_body_131_3mm=hub[2]*1000,point_up_highest_mm=float(shin.vertices[:,2].max()*1000),point_up_lowest_mm=float(shin.vertices[:,2].min()*1000),shoulder_gap_mm=a.measure(L,np.pi,leg,pose)[0]*1000,collision=a.measure(L,np.pi,leg,pose)[1]))
    rows=[]
    for L in [TIP,.060,.057,.054,.050,.048]:
        section,minima=a.sweep(L)
        rows.extend(section);result['sweeps'].append(dict(length_mm=L*1000,leg='front_r',pose='neutral',minima=minima))
        print('SWEEP',L*1000,minima,flush=True)
    # Confirm the proposed 54 mm candidate for all legs at neutral and existing v5 apex.
    for leg in range(4):
        for pose in ['neutral','apex']:
            if leg==0 and pose=='neutral':continue
            section,minima=a.sweep(.054,leg,pose)
            rows.extend(section);result['sweeps'].append(dict(length_mm=54.,leg=LEGS[leg],pose=pose,minima=minima))
            print('CONFIRM',LEGS[leg],pose,minima,flush=True)
    # Independent ray/triangle check of a colliding original and separated candidate.
    a.pose(0,'neutral',np.pi)
    motor=r.world_mesh('leg_front_r_1')
    result['independent_surface_checks']={str(L*1000):len(intersection(r.world_mesh('leg_front_r_3',custom=shortened(a.raw,L)),motor)) for L in [TIP,.054]}
    result['source_mesh_sha256']=hashlib.sha256((r.source/'meshes/stl/CustomLegFoot.stl').read_bytes()).hexdigest()
    (OUT/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (OUT/'rotation-clearance.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('DONE',OUT/'measurements.json',flush=True)

if __name__=='__main__':main()
