"""Independent nonconvex CAD self-clearance; no MuJoCo contact hull shortcuts."""
import xml.etree.ElementTree as ET
import io
import numpy as np
import trimesh
import fcl
import mujoco as mj
from .core import HERE,LEGS

class CADClearance:
    def __init__(self,robot):
        self.r=robot;self.objects={};self.body_ids={}
        root=ET.fromstring(robot.model_xml)
        meshes={e.get('name'):e for e in root.iter('mesh')}
        for body in root.iter('body'):
            for g in body.findall('geom'):
                if g.get('type')!='mesh' or g.get('group')!='1':continue
                mesh=meshes[g.get('mesh')]
                if robot.model_assets is None:
                    t=trimesh.load(HERE/'assets'/mesh.get('file'),process=True)
                else:
                    t=trimesh.load(io.BytesIO(robot.model_assets[mesh.get('file')]),file_type='stl',process=True)
                v=t.vertices*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                quat=np.fromstring(g.get('quat','1 0 0 0'),sep=' ');quat/=np.linalg.norm(quat)
                R=np.empty(9);mj.mju_quat2Mat(R,quat)
                v=v@R.reshape(3,3).T+np.fromstring(g.get('pos','0 0 0'),sep=' ')
                b=fcl.BVHModel();b.beginModel();b.addSubModel(v.astype(np.float64),t.faces.astype(np.int32));b.endModel()
                name=body.get('name');self.objects[name]=fcl.CollisionObject(b);self.body_ids[name]=robot.m.body(name).id
        self.pairs=[]
        for leg in LEGS:
            name=f'leg_{leg}_3'
            for other in self.objects:
                if other==name:continue
                pair=tuple(sorted((name,other)))
                if pair not in self.pairs:self.pairs.append(pair)
    def measure(self):
        for name,obj in self.objects.items():
            bid=self.body_ids[name]
            obj.setTransform(fcl.Transform(self.r.d.xmat[bid].reshape(3,3),self.r.d.xpos[bid]))
        lowest=(float('inf'),None);front_rear=(float('inf'),None);hits=[]
        for a,b in self.pairs:
            count=fcl.collide(self.objects[a],self.objects[b],fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult())
            gap=0. if count else float(fcl.distance(self.objects[a],self.objects[b],fcl.DistanceRequest(),fcl.DistanceResult()))
            if count:hits.append([a,b])
            if gap<lowest[0]:lowest=(gap,[a,b])
            if a.endswith('_3') and b.endswith('_3') and (('front_' in a and 'back_' in b) or ('back_' in a and 'front_' in b)):
                if gap<front_rear[0]:front_rear=(gap,[a,b])
        return dict(minimum_m=lowest[0],nearest_pair=lowest[1],intersections=hits,
                    front_rear_minimum_m=front_rear[0],front_rear_nearest_pair=front_rear[1])
