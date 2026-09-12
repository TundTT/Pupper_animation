"""Offline geometry review; never connects to or commands a robot.

Sources are pinned. CAD surface meshes are used, not the training foot capsule.
"""
from pathlib import Path
import io
import json
import re
import subprocess
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
LEG_REF = '6b55e30ff224193a2a21c8dc3dbb88e029a417c8'
PREFIX = 'Stanford/training/pupper_v3_description/description/'
LEGS = ['front_r', 'front_l', 'back_r', 'back_l']

def git_bytes(path):
    return subprocess.check_output(['git', 'show', LEG_REF + ':' + PREFIX + path], cwd=REPO)

def vector(value, default):
    return np.array(default, dtype=float) if value is None else np.fromstring(value, sep=' ')

def rotation(value):
    q = vector(value, [1, 0, 0, 0]); q /= np.linalg.norm(q)
    mat = np.empty(9); mj.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)

class Review:
    def __init__(self):
        self.raw_xml = git_bytes('mujoco_xml/pupper_v3_complete.mjx.position.xml')
        self.root = ET.fromstring(re.sub(rb'<!--.*?-->', b'', self.raw_xml, flags=re.S))
        self.meshes = {}
        assets = {}
        self.source = OUT / 'source' / 'description'
        (self.source / 'mujoco_xml').mkdir(parents=True, exist_ok=True)
        (self.source / 'meshes' / 'stl').mkdir(parents=True, exist_ok=True)
        (self.source / 'mujoco_xml' / 'pupper_v3_complete.mjx.position.xml').write_bytes(self.raw_xml)
        for e in self.root.iter('mesh'):
            data = git_bytes('meshes/stl/' + e.get('file'))
            assets[e.get('file')] = data
            (self.source / 'meshes' / 'stl' / e.get('file')).write_bytes(data)
            mesh = trimesh.load(io.BytesIO(data), file_type='stl', process=True)
            mesh.vertices *= vector(e.get('scale'), [1, 1, 1])
            self.meshes[e.get('name')] = mesh
        self.root.find('compiler').set('meshdir', '')
        self.model = mj.MjModel.from_xml_string(ET.tostring(self.root, encoding='unicode'), assets)
        self.data = mj.MjData(self.model)
        self.visuals = {}
        for body in self.root.iter('body'):
            self.visuals[body.get('name')] = [g for g in body.findall('geom') if g.get('type') == 'mesh']
        self.set_pose()

    def set_pose(self, leg=0, hip=0., abd=None, spin=np.pi):
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[2] = .1313
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[8+3*leg] = hip
        if abd is not None: self.data.qpos[7+3*leg] = abd
        self.data.qpos[9+3*leg] += spin
        mj.mj_forward(self.model, self.data)

    def world_mesh(self, body, which=0, custom=None):
        geom = self.visuals[body][which]
        mesh = self.meshes[geom.get('mesh')].copy() if custom is None else custom.copy()
        local = mesh.vertices @ rotation(geom.get('quat')).T + vector(geom.get('pos'), [0, 0, 0])
        bid = self.model.body(body).id
        mesh.vertices = local @ self.data.xmat[bid].reshape(3,3).T + self.data.xpos[bid]
        return mesh

    def local_mesh(self, body, which=0):
        geom = self.visuals[body][which]
        return self.meshes[geom.get('mesh')]

    def to_body(self, points, body):
        bid = self.model.body(body).id
        return (points-self.data.xpos[bid]) @ self.data.xmat[bid].reshape(3,3)

def inspect():
    r = Review()
    print('MODEL', r.model.nq, r.model.nu)
    for name in ['CustomLegFoot','LegAssemblyForFlangedv26_001','LegAssemblyForFlangedv26_002']:
        m = r.meshes[name]
        pieces = m.split(only_watertight=False, repair=False)
        print(name, 'vertices',len(m.vertices),'faces',len(m.faces),'bounds_mm',(m.bounds*1000).round(3).tolist(),'watertight',m.is_watertight)
        print('COMPONENTS',[(len(p.faces),p.is_watertight,(p.bounds*1000).round(2).tolist()) for p in pieces if len(p.faces)>30])
    for leg in range(4):
        r.set_pose(leg=leg)
        shin=r.world_mesh('leg_'+LEGS[leg]+'_3')
        print('POINT_UP',LEGS[leg], 'hub',r.data.xpos[r.model.body('leg_'+LEGS[leg]+'_3').id].round(5).tolist(),'shin_bounds_mm',(shin.bounds*1000).round(2).tolist())
        for j in [1,2]:
            motor=r.world_mesh('leg_'+LEGS[leg]+'_'+str(j))
            candidates=shin.vertices[shin.vertices[:,2] > r.data.xpos[r.model.body('leg_'+LEGS[leg]+'_3').id,2]+.025]
            near,dist,tri=trimesh.proximity.closest_point(motor,candidates)
            idx=np.argmin(dist)
            print('motor',j,'vertex_surface_min_mm',float(dist[idx]*1000),'shin_nearest_world',candidates[idx].tolist(),'motor_nearest_world',near[idx].tolist(),flush=True)

if __name__ == '__main__': inspect()
