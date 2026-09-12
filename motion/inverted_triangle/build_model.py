"""Build the post-cooled transition model from pinned leg CAD.

The whole-shin convex hull is used ONLY against the flat floor. It preserves
the lowest surface/support envelope against a plane; it is not an exact hollow
ring collider and must never be used for shin/body interference certification.
Detailed source meshes are retained for independent self-clearance auditing.
"""
from pathlib import Path
import hashlib, io, json, re, subprocess, xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
SOURCE='27bc66823478758bd9dc701a5aec27cceaa0710a'
PREFIX='Stanford/training/pupper_v3_description/description/'
LEGS=('front_r','front_l','back_r','back_l')

def build():
    raw=subprocess.check_output(['git','show',SOURCE+':'+PREFIX+'mujoco_xml/pupper_v3_complete.mjx.position.xml'],cwd=REPO)
    (HERE/'source.xml').write_bytes(raw)
    root=ET.fromstring(re.sub(rb'<!--.*?-->',b'',raw,flags=re.S))
    assets=HERE/'assets';assets.mkdir(exist_ok=True)
    hashes={}
    for mesh in root.iter('mesh'):
        name=mesh.get('file')
        data=subprocess.check_output(['git','show',SOURCE+':'+PREFIX+'meshes/stl/'+name],cwd=REPO)
        (assets/name).write_bytes(data);hashes[name]=hashlib.sha256(data).hexdigest()
    root.find('compiler').set('meshdir','assets')
    root.find('option').set('timestep',str(1/520))
    root.find('option').set('iterations','30')
    root.find('option').set('ls_iterations','10')
    root.find('option').set('integrator','implicitfast')
    # Explicit mass properties already exist. Add no fictitious mass for the gap.
    raw_mesh=trimesh.load(assets/'CustomLegFoot.stl',process=True)
    hull=raw_mesh.convex_hull
    hull.export(assets/'ShinFloorHull.stl')
    ET.SubElement(root.findall('asset')[-1],'mesh',name='ShinFloorHull',file='ShinFloorHull.stl',scale='0.001 0.001 0.001')
    for leg in LEGS:
        body=root.find(f'.//body[@name="leg_{leg}_3"]')
        body.remove(body.find('geom[@type="capsule"]'))
        visual=body.find('geom[@type="mesh"]')
        visual.set('name',f'{leg}_shin_visual')
        attrib={k:visual.get(k) for k in ['pos','quat'] if visual.get(k) is not None}
        ET.SubElement(body,'geom',**attrib,name=f'{leg}_floor_contact',type='mesh',mesh='ShinFloorHull',density='0',group='3',contype='0',conaffinity='1',friction='0.8 0.02 0.001',condim='3',solref='0.01 1')
        joint=body.find('joint');joint.set('limited','false');joint.attrib.pop('range',None)
    # The original spheres miss portions of the lower motor housing. Use each
    # CAD envelope against the floor as well, so a search cannot stand on a
    # hidden motor intersection or an overly convenient collision proxy.
    mesh_assets={e.get('name'):e for e in root.iter('mesh')}
    for body in root.iter('body'):
        if body.get('name','').endswith('_3'):continue
        for g in list(body.findall('geom')):
            if g.get('class')=='collision':body.remove(g)
        for visual in list(body.findall('geom')):
            if visual.get('type')!='mesh':continue
            source_asset=mesh_assets[visual.get('mesh')]
            mesh=trimesh.load(assets/source_asset.get('file'),process=True)
            name=source_asset.get('name')+'_FloorHull'
            mesh.convex_hull.export(assets/(name+'.stl'))
            ET.SubElement(root.findall('asset')[-1],'mesh',name=name,file=name+'.stl',scale=source_asset.get('scale','1 1 1'))
            hashes[name+'.stl']=hashlib.sha256((assets/(name+'.stl')).read_bytes()).hexdigest()
            attrib={k:visual.get(k) for k in ['pos','quat'] if visual.get(k) is not None}
            ET.SubElement(body,'geom',**attrib,name=body.get('name')+'_floor_envelope',type='mesh',mesh=name,density='0',group='3',contype='0',conaffinity='1',friction='0.8 0.02 0.001',condim='3',solref='0.01 1')
    # Match the current soft proximal limits, not the wider legacy training limits.
    hardware=ET.parse(REPO/'ros2_ws/src/pupper_v3_description/description/components.xacro')
    for i,leg in enumerate(LEGS):
        for j in [1,2]:
            name=f'leg_{leg}_{j}'
            hw=hardware.find(f'.//joint[@name="{name}"]')
            lo=hw.find('param[@name="position_min"]').text
            hi=hw.find('param[@name="position_max"]').text
            root.find(f'.//joint[@name="{name}"]').set('range',f'{lo} {hi}')
    # Torque input makes the outer simulator PD and the total torque cap explicit.
    actuator=root.find('actuator');actuator.clear()
    for leg in LEGS:
        for j in [1,2,3]:
            ET.SubElement(actuator,'motor',name=f'leg_{leg}_{j}',joint=f'leg_{leg}_{j}',ctrlrange='-3 3',ctrllimited='true')
    root.find('keyframe').clear()
    root.find('worldbody/geom[@name="floor"]').set('conaffinity','0')
    root.insert(0,ET.Comment(' POST-COOLED RIGID SHINS. Additional axial gap 9 mm; no heating or deformation. Floor-only hulls; exact self-clearance audited separately. '))
    ET.indent(root)
    (HERE/'model.xml').write_text(ET.tostring(root,encoding='unicode'))
    model=mujoco.MjModel.from_xml_path(str(HERE/'model.xml'))
    q=model.qpos0.copy();q[2]=.2;q[3:7]=[1,0,0,0];q[9::3]+=np.pi
    # Candidate post-cooled stance: a small lateral joint-2 splay gives the
    # lower motor housings clearance. This is not a captured hardware pose.
    q[8::3]=[-.29,.29,-.29,.29]
    data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
    lowest=1.
    for leg in LEGS:
        g=model.geom(f'{leg}_floor_contact').id;mid=model.geom_dataid[g]
        v=model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
        points=v@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        lowest=min(lowest,points[:,2].min())
    q[2]-=lowest-.0005
    root.find('keyframe').append(ET.Element('key',name='inverted',qpos=' '.join(f'{v:.12g}' for v in q),ctrl=' '.join(['0']*12)))
    ET.indent(root);(HERE/'model.xml').write_text(ET.tostring(root,encoding='unicode'))
    hashes['ShinFloorHull.stl']=hashlib.sha256((assets/'ShinFloorHull.stl').read_bytes()).hexdigest()
    manifest=dict(source_commit=SOURCE,source_xml_sha256=hashlib.sha256(raw).hexdigest(),asset_sha256=hashes,model_sha256=hashlib.sha256((HERE/'model.xml').read_bytes()).hexdigest(),gap_m=.009,initial_joint2_rad=[-.29,.29,-.29,.29],initial_pose_status='Model candidate for user-reported approximately 5 mm lower housing floor clearance; not a captured hardware pose',components_sha256=hashlib.sha256((REPO/'ros2_ws/src/pupper_v3_description/description/components.xacro').read_bytes()).hexdigest(),floor_collision='Convex hull of original CAD, floor-only mask; detailed self-collision audit required',modeling='rigid locked post-morph; no heating; original mass and COM inertia; explicit total torque saturation 3 Nm; 520 Hz')
    (HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Built',HERE/'model.xml')

if __name__=='__main__':build()
