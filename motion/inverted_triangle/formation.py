"""Synthetic rigid distal-tip errors; nominal assets and gap stay unchanged.

Visuals, floor hulls, foot sites and detailed CAD audits share the same changed
geometry. This is a sensitivity model, not a thermomechanical polymer model.
"""
import hashlib
import io
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from .core import HERE, LEGS, smooth

ANCHOR_MM=40.

def validate(spec):
    if set(spec)!= {'length_mm','bend_deg'}:raise ValueError('Expected length_mm and bend_deg')
    length=np.asarray(spec['length_mm'],dtype=float);bend=np.asarray(spec['bend_deg'],dtype=float)
    if length.shape!=(4,) or bend.shape!=(4,) or not np.isfinite([length,bend]).all():
        raise ValueError('Four finite independent errors in FR/FL/BR/BL order required')
    if np.max(abs(length))>10 or np.max(abs(bend))>10:
        raise ValueError('Distal-tip model scope: +/-10 mm and +/-10 degrees')
    return length,bend

def deform(vertices, length_mm, bend_deg, tip_extent_mm):
    v=np.array(vertices,dtype=float,copy=True)
    weight=smooth((-v[:,1]-ANCHOR_MM)/(tip_extent_mm-ANCHOR_MM))
    v[:,1]-=float(length_mm)*weight
    angle=np.deg2rad(float(bend_deg))*weight
    x=v[:,0].copy();y=v[:,1]+ANCHOR_MM
    v[:,0]=np.cos(angle)*x-np.sin(angle)*y
    v[:,1]=np.sin(angle)*x+np.cos(angle)*y-ANCHOR_MM
    return v

def build(spec, nominal_manifest):
    length,bend=validate(spec)
    root=ET.parse(HERE/'model.xml').getroot()
    assets={m.get('file'):(HERE/'assets'/m.get('file')).read_bytes() for m in root.iter('mesh')}
    original=trimesh.load(io.BytesIO(assets['CustomLegFoot.stl']),file_type='stl',process=True)
    extent=float(-original.vertices[:,1].min())
    asset=root.findall('asset')[-1]
    generated={}
    for i,leg in enumerate(LEGS):
        if length[i]==0 and bend[i]==0:continue
        body=root.find(f'.//body[@name="leg_{leg}_3"]')
        mesh=trimesh.Trimesh(vertices=deform(original.vertices,length[i],bend[i],extent),faces=original.faces,process=False)
        for kind,shape in [('visual',mesh),('floor',mesh.convex_hull)]:
            name=f'formation_{leg}_{kind}';file=name+'.stl'
            assets[file]=shape.export(file_type='stl')
            generated[file]=hashlib.sha256(assets[file]).hexdigest()
            ET.SubElement(asset,'mesh',name=name,file=file,scale='.001 .001 .001')
            suffix='shin_visual' if kind=='visual' else 'floor_contact'
            body.find(f'geom[@name="{leg}_{suffix}"]').set('mesh',name)
        for site in body.findall('site'):
            pos=np.fromstring(site.get('pos'),sep=' ')*1000
            if i%2:pos[:2]*=-1
            pos=deform(pos[None,:],length[i],bend[i],extent)[0]
            if i%2:pos[:2]*=-1
            site.set('pos',' '.join(f'{x/1000:.14g}' for x in pos))
    xml=ET.tostring(root,encoding='utf-8')
    manifest=dict(nominal_manifest)
    manifest.update(nominal_model_sha256=nominal_manifest['model_sha256'],model_sha256=hashlib.sha256(xml).hexdigest(),
        formation=dict(length_mm=length.tolist(),bend_deg=bend.tolist()),
        formation_status='Synthetic sensitivity geometry; user reports 10 mm longest-shortest spread, rigid limbs',
        deformation='Smooth distal-tip length/bend beyond 40 mm in negative mesh Y; attachment and Z gap unchanged',
        distal_anchor_mm=ANCHOR_MM,nominal_tip_extent_mm=extent,generated_asset_sha256=generated,
        inertia_status='Nominal mass/COM/inertia retained; redistribution due to shape is not modeled',
        deformable_material=False)
    return xml,assets,manifest
