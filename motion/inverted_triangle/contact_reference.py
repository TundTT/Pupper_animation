"""Explicit rigid_flush contact counterfactual, retaining CAD and backpack.

Geometry follows the selected training run j3xez9z5's effective capsule values.
This changes physical contact at a declared model boundary, never secretly in the
primary CAD-floor handoff. Distorted capsules are synthetic flush approximations.
"""
import hashlib,xml.etree.ElementTree as ET
import numpy as np
from .core import HERE,LEGS

def build(xml,assets,manifest):
    root=ET.fromstring(xml)
    if assets is None:assets={m.get('file'):(HERE/'assets'/m.get('file')).read_bytes() for m in root.iter('mesh')}
    params=[]
    for i,leg in enumerate(LEGS):
        proximal=np.array([.00063,-.012,.038884]);tip=np.array([.00063,-.0626499610000004,.038884])
        formation=manifest.get('formation')
        if formation:
            from .formation import deform
            tip=deform((tip*1000)[None,:],formation['length_mm'][i],formation['bend_deg'][i],manifest['nominal_tip_extent_mm'])[0]/1000
        axis=(tip-proximal)/np.linalg.norm(tip-proximal);distal=tip-.012*axis
        if i%2:proximal[:2]*=-1;distal[:2]*=-1
        g=root.find(f'.//geom[@name="{leg}_floor_contact"]')
        for key in ['mesh','pos','quat']:g.attrib.pop(key,None)
        g.set('type','capsule');g.set('size','.012');g.set('fromto',' '.join(map(str,np.r_[proximal,distal])))
        g.set('solimp','.99 .99 .001 .5 2');g.set('solref','.008 1')
        params.append(dict(leg=leg,fromto=np.r_[proximal,distal].tolist(),radius_m=.012))
    output=ET.tostring(root,encoding='utf-8');result=dict(manifest)
    result.update(contact_parent_model_sha256=manifest['model_sha256'],model_sha256=hashlib.sha256(output).hexdigest(),contact_reference=dict(type='rigid_flush',training_run='j3xez9z5',capsules=params,formation_approximation='Fit capsule distal surface to deformed nominal tip; not an exact uneven CAD hull',backpack_retained=True))
    return output,assets,result
