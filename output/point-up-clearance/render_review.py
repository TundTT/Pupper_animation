"""Render the measured CAD geometry and kinematic-only XML candidates."""
import copy
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from geometry_review import OUT, Review
from surface_check import TIP, shortened

def build(review,length):
    root=copy.deepcopy(review.root)
    root.find('compiler').set('meshdir','')
    # Hold the displayed pose if opened in a simulator; this is not a controller model.
    root.remove(root.find('actuator'))
    root.find('option').set('gravity','0 0 0')
    root.find('keyframe/key').attrib.pop('ctrl',None)
    for joint in root.iter('joint'):
        if joint.get('name','').endswith('_3'):joint.set('limited','false')
    mesh_dir=OUT/'candidate_meshes';mesh_dir.mkdir(exist_ok=True)
    if length != TIP:
        name=f'CustomLegFoot_tip_{length*1000:g}mm.stl'
        shortened(review.meshes['CustomLegFoot'],length).export(mesh_dir/name)
    for e in root.iter('mesh'):
        if e.get('name')=='CustomLegFoot' and length != TIP:
            e.set('file','candidate_meshes/'+name);e.set('scale','1 1 1')
        else:e.set('file','source/description/meshes/stl/'+e.get('file'))
    # This file is for geometry inspection only. Physical parameters are not recalibrated.
    root.insert(0,ET.Comment(' KINEMATIC PREVIEW ONLY. Actuators removed, gravity disabled, hub limits disabled for viewing. No validated control, inertia update, or SMP constitutive model. '))
    root.find('keyframe/key').set('qpos',' '.join(map(str,review.model.qpos0)))
    path=OUT/f'point_up_tip_{length*1000:.1f}mm.xml'
    q=review.model.qpos0.copy();q[2]=.1313;q[9]+=np.pi
    root.find('keyframe/key').set('qpos',' '.join(map(str,q)))
    ET.indent(root)
    path.write_text(ET.tostring(root,encoding='unicode'),encoding='utf8')
    model=mj.MjModel.from_xml_path(str(path))
    return model,path

def render(model,spin=np.pi,hip=0,abd=1,angle=-90):
    data=mj.MjData(model);data.qpos[:]=model.qpos0;data.qpos[2]=.1313;data.qpos[7]=abd;data.qpos[8]=hip;data.qpos[9]+=spin;mj.mj_forward(model,data)
    model.vis.global_.offwidth=900;model.vis.global_.offheight=700
    active={model.body('leg_front_r_'+str(j)).id:j for j in [1,2,3]}
    colors={1:[.3,.42,.5,1],2:[.66,.7,.72,1],3:[.94,.52,.18,1]}
    for g in range(model.ngeom):
        bid=int(model.geom_bodyid[g])
        if bid not in active:model.geom_rgba[g,3]=0
        elif model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:model.geom_rgba[g]=colors[active[bid]]
    opt=mj.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
    camera=mj.MjvCamera();mj.mjv_defaultCamera(camera)
    camera.lookat[:]=[.078,-.103,.105];camera.distance=.255;camera.azimuth=angle;camera.elevation=-2;camera.orthographic=1
    with mj.Renderer(model,height=700,width=900) as renderer:
        renderer.update_scene(data,camera=camera,scene_option=opt)
        return Image.fromarray(renderer.render())

def main():
    review=Review();original,_=build(review,TIP);short,_=build(review,.054)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    canvas=Image.new('RGB',(1800,1710),'#eef2f5');draw=ImageDraw.Draw(canvas)
    items=[(original,np.pi,0,1,90,'CURRENT | 62.7 mm point-up','CAD surfaces intersect'),
           (short,np.pi,0,1,90,'CANDIDATE | 54 mm point-up','About 5.7 mm upper-assembly clearance'),
           (original,np.pi,0,1,45,'CURRENT | oblique view','Blue-gray: upper motor assembly; orange: shin'),
           (short,np.pi,0,1,45,'54 mm | oblique view','Hub, mount, side petals and thickness retained')]
    for i,(m,spin,hip,abd,az,title,subtitle) in enumerate(items):
        x=(i%2)*900;y=(i//2)*820
        draw.text((x+24,y+18),title,font=font,fill='#1b3142')
        draw.text((x+24,y+58),subtitle,font=small,fill='#334f60')
        canvas.paste(render(m,spin,hip,abd,az),(x,y+105))
    draw.text((24,1650),'Actual leg-branch CAD, front-right limb | Static geometry study; not a hardware test or validated morph trajectory',font=small,fill='#334f60')
    canvas.save(OUT/'point-up-comparison.png')
    print(OUT/'point-up-comparison.png')

if __name__=='__main__':main()
