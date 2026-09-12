"""Static CAD pose comparison, explicitly not a dynamics or success video."""
import argparse
import json
from pathlib import Path
import subprocess
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw
from .core import LEGS, HERE
from .roll_to_stand import initialize

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r,home,goal=initialize()
    old=json.loads(subprocess.check_output(['git','show',
        '8904c2a:motion/inverted_triangle/results/robustness_20260912/policy_compatibility.json'],cwd=HERE.parents[1]))
    r.m.vis.global_.offwidth=640;r.m.vis.global_.offheight=480
    for leg in LEGS:r.m.geom_rgba[r.m.geom(leg+'_shin_visual').id]=[1.,.55,.12,1.]
    renderer=mj.Renderer(r.m,height=480,width=640)
    sheet=Image.new('RGB',(1920,480));camera=mj.MjvCamera()
    camera.distance=.65;camera.azimuth=100;camera.elevation=-18
    records=[]
    try:
        for i,(title,q) in enumerate([('Inverted start',home),('OLD endpoint: not walk-ready',old['final_command']),('NEW target: walking default',goal)]):
            r.d.qpos[:]=r.initial;r.d.qpos[7:]=q;mj.mj_forward(r.m,r.d)
            r.d.qpos[2]-=float(r.bottoms().min());mj.mj_forward(r.m,r.d)
            camera.lookat[:]=[0,0,.12];renderer.update_scene(r.d,camera=camera)
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,50),fill='black');draw.text((8,7),title,fill='white')
            draw.text((8,27),'STATIC CAD TARGET | 9 mm + backpack | not a motion result',fill='white')
            sheet.paste(im,(i*640,0));records.append(dict(label=title,qpos=r.d.qpos.tolist()))
    finally:renderer.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output)
    a.output.with_suffix('.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__':main()
