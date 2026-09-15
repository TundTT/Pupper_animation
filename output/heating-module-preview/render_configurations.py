"""Render the exact XML geometry for review, without stepping a policy."""
from pathlib import Path
import json
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def main():
    records=json.loads((HERE/'propagation-validation.json').read_text())['models']
    for config,title in [('wheel','Wheel'),('leg','Leg'),('align','Alignment'),('triangle','Point-up triangle / flip')]:
        record=next(r for r in records if f'Pupper_backpack_{config}' in r['path'])
        path=Path(record.get('review_xml',record['path']))
        model=mj.MjModel.from_xml_path(str(path));data=mj.MjData(model)
        mj.mj_resetDataKeyframe(model,data,0);mj.mj_forward(model,data)
        bottom=float('inf')
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid]==0 or model.geom_type[gid]!=mj.mjtGeom.mjGEOM_MESH or model.geom_group[gid]!=1:continue
            mid=model.geom_dataid[gid];start=model.mesh_vertadr[mid];n=model.mesh_vertnum[mid]
            vertices=model.mesh_vert[start:start+n]@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            bottom=min(bottom,float(vertices[:,2].min()))
        data.qpos[2]-=bottom;mj.mj_forward(model,data)
        model.vis.global_.offwidth=900;model.vis.global_.offheight=680
        opt=mj.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
        cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
        cam.lookat[:]=data.xpos[model.body('base_link').id]+[0,0,.01]
        board=Image.new('RGB',(1800,1595),'#f2f5f7');draw=ImageDraw.Draw(board)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
        bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',34)
        draw.text((24,12),title+' | Approved backpack + 9 mm spacer offset',font=bold,fill='#173044')
        draw.text((24,58),'Actual XML / STL geometry | Static pose for visual review',font=font,fill='#314d60')
        with mj.Renderer(model,width=900,height=680) as renderer:
            for index,(label,az,el) in enumerate([('Angled view',135,-22),('Side',90,0),('Top',90,-90),('End view',0,-5)]):
                cam.azimuth=az;cam.elevation=el;cam.distance=.60 if label=='Angled view' else .52
                renderer.update_scene(data,camera=cam,scene_option=opt)
                renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False
                image=Image.fromarray(renderer.render())
                col,row=index%2,index//2;x,y=col*900,105+row*715
                draw.text((x+20,y),label,font=font,fill='#173044');board.paste(image,(x,y+30))
                image.save(HERE/f'approved_{config}_{label.lower().replace(" ","_")}.png')
        draw.text((24,1560),'Orange: 0.602 kg module | Spacer offset is modeled; the printed spacer solid is not included.',font=font,fill='#314d60')
        target=HERE/f'approved_{config}_views.png';board.save(target);print(target,flush=True)


if __name__=='__main__':main()
