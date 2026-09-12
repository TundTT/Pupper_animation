"""Actual CAD preview: original versus translated full-length shin."""
from pathlib import Path
import sys, copy, xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'point-up-clearance'))
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from geometry_review import Review, vector
from render_review import render

OUT=Path(__file__).resolve().parent
r=Review()
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
canvas=Image.new('RGB',(1800,1640),'#eef2f5');draw=ImageDraw.Draw(canvas)
for col,mm in enumerate([0,16]):
    root=copy.deepcopy(r.root)
    assets={e.get('file'):(r.source/'meshes/stl'/e.get('file')).read_bytes() for e in root.iter('mesh')}
    for body in root.iter('body'):
        if body.get('name','').endswith('_3'):
            for g in body.findall('geom'):
                if g.get('mesh')=='CustomLegFoot':
                    pos=vector(g.get('pos'),[0,0,0]);pos[2]+=mm/1000
                    g.set('pos',' '.join(map(str,pos)))
    m=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'),assets)
    for row,angle in enumerate([0,45]):
        x=col*900;y=row*800
        draw.text((x+22,y+14),f'{mm} mm added spacer | original 62.7 mm tip',font=font,fill='#1b3142')
        draw.text((x+22,y+55),('Side view across hub axis' if row==0 else 'Oblique view')+' | point-up pose',font=small,fill='#334f60')
        canvas.paste(render(m,angle=angle),(x,y+95))
draw.text((20,1604),'Blue-gray: upper motor assembly | Orange: full shin | Spacer hardware itself is not modeled',font=small,fill='#334f60')
canvas.save(OUT/'spacer-comparison.png')
print(OUT/'spacer-comparison.png')
