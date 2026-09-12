from pathlib import Path
import hashlib,xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
MODEL=ROOT/'training/wheel_align/model.xml'
def rotation(e):
 w,x,y,z=np.array([float(x) for x in e.get('quat','1 0 0 0').split()]);n=np.linalg.norm([w,x,y,z]);w,x,y,z=np.array([w,x,y,z])/n
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def vector(e,k='pos'):return np.array([float(x) for x in e.get(k,'0 0 0').split()])
def generate():
 root=ET.parse(MODEL).getroot();base=root.find('.//body[@name="base_link"]');box=next(g for g in base.findall('geom') if g.get('type')=='box')
 boxes=[(vector(box),rotation(box),vector(box,'size'))];heater=base.find('body[@name="heating_module"]')
 for g in heater.findall('geom'):
  if g.get('type')=='box':boxes.append((vector(heater)+rotation(heater)@vector(g),rotation(heater)@rotation(g),vector(g,'size')))
 offsets=[]
 for leg in ('front_r','front_l','back_r','back_l'):
  g=root.find(f'.//geom[@name="leg_{leg}_3_wheel_collision"]');p=vector(g)
  assert np.allclose(p[:2],0) and np.allclose(vector(g,'size'),[.048,.01675]);offsets.append(p[2])
 assert np.allclose(offsets,offsets[0])
 def array(name,v):
  v=np.asarray(v).ravel();return f'inline constexpr std::array<double,{len(v)}> {name}'+'{'+','.join(format(x,'.17g') for x in v)+'};\n'
 s='#pragma once\n#include <array>\nnamespace keyframe_align::model_geometry {\n'
 s+='// Generated from model.xml SHA256 '+hashlib.sha256(MODEL.read_bytes()).hexdigest()+'\n'
 s+=f'inline constexpr double wheel_offset={offsets[0]:.17g};\ninline constexpr int box_count={len(boxes)};\n'
 for name,index in [('box_positions',0),('box_rotations',1),('box_sizes',2)]:s+=array(name,[b[index] for b in boxes])
 return s+'}\n'
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();path=Path(__file__).with_name('model_geometry.hpp');s=generate()
 if args.check:assert path.read_text()==s,'Regenerate model_geometry.hpp after XML edits'
 else:path.write_text(s)
