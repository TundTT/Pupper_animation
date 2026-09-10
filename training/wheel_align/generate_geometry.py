"""Regenerate Python/C++ fixed geometry from model.xml; no simulation or training."""
import argparse
import json
import re
from pathlib import Path
import mujoco
import numpy as np
from .configs import MODEL_PATH

ROOT=Path(__file__).resolve().parents[2]

def generated():
    m=mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    legs=['front_r','front_l','back_r','back_l']
    ids=[[m.body(f'leg_{leg}_{j}').id for j in (1,2,3)] for leg in legs]
    def rotation(quat):
        out=np.empty(9);mujoco.mju_quat2Mat(out,quat);return out.reshape(3,3).tolist()
    data={}
    for key,j in [('p1',0),('p3',2)]:data[key]=[m.body_pos[row[j]].tolist() for row in ids]
    for key,j in [('r1',0),('r2',1),('r3',2)]:data[key]=[rotation(m.body_quat[row[j]]) for row in ids]
    for row in ids:
        assert np.allclose(m.body_pos[row[1]],0), 'FK needs extension for a nonzero second joint offset'
    boxes=np.where((m.geom_bodyid==m.body('base_link').id)&(m.geom_type==mujoco.mjtGeom.mjGEOM_BOX))[0]
    assert len(boxes)==1
    b=boxes[0];data.update(box_pos=m.geom_pos[b].tolist(),box_rot=rotation(m.geom_quat[b]),box_size=m.geom_size[b].tolist())
    header='#pragma once\n#include <array>\n#include <cmath>\n#include <algorithm>\nnamespace neural_controller {\nnamespace align_geometry {\n'
    for key,value in data.items():
        v=np.asarray(value).ravel()
        header+=f'inline constexpr std::array<double, {len(v)}> {key}{{'+', '.join(format(x,'.17g') for x in v)+'};\n'
    header+='}\n}\n'
    return data,header

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');args=p.parse_args()
    data,header=generated();json_path=MODEL_PATH.with_name('geometry.json')
    cpp_path=ROOT/'ros2_ws/src/neural_controller/include/neural_controller/wheel_align_geometry_data.hpp'
    if args.check:
        saved=json.loads(json_path.read_text());cpp=cpp_path.read_text()
        for key,value in data.items():
            np.testing.assert_allclose(saved[key],value,atol=1e-14,rtol=0)
            block=re.search(r'\b'+key+r'\{([^}]+)\}',cpp).group(1)
            np.testing.assert_allclose(np.fromstring(block,sep=','),np.asarray(value).ravel(),atol=1e-14,rtol=0)
        print('PASS: generated geometry matches XML')
    else:
        json_path.write_text(json.dumps(data,indent=2)+'\n');cpp_path.write_text(header)
        print('Updated geometry.json and wheel_align_geometry_data.hpp; rebuild, test and retrain after geometry changes.')

if __name__=='__main__':main()
