#!/usr/bin/env python3
"""Offline export of the pinned roll controller's nominal kinematics and fixtures."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE='221e16681952bb9350bd2c1019c2b092c347432a'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    src=a.source.resolve()
    # Source may be a materialized Git archive; compare bytes rather than trust its path.
    for path in list((src/'motion/inverted_triangle').glob('*.py'))+list((src/'motion/inverted_triangle').glob('*.xml'))+list((src/'motion/inverted_triangle').glob('*.json')):
        expected=subprocess.check_output(['git','show',SOURCE+':'+path.relative_to(src).as_posix()],cwd=ROOT)
        if path.read_bytes().replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):raise ValueError('Edited source: '+str(path))
    sys.path.insert(0,str(src))
    import numpy as np
    import mujoco as mj
    from motion.inverted_triangle.core import Robot,KP,KD,LEGS
    from motion.inverted_triangle.balanced_support import BalancedSupport
    r=Robot();goal=np.tile([1.,0.,-1.,-1.,0.,1.],2)
    initial=goal.copy();initial[[1,7]]=-.29;initial[[4,10]]=.29
    initial[[2,8]]+=np.pi;initial[[5,11]]-=np.pi
    config=json.loads(subprocess.check_output(['git','show',SOURCE+':motion/inverted_triangle/roll_validation/selected_controller.json'],cwd=ROOT))
    v=lambda x:'V3('+','.join(format(float(t),'.17g') for t in x)+')'
    links=[];tips=[];records=[]
    for i,body in enumerate(r.bodies):
        chain=[r.m.body_parentid[r.m.body_parentid[body]],r.m.body_parentid[body],body]
        for j,b in enumerate(chain):
            assert r.m.body_jntnum[b]==1 and np.max(abs(r.m.jnt_pos[1+3*i+j]))==0
            if j==0:assert r.m.body_parentid[b]==r.base
            q=r.m.body_quat[b];axis=r.m.jnt_axis[1+3*i+j]
            links.append('{'+v(r.m.body_pos[b])+',Q4('+','.join(format(float(t),'.17g') for t in q)+'),'+v(axis)+','+format(float(r.m.body_mass[b]),'.17g')+','+v(r.m.body_ipos[b])+','+format(float(r.m.qpos0[7+3*i+j]),'.17g')+'}')
        local=(r.points(i)[r.tip_masks[i]]-r.d.xpos[body])@r.d.xmat[body].reshape(3,3)
        tips.append('{'+','.join(v(x) for x in local)+'}')
    header='#pragma once\n#include <Eigen/Geometry>\n#include <array>\n#include <vector>\nnamespace triangle_roll {\nusing V3=Eigen::Vector3d;using Q4=Eigen::Quaterniond;\nstruct Link { V3 pos;Q4 rot;V3 axis;double mass;V3 com;double ref; };\n'
    header+='inline const std::array<Link,12> links={{'+','.join(links)+'}};\n'
    header+='inline const std::array<std::vector<V3>,4> tips={{'+','.join(tips)+'}};\n}\n'
    evidence=ROOT/'hardware_testing/triangle_roll';evidence.mkdir(parents=True,exist_ok=True)
    dest=ROOT/'ros2_ws/src/neural_controller/include/neural_controller/triangle_roll_geometry.hpp'
    dest.write_text(header,encoding='utf-8',newline='\n')
    plan=dict(schema_version=2,source_commit=SOURCE,model_sha256=r.manifest['model_sha256'],axial_gap_m=.009,hz=520,
        joint_names=['leg_'+leg+'_'+str(j) for leg in LEGS for j in [1,2,3]],initial=initial.tolist(),goal=goal.tolist(),kp=KP.tolist(),kd=KD.tolist(),
        controller=config,geometry_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),walking_enabled=False)
    plan['plan_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    planpath=ROOT/'ros2_ws/src/neural_controller/launch/triangle_roll_plan.json'
    planpath.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8',newline='\n')
    contract=ROOT/'ros2_ws/src/neural_controller/include/neural_controller/triangle_roll_contract.hpp'
    contract.write_text('#pragma once\nnamespace triangle_roll { inline constexpr const char* contract_json=R"ROLL('+json.dumps(plan,separators=(',',':'))+')ROLL"; }\n',encoding='utf-8',newline='\n')
    rng=np.random.default_rng(431);feedback=BalancedSupport(goal,config['balanced_support'])
    for k in range(520):
        q=goal+rng.uniform(-.15,.15,12);qd=rng.uniform(-.1,.1,12);command=goal+rng.uniform(-.12,.12,12)
        rot=np.array([1.,*rng.uniform(-.04,.04,3)]);rot/=np.linalg.norm(rot)
        R=np.empty(9);mj.mju_quat2Mat(R,rot);gravity=R.reshape(3,3).T@np.array([0.,0.,-1.])
        output=feedback.update(rot,q,qd,command,10/520)
        records.append(np.r_[gravity,q,qd,command,output,feedback.estimated_load_N])
    np.savetxt(ROOT/'ros2_ws/src/neural_controller/test/triangle_roll_support_reference.csv',records,delimiter=',',fmt='%.17g')
    (evidence/'export_provenance.json').write_text(json.dumps(dict(source_commit=SOURCE,model_sha256=plan['model_sha256'],plan_sha256=plan['plan_sha256'],export_sha256=hashlib.sha256(planpath.read_bytes()).hexdigest(),geometry_sha256=plan['geometry_sha256'],fixture_rows=len(records),fixture_scope='Sequential 52 Hz BalancedSupport updates over 10 seconds, random joint and IMU inputs, seed 431'),indent=2)+'\n')
    print(plan['plan_sha256'])

if __name__=='__main__':main()
