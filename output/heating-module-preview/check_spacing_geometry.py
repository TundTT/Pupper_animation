"""Independent MuJoCo/Python/C++ wheel-center and clearance checks; no rollout."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALIGN = REPO.parent / 'Pupper_backpack_align'


def linux(path):
    p = str(Path(path).resolve()).replace('\\', '/')
    return '/mnt/' + p[0].lower() + p[2:]


def main():
    template = '''#include <iostream>
#include <iomanip>
#include "HEADER"
int main(){std::array<double,12> q;std::array<double,3> g;
 std::cout<<std::setprecision(17);
 while(std::cin>>q[0]){for(int i=1;i<12;++i)std::cin>>q[i];for(auto& x:g)std::cin>>x;
 for(int k=0;k<4;++k){auto m=neural_controller::WheelAlignMotion::margins(q,g,k);for(auto x:m)std::cout<<x<<' ';}
 for(int k=0;k<4;++k){auto m=keyframe_align::Geometry::margins(q,g,k);for(auto x:m)std::cout<<x<<' ';}std::cout<<'\\n';}}
'''
    sys.path.insert(0,str(ALIGN))
    from training.wheel_align import geometry
    xml = ALIGN / 'training/wheel_align/.backpack-spacing-for-commit.xml'
    m = mj.MjModel.from_xml_path(str(xml)); d = mj.MjData(m)
    rng = np.random.default_rng(20260912)
    rows=[]; expected=[]; actual_floor=[]; max_center_error=0.; max_axis_error=0.
    names=['front_r','front_l','back_r','back_l']
    for _ in range(300):
        q=np.tile([1.,0.,0.,-1.,0.,0.],2)+rng.uniform(-.3,.3,12)
        quat=rng.normal(size=4);quat/=np.linalg.norm(quat)
        mj.mj_resetDataKeyframe(m,d,0);d.qpos[7:]=q;d.qpos[3:7]=quat;mj.mj_forward(m,d)
        rot=d.xmat[m.body('base_link').id].reshape(3,3);g=rot.T@np.array([0.,0.,-1.])
        centers,axes=geometry.wheel_frames(q)
        bottoms=[]
        for i,name in enumerate(names):
            gid=m.geom(f'leg_{name}_3_wheel_collision').id
            observed_center=(d.geom_xpos[gid]-d.xpos[m.body('base_link').id])@rot
            observed_axis=rot.T@d.geom_xmat[gid].reshape(3,3)[:,2]
            max_center_error=max(max_center_error,float(np.max(abs(centers[i]-observed_center))))
            max_axis_error=max(max_axis_error,float(np.max(abs(axes[i]-observed_axis))))
            az=d.geom_xmat[gid].reshape(3,3)[2,2]
            bottoms.append(d.geom_xpos[gid,2]-.048*np.sqrt(max(0,1-az*az))-.01675*abs(az))
        rows.append(np.r_[q,g]);expected.append([geometry.margins(q,g,k) for k in range(4)])
        actual_floor.append([bottoms[k]-min(bottoms[j] for j in range(4) if j!=k) for k in range(4)])
    assert max_center_error<1e-10 and max_axis_error<1e-10
    payload=''.join(' '.join(format(x,'.17g') for x in r)+'\n' for r in rows)
    results={}
    variants=[('robot',REPO/'ros2_ws/src/neural_controller/include/neural_controller/keyframe_align/geometry.hpp',REPO),
              ('align',HERE/'alignment-geometry-for-commit.hpp',ALIGN)]
    for name,header,root in variants:
        cpp=HERE/f'geometry-check-{name}.cpp';cpp.write_text(template.replace('HEADER',linux(header)))
        binary='/tmp/quadmorph-xml-spacing-'+name
        subprocess.run(['wsl','-d','Ubuntu','-e','g++','-std=c++17','-O2',
                        '-I'+linux(root/'ros2_ws/src/neural_controller/include'),linux(cpp),'-o',binary],check=True)
        output=subprocess.check_output(['wsl','-d','Ubuntu','-e',binary],input=payload.encode()).decode()
        values=np.fromstring(output,sep=' ').reshape(300,8,3)
        np.testing.assert_allclose(values[:,:4],expected,atol=1e-12)
        conservative_gap=np.array(actual_floor)-values[:,4:,0]
        assert conservative_gap.min()>-1e-10 and conservative_gap.max()<.0051
        results[name]=dict(python_cpp_max_error=float(np.max(abs(values[:,:4]-expected))),
                           floor_bound_slack_min_m=float(conservative_gap.min()),
                           floor_bound_slack_max_m=float(conservative_gap.max()))
    record=dict(poses=300,limbs_per_pose=4,seed=20260912,
                wheel_center_z_m=geometry.PARAMS['wheel_center_z'],
                mujoco_center_max_error_m=max_center_error,mujoco_axis_max_error=max_axis_error,
                cpp=results,status='geometry consistency only; no motion or policy validation')
    (HERE/'spacing-geometry-validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
