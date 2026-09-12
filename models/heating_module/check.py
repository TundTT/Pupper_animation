"""Check active backpack/9 mm-gap MJCFs without running a policy or robot."""
from pathlib import Path
import argparse
import json
import re
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def default_models():
    for relative in ['motion/inverted_triangle/model.xml','training/wheel_align/model.xml',
                     'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml']:
        path=REPO/relative
        if path.exists():return [path]
    return sorted((REPO/'ros2_ws/src/pupper_v3_description/description/mujoco_xml').glob('pupper_v3_complete*.xml'))


def check(path):
    spec=json.loads((HERE/'spec.json').read_text())
    root=ET.fromstring(re.sub(r'<!--.*?-->','',path.read_text(),flags=re.S))
    assert len(root.findall(".//body[@name='heating_module']"))==1
    marker=root.find("./custom/numeric[@name='quadmorph_additional_hub_gap_m']")
    assert marker is not None and float(marker.get('data'))==.009
    for leg in ['front_r','front_l','back_r','back_l']:
        body=root.find(f".//body[@name='leg_{leg}_3']")
        mesh=next((g for g in body.findall('geom') if g.get('mesh') in ('AdaptiveWheel','CustomLegFoot')),None)
        if mesh is None:
            expected_com_z=.02733 # original legacy COM 18.33 mm plus 9 mm
        else:
            np.testing.assert_allclose(np.fromstring(mesh.get('pos'),sep=' ')[2],.0226,atol=1e-12)
            expected_com_z=.03898 if mesh.get('mesh')=='AdaptiveWheel' else .039671
            cylinder=body.find('geom[@type="cylinder"]')
            if cylinder is not None:
                np.testing.assert_allclose(np.fromstring(cylinder.get('pos'),sep=' ')[2],.03935,atol=1e-12)
        np.testing.assert_allclose(np.fromstring(body.find('inertial').get('pos'),sep=' ')[2],expected_com_z,atol=1e-12)
    model=mj.MjModel.from_xml_path(str(path));body=model.body('heating_module')
    assert model.body_jntnum[body.id]==0
    np.testing.assert_allclose(body.pos,spec['body_pos_m'],atol=1e-12)
    np.testing.assert_allclose(body.quat,spec['body_quat_wxyz'],atol=1e-12)
    np.testing.assert_allclose(body.mass,spec['mass_lb']*.45359237,atol=1e-12)
    np.testing.assert_allclose(body.ipos,np.array(spec['com_in'])*.0254,atol=1e-12)
    rotation=np.zeros(9);mj.mju_quat2Mat(rotation,body.iquat);rotation=rotation.reshape(3,3)
    tensor=rotation@np.diag(body.inertia)@rotation.T
    expected=np.array(spec['inertia_lb_in2'])*(.45359237*.0254**2)
    assert np.max(abs(tensor-expected))<1e-7*np.linalg.norm(expected,ord=2)
    data=mj.MjData(model)
    if model.nkey:mj.mj_resetDataKeyframe(model,data,0)
    mj.mj_forward(model,data)
    assert np.isfinite(data.qM).all() and np.isfinite(data.qacc).all()
    return dict(path=str(path),mass_kg=float(model.body_mass.sum()),status='PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml',nargs='*',type=Path)
    args=parser.parse_args();paths=args.xml or default_models()
    assert paths,'No active models found; pass XML paths explicitly'
    for path in paths:print(json.dumps(check(path)))
