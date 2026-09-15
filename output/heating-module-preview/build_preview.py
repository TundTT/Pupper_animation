"""Build one review-only backpack MJCF from the pinned 9 mm wheel model.

Never modifies training checkouts, hardware configuration, or policy artifacts.
Run from this repository with Python containing MuJoCo 3.3.7, numpy and Pillow.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
REF = '88f6a8888644c8072df36d4878d77684972c9231'
PREFIX = 'Stanford/training/pupper_v3_description/description/'
SOURCE = Path('C:/Users/tundt/Downloads/Heating module.stl')


def git_blob(path):
    return subprocess.check_output(['git', 'show', REF + ':' + PREFIX + path], cwd=REPO)


def numbers(values):
    return ' '.join(format(float(v), '.14g') for v in values)


def clean_parse(raw):
    # Historical source has double hyphens inside XML comments. Remove comments
    # for strict parsing, without changing any model element or numeric attribute.
    return ET.fromstring(re.sub(rb'<!--.*?-->', b'', raw, flags=re.S))


def main():
    spec = json.loads((OUT / 'module_spec.json').read_text())
    meshdir = OUT / 'meshes'
    meshdir.mkdir(exist_ok=True)
    raw = git_blob('mujoco_xml/pupper_v3_complete.mjx.position.xml')
    root = clean_parse(raw)
    assets = {}
    for entry in root.iter('mesh'):
        filename = entry.get('file')
        contents = git_blob('meshes/stl/' + filename)
        if contents.startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError('Resolve source LFS mesh before generating: ' + filename)
        (meshdir / filename).write_bytes(contents)
        assets[filename] = contents
    root.find('compiler').set('meshdir', 'meshes')
    baseline = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'), assets)
    mass = spec['mass_lb'] * .45359237
    com = np.array(spec['com_in']) * .0254
    tensor = np.array(spec['inertia_lb_in2']) * (.45359237 * .0254 ** 2)
    eigenvalues = np.linalg.eigvalsh(tensor)
    assert np.all(eigenvalues > 0)
    assert eigenvalues[-1] < eigenvalues[0] + eigenvalues[1]
    meshpath = meshdir / 'Heating_module.stl'
    if not meshpath.exists():
        shutil.copy2(SOURCE, meshpath)
    ET.SubElement(root.find('asset'), 'mesh', name='heating_module_mesh',
                  file=meshpath.name, scale=numbers([spec['stl_to_m']] * 3))
    base = root.find(".//body[@name='base_link']")
    base.append(ET.Comment(' Backpack review candidate: fixed body; no joint or added actuator. '
                           'CAD COM and full tensor remain in the original CAD axes. '
                           'Change this body pose to move visual, collision and inertia together. '))
    body = ET.SubElement(base, 'body', name='heating_module',
                         pos=numbers(spec['body_pos_m']), quat=numbers(spec['body_quat_wxyz']))
    ET.SubElement(body, 'inertial', pos=numbers(com), mass=numbers([mass]),
                  fullinertia=numbers(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]))
    ET.SubElement(body, 'geom', name='heating_module_visual', type='mesh',
                  mesh='heating_module_mesh', group='1', contype='0', conaffinity='0',
                  density='0', rgba='0.96 0.49 0.08 1')
    body.append(ET.Comment(' Conservative box contact proxies only; visual CAD is unchanged. '
                           'Explicit inertial above prevents geometry-based double counting. '))
    for proxy in spec['collision_boxes_cad_mm']:
        bounds = np.array(proxy['bounds']) * .001
        ET.SubElement(body, 'geom', name='heating_module_' + proxy['name'], type='box',
                      pos=numbers(bounds.mean(0)), size=numbers((bounds[1] - bounds[0]) / 2),
                      density='0', group='3', contype='1', conaffinity='1',
                      rgba='0.9 0.35 0.08 0.25', friction='0.8 0.02 0.01')
    ET.indent(root, space='  ')
    xmlpath = OUT / 'wheel_gap9_backpack_preview.xml'
    xmlpath.write_text(ET.tostring(root, encoding='unicode') + '\n', encoding='utf-8')
    model = mj.MjModel.from_xml_path(str(xmlpath))
    bid = model.body('heating_module').id
    assert model.body_jntnum[bid] == 0
    assert (model.nq, model.nv, model.nu) == (baseline.nq, baseline.nv, baseline.nu)
    assert np.isclose(model.body_mass.sum() - baseline.body_mass.sum(), mass, atol=1e-12)
    # Name-based comparisons avoid assumptions about compiled body/geom indices.
    for i in range(1, baseline.nbody):
        original = baseline.body(i)
        added = model.body(original.name)
        for field in ['pos', 'quat', 'mass', 'ipos', 'iquat', 'inertia']:
            np.testing.assert_allclose(getattr(original, field), getattr(added, field), atol=1e-12)
    for field in ['jnt_pos', 'jnt_axis', 'jnt_range', 'qpos0', 'actuator_gainprm',
                  'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange', 'key_qpos']:
        np.testing.assert_allclose(getattr(model, field), getattr(baseline, field), atol=1e-12)
    principal_rotation = np.zeros(9)
    mj.mju_quat2Mat(principal_rotation, model.body_iquat[bid])
    rotation = principal_rotation.reshape(3, 3)
    reconstructed = rotation @ np.diag(model.body_inertia[bid]) @ rotation.T
    # MuJoCo's iterative eigensolver has finite tolerance. Require tensor error
    # below 1e-7 of its spectral norm, far below the screenshot's rounding.
    tensor_error = float(np.max(np.abs(reconstructed - tensor)))
    assert tensor_error < 1e-7 * np.linalg.norm(tensor, ord=2)
    d = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, d, 0)
    mj.mj_forward(model, d)
    assert np.isfinite(d.qM).all() and np.isfinite(d.qacc).all()
    module_to_base = np.zeros(9)
    mj.mju_quat2Mat(module_to_base, model.body_quat[bid])
    r = module_to_base.reshape(3, 3)
    record = dict(source_branch='codex/9mm-gap-wheel', source_commit=REF,
                  source_xml_sha256=hashlib.sha256(raw).hexdigest(),
                  heating_mesh_sha256=hashlib.sha256(meshpath.read_bytes()).hexdigest(),
                  preview_xml_sha256=hashlib.sha256(xmlpath.read_bytes()).hexdigest(),
                  mujoco_version=mj.__version__, inertia_roundtrip_error=tensor_error,
                  mass_kg=mass, cad_com_m=com.tolist(),
                  cad_com_inertia_kg_m2=tensor.tolist(), principal_inertias=eigenvalues.tolist(),
                  module_com_in_base_m=(model.body_pos[bid] + r @ com).tolist(),
                  original_robot_mass_kg=float(baseline.body_mass.sum()),
                  robot_plus_module_mass_kg=float(model.body_mass.sum()),
                  nq=model.nq, nv=model.nv, actuators=model.nu,
                  checks=['loads successfully', 'positive physical inertia', 'full tensor round trip',
                          'exact mass increment', 'fixed attachment without added joint',
                          'existing body inertias and transforms unchanged',
                          'joints, actuators, keyframes unchanged', 'finite forward dynamics'],
                  status='Placement review only; no policy rollout, training, hardware motion or propagation.')
    (OUT / 'validation.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
