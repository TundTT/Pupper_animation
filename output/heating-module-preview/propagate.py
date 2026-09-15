"""Apply the approved module to isolated derivative checkouts and local ROS XMLs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import runpy
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DESKTOP = REPO.parent
SHARED = REPO / 'models/heating_module'
LOCOMOTION = 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
ROS = 'ros2_ws/src/pupper_v3_description'


def parse(path):
    return ET.fromstring(re.sub(rb'<!--.*?-->', b'', path.read_bytes(), flags=re.S))


def module(root):
    spec = importlib.util.spec_from_file_location('backpack_apply', root / 'models/heating_module/apply.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hydrate(root, path):
    """Materialize only referenced meshes in a sparse checkout, from its own HEAD."""
    tree = parse(path)
    compiler = tree.find('compiler')
    meshdir = path.parent / compiler.get('meshdir', '.')
    for mesh in tree.iter('mesh'):
        name = mesh.get('file')
        if not name:
            continue
        target = (meshdir / name).resolve()
        if target.exists():
            continue
        relative = target.relative_to(root).as_posix()
        contents = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=root)
        if contents.startswith(b'version https://git-lfs'):
            oid = re.search(rb'oid sha256:([0-9a-f]+)', contents)[1].decode()
            cached = REPO / '.git/lfs/objects' / oid[:2] / oid[2:4] / oid
            if cached.exists():
                contents = cached.read_bytes()
            else:
                original = DESKTOP / 'Pupper_alignment_keyframes' / relative
                contents = original.read_bytes()
            assert hashlib.sha256(contents).hexdigest() == oid
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)


def baseline(root, path):
    raw = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(root).as_posix()], cwd=root)
    tree = ET.fromstring(re.sub(rb'<!--.*?-->', b'', raw, flags=re.S))
    compiler = tree.find('compiler')
    compiler.set('meshdir', str((path.parent / compiler.get('meshdir', '.')).resolve()))
    return mj.MjModel.from_xml_string(ET.tostring(tree, encoding='unicode'))


def verify(before, path, gap_delta=0):
    after = mj.MjModel.from_xml_path(str(path))
    assert (before.nq, before.nv, before.nu) == (after.nq, after.nv, after.nu)
    np.testing.assert_allclose(after.body_mass.sum() - before.body_mass.sum(), .60191707499, atol=1e-12, rtol=0)
    body = after.body('heating_module')
    assert after.body_jntnum[body.id] == 0
    np.testing.assert_allclose(body.pos, [-.036349545, .000031565, .032506671263], atol=1e-13)
    np.testing.assert_allclose(body.quat, [.5, -.5, -.5, .5], atol=1e-13)
    for i in range(1, before.nbody):
        old = before.body(i)
        new = after.body(old.name)
        shifted_limb = bool(re.fullmatch(r'leg_(front|back)_[rl]_3', old.name))
        for field in ['pos', 'quat', 'mass', 'ipos', 'iquat', 'inertia']:
            expected_value = np.array(getattr(old, field), copy=True)
            if field == 'ipos' and shifted_limb:
                expected_value[2] += gap_delta
            np.testing.assert_allclose(expected_value, getattr(new, field), atol=1e-12)
        for kind in ['geom', 'site']:
            old_ids = np.flatnonzero(getattr(before, kind + '_bodyid') == old.id)
            new_ids = np.flatnonzero(getattr(after, kind + '_bodyid') == new.id)
            assert len(old_ids) == len(new_ids)
            expected_pos = getattr(before, kind + '_pos')[old_ids].copy()
            if shifted_limb:
                expected_pos[:, 2] += gap_delta
            np.testing.assert_allclose(expected_pos, getattr(after, kind + '_pos')[new_ids], atol=1e-10)
            for field in ['quat', 'size']:
                np.testing.assert_allclose(getattr(before, kind + '_' + field)[old_ids],
                                           getattr(after, kind + '_' + field)[new_ids], atol=1e-10)
    for field in ['jnt_pos', 'jnt_axis', 'jnt_range', 'jnt_limited', 'qpos0',
                  'actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange',
                  'actuator_forcerange', 'key_qpos']:
        np.testing.assert_allclose(getattr(before, field), getattr(after, field), atol=1e-12)
    rot = np.zeros(9)
    mj.mju_quat2Mat(rot, body.iquat)
    tensor = rot.reshape(3, 3) @ np.diag(body.inertia) @ rot.reshape(3, 3).T
    spec = json.loads((SHARED / 'spec.json').read_text())
    expected = np.array(spec['inertia_lb_in2']) * (.45359237 * .0254**2)
    assert np.max(np.abs(tensor - expected)) < 1e-7 * np.linalg.norm(expected, ord=2)
    data = mj.MjData(after)
    if after.nkey:
        mj.mj_resetDataKeyframe(after, data, 0)
    mj.mj_forward(after, data)
    assert np.isfinite(data.qM).all() and np.isfinite(data.qacc).all()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mass_before_kg=float(before.body_mass.sum()), mass_after_kg=float(after.body_mass.sum()),
                additional_hub_gap_m=.009, gap_translation_applied_m=gap_delta,
                nq=after.nq, nv=after.nv, nu=after.nu, status='structural checks passed; no motion audit')


def composer_hook(path):
    text = path.read_text()
    if '# Approved heating backpack' in text:
        return
    anchor = '    print("Writing to:", output_path)'
    assert text.count(anchor) == 1
    hook = '''    # Approved heating backpack: insert after other geometry processing.
    import runpy
    repository = next(p for p in pathlib.Path(__file__).resolve().parents
                      if (p / 'models/heating_module/apply.py').exists())
    backpack = runpy.run_path(str(repository / 'models/heating_module/apply.py'))
    compiler = root.find('compiler')
    backpack['add_to_tree'](root, output_path.parent / compiler.get('meshdir', '.'))

'''
    path.write_text(text.replace(anchor, hook + anchor))


def main(skip_align=False):
    targets = [('wheel', LOCOMOTION), ('leg', LOCOMOTION),
               ('align', 'training/wheel_align/model.xml'),
               ('triangle', 'motion/inverted_triangle/model.xml')]
    if skip_align:
        targets = [(name, relative) for name, relative in targets if name != 'align']
    report = dict(placement_revision=3, approved=True, models=[], generators=[],
                  previous_motion_results_apply=False)
    for name, relative in targets:
        root = DESKTOP / ('Pupper_backpack_' + name)
        dest = root / 'models/heating_module'
        dest.mkdir(parents=True, exist_ok=True)
        for source in SHARED.iterdir():
            if source.is_file():
                shutil.copy2(source, dest / source.name)
        path = root / relative
        hydrate(root, path)
        before = baseline(root, path)
        raw = subprocess.check_output(['git', 'show', 'HEAD:' + relative], cwd=root)
        source_root = ET.fromstring(re.sub(rb'<!--.*?-->', b'', raw, flags=re.S))
        gap_delta = .009 - float(runpy.run_path(str(SHARED / 'gap.py'))['current_gap'](source_root))
        if name == 'triangle':
            builder = path.with_name('build_model.py')
            text = builder.read_text()
            anchor = "    root.insert(0,ET.Comment(' POST-COOLED"
            hook = '''    # Approved heating backpack: floor-only proxies preserve this model's mask contract.
    import runpy
    backpack = runpy.run_path(str(REPO/'models/heating_module/apply.py'))
    backpack_manifest = backpack['add_to_tree'](root, assets, floor_only=True)
    hashes['Heating_module.stl'] = backpack_manifest['mesh_sha256']
'''
            assert text.count(anchor) == 1
            if '# Approved heating backpack' not in text:
                text = text.replace(anchor, hook + anchor)
                text = text.replace("    (HERE/'source_manifest.json').write_text", "    manifest['heating_module'] = backpack_manifest\n    manifest['modeling'] += '; approved fixed 0.60191707499 kg heating backpack'\n    (HERE/'source_manifest.json').write_text")
            builder.write_bytes(text.encode('utf-8'))
            subprocess.run([sys.executable, '-m', 'motion.inverted_triangle.build_model'], cwd=root, check=True)
            report['generators'].append(str(builder))
        else:
            helper = module(root)
            helper.patch_file(path)
            once = path.read_bytes()
            helper.patch_file(path)
            assert path.read_bytes() == once, 'Backpack insertion must be idempotent'
            if name in ['wheel', 'leg']:
                composer = root / 'Stanford/training/pupper_v3_description/scripts/create_mujoco_xml.py'
                composer_hook(composer)
                report['generators'].append(str(composer))
        report['models'].append(verify(before, path, gap_delta))
        print('Validated', name, flush=True)
    # ROS simulator's floating, fixed and backlash entrypoints, plus its existing
    # MJX/position variants. Do not touch archived output or third-party examples.
    helper = module(REPO)
    for path in sorted((REPO / ROS / 'description/mujoco_xml').glob('pupper_v3_complete*.xml')):
        before = baseline(REPO, path)
        helper.patch_file(path)
        report['models'].append(verify(before, path, .009))
        print('Validated', path.name, flush=True)
    composer = REPO / ROS / 'scripts/create_mujoco_xml.py'
    composer_hook(composer)
    report['generators'].append(str(composer))
    (HERE / 'propagation-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Validated', len(report['models']), 'models')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-align', action='store_true', help='Leave the concurrently edited alignment checkout untouched')
    main(parser.parse_args().skip_align)
