"""Prepare only model/spacing changes, preserving the paused controller work."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import importlib.util
import numpy as np
import mujoco as mj

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALIGN = REPO.parent / 'Pupper_backpack_align'
REL = 'training/wheel_align/model.xml'


def main():
    spec = importlib.util.spec_from_file_location('propagate', HERE / 'propagate.py')
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    for source in (REPO / 'models/heating_module').iterdir():
        if source.is_file():
            shutil.copy2(source, ALIGN / 'models/heating_module' / source.name)
    helper = p.module(ALIGN)
    current = ALIGN / REL
    before = mj.MjModel.from_xml_path(str(current))
    helper.patch_file(current)
    after = mj.MjModel.from_xml_path(str(current))
    # Other task already applied the gap: keep its actuator/model changes intact.
    for name in ['body_pos','body_quat','body_mass','body_ipos','body_inertia',
                 'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','key_qpos']:
        np.testing.assert_allclose(getattr(before, name), getattr(after, name), atol=1e-10)
    candidate = current.with_name('.backpack-spacing-for-commit.xml')
    candidate.write_bytes(subprocess.check_output(['git','show','HEAD:'+REL],cwd=ALIGN))
    helper.patch_file(candidate)
    once = candidate.read_bytes(); helper.patch_file(candidate); assert once == candidate.read_bytes()
    record = p.verify(p.baseline(ALIGN,current),candidate,.009)
    record['path'] = str(current)
    record['review_xml'] = str(candidate)
    record['scope'] = 'Commit candidate preserves HEAD actuator settings; paused controller changes remain in working tree.'
    # Extract the same wheel centers from the current XML; actuator type is irrelevant here.
    subprocess.run([sys.executable,'-m','training.wheel_align.generate_geometry'],cwd=ALIGN,check=True)
    subprocess.run([sys.executable,'-m','training.wheel_align.generate_geometry','--check'],cwd=ALIGN,check=True)
    subprocess.run([sys.executable,'-m','motion.keyframe_align.generate_geometry'],cwd=ALIGN,check=True)
    header_rel = 'ros2_ws/src/neural_controller/include/neural_controller/wheel_align_geometry_data.hpp'
    old = (REPO / header_rel).read_text()
    new = (ALIGN / header_rel).read_text()
    # Existing FK/body-box constants must be identical before sharing the new center offsets.
    import re
    for key, values in re.findall(r'\b(\w+)\{([^{}]+)\};',old):
        target = re.search(r'\b'+key+r'\{([^{}]+)\};',new)
        assert target, key
        np.testing.assert_allclose(np.fromstring(values,sep=','),np.fromstring(target[1],sep=','),atol=1e-12)
    (REPO / header_rel).write_bytes(new.encode())
    # Prepare the preexisting controller's geometry change for the index only.
    geometry_rel = 'motion/keyframe_align/geometry.hpp'
    geometry = subprocess.check_output(['git','show','HEAD:'+geometry_rel],cwd=ALIGN).decode()
    assert geometry.count('.03035') == 1
    geometry = geometry.replace('.03035','wheel_center_z[k]')
    (HERE / 'alignment-geometry-for-commit.hpp').write_bytes(geometry.encode())
    report_path = HERE / 'propagation-validation.json'
    report = json.loads(report_path.read_text())
    report['models'] = [r for r in report['models'] if 'Pupper_backpack_align' not in r['path']] + [record]
    report['alignment_generated_geometry_matches'] = True
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: alignment gap/backpack candidate validated; paused controller edits preserved')


if __name__ == '__main__': main()
