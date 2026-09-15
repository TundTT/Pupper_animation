"""Stage this task only; leave the paused alignment controller edits untouched."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOTS={'robot-code':REPO,**{'codex/backpack-'+n:REPO.parent/('Pupper_backpack_'+n) for n in ['wheel','leg','align','triangle']}}
MODULE_FILES=['.gitattributes','README.md','apply.py','gap.py','check.py','spec.json','Heating_module.stl','review']


def git(root,*args,input=None):
    return subprocess.check_output(['git','-c','core.safecrlf=false',*args],cwd=root,input=input)


def stage_blob(root,relative,contents):
    oid=git(root,'hash-object','-w','--stdin',input=contents).decode().strip()
    git(root,'update-index','--add','--cacheinfo','100644',oid,relative)


def main():
    for branch,root in ROOTS.items():
        if git(root,'diff','--cached','--name-only').strip():
            raise RuntimeError('Unexpected preexisting staged changes: '+branch)
        paths=['models/heating_module/'+f for f in MODULE_FILES]
        if branch=='robot-code':
            paths += [str(p.relative_to(root)).replace('\\','/') for p in
                      (root/'ros2_ws/src/pupper_v3_description/description/mujoco_xml').glob('pupper_v3_complete*.xml')]
            paths += ['ros2_ws/src/pupper_v3_description/description/meshes/stl/Heating_module.stl',
                      'ros2_ws/src/pupper_v3_description/scripts/create_mujoco_xml.py']
            paths += ['ros2_ws/src/neural_controller/include/neural_controller/'+f for f in
                      ['wheel_align_motion.hpp','wheel_align_geometry_data.hpp','keyframe_align/geometry.hpp']]
        elif branch.endswith(('-wheel','-leg')):
            paths += ['Stanford/training/pupper_v3_description/'+f for f in
                      ['description/mujoco_xml/pupper_v3_complete.mjx.position.xml',
                       'description/meshes/stl/Heating_module.stl','scripts/create_mujoco_xml.py']]
        elif branch.endswith('-triangle'):
            paths += ['motion/inverted_triangle/'+f for f in
                      ['build_model.py','model.xml','source_manifest.json','assets/Heating_module.stl']]
        else:
            paths += ['training/wheel_align/'+f for f in
                      ['generate_geometry.py','geometry.py','geometry.json','meshes/Heating_module.stl']]
            paths += ['ros2_ws/src/neural_controller/include/neural_controller/'+f for f in
                      ['wheel_align_motion.hpp','wheel_align_geometry_data.hpp']]
        git(root,'add','--sparse','--',*paths)
        if branch.endswith('-align'):
            stage_blob(root,'training/wheel_align/model.xml',
                       (root/'training/wheel_align/.backpack-spacing-for-commit.xml').read_bytes())
            stage_blob(root,'motion/keyframe_align/geometry.hpp',
                       (HERE/'alignment-geometry-for-commit.hpp').read_bytes())
            # Add only this task's provenance fields to the original runner.
            for relative in ['motion/keyframe_align/audit_suite.py','motion/keyframe_align/simulate.py']:
                text=git(root,'show','HEAD:'+relative).decode()
                if relative.endswith('audit_suite.py'):
                    anchor="    files={str(p.relative_to(root))"
                    addition="    paths += [root/'models/heating_module'/f for f in ('spec.json','apply.py','gap.py','Heating_module.stl')]\n    paths += [root/'training/wheel_align/meshes/Heating_module.stl']\n"
                else:
                    anchor="    report['dependency_sha256']="
                    addition="    dependencies += ['models/heating_module/'+name for name in ('spec.json','apply.py','gap.py','Heating_module.stl')]\n    dependencies += ['training/wheel_align/meshes/Heating_module.stl']\n"
                assert text.count(anchor)==1
                stage_blob(root,relative,text.replace(anchor,addition+anchor).encode())
        git(root,'diff','--cached','--check')
        print('Staged',branch,flush=True)
    # Record the bytes actually staged, so newline normalization cannot invalidate hashes.
    report=json.loads((HERE/'propagation-validation.json').read_text())
    for item in report['models']:
        absolute=Path(item['path'])
        branch,root=next((b,r) for b,r in ROOTS.items() if absolute.is_relative_to(r))
        relative=absolute.relative_to(root).as_posix()
        item['branch']=branch;item['path']=relative
        item['sha256']=hashlib.sha256(git(root,'show',':'+relative)).hexdigest()
        item.pop('review_xml',None)
    report['hash_reference']='Git blob bytes, after staging and text normalization'
    report['generators']=['models/heating_module/apply.py','models/heating_module/gap.py',
                          'description composer hooks','motion/inverted_triangle/build_model.py']
    report.update(mujoco_version='3.3.7',triangle_rebuild_identical=True,
                  triangle_source_manifest_verified=True,original_training_checkouts_unchanged=True)
    raw=(json.dumps(report,indent=2)+'\n').encode()
    geometry=(HERE/'spacing-geometry-validation.json').read_text().encode()
    for branch,root in ROOTS.items():
        (root/'models/heating_module/validation.json').write_bytes(raw)
        (root/'models/heating_module/spacing-geometry-validation.json').write_bytes(geometry)
        git(root,'add','--sparse','--','models/heating_module/validation.json','models/heating_module/spacing-geometry-validation.json')
        git(root,'diff','--cached','--check')
    print('PASS: staged models, review images, and normalized validation records')


if __name__=='__main__':main()
