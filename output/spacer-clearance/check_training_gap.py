"""Offline validation of the two training XML edits; no policies or hardware."""
from pathlib import Path
import ast, hashlib, json, re, subprocess, importlib.util
import numpy as np
import mujoco as mj

REL='Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
OUT=Path(__file__).resolve().parent
results=[]
for branch in ['leg','wheel']:
    root=Path(f'C:/Users/tundt/Desktop/Pupper_gap_{branch}')
    path=root/REL
    before=subprocess.check_output(['git','show','HEAD:'+REL],cwd=root).decode()
    old=mj.MjModel.from_xml_string(before.replace('../meshes/stl/',(path.parent.parent/'meshes/stl').as_posix()+'/'))
    new=mj.MjModel.from_xml_path(str(path))
    assert (old.nq,old.nv,old.nu)==(new.nq,new.nv,new.nu)==(19,18,12)
    for key in ['body_pos','body_quat','body_mass','body_inertia','body_iquat',
                'jnt_pos','jnt_axis','jnt_range','jnt_limited','qpos0','key_qpos','key_ctrl',
                'actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange',
                'geom_size','geom_quat','geom_contype','geom_conaffinity','site_quat']:
        np.testing.assert_array_equal(getattr(old,key),getattr(new,key),err_msg=key)
    changed_bodies=[new.body('leg_'+leg+'_3').id for leg in ['front_r','front_l','back_r','back_l']]
    for positions,owners in [('geom_pos','geom_bodyid'),('site_pos','site_bodyid')]:
        expected=np.zeros_like(getattr(new,positions))
        expected[np.isin(getattr(new,owners),changed_bodies),2]=.009
        np.testing.assert_allclose(getattr(new,positions)-getattr(old,positions),expected,atol=1e-12)
    expected=np.zeros_like(new.body_ipos);expected[changed_bodies,2]=.009
    np.testing.assert_allclose(new.body_ipos-old.body_ipos,expected,atol=1e-12)
    a,b=mj.MjData(old),mj.MjData(new);rng=np.random.default_rng(42)
    for _ in range(20):
        q=old.qpos0.copy();q[2]=.3;q[7:]+=rng.uniform(-.5,.5,12)
        a.qpos[:]=q;b.qpos[:]=q;mj.mj_forward(old,a);mj.mj_forward(new,b)
        np.testing.assert_allclose(a.xpos,b.xpos,atol=1e-12)
        for bid in changed_bodies:
            delta=b.xmat[bid].reshape(3,3)[:,2]*.009
            for positions,owners in [('geom_xpos','geom_bodyid'),('site_xpos','site_bodyid')]:
                ids=np.flatnonzero(getattr(new,owners)==bid)
                actual=getattr(b,positions)[ids]-getattr(a,positions)[ids]
                np.testing.assert_allclose(actual,np.tile(delta,(len(ids),1)),atol=1e-12)
    # Exercise the contact/inertia model at the existing home controls, no policy.
    mj.mj_resetDataKeyframe(new,b,0)
    for _ in range(500):mj.mj_step(new,b)
    assert np.all(np.isfinite(b.qpos)) and np.all(np.isfinite(b.qvel))
    assert not np.any(b.warning.number),b.warning.number
    if branch=='wheel':
        py=root/'mujoco_playground/workspace/configs.py'
        tree=ast.parse(py.read_text())
        value=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WHEEL_CENTER_LOCAL_Z' for t in n.targets))
        for leg in ['front_r','front_l','back_r','back_l']:
            assert abs(new.geom('leg_'+leg+'_3_wheel_collision').pos[2]-value)<1e-12
    else:
        py=root/'mujoco_playground/workspace/tools/prepare_walk_model.py'
        compile(py.read_text(),str(py),'exec')
        assert '0.029884' not in py.read_text() and '0.038884' in py.read_text()
        spec=importlib.util.spec_from_file_location('walk_geometry',root/'mujoco_playground/workspace/walk_geometry.py')
        geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
        _,_,old_ring,_=geometry.load_geometry(old,path)
        _,_,new_ring,_=geometry.load_geometry(new,path)
        expected=np.zeros_like(old_ring);expected[:,:,2]=.009
        np.testing.assert_allclose(new_ring-old_ring,expected,atol=1e-12)
    results.append(dict(branch=branch,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),validated_limbs=4,random_fk_poses=20,uncontrolled_home_steps=500,mujoco_warnings=0,xml_path=str(path)))
    print('PASS',branch,'exact 9 mm shift of visuals, contacts, sites and CoM; joints and actuation preserved; model loads and steps',flush=True)
(OUT/'training-gap-validation.json').write_text(json.dumps(results,indent=2))
