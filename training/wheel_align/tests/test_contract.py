import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from training.wheel_align import configs as c,contract as ct,geometry

def test_reference_paths_have_self_clearance():
    # Geometry-only path check. Actual loaded dynamics are checked separately.
    for command in range(1,5):
        q=c.DEFAULT_POSE.copy();s=ct.select(ct.reset(q,q[ct.WHEEL]),command,q)
        for _ in range(400):
            s=ct.prepare(s);q[ct.POS]=s['motion_reference']
            _,gap,body=geometry.margins(q,np.array([0.,0.,-1.]),ct.leg(s))
            assert gap>.010 and body>.005


def test_interrupt_limits_and_completion():
    q=c.DEFAULT_POSE.copy();s=ct.reset(q,q[ct.WHEEL]);s=ct.select(s,1,q)
    for n in range(800):
        s=ct.prepare(s);s,_=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),np.zeros(8))
        for _ in range(10):
            previous=s['velocity'].copy();s=ct.integrate(s)
            assert np.max(np.abs(s['velocity']-previous))/c.PHYSICS_DT<=2+1e-9
        q[ct.POS]=s['applied'];s=ct.finish(s,q,np.zeros(12))
        if n==70:s=ct.select(s,2,q)
        if n>70 and s['phase']==ct.LOWER:assert s['active_command']==1
        if n>70 and s['phase']==ct.HOLD:break
    assert s['phase']==ct.HOLD and not s['completed'][1]
    assert np.max(np.abs(s['velocity']))<.03

def test_cpp_trace_parity():
    exe=os.environ.get('ALIGN_MOTION_TEST_EXE')
    if not exe:pytest.skip('Compile align_motion_test and set ALIGN_MOTION_TEST_EXE for cross-language parity')
    rng=np.random.default_rng(20260910);q=c.DEFAULT_POSE.copy();q[ct.WHEEL]=[.3,-.4,.5,-.6]
    s=ct.reset(q,q[ct.WHEEL]);records=[];expected=[];phases=set()
    for n in range(10000):
        dt=c.CONTROL_DT;cmd=[0,1,2,3,4][min(n//2000,4)]
        if n%2000 in range(100,150):cmd=0
        qd=np.zeros(12);q[ct.POS]=s['applied']
        if s['phase'] in (ct.ROTATE,ct.VERIFY):q[3*ct.leg(s)+2]=s['target'][ct.leg(s)]
        a=rng.uniform(-.03,.03,8);angular=np.zeros(3);g=np.array([0.,0.,-1.])
        records.append(f'{dt} {cmd} '+ ' '.join(map(str,np.r_[q,qd,angular,g,a])))
        s=ct.finish(s,q,qd);s=ct.select(s,cmd,q);s=ct.prepare(s,dt)
        prefix=np.r_[s['phase'],s['active_command'],s['gate_steps'],s['settled_steps'],s['progress'],s['motion_reference']]
        phases.add(int(s['phase']));s,w=ct.begin(s,q,qd,angular,g,a,dt)
        for _ in range(10):s=ct.integrate(s,dt/10)
        expected.append(np.r_[prefix,s['applied'],s['velocity'],w,s['completed'],s['residual_gain'],geometry.margins(q,g,ct.leg(s))])
    proc=subprocess.run([exe,'trace'],input='\n'.join(records)+'\n',text=True,capture_output=True,check=True)
    actual=np.array([[float(v) for v in line.split()] for line in proc.stdout.splitlines()])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-10)
    assert phases==set(range(6))

def test_geometry_matches_mujoco():
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(c.MODEL_PATH));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);d.qpos[:7]=[0,0,0,1,0,0,0]
    ids=[m.geom(n).id for n in c.WHEEL_COLLISION_GEOM_NAMES]
    rng=np.random.default_rng(42)
    for _ in range(50):
        q=c.DEFAULT_POSE+rng.uniform(-.15,.15,12);d.qpos[7:]=q;mujoco.mj_forward(m,d)
        p,axis=geometry.wheel_frames(q)
        np.testing.assert_allclose(p,d.geom_xpos[ids],atol=1e-12)
        np.testing.assert_allclose(axis,d.geom_xmat[ids].reshape(4,3,3)[:,:,2],atol=1e-12)
    assert np.all(m.geom_contype[ids]==2) and np.all(m.geom_conaffinity[ids]==1)
