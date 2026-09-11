import json
from pathlib import Path
import numpy as np
import pytest
from training.wheel_align import configs as c,contract as ct,schedule,rewards
from training.wheel_align.select_checkpoint import select,digest


def test_verify_recovers_smoothly_and_does_not_restore_bad_correction(monkeypatch):
    q=c.DEFAULT_POSE.copy();s=ct.select(ct.reset(q,q[ct.WHEEL]),1,q)
    s=ct.prepare(s);s['phase']=np.asarray(ct.VERIFY);s['progress']=np.asarray(1.)
    s['motion_reference']=c.APEX_POSES[1].copy();s['desired']=s['motion_reference']+.02
    s['applied']=s['desired'].copy();home=s['home'].copy()
    monkeypatch.setattr(ct.geometry,'ready',lambda *a:False)
    monkeypatch.setattr(ct.geometry,'margins',lambda *a:(.025,.009,.020))
    for tick in range(106):
        previous=s['desired'].copy()
        s,wheels=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),np.ones(8))
        assert not s['was_rotating'] or tick>=10
        assert np.max(np.abs(s['desired']-previous))<=.02*c.CONTROL_DT/c.RECOVERY_SECONDS+1e-10
        old=s['velocity'].copy();s=ct.integrate(s)
        assert np.max(np.abs(s['velocity']-old))<=2*c.PHYSICS_DT+1e-10
        if tick==10:
            monkeypatch.setattr(ct.geometry,'ready',lambda *a:True)
            monkeypatch.setattr(ct.geometry,'margins',lambda *a:(.025,.025,.020))
    assert s['residual_gain']==0
    np.testing.assert_allclose(s['desired'],s['motion_reference'])
    assert ct.observation(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.])).shape==(83,)
    s['phase']=np.asarray(ct.HOLD);s=ct.prepare(s);s=ct.select(s,2,q);s=ct.prepare(s)
    assert s['residual_gain']==1
    np.testing.assert_equal(s['home'],home)


def test_timeout_waits_for_actual_lower_and_settle():
    m=ct.reset(c.DEFAULT_POSE,np.zeros(4));m['phase']=np.asarray(ct.ROTATE)
    s=schedule.reset();s['age']=np.asarray(c.LEG_TIMEOUT_STEPS)
    s,command=schedule.advance(s,m,np.arange(1,5),count=1)
    assert command==0 and s['timeouts'][1] and s['index']==1
    m['phase']=np.asarray(ct.LOWER)
    for _ in range(600):
        s,_=schedule.advance(s,m,np.arange(1,5),count=1)
        assert not schedule.finished(s,m,1)
    m['phase']=np.asarray(ct.HOLD)
    for _ in range(103):s,_=schedule.advance(s,m,np.arange(1,5),count=1)
    assert not schedule.finished(s,m,1)
    s,_=schedule.advance(s,m,np.arange(1,5),count=1)
    assert schedule.finished(s,m,1) and not m['completed'].any()


def test_sustained_residual_is_penalized():
    m=ct.reset(c.DEFAULT_POSE,np.zeros(4));m['phase']=np.asarray(ct.ROTATE)
    m['last_action']=np.ones(8)
    assert rewards.residual_cost(np.ones(8),m,c.CONTROL_DT)>0
    assert rewards.residual_cost(np.zeros(8),m,c.CONTROL_DT)==0
    m['phase']=np.asarray(ct.LOWER)
    assert rewards.residual_cost(np.ones(8),m,c.CONTROL_DT)==0


def setup_run(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(motion_contract_version=c.MOTION_VERSION,
        source_hashes={'source':'unchanged'},curriculum_stage='foundation')))
    for step in (0,100,200):(tmp_path/f'params_{step}').write_bytes(str(step).encode())
    (tmp_path/'mjx_params').write_bytes(b'preserve-final')


def test_selection_preserves_final_and_requires_holdout(tmp_path):
    setup_run(tmp_path);seen=[]
    def audit(path,**kwargs):
        seen.append((path.name,kwargs['seed']))
        return dict(**kwargs,checkpoint_sha256=digest(path),passes_simulation_gate=path.name=='params_100')
    report=select(tmp_path,audit,{'source':'unchanged'})
    assert report['step']==100 and len(seen)==3
    assert Path(report['params']).read_bytes()==b'100'
    assert (tmp_path/'mjx_params').read_bytes()==b'preserve-final'
    assert (tmp_path/'selection/step-200/audit-foundation.json').exists()
    assert seen[-1][1]==20260912
    # A cached, passing candidate must remain reproducible without another audit.
    again=select(tmp_path,lambda *a,**kw:pytest.fail('Unexpected repeat audit'),{'source':'unchanged'})
    assert again['checkpoint_sha256']==report['checkpoint_sha256']


def test_selection_stops_if_holdout_fails(tmp_path):
    setup_run(tmp_path)
    def audit(path,**kwargs):
        return dict(**kwargs,checkpoint_sha256=digest(path),passes_simulation_gate=kwargs['seed']==20260910)
    result=select(tmp_path,audit,{'source':'unchanged'})
    assert result['status']=='failed' and not (tmp_path/'selected_checkpoint.json').exists()
    with pytest.raises(ValueError):select(tmp_path,audit,{'source':'changed'})
    with pytest.raises(ValueError):select(tmp_path,audit,{'source':'unchanged'},16)
