import json
from types import SimpleNamespace
import numpy as np
import pytest
from training.wheel_align import configs as c,contract as ct,contact_metrics,schedule,curriculum

def test_hovering_and_self_contact_do_not_earn_support_load():
    contact=SimpleNamespace(efc_address=np.array([0,4,8]),dist=np.array([-.001,.004,-.001]),
        geom=np.array([[0,7],[0,13],[19,25]]))
    loads=contact_metrics.wheel_loads(contact,np.ones(12),np.array([7,13,19,25]),0)
    np.testing.assert_equal(loads,[4,0,0,0])

def test_peak_impact_cost_is_not_diluted_or_refunded():
    peak=np.zeros(4);clear=np.array([0.,.1,.1,.1]);speed=np.array([.22,0.,0.,0.])
    peak,cost=contact_metrics.approach_cost(peak,clear,speed)
    assert cost>30
    peak,again=contact_metrics.approach_cost(peak,clear,speed);assert again==0
    peak,refund=contact_metrics.approach_cost(peak,np.ones(4),np.zeros(4));assert refund==0
    _,next_event=contact_metrics.approach_cost(peak,clear,speed);assert next_event==cost

def test_force_decoder_matches_native_mujoco():
    import mujoco
    m=mujoco.MjModel.from_xml_path(str(c.MODEL_PATH));d=mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m,d,0);d.ctrl[ct.POS]=ct.NEUTRAL
    for _ in range(1560):mujoco.mj_step(m,d)
    wheels=np.array([m.geom(n).id for n in c.WHEEL_COLLISION_GEOM_NAMES])
    expected=np.zeros(4)
    for i in range(d.ncon):
        force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
        if 0 in d.contact.geom[i] and d.contact.dist[i]<=0:
            for k,wheel in enumerate(wheels):
                if wheel in d.contact.geom[i]:expected[k]+=force[0]
    np.testing.assert_allclose(contact_metrics.wheel_loads(d.contact,d.efc_force,wheels,0),expected,atol=1e-10)
    assert np.all(expected>0)

def test_actor_cannot_cancel_lift_or_change_verification_and_lowering():
    q=c.DEFAULT_POSE.copy();s=ct.select(ct.reset(q,np.zeros(4)),1,q)
    for _ in range(400):s=ct.prepare(s)
    s,_=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),np.ones(8))
    assert s['desired'][3]<=s['motion_reference'][3]
    s['phase']=np.asarray(ct.ROTATE);obs_rotate=ct.observation(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]))
    saved=s['desired'].copy();s['phase']=np.asarray(ct.VERIFY)
    obs_verify=ct.observation(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]))
    np.testing.assert_array_equal(obs_rotate,obs_verify)
    s,_=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),-np.ones(8))
    np.testing.assert_array_equal(saved,s['desired'])
    s['phase']=np.asarray(ct.LOWER);s=ct.prepare(s)
    a,_=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),np.ones(8))
    b,_=ct.begin(s,q,np.zeros(12),np.zeros(3),np.array([0.,0.,-1.]),-np.ones(8))
    np.testing.assert_array_equal(a['desired'],b['desired'])

def test_sequence_waits_for_completion_and_records_timeout():
    q=c.DEFAULT_POSE.copy();m=ct.select(ct.reset(q,np.zeros(4)),1,q)
    order=np.arange(1,5);s=schedule.reset()
    for _ in range(32*52):s,command=schedule.advance(s,m,order)
    assert command==1 and s['index']==0  # v3 would have cancelled here
    m['completed'][1]=True;m['phase']=np.asarray(ct.HOLD)
    s,command=schedule.advance(s,m,order);assert s['index']==1 and command==0
    m['phase']=np.asarray(ct.LIFT);s['age']=np.asarray(c.LEG_TIMEOUT_STEPS)
    s,command=schedule.advance(s,m,order);assert s['timeouts'][0] and s['index']==2

def test_interruption_gets_time_to_retry_same_wheel():
    q=c.DEFAULT_POSE.copy();m=ct.select(ct.reset(q,np.zeros(4)),1,q);s=schedule.reset()
    commands=[]
    for _ in range(52*52):
        s,command=schedule.advance(s,m,np.arange(1,5),interrupt=True);commands.append(int(command))
    assert commands[519:623]==[0]*104
    assert s['index']==0 and commands[-1]==1 and not s['timeouts'].any()

def test_warm_start_checks_contract_and_provenance(tmp_path,monkeypatch):
    from brax.io import model
    p=tmp_path/'mjx_params';p.write_bytes(b'TEST ONLY')
    cfg=dict(motion_contract_version=4,source_hashes={'a':'b'},curriculum_stage='foundation')
    (tmp_path/'config.json').write_text(json.dumps(cfg))
    params=object();monkeypatch.setattr(model,'load_params',lambda path:params)
    result,meta=curriculum.initialization(p,'single',{'a':'b'})
    assert result is params and meta['stage']=='foundation' and 'fresh optimizer' in meta['semantics']
    with pytest.raises(ValueError):curriculum.initialization(p,'sequence',{'a':'b'})
    with pytest.raises(ValueError):curriculum.initialization(p,'single',{'a':'changed'})
    cfg['motion_contract_version']=3;(tmp_path/'config.json').write_text(json.dumps(cfg))
    with pytest.raises(ValueError):curriculum.initialization(p,'single',{'a':'b'})

def test_generated_runtime_reference_matches():
    from training.wheel_align.generate_reference import OUTPUT,render
    assert OUTPUT.read_text()==render()
