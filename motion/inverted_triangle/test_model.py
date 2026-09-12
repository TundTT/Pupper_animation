"""Physical-contract tests; no robot or learned policy is involved."""
import numpy as np
import mujoco as mj
from .core import Robot,HUB,PROX,LEGS,smooth
from .search_lift import envelope

def test_all_inverted_shins_carry_weight():
    r=Robot();target=r.initial[7:].copy()
    for _ in range(520*5):r.tick(target)
    assert r.tilt()<np.deg2rad(1)
    assert np.all(r.contacts_precise()>4)
    assert abs(r.contacts_precise().sum()-r.m.body_mass.sum()*9.81)<.6
    assert np.max(np.abs(r.bottoms()))<.001
    assert r.unintended_floor_force()<.1

def test_no_hub_stops_and_floor_only_shin_hulls():
    r=Robot()
    assert np.all(r.m.jnt_limited[1:][HUB]==0)
    assert np.all(r.m.jnt_limited[1:][PROX]==1)
    assert np.all(r.m.geom_contype[r.geoms]==0)
    assert np.all(r.m.geom_conaffinity[r.geoms]==1)
    np.testing.assert_allclose(r.m.actuator_ctrlrange,np.tile([-3,3],(12,1)))

def test_sweep_bound_dominates_sampled_rotation():
    r=Robot();rng=np.random.default_rng(4)
    for leg in range(4):
        for _ in range(5):
            r.reset();r.d.qpos[7:][PROX]+=rng.uniform(-.25,.25,8);mj.mj_forward(r.m,r.d)
            bound=envelope(r,leg);angle=r.d.qpos[7+HUB[leg]]
            for spin in np.linspace(0,2*np.pi,25):
                r.d.qpos[7+HUB[leg]]=angle+spin;mj.mj_forward(r.m,r.d)
                assert r.points(leg)[:,2].min()>=bound-1e-7

def test_quintic_starts_and_stops_smoothly():
    u=np.linspace(0,1,10001);v=smooth(u)
    assert v[0]==0 and v[-1]==1 and np.min(np.diff(v))>=0
    assert max(v[1]-v[0],v[-1]-v[-2])<1e-9

def test_torque_is_saturated():
    r=Robot();r.tick(r.initial[7:]+100)
    assert np.max(np.abs(r.d.ctrl))<=3
    np.testing.assert_allclose(r.d.actuator_force,r.d.ctrl)


def test_friction_sweep_changes_actual_contacts():
    for mu in [.5,.8,1.]:
        r=Robot(friction=mu);target=r.initial[7:].copy()
        for _ in range(520):r.tick(target)
        assert r.d.ncon>0
        for k in range(r.d.ncon):assert abs(r.d.contact[k].friction[0]-mu)<1e-10


def test_continuation_preserves_time_state_and_dynamics():
    a=Robot();q=a.initial[7:].copy();q[1]+=.03
    for _ in range(217):a.tick(q)
    snapshot=a.snapshot(q);b=Robot();np.testing.assert_allclose(b.restore(snapshot),q)
    assert b.d.time==a.d.time and b.d.time>0
    for _ in range(20):a.tick(q);b.tick(q)
    np.testing.assert_allclose(a.d.qpos,b.d.qpos,atol=1e-9)
    np.testing.assert_allclose(a.d.qvel,b.d.qvel,atol=1e-9)


def test_long_tips_start_above_triangle_bases():
    r=Robot()
    assert np.all(r.tip_bottoms()-r.bottoms()>.05)


def test_candidate_hash_mismatch_is_rejected(tmp_path):
    import json,pytest
    from .simulate import run
    candidate=tmp_path/'candidate.json'
    candidate.write_text(json.dumps(dict(leg='back_r',full_pose=Robot().initial[7:].tolist(),model_sha256='wrong')))
    with pytest.raises(ValueError,match='hash mismatch'):run(candidate,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
