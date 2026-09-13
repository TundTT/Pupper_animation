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


def test_extended_trajectory_preserves_hubs_and_obeys_command_rates():
    from .simulate import trajectory
    from .core import duration,SPEED
    r=Robot();home=r.initial[7:].copy();pre=home.copy();pre[PROX]+=.02
    lift=home.copy();lift[7]=1.1;landing=lift.copy();landing[8]-=np.pi;landing[7]=.7
    phases=trajectory(home,lift,2,pre_shift_pose=pre,landing_pose=landing)
    assert [p[0] for p in phases][:3]==['initial_hold','body_shift','body_shift_hold']
    previous=home
    for phase,target,seconds in phases:
        dt=duration(previous,target,seconds)
        assert np.all(1.875*np.abs(target-previous)/dt <= SPEED+1e-12)
        np.testing.assert_allclose(target[HUB[[0,1,3]]],home[HUB[[0,1,3]]])
        previous=target
    np.testing.assert_array_equal(phases[-1][1],landing)


def test_extended_trajectory_rejects_silent_support_hub_rotation():
    import pytest
    from .simulate import trajectory
    r=Robot();home=r.initial[7:].copy();bad=home.copy();bad[2]+=.1
    with pytest.raises(ValueError,match='references'):trajectory(home,bad,2)
    with pytest.raises(ValueError,match='references'):trajectory(home,home,2,pre_shift_pose=bad)
    with pytest.raises(ValueError,match='references'):trajectory(home,home,2,landing_pose=bad)


def test_delay_and_dynamics_continue_without_resetting_controller_history():
    settings=dict(mass_scale=1.1,base_com_offset_m=[.003,-.002,0.],kp_scale=.9,kd_scale=1.1,torque_limit_Nm=1.5,delay_steps=5)
    a=Robot(dynamics=settings);home=a.initial[7:].copy()
    for i in range(137):a.tick(home+np.sin(i*.03)*.01)
    snapshot=a.snapshot(home);b=Robot(dynamics=settings);b.restore(snapshot)
    assert len(snapshot['command_history'])==5
    for i in range(50):
        q=home+np.cos(i*.1)*.015;a.tick(q);b.tick(q)
        np.testing.assert_allclose(a.d.qpos,b.d.qpos,atol=1e-10)
        np.testing.assert_allclose(a.d.qvel,b.d.qvel,atol=1e-10)
        assert np.max(np.abs(a.d.ctrl))<=1.5


def test_runtime_uncertainty_is_explicit_and_cannot_increase_torque():
    import pytest
    a=Robot();b=Robot(dynamics=dict(mass_scale=1.1,base_com_offset_m=[.003,0.,0.]))
    np.testing.assert_allclose(b.m.body_mass,1.1*a.m.body_mass)
    np.testing.assert_allclose(b.m.body_inertia,1.1*a.m.body_inertia)
    np.testing.assert_allclose(b.m.body_ipos[b.base]-a.m.body_ipos[a.base],[.003,0.,0.])
    with pytest.raises(ValueError,match='dynamics mismatch'):b.restore(a.snapshot(a.initial[7:]))
    with pytest.raises(ValueError,match='increase'):Robot(dynamics=dict(torque_limit_Nm=3.1))
    with pytest.raises(ValueError,match='integer'):Robot(dynamics=dict(delay_steps=.5))


def test_initial_offsets_are_not_reapplied_to_continuation(tmp_path):
    import json
    from .core import HERE
    from .simulate import run
    r=Robot();home=r.initial[7:].copy()
    for _ in range(50):r.tick(home)
    state=r.snapshot(home)
    candidate=tmp_path/'continuation_fixture.json'
    candidate.write_text(json.dumps(dict(leg='back_r',full_pose=home.tolist(),model_sha256=r.manifest['model_sha256'])))
    run(candidate,tmp_path/'continued',cad=False,landing_delta=0.,
        start_override=state,scenario=dict(initial_offset=dict(height_m=.002,roll_rad=.02)))
    actual=json.loads((tmp_path/'continued/start_state.json').read_text())
    np.testing.assert_allclose(actual['state'][1:20],state['state'][1:20],atol=1e-12)


def test_touchdown_preserves_hubs_rates_and_landing_audit_scope():
    import pytest
    from .simulate import trajectory
    from .core import duration,SPEED
    r=Robot();home=r.initial[7:].copy();lift=home.copy();lift[4]=-1.2
    touch=lift.copy();touch[5]-=np.pi;touch[4]=-.9
    landed=touch.copy();landed[0]+=.3
    phases=trajectory(home,lift,1,landing_pose=landed,touchdown_pose=touch)
    assert [p[0] for p in phases][-4:]==['land','land','land','planted_hold']
    previous=home
    for _,target,seconds in phases:
        assert np.all(1.875*np.abs(target-previous)/duration(previous,target,seconds)<=SPEED+1e-12)
        np.testing.assert_array_equal(target[HUB[[0,2,3]]],home[HUB[[0,2,3]]])
        previous=target
    bad=touch.copy();bad[8]+=.1
    with pytest.raises(ValueError,match='references'):
        trajectory(home,lift,1,landing_pose=landed,touchdown_pose=bad)


def test_held_out_cases_are_deterministic_and_separate():
    from .robustness import conditions,held_out_conditions
    assert held_out_conditions()==held_out_conditions()
    assert len(conditions())==36 and len(held_out_conditions())==12
    assert not ({c['seed'] for c in conditions()} & {c['seed'] for c in held_out_conditions()})
    assert all(c['scenario']['dynamics']['torque_limit_Nm']<=3 for c in held_out_conditions())
