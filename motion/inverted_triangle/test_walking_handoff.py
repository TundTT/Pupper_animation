import numpy as np
from .walking_policy import WalkingPolicy,POLICY
from .core import Robot
from .roll_to_stand import initialize
from .sensors import JointImuSensors

def test_pinned_inference_reference():
    rows=np.loadtxt(POLICY.parents[1]/'test/walk_policy_reference.csv',delimiter=',')
    policy=WalkingPolicy(np.zeros(12))
    assert abs(policy.infer(rows[:,:144])-rows[:,144:]).max()<3e-5

def test_runtime_ramp_history_and_fade():
    p=WalkingPolicy(np.zeros(12))
    np.testing.assert_array_equal(p.target(0,[0]*3,[0,0,-1],p.home,[0]*3),np.zeros(12))
    np.testing.assert_allclose(p.target(1,[0]*3,[0,0,-1],p.home,[0]*3),p.home/2)
    np.testing.assert_allclose(p.target(2,[0]*3,[0,0,-1],p.home,[0]*3),p.home)
    first=p.history[0].copy();np.testing.assert_array_equal(p.history,np.tile(first,(4,1)))
    p.target(3,[.1,0,0],[0,0,-1],p.home,[.1,0,0]);old_action=p.last_action.copy()
    p.target(4,[.2,0,0],[0,0,-1],p.home,[.1,0,0])
    np.testing.assert_array_equal(p.history[0,-12:],old_action)
    np.testing.assert_array_equal(p.history[2],first)
    assert p.history[0,0]==np.float32(.2) and p.history[1,0]==np.float32(.1)

def test_delayed_winding_and_sensor_latency():
    r,home,goal=initialize(dynamics=dict(delay_steps=3))
    for q in r.command_history:np.testing.assert_array_equal(q,home)
    assert abs(r.tick(home)).max()<1e-10
    sensor=JointImuSensors(dict(delay_steps=1,joint_bias_rad=[.01]*12),seed=4)
    a=sensor.read(r);r.tick(home+.005);b=sensor.read(r)
    np.testing.assert_array_equal(a['q'],b['q'])
    assert set(a)=={'quaternion','q','qd','omega','gravity'}

def test_capsule_reference_preserves_mass_geometry_and_rejects_restore():
    import pytest
    a=Robot();b=Robot(contact_model='rigid_flush')
    np.testing.assert_array_equal(a.m.body_mass,b.m.body_mass)
    for i in range(4):
        # Floor hull has a 23 um tessellation difference from the visual CAD.
        name=['front_r','front_l','back_r','back_l'][i]+'_shin_visual'
        for field in ['geom_pos','geom_quat','geom_size']:
            np.testing.assert_array_equal(getattr(a.m,field)[a.m.geom(name).id],getattr(b.m,field)[b.m.geom(name).id])
        g=b.geoms[i]
        np.testing.assert_allclose(b.m.geom_size[g,:2],[.012,.0193249805],atol=1e-10)
    with pytest.raises(ValueError,match='model/friction'):b.restore(a.snapshot(a.initial[7:]))

def test_action_correction_history_is_effective_action():
    p=WalkingPolicy(np.zeros(12),action_multiplier=[.8,.8,.6]);p.target(4,[0]*3,[0,0,-1],p.home,[0]*3)
    effective=p.last_action.copy();p.target(5,[0]*3,[0,0,-1],p.home,[0]*3)
    np.testing.assert_allclose(p.history[0,-12:],effective,atol=1e-7)

def test_broadphase_preserves_all_cad_minima_and_intersections():
    import fcl,mujoco as mj
    from .clearance import CADClearance
    r=Robot();checker=CADClearance(r);rng=np.random.default_rng(27);collisions=0
    for k in range(19):
        r.d.qpos[7:]+=rng.uniform(-.2,.2,12)
        if k==18:
            import json
            from .core import HERE
            r.d.qpos[:]=json.loads((HERE/'roll_validation/cad_intersection_fixture.json').read_text())['qpos']
        mj.mj_forward(r.m,r.d);actual=checker.measure()
        lowest=(float('inf'),None);front=(float('inf'),None);hits=[]
        for a,b in checker.pairs:
            count=fcl.collide(checker.objects[a],checker.objects[b],fcl.CollisionRequest(num_max_contacts=1),fcl.CollisionResult())
            gap=0. if count else float(fcl.distance(checker.objects[a],checker.objects[b],fcl.DistanceRequest(),fcl.DistanceResult()))
            if count:hits.append([a,b])
            if gap<lowest[0]:lowest=(gap,[a,b])
            if a.endswith('_3') and b.endswith('_3') and (('front_' in a and 'back_' in b) or ('back_' in a and 'front_' in b)) and gap<front[0]:front=(gap,[a,b])
        assert actual==dict(minimum_m=lowest[0],nearest_pair=lowest[1],intersections=hits,front_rear_minimum_m=front[0],front_rear_nearest_pair=front[1])
        collisions+=bool(hits)
    assert collisions>0

def test_gain_schedule_and_command_delay_replay_exactly(tmp_path):
    from .core import KP,KD
    from .roll_audit import RollRecorder
    r,home,goal=initialize(dynamics=dict(delay_steps=3,mass_scale=1.1))
    recorder=RollRecorder(r,home)
    for k in range(22):
        target=home+(.001 if k>10 else 0);damping=KD*(2 if k<12 else 1)
        r.tick(target,kd=damping);recorder.add(r,target,kd=damping)
    result=recorder.finish(r,tmp_path,cad=True)
    assert result['replay_max_state_error']==0
    data=np.load(tmp_path/'integration.npz')
    np.testing.assert_array_equal(data['kd'][0],2*KD)
    np.testing.assert_array_equal(data['kd'][-1],KD)
