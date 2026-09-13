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
