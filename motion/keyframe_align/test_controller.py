import copy
import json
import math
import numpy as np
import pytest
from .native import Controller, CONFIG

POS=[0,1,3,4,6,7,9,10];WHEEL=[2,5,8,11];DT=.01
Q=np.array([1.,0.,.3,-1.,0.,-.4,1.,0.,.5,-1.,0.,-.6])

def tick(c,q,qd,command,g=None,deadband=0):
    # The original coordinated front poses require a support lean. This is an
    # ideal tracking fixture, not physical evidence; native MuJoCo tests balance.
    if g is None:
        y={1:-.06,2:.06}.get(command,0.)
        g=(0,y,-math.sqrt(1-y*y))
    o=c.step(DT,command,q,qd,[0,0,0],g)
    previous=q.copy();q[POS]=o[:8]
    w=o[8:12].copy();w[np.abs(w)<deadband]=0;q[WHEEL]+=w*DT
    qd[:]=(q-previous)/DT
    return o

def init(config=None):
    c=Controller(config=config);q=Q.copy();qd=np.zeros(12);c.reset(q,q[WHEEL])
    for _ in range(700):o=tick(c,q,qd,0)
    assert o[12]==6
    return c,q,qd

def test_full_sequence_with_synthetic_wheel_deadband():
    c,q,qd=init()
    for command,leg in [(1,1),(2,0),(3,2),(4,3)]:
        for n in range(6000):
            o=tick(c,q,qd,command,deadband=.22)
            assert o[23] and o[16]<0
            assert np.max(np.abs(o[8:12]))<=.500001
            if o[12]==6 and int(o[14])&(1<<leg):break
        else:pytest.fail(f'Wheel {leg} failed to finish')
        assert abs(math.atan2(math.sin(Q[WHEEL[leg]]+math.pi-q[WHEEL[leg]]),math.cos(Q[WHEEL[leg]]+math.pi-q[WHEEL[leg]])))<.025
    c.close()

def test_gate_loss_resets_integral_and_rotation():
    c,q,qd=init()
    for _ in range(4000):
        o=tick(c,q,qd,3,deadband=.22)
        if abs(o[22])>.02:break
    assert abs(o[22])>.02
    o=tick(c,q,qd,3,g=(.2,0,-math.sqrt(.96)))
    assert int(o[15])&16 and o[22]==0 and o[10]==0
    c.close()

def test_interruption_preserves_old_leg_until_lowered():
    c,q,qd=init()
    for _ in range(350):o=tick(c,q,qd,1)
    previous=o[:8].copy();o=tick(c,q,qd,2)
    assert o[12]==4 and o[13]==1
    assert np.max(np.abs(o[:8]-previous))<.007
    for _ in range(2000):
        o=tick(c,q,qd,2)
        if o[13]==2:break
    assert o[13]==2 and not int(o[14])&2
    c.close()

def test_timeout_lowers_and_latches_without_retry():
    config=json.loads(CONFIG.read_text());config['attempt_timeout_seconds']=8
    c,q,qd=init(config)
    for _ in range(2400):o=tick(c,q,qd,1,g=(.2,0,-math.sqrt(.96)))
    assert o[12]==6 and o[16]==1 and o[14]==0
    for _ in range(200):o=tick(c,q,qd,1)
    assert o[12]==6
    tick(c,q,qd,0);o=tick(c,q,qd,1)
    assert o[12]==1
    c.close()

def test_stop_and_bad_sensor_data_remove_authority_until_reset():
    c,q,qd=init();o=c.step(DT,1,q,qd,[0,0,0],[0,0,-1],True)
    assert o[12]==7 and o[23]==0 and np.all(o[8:12]==0)
    assert c.step(DT,1,q,qd,[0,0,0],[0,0,-1])[23]==0
    c.reset(q,q[WHEEL]);bad=q.copy();bad[1]=np.nan
    assert c.step(DT,1,bad,qd,[0,0,0],[0,0,-1])[23]==0
    c.close()

def test_invalid_config_and_calibration_rejected():
    config=json.loads(CONFIG.read_text());config['hip_speed_limit']=20
    with pytest.raises(ValueError):Controller(config=config)
    c=Controller()
    with pytest.raises(ValueError):c.reset(Q,[0,0,np.nan,0])
    c.close()

def test_calibration_target_survives_arbitrary_entry_revolutions():
    c,q,qd=init();home=Q[WHEEL].copy();q[WHEEL]+=np.array([12,-13,14,-15])*2*math.pi
    c.reset(q,home)
    for _ in range(700):tick(c,q,qd,0)
    for _ in range(6000):
        o=tick(c,q,qd,3)
        if int(o[14])&4:break
    assert int(o[14])&4 and abs(o[20])<.025
    c.close()
