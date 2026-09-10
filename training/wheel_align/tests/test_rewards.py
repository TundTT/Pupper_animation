import numpy as np
from training.wheel_align import configs as c, contract as ct, rewards

def sample(phase=ct.ROTATE):
    q=c.DEFAULT_POSE.copy();s=ct.reset(q,np.zeros(4))
    s.update(phase=np.asarray(phase),active_command=np.asarray(1),progress=np.asarray(1.),
             was_rotating=np.asarray(phase in (ct.ROTATE,ct.VERIFY)))
    return s,q

def score(s,q,next_q=None,after=None,**kw):
    args=dict(floor=.015,gap=.015,bodygap=.010,tilt=.02,angular_speed=.01,unsafe=np.asarray(False))
    args.update(kw)
    return rewards.task_terms(s,after or s,q,q if next_q is None else next_q,dt=c.CONTROL_DT,**args)

def test_safe_clearance_beats_stalled_hover():
    s,q=sample(ct.LIFT)
    assert score(s,q)['reward']>score(s,q,floor=-.017,tilt=.17)['reward']
    assert score(s,q)['reward']>score(s,q,gap=.004)['reward']
    assert score(s,q)['reward']<0  # no endless positive apex bonus

def test_signed_progress_and_no_ground_rotation_reward():
    s,q=sample();q[5]=1.;forward=q.copy();forward[5]+=.02
    back=q.copy();back[5]-=.02
    assert score(s,q,forward)['angle_progress']>0
    assert score(s,q,back)['angle_progress']<0
    assert score(s,q,forward,floor=0)['angle_progress']==0
    assert score(s,q,forward,unsafe=np.asarray(True))['angle_progress']==0
    # A closed angular cycle has zero potential credit.
    assert abs(score(s,q,forward)['angle_progress']+score(s,forward,q)['angle_progress'])<1e-12

def test_completion_events_are_one_shot_and_verified():
    s,q=sample(ct.LOWER);s['verified']=np.asarray(True)
    after=dict(s);after['completed']=np.array([False,True,False,False]);after['phase']=np.asarray(ct.HOLD)
    assert score(s,q,after=after)['completed_event']==1
    assert score(after,q)['completed_event']==0
    pending,_=sample(ct.LOWER)
    pending.update(progress=np.asarray(1.),step_pending=np.asarray(True),phase_steps=np.asarray(300))
    finished=ct.finish(pending,q,np.zeros(12))
    assert not finished['completed'].any()  # an interrupted descent is not success

def test_real_servo_traverses_half_turn_and_all_four_completions():
    # Ideal proximal tracking isolates the supervisor/hub servo, without
    # teleporting the hub to its target as the language-parity fixture does.
    q=c.DEFAULT_POSE.copy();qd=np.zeros(12);s=ct.reset(q,np.zeros(4))
    for command in (1,2,3,4):
        s=ct.select(s,command,q);enabled=0
        for _ in range(1664):
            s=ct.prepare(s);s,w=ct.begin(s,q,qd,np.zeros(3),np.array([0.,0.,-1.]),np.zeros(8))
            enabled+=int(s['was_rotating'])
            for _ in range(10):
                old=q.copy();s=ct.integrate(s);q[ct.POS]=s['applied']
                # First-order velocity actuator response, positive damping.
                qd[ct.WHEEL]+=(w-qd[ct.WHEEL])*c.PHYSICS_DT/.04
                q[ct.WHEEL]+=qd[ct.WHEEL]*c.PHYSICS_DT
                qd[ct.POS]=(q[ct.POS]-old[ct.POS])/c.PHYSICS_DT
            s=ct.finish(s,q,qd)
            if s['completed'][ct.COMMAND_LEG[command]]:break
        assert enabled>500  # actually sweeps through ~pi, not instant alignment
        assert s['completed'][ct.COMMAND_LEG[command]]
    assert s['completed'].all()
    assert np.max(np.abs(ct.wrap(s['target']-q[ct.WHEEL])))<.035

def test_native_front_lift_can_clear_rotate_and_lower():
    from training.wheel_align.diagnose import probe,FEASIBLE_FL_ACTION
    result=probe(FEASIBLE_FL_ACTION)
    assert result['completed'][1]
    assert all(result['phase_seconds'][phase]>0 for phase in ('lift','rotate','verify','lower','hold'))
    assert np.all(np.array(result['min_rotation_gate_margins_m'])>np.array([.010,.010,.005]))
