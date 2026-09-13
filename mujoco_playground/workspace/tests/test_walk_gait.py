"""Command exposure and tip/touchdown objective checks."""
import jax
from jax import numpy as jp
import numpy as np
from workspace.walk_env import PupperWalkEnv
from workspace import walk_geometry as g


def test_explicit_straight_commands_cover_both_directions():
    env=PupperWalkEnv()
    commands=np.asarray(jax.jit(jax.vmap(env.sample_command))(jax.random.split(jax.random.PRNGKey(41),20000)))
    straight=(commands[:,1:]==0).all(axis=1)
    forward=straight&(commands[:,0]>0);reverse=straight&(commands[:,0]<0)
    stand=(commands==0).all(axis=1)
    np.testing.assert_allclose([stand.mean(),forward.mean(),reverse.mean()],[.2,.2,.2],atol=.015)
    assert np.all(np.abs(commands[forward|reverse,0])>=.1)
    assert np.all(np.abs(commands[:,0])<=.35)
    lateral=(commands[:,0]==0)&(commands[:,1]!=0)&(commands[:,2]==0)
    turn=(commands[:,0]==0)&(commands[:,1]==0)&(commands[:,2]!=0)
    np.testing.assert_allclose([lateral.mean(),turn.mean()],[.1,.1],atol=.012)
    assert np.mean(~straight&~lateral&~turn)>.18


def test_short_swing_has_a_duration_gradient_and_standing_is_free():
    scores=[g.swing_duration_reward(np.full(4,t),np.ones(4,dtype=bool)) for t in (.02,.04,.08,.16)]
    np.testing.assert_allclose(scores,[-.24,-.16,0.,.32])
    assert g.swing_duration_reward(np.full(4,.02),np.zeros(4,dtype=bool))==0


def test_slow_tracking_rejects_freezing_and_preserves_fast_tolerance():
    assert g.linear_tracking_reward(np.zeros(2),np.zeros(2))==1.
    assert g.linear_tracking_reward(np.zeros(2),np.array([0.,.1]))<.3
    for sign in (-1.,1.):
        target=np.array([sign*.35,0.]);actual=np.array([sign*.30,0.])
        np.testing.assert_allclose(g.linear_tracking_reward(actual,target),np.exp(-.05**2/.07),rtol=.001)


def test_turning_preserves_translation_tolerance_and_rewards_slow_yaw():
    np.testing.assert_allclose(g.linear_tracking_reward(np.array([.03,0]),np.zeros(2),yaw_rate=.5),np.exp(-.03**2/.07))
    assert g.yaw_tracking_reward(0.,.2)<.65
    assert g.yaw_tracking_reward(.2,.2)==1.


def test_clearance_shortfall_scores_actual_touchdown_not_stance():
    peak=np.array([.001,.004,.008,.0]);previous=np.zeros(4,dtype=bool)
    current=np.array([True,True,True,False])
    np.testing.assert_allclose(g.clearance_shortfall_cost(peak,previous,current),.75**2)
    assert g.clearance_shortfall_cost(peak,current,current)==0.


def test_touchdown_uses_previous_airborne_velocity_once():
    previous=np.array([False,True,False,False])
    contact=np.array([True,True,False,True])
    velocity=np.array([-.35,-.6,-.8,.2])
    np.testing.assert_allclose(g.touchdown_cost(previous,contact,velocity),.04)
    assert g.touchdown_cost(contact,contact,velocity)==0
    # A touchdown straddling control boundaries must still use the previous
    # sample, even when current contact solver velocity is zero.
    saved_contact=np.zeros(4,dtype=bool);saved_velocity=np.full(4,-.25)
    np.testing.assert_allclose(g.touchdown_cost(saved_contact,np.ones(4,dtype=bool),saved_velocity),.04)


def test_tip_guard_allows_rolling_and_swing_but_penalizes_side_support():
    allowance=np.cos(np.deg2rad(15.));scale=np.cos(np.deg2rad(45.))
    contact=np.ones(4,dtype=bool)
    assert g.tip_support_cost(np.cos(np.deg2rad([0.,5.,10.,15.])),contact,allowance,scale)==0
    assert g.tip_support_cost(np.zeros(4),~contact,allowance,scale)==0
    np.testing.assert_allclose(g.tip_support_cost(np.full(4,scale),contact,allowance,scale),1.)
    np.testing.assert_allclose(g.tip_support_cost(np.zeros(4),contact,allowance,scale),4.)


def test_diagnostic_step_matches_plain_and_keeps_impact_history():
    env=PupperWalkEnv();state=jax.jit(env.reset)(jax.random.PRNGKey(18))
    plain=jax.jit(env.step);measured=jax.jit(env.step_with_diagnostics)
    for action in (jp.zeros(12),jp.full(12,.2)):
        normal=plain(state,action);state,trace=measured(state,action)
        np.testing.assert_allclose(state.pipeline_state.qpos,normal.pipeline_state.qpos,atol=2e-5,rtol=0.)
        np.testing.assert_allclose(state.reward,normal.reward,atol=1e-6)
        assert trace['z'].shape==(5,4)
        np.testing.assert_array_equal(state.info['impact_contact'],trace['contact'][-1])
        np.testing.assert_allclose(state.info['impact_normal_velocity'],trace['normal_velocity'][-1])


def test_target_acceleration_penalizes_reversal_and_clears_on_reset():
    from workspace.walk_config import get_config
    c=get_config();c.reward_scales.action_acceleration=-1.
    c.action_scale=c.smoothness_reference_scale
    env=PupperWalkEnv(c);state=jax.jit(env.reset)(jax.random.PRNGKey(3))
    step=jax.jit(env.step)
    # A ramp pays only when it starts, whereas reversing pays again.
    for target,expected in [(0.,0.),(.1,-.01),(.2,0.),(.1,-.04)]:
        state=step(state,jp.full(12,target))
        np.testing.assert_allclose(state.metrics['action_acceleration'],expected,atol=1e-6)
    state.info.update(reset_next=jp.array(True),last_action=jp.ones(12),last_action_delta=jp.ones(12))
    state=step(state,jp.zeros(12))
    assert float(state.metrics['action_acceleration'])==0.
    np.testing.assert_array_equal(state.info['last_action_delta'],np.zeros(12))


def test_planned_swing_cannot_be_erased_or_restarted_by_early_contact():
    elapsed=np.full(4,-1.);stance=np.ones(4,dtype=bool);air=~stance
    elapsed,cost,reference,active=g.planned_swing_cost(elapsed,stance,air,np.zeros(4),.004)
    assert active.all() and cost==0.
    for _ in range(20):
        elapsed,cost,reference,active=g.planned_swing_cost(elapsed,stance,stance,np.zeros(4),.004)
    np.testing.assert_allclose(reference,.003,atol=1e-7)
    np.testing.assert_allclose(cost,4.,atol=1e-7)
    # A second lift during the planned window does not restart its phase.
    later,_,_,_=g.planned_swing_cost(elapsed,stance,air,np.zeros(4),.004)
    np.testing.assert_allclose(later,.084)
    for _ in range(21):
        elapsed,_,_,active=g.planned_swing_cost(elapsed,stance,stance,np.zeros(4),.004)
    assert not active.any() and np.all(elapsed==-1.)
    _,cost,_,active=g.planned_swing_cost(elapsed,stance,stance,np.zeros(4),.004)
    assert cost==0. and not active.any()


def test_swing_credit_prefers_full_arc_over_ground_hover_or_early_contact():
    def cycle(kind):
        elapsed=np.full(4,-1.);previous=np.ones(4,dtype=bool);total=0.
        for i in range(40):
            reference=.006*np.sin(np.pi*i/40)**2
            height=np.full(4,reference if kind=='arc' else (0. if kind=='hover' else -.0015))
            contact=np.full(4,kind=='contact' and i>0)
            elapsed,cost,_,_=g.planned_swing_cost(elapsed,previous,contact,height,.004,apex=.006,bonus=.25)
            total+=cost*.004;previous=contact
        return total
    # Negative cost earns credit under a negative weight. Neither cheating
    # trajectory can profit over the full planned window.
    assert cycle('arc')<0.
    assert cycle('hover')>0.
    assert cycle('contact')>cycle('hover')


def test_front_correction_does_not_increase_perfect_arc_bonus():
    # At the 16mm apex, an early front landing must cost more than the identical
    # rear landing. Following the curve earns exactly the original bonus.
    elapsed=np.full(4,.096);air=np.zeros(4,dtype=bool);target=np.full(4,.016)
    kwargs=dict(apex=.016,duration=.2,bonus=.25)
    base=g.planned_swing_cost(elapsed,air,air,target,.004,**kwargs)[1]
    weighted=g.planned_swing_cost(elapsed,air,air,target,.004,weights=np.array([3.,3.,1.,1.]),**kwargs)[1]
    np.testing.assert_allclose(base,weighted)
    costs=[]
    for foot in (0,2):
        height=target.copy();height[foot]=0.;contact=air.copy();contact[foot]=True
        costs.append(g.planned_swing_cost(elapsed,air,contact,height,.004,weights=np.array([3.,3.,1.,1.]),**kwargs)[1])
    assert costs[0]>costs[1]
