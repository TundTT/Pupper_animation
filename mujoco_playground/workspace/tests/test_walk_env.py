"""Integration checks for the JIT environment and its training-only signals."""
import jax
from jax import numpy as jp
import numpy as np
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.walk_randomize import domain_randomize


def test_jit_reset_rollout_and_episode_bookkeeping():
    c=get_config();c.sensor_noise=0.;c.latency_probability=0.;c.reset_joint_noise=0.;c.reset_yaw_noise=0.;c.reset_xy_noise=0.
    c.stand_probability=1.
    env=PupperWalkEnv(c)
    state=jax.jit(env.reset)(jax.random.PRNGKey(5))
    assert state.obs.shape==(144,)
    # Brax adds metrics; the env must preserve the pytree across wrapped scans.
    state.metrics['reward']=jp.array(0.)
    step=jax.jit(env.step)
    for _ in range(100):state=step(state,jp.zeros(12))
    assert float(state.done)==0
    assert float(state.metrics['foot_contacts'])==4
    assert float(state.metrics['ring_side'])==0
    assert float(state.metrics['ring_bottom'])==0
    assert 'reward' in state.metrics
    assert float(state.metrics['tilt_deg'])<2
    assert np.isfinite(float(state.reward))
    # Simulate the non-physics bookkeeping of Brax's autoreset: history/physics
    # are reset by the wrapper, while env-owned command/air-time state must reset.
    state.info.update(reset_next=jp.array(True),air_time=jp.ones(4),last_action=jp.ones(12),command=jp.ones(3))
    state=step(state,jp.zeros(12))
    np.testing.assert_array_equal(state.info['command'],np.zeros(3))
    np.testing.assert_array_equal(state.info['air_time'],np.zeros(4))
    assert float(state.metrics['action_rate'])==0


def test_shaping_term_ceilings_are_not_negligible():
    """A positive shaping term whose per-second ceiling is tiny next to
    tracking_linear's cannot move the gait no matter how it's tuned -- this has
    happened twice now (air_time, then swing_clearance at its original weight of
    .5, both realizing under 0.1% of total reward). Guards the weights chosen to
    fix that, using the measured gait characteristics from the walk-policy review
    (duty factor ~.6 -> ~1.4 of 4 feet in swing at any instant; moving ~80% of the
    time given stand_probability=.2)."""
    c=get_config()
    tracking_ceiling=c.reward_scales.tracking_linear*1.
    swing_ceiling=max(c.reward_scales.swing_clearance*1.4*c.swing_clearance_target*.8,
                      abs(c.reward_scales.planned_swing)*c.planned_swing_bonus*2*.8)
    assert swing_ceiling/tracking_ceiling>.02,'swing_clearance is too small to shape the gait'
    # A single foot held up at rest must cost noticeably more than stand_pose's
    # per-joint-averaged penalty for the same deviation (~.006/s measured), or the
    # dilution over 12 joints makes it invisible again.
    assert abs(c.reward_scales.stance_feet)*1.>.05,'stance_feet is too small to fix tripod standing'


def test_randomized_gain_preserves_home_target():
    env=PupperWalkEnv()
    model,axes=domain_randomize(env.sys,jax.random.split(jax.random.PRNGKey(8),3))
    np.testing.assert_allclose(model.actuator_biasprm[:,:,0],model.actuator_gainprm[:,:,0]*env.home,rtol=1e-6)
    np.testing.assert_allclose(model.actuator_biasprm[:,:,1],-model.actuator_gainprm[:,:,0])
    assert axes.actuator_gainprm==0
    assert np.all(np.asarray(model.body_mass)>=0)

def test_randomized_brax_wrapper_can_reset_and_step():
    import functools
    from brax.envs.wrappers import training
    c=get_config();c.sensor_noise=0.;c.reset_joint_noise=0.
    env=PupperWalkEnv(c)
    wrapper=training.wrap(env,episode_length=8,randomization_fn=functools.partial(domain_randomize,rng=jax.random.split(jax.random.PRNGKey(12),2)))
    state=jax.jit(wrapper.reset)(jax.random.split(jax.random.PRNGKey(13),2))
    metric_keys=set(state.metrics)
    state=jax.jit(wrapper.step)(state,jp.zeros((2,12)))
    assert state.obs.shape==(2,144)
    assert np.all(np.isfinite(np.asarray(state.reward)))
    assert set(state.metrics)==metric_keys
    references=np.asarray(state.info['height_reference'])
    assert np.all(references < env.height)
    assert abs(references[0]-references[1])>1e-5


def test_height_reference_tracks_length_without_reset_noise():
    from workspace.walk_randomize import LEG_BODY_IDS
    c=get_config()
    c.foot_model='legacy_soft'  # The audited heights below used the old capsule.
    # Reference must be independent of randomized reset pose and sensor noise.
    c.reset_joint_noise=.1;c.reset_xy_noise=.2;c.reset_yaw_noise=.3
    env=PupperWalkEnv(c)
    nominal=env.sys
    for scale,expected in ((1.,.142463661),(.864,.131007),(.94,.137409),(1.0192,.144081)):
        env.sys=nominal.tree_replace({'body_pos':nominal.body_pos.at[LEG_BODY_IDS].multiply(scale)})
        reference=float(jax.jit(env.neutral_height_reference)())
        # Independent four-second MJX standing heights from the gait audit.
        np.testing.assert_allclose(reference,expected,atol=8e-5)


def test_height_reward_and_survival_are_invariant_to_travel_on_slope():
    from workspace.walk_randomize import _small_tilt_quat
    from workspace import walk_geometry as geom
    c=get_config();c.sensor_noise=0.;c.latency_probability=0.;c.push_probability=0.
    env=PupperWalkEnv(c)
    env.sys=env.sys.tree_replace({'geom_quat':env.sys.geom_quat.at[env.floor].set(_small_tilt_quat(jp.array(0.),jp.array(.04)))})
    state=jax.jit(env.reset)(jax.random.PRNGKey(8))
    normal=geom.quat_axis_z(env.sys.geom_quat[env.floor],jp)
    # Two metres downslope changes world Z by -80mm, formerly a false fall.
    translation=jp.array([2.,0.,-2.*normal[0]/normal[2]])
    shifted=state.replace(pipeline_state=state.pipeline_state.replace(qpos=state.pipeline_state.qpos.at[:3].add(translation)))
    step=jax.jit(env.step)
    original=step(state,jp.zeros(12));translated=step(shifted,jp.zeros(12))
    np.testing.assert_allclose(translated.metrics['height'],original.metrics['height'],atol=1e-5)
    np.testing.assert_allclose(translated.metrics['torso_height'],original.metrics['torso_height'],atol=1e-5)
    assert float(translated.done)==float(original.done)==0.
