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
