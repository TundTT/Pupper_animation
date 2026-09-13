import jax
import mujoco
import numpy as np

from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.walk_randomize import domain_randomize


def test_floor_override_reaches_real_capsule_contacts():
    c = get_config()
    c.floor_friction = 2.5
    env = PupperWalkEnv(c)
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = np.asarray(env.init_q)
    mujoco.mj_forward(env.mj_model, d)
    contacts = [contact for contact in d.contact
                if contact.dist <= 0 and env.floor in contact.geom]
    assert len(contacts) == 4
    np.testing.assert_allclose([contact.friction[:2] for contact in contacts], 2.5)
    assert float(env.sys.geom_friction[env.floor, 0]) == 2.5


def test_training_randomization_honors_high_friction_range():
    env = PupperWalkEnv()
    keys = jax.random.split(jax.random.PRNGKey(907), 16)
    old, _ = domain_randomize(env.sys, keys)
    high, axes = domain_randomize(env.sys, keys, friction_range=(1.2, 2.5))
    friction = np.asarray(high.geom_friction[:, :, 0])
    assert np.all((friction >= 1.2) & (friction <= 2.5))
    assert np.ptp(friction[:, env.floor]) > .5
    assert axes.geom_friction == 0
    # Same seeds retain the other physical perturbations for the causal trial.
    for field in ('body_pos', 'body_quat', 'geom_quat', 'geom_solref', 'actuator_gainprm'):
        np.testing.assert_array_equal(np.asarray(getattr(old, field)), np.asarray(getattr(high, field)))
    np.testing.assert_array_equal(np.asarray(old.geom_friction[:, :, 1:]),
                                  np.asarray(high.geom_friction[:, :, 1:]))
