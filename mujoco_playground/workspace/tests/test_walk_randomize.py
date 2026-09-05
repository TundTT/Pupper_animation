"""Guards the ring-clearance reward against desyncing from leg-length DR.

`ring_local` (walk_geometry.py) and `self.sizes` (walk_env.py) are baked once from
the foot body's nominal local mesh/geom mounts. A leg-length randomization that
scales `body_pos` upstream of the foot body should stay perfectly consistent with
them (the reward re-projects through the real per-step body pose every step); one
that instead moves the foot's own mesh/geom mount would silently desync. These
tests catch a future change to walk_randomize.py that crosses that line.
"""
import jax
from jax import numpy as jp
from mujoco import mjx
import numpy as np
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.walk_randomize import LEG_BODY_IDS, domain_randomize

_UNSAFE_FIELDS = {'mesh_pos', 'mesh_quat', 'geom_pos', 'geom_quat', 'geom_size'}


def test_domain_randomize_never_touches_the_foots_local_mesh_mount():
    import inspect
    from workspace import walk_randomize
    source = inspect.getsource(walk_randomize)
    for field in _UNSAFE_FIELDS:
        assert f"'{field}'" not in source and f'"{field}"' not in source, (
            f'domain_randomize touches {field}, which desyncs ring_local from the '
            'true foot mesh mount -- see walk_randomize.py module docstring')


def test_thigh_length_randomization_keeps_ring_calibrated():
    c = get_config()
    env = PupperWalkEnv(c)
    for scale in (0.94, 1.0, 1.06):
        sys = env.sys.tree_replace({'body_pos': env.sys.body_pos.at[LEG_BODY_IDS].multiply(scale)})
        step = jax.jit(lambda d: mjx.step(sys, d))
        d = jax.jit(lambda: mjx.forward(sys, mjx.make_data(sys).replace(qpos=env.init_q)))()
        for _ in range(50):
            d = step(d)
        d = jax.jit(lambda d: mjx.forward(sys, d))(d)
        ring = env.ring_signals(d)
        contact, _ = env.contact_mask(d)
        rot = d.xmat[env.torso].reshape(3, 3)
        tilt_deg = float(jp.degrees(jp.arccos(jp.clip(rot[2, 2], -1., 1.))))
        assert float(ring['ring_side']) == 0., scale
        assert float(ring['ring_bottom']) == 0., scale
        assert float(ring['ring_penetration_m']) < c.bottom_allowance, scale
        assert tilt_deg < 3., scale
        assert int(jp.sum(contact)) == 4, scale


def test_randomize_adds_leg_length_and_contact_softness_fields():
    env = PupperWalkEnv()
    model, axes = domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(3), 4))
    assert axes.body_pos == 0
    assert axes.geom_solref == 0
    # Per-env body_pos for the leg bodies must actually differ across the batch,
    # proving the field is live rather than silently constant.
    leg_pos = np.asarray(model.body_pos)[:, LEG_BODY_IDS]
    assert not np.allclose(leg_pos[0], leg_pos[1])
