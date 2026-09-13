"""Guards the ring-clearance reward against desyncing from leg-length/tilt/terrain DR.

`ring_local` (walk_geometry.py) and `self.sizes` (walk_env.py) are baked once from
the foot body's nominal local mesh/geom mounts. Randomization that acts upstream in
the kinematic chain (body_pos, body_quat, the floor's own geom_quat) stays perfectly
consistent with them (the reward re-projects through the real per-step body pose /
floor normal every step); randomization that instead moves the FOOT's own visual
mesh mount or capsule size would silently desync. This checks actual per-index
values (not just "was this field name touched anywhere"), since geom_quat is now
legitimately randomized for the floor -- a plain field-name check can't distinguish
that from an unsafe change to the foot's own geom_quat row.
"""
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.walk_randomize import LEG_BODY_IDS, FOOT_GEOM_IDS, domain_randomize
from workspace import walk_geometry as wgeom


def test_domain_randomize_never_touches_the_foots_local_mesh_mount():
    m = mujoco.MjModel.from_xml_path(str(wgeom.MODEL_PATH))
    mesh_id = m.mesh('CustomLegFoot').id
    visual_ids = np.array([
        np.flatnonzero((m.geom_bodyid == body) & (m.geom_dataid == mesh_id) & (m.geom_group == 1))[0]
        for body in LEG_BODY_IDS
    ])
    env = PupperWalkEnv()
    model, _ = domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(5), 4))
    # mesh_pos, mesh_quat, geom_pos, and geom_size are not in `fields` at all, so
    # they pass through completely untouched (no per-env batch axis, still the
    # original single copy) -- confirms that indirectly too.
    np.testing.assert_allclose(np.asarray(model.mesh_pos)[mesh_id], m.mesh_pos[mesh_id])
    np.testing.assert_allclose(np.asarray(model.mesh_quat)[mesh_id], m.mesh_quat[mesh_id])
    np.testing.assert_allclose(np.asarray(model.geom_pos)[visual_ids], m.geom_pos[visual_ids])
    np.testing.assert_allclose(np.asarray(model.geom_size)[FOOT_GEOM_IDS], env.mj_model.geom_size[FOOT_GEOM_IDS])
    # geom_quat IS in `fields` now (floor tilt), so it does have a per-env batch
    # axis -- check every env's visual-geom rows still match nominal.
    for env_i in range(4):
        np.testing.assert_allclose(np.asarray(model.geom_quat)[env_i, visual_ids], m.geom_quat[visual_ids])


def test_thigh_length_randomization_keeps_ring_calibrated():
    c = get_config()
    env = PupperWalkEnv(c)
    for scale in (0.864, 0.94, 1.0192):
        sys = env.sys.tree_replace({'body_pos': env.sys.body_pos.at[LEG_BODY_IDS].multiply(scale)})
        step = jax.jit(lambda d: mjx.step(sys, d))
        d = jax.jit(lambda: mjx.forward(sys, mjx.make_data(sys).replace(qpos=env.init_q)))()
        for _ in range(1000):
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


def test_randomize_varies_length_and_mounts_but_preserves_rigid_contacts():
    env = PupperWalkEnv()
    model, axes = domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(3), 4))
    assert axes.body_pos == 0
    assert axes.geom_solref == 0
    assert axes.body_quat == 0
    assert axes.geom_quat == 0
    for name in ('geom_solref','geom_solimp'):
        expected=np.broadcast_to(np.asarray(getattr(env.sys,name)),np.asarray(getattr(model,name)).shape)
        np.testing.assert_array_equal(np.asarray(getattr(model,name)),expected)
    # Per-env body_pos for the leg bodies must actually differ across the batch,
    # proving the field is live rather than silently constant.
    leg_pos = np.asarray(model.body_pos)[:, LEG_BODY_IDS]
    assert not np.allclose(leg_pos[0], leg_pos[1])
    scales=np.linalg.norm(leg_pos,axis=-1)/np.linalg.norm(np.asarray(env.sys.body_pos)[LEG_BODY_IDS],axis=-1)
    assert np.all((scales>=.864)&(scales<=1.0192))
    leg_quat = np.asarray(model.body_quat)[:, LEG_BODY_IDS]
    assert not np.allclose(leg_quat[0], leg_quat[1])
    from workspace.walk_randomize import FLOOR_GEOM_ID
    floor_quat = np.asarray(model.geom_quat)[:, FLOOR_GEOM_ID]
    assert not np.allclose(floor_quat[0], floor_quat[1])


def test_quat_axis_z_matches_known_rotations():
    identity = jp.array([1., 0., 0., 0.])
    np.testing.assert_allclose(wgeom.quat_axis_z(identity, jp), [0., 0., 1.], atol=1e-6)
    # 90deg roll about local X: local +Z maps to world +Y (or -Y depending on
    # handedness) -- just confirm it's now horizontal (near-zero Z component).
    half = jp.cos(jp.pi/4)
    roll90 = jp.array([half, half, 0., 0.])
    n = wgeom.quat_axis_z(roll90, jp)
    assert abs(float(n[2])) < 1e-5
    np.testing.assert_allclose(float(jp.linalg.norm(n)), 1., atol=1e-6)


def test_leg_mount_tilt_keeps_standing_within_allowance():
    """A few-degree body_quat tilt on the leg bodies (manufacturing-tolerance proxy)
    should still let the robot settle without blowing past the ring's bottom
    allowance, even though it's no longer bit-identical to the untilted case the
    way pure leg-length scaling is."""
    c = get_config()
    env = PupperWalkEnv(c)
    from workspace.walk_randomize import _small_tilt_quat, _quat_mul
    for rx, ry in [(0.07, 0.), (-0.07, 0.07), (0., -0.07)]:
        tilt = _small_tilt_quat(jp.full((4,), rx), jp.full((4,), ry))
        sys = env.sys.tree_replace({'body_quat': env.sys.body_quat.at[LEG_BODY_IDS].set(
            _quat_mul(tilt, env.sys.body_quat[LEG_BODY_IDS]))})
        step = jax.jit(lambda d: mjx.step(sys, d))
        d = jax.jit(lambda: mjx.forward(sys, mjx.make_data(sys).replace(qpos=env.init_q)))()
        for _ in range(50):
            d = step(d)
        d = jax.jit(lambda d: mjx.forward(sys, d))(d)
        ring = env.ring_signals(d)
        assert float(ring['ring_penetration_m']) < c.bottom_allowance, (rx, ry)
        rot = d.xmat[env.torso].reshape(3, 3)
        tilt_deg = float(jp.degrees(jp.arccos(jp.clip(rot[2, 2], -1., 1.))))
        assert tilt_deg < 10., (rx, ry)


def test_floor_tilt_ring_reward_uses_the_true_floor_normal():
    """Settling on a tilted floor with the STALE (flat) normal should look wrong;
    with the correct per-episode floor normal it should look like ordinary standing,
    same as the flat-floor case -- proving walk_env.py's floor_normal plumbing (not
    just this test file) actually uses the randomized geom_quat, not raw world Z."""
    from workspace.walk_randomize import _small_tilt_quat, FLOOR_GEOM_ID
    c = get_config()
    env = PupperWalkEnv(c)
    tilt = _small_tilt_quat(jp.array(0.06), jp.array(-0.04))
    sys = env.sys.tree_replace({'geom_quat': env.sys.geom_quat.at[FLOOR_GEOM_ID].set(tilt)})
    step = jax.jit(lambda d: mjx.step(sys, d))
    d = jax.jit(lambda: mjx.forward(sys, mjx.make_data(sys).replace(qpos=env.init_q)))()
    for _ in range(50):
        d = step(d)
    d = jax.jit(lambda d: mjx.forward(sys, d))(d)
    floor_normal = wgeom.quat_axis_z(sys.geom_quat[FLOOR_GEOM_ID], jp)
    ring_correct = env.ring_signals(d, floor_normal)
    ring_stale = env.ring_signals(d, jp.array([0., 0., 1.]))
    assert float(ring_correct['ring_penetration_m']) < c.bottom_allowance
    # The stale (world-Z) reading should disagree with the correct one -- otherwise
    # this test isn't actually exercising the tilt.
    assert abs(float(ring_correct['ring_penetration_m']) - float(ring_stale['ring_penetration_m'])) > 1e-4
