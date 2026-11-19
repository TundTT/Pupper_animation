"""Physical geometry/contact checks for the flush rigid capsule model."""
import jax
import mujoco
import numpy as np
from workspace import walk_geometry as g
from workspace.walk_config import get_config
from workspace.walk_env import PupperWalkEnv
from workspace.walk_randomize import domain_randomize, LEG_BODY_IDS


def test_flush_tip_preserves_radius_proximal_endpoint_and_visual_ring():
    old=g.load_walk_model(foot_model='legacy_soft')
    new=g.load_walk_model()
    _,feet,ring,_=g.load_geometry(new)
    np.testing.assert_array_equal(ring,g.load_geometry(old)[2])
    for i,foot in enumerate(feet):
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,new.geom_quat[foot])
        axis=rot.reshape(3,3)[:,2]  # +Z points toward the proximal cap here.
        np.testing.assert_allclose(new.geom_size[foot,0],old.geom_size[foot,0])
        np.testing.assert_allclose(new.geom_pos[foot]+axis*new.geom_size[foot,1],
                                   old.geom_pos[foot]+axis*old.geom_size[foot,1],atol=1e-12)
        tip=(new.geom_pos[foot]@axis)-sum(new.geom_size[foot,:2])
        np.testing.assert_allclose(tip,np.min(ring[i]@axis),atol=1e-12)
        np.testing.assert_allclose(2*(new.geom_size[foot,1]-old.geom_size[foot,1]),.002649961,atol=1e-9)
    assert get_config().reward_scales.ring_bottom==0.
    assert get_config().reward_scales.ring_side_contact<0.


def test_stiff_support_at_high_friction_and_length_endpoints():
    for scale in (.864,1.,1.0192):
        m=g.load_walk_model()
        _,feet,_,_=g.load_geometry(m)
        m.body_pos[LEG_BODY_IDS]*=scale
        for mu in (1.,2.,3.):
            m.geom_friction[m.geom('floor').id,0]=mu
            d=mujoco.MjData(m);d.qpos[:]=m.key('home').qpos
            for _ in range(1000):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d)
            bottom=g.capsule_bottom(d.geom_xpos[feet],d.geom_xmat[feet],m.geom_size[feet])[:,2]
            assert np.min(bottom)>-.0001,(scale,mu,bottom)
            assert len([x for x in d.contact if x.dist<=0 and m.geom('floor').id in x.geom])==4
            assert np.linalg.norm(d.qvel)<.01
            np.testing.assert_allclose([x.solref[0] for x in d.contact],.008)
            np.testing.assert_allclose([x.friction[0] for x in d.contact],mu)


def test_length_ranges_apply_around_flush_assembly_without_softening():
    env=PupperWalkEnv()
    keys=jax.random.split(jax.random.PRNGKey(42),4)
    sample,_=domain_randomize(env.sys,keys,leg_length_common_range=(.95,.95),
                             leg_length_per_leg_range=(1.01,1.01))
    np.testing.assert_allclose(np.asarray(sample.body_pos)[:,LEG_BODY_IDS],
        np.broadcast_to(np.asarray(env.sys.body_pos)[LEG_BODY_IDS]*.9595,(4,4,3)),atol=1e-8)
    # Only upstream transforms change: local capsule and ring retain the flush fit.
    np.testing.assert_array_equal(sample.geom_pos,env.sys.geom_pos)
    np.testing.assert_array_equal(sample.geom_size,env.sys.geom_size)
    for field in ('geom_solref','geom_solimp'):
        actual=np.asarray(getattr(sample,field))
        np.testing.assert_array_equal(actual,np.broadcast_to(getattr(env.sys,field),actual.shape))


def test_effective_model_snapshot_reloads_outside_asset_directory(tmp_path):
    m=g.load_walk_model()
    path=tmp_path/'effective_model.xml'
    g.save_effective_model(m,path)
    restored=mujoco.MjModel.from_xml_path(str(path))
    # MuJoCo's XML writer rounds decimal values; require sub-micron geometry.
    np.testing.assert_allclose(restored.geom_size,m.geom_size,atol=1e-7)
    np.testing.assert_allclose(restored.geom_pos,m.geom_pos,atol=1e-7)
    np.testing.assert_allclose(restored.geom_solref,m.geom_solref)
    np.testing.assert_allclose(restored.geom_solimp,m.geom_solimp)
    np.testing.assert_allclose(restored.key('home').qpos,m.key('home').qpos,atol=5e-6)
