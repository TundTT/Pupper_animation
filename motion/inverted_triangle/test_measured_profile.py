import numpy as np
import pytest
import trimesh
from .core import HERE,Robot
from .measured_profile import deform,NOMINAL_TOP_MM,NOMINAL_BOTTOM_MM
from .handoff_probe import initialize


def test_extents_fit_preserves_mount_width_and_axial_gap():
    v=trimesh.load(HERE/'assets/CustomLegFoot.stl',process=True).vertices
    for top,bottom in [(55,37),(57.5,34),(60,32),(56,37),(57,35)]:
        w=deform(v,top,bottom)
        assert -w[:,1].min()==pytest.approx(top)
        assert w[:,1].max()==pytest.approx(bottom)
        np.testing.assert_array_equal(w[:,[0,2]],v[:,[0,2]])
        mask=np.linalg.norm(v[:,:2],axis=1)<=18
        np.testing.assert_array_equal(w[mask],v[mask])
        np.testing.assert_array_equal(deform(v[::137],top,bottom),w[::137])
    np.testing.assert_array_equal(deform(v,NOMINAL_TOP_MM,NOMINAL_BOTTOM_MM),v)


def test_measured_profile_compiles_and_replays():
    import json
    config=json.loads((HERE/'measured_baseline_config.json').read_text())
    r,_=initialize(config);original=Robot()
    assert r.manifest['gap_m']==.009
    assert len(r.contact_owners)==8
    for name in ('body_mass','body_inertia','body_ipos','body_pos'):
        np.testing.assert_array_equal(getattr(r.m,name),getattr(original.m,name))
    cmd=r.initial[7:].copy()
    for _ in range(20):r.tick(cmd)
    s=r.snapshot(cmd);clone=Robot(formation=config['formation']);clone.restore(s)
    for _ in range(20):r.tick(cmd);clone.tick(cmd)
    np.testing.assert_allclose(r.d.qpos,clone.d.qpos,atol=1e-12,rtol=0)
    np.testing.assert_allclose(r.d.qvel,clone.d.qvel,atol=1e-12,rtol=0)
    assert all(mask.any() for mask in r.tip_masks)


def test_unmeasured_angle_not_silently_invented():
    with pytest.raises(ValueError,match='angle'):deform(np.zeros((3,3)),57,35,2)


def test_labelled_synthetic_angle_preserves_mount_and_shared_geometry():
    from .clearance import CADClearance
    v=trimesh.load(HERE/'assets/CustomLegFoot.stl',process=True).vertices
    w=deform(v,57,35,synthetic_bend_deg=3)
    mask=np.linalg.norm(v[:,:2],axis=1)<=18
    np.testing.assert_allclose(w[mask],v[mask],atol=1e-14,rtol=0)
    np.testing.assert_array_equal(w[:,2],v[:,2])
    assert np.max(abs(w-deform(v,57,35)))>1
    r=Robot(formation=dict(top_mm=[57]*4,bottom_mm=[35]*4,bend_deg=[0]*4,synthetic_bend_deg=[3,-3,3,-3],contact_parts=4))
    assert 'synthetic_angle_status' in r.manifest
    assert r.manifest['gap_m']==.009
    checker=CADClearance(r)
    assert len(checker.pairs)>0
    state=r.snapshot(r.initial[7:]);clone=Robot(formation=r.manifest['formation']);clone.restore(state)
    for _ in range(30):r.tick(r.initial[7:]);clone.tick(r.initial[7:])
    np.testing.assert_array_equal(r.d.qpos,clone.d.qpos)
    with pytest.raises(ValueError,match='Synthetic bend'):deform(v,57,35,synthetic_bend_deg=4)
