"""Geometry consistency and nominal-regression tests, not hardware certification."""
import hashlib
import json
import numpy as np
import pytest
from .core import Robot, HERE
from .formation import deform, validate
from .clearance import CADClearance
from .roll_to_stand import initialize


def test_tip_length_changes_leave_attachment_and_axial_gap_fixed():
    vertices=np.array([[1.,-20.,5.],[2.,-40.,6.],[0.,-62.694,7.]])
    changed=deform(vertices,5.,0.,62.694)
    np.testing.assert_array_equal(changed[:2],vertices[:2])
    np.testing.assert_array_equal(changed[:,2],vertices[:,2])
    assert changed[2,1]==pytest.approx(vertices[2,1]-5.)
    bent=deform(vertices,-5.,5.,62.694)
    np.testing.assert_array_equal(bent[:2],vertices[:2])
    np.testing.assert_array_equal(bent[:,2],vertices[:,2])


def test_zero_error_model_preserves_nominal_integration():
    nominal,home,_=initialize()
    zero,home_zero,_=initialize(formation=dict(length_mm=[0]*4,bend_deg=[0]*4))
    for _ in range(260):
        nominal.tick(home);zero.tick(home_zero)
    np.testing.assert_allclose(zero.d.qpos,nominal.d.qpos,atol=1e-12,rtol=0)
    np.testing.assert_allclose(zero.d.qvel,nominal.d.qvel,atol=1e-12,rtol=0)


def test_independent_variant_preserves_nominal_assets_and_inertias():
    original=Robot()
    changed=Robot(formation=dict(length_mm=[-10,0,0,0],bend_deg=[3,0,0,0]))
    assert changed.manifest['nominal_model_sha256']==original.manifest['model_sha256']
    assert changed.manifest['model_sha256']!=original.manifest['model_sha256']
    assert hashlib.sha256((HERE/'model.xml').read_bytes()).hexdigest()==original.manifest['model_sha256']
    assert set(changed.manifest['generated_asset_sha256'])=={
        'formation_front_r_visual.stl','formation_front_r_floor.stl'}
    for name,digest in original.manifest['asset_sha256'].items():
        assert hashlib.sha256((HERE/'assets'/name).read_bytes()).hexdigest()==digest
    for field in ['body_mass','body_inertia','body_ipos','body_pos']:
        np.testing.assert_array_equal(getattr(original.m,field),getattr(changed.m,field))
    for i in range(1,4):
        np.testing.assert_array_equal(original.points(i),changed.points(i))
    assert all(mask.any() for mask in changed.tip_masks)
    with pytest.raises(ValueError,match='Continuation model'):
        original.restore(changed.snapshot(changed.initial[7:]))
    # Both the floor hull and detailed CAD reader must compile the derived model.
    assert np.isfinite(changed.tip_bottoms()).all()
    assert np.isfinite(CADClearance(changed).measure()['minimum_m'])


def test_cases_use_ten_mm_total_spread_and_labelled_angle_scope():
    cases=json.loads((HERE/'formation_cases.json').read_text())
    for case in cases:
        if 'formation' not in case:continue
        length,bend=validate(case['formation'])
        assert np.ptp(length)<=10
        assert np.max(abs(bend))<=5
    with pytest.raises(ValueError):validate(dict(length_mm=[0,0,0,float('nan')],bend_deg=[0]*4))


def test_variant_continuation_checks_mesh_identity_and_preserves_dynamics():
    spec=dict(length_mm=[-5,5,5,-5],bend_deg=[3,-3,-3,3])
    a=Robot(formation=spec);b=Robot(formation=spec);command=a.initial[7:].copy()
    for _ in range(100):a.tick(command)
    snapshot=a.snapshot(command);b.restore(snapshot)
    for _ in range(20):a.tick(command);b.tick(command)
    np.testing.assert_allclose(a.d.qpos,b.d.qpos,atol=1e-10,rtol=0)
    np.testing.assert_allclose(a.d.qvel,b.d.qvel,atol=1e-10,rtol=0)
    # Identical XML text does not guarantee identical external STL asset bytes.
    snapshot['generated_asset_sha256']=dict(snapshot['generated_asset_sha256'])
    snapshot['generated_asset_sha256']['formation_front_r_visual.stl']='wrong'
    with pytest.raises(ValueError,match='generated geometry'):b.restore(snapshot)
