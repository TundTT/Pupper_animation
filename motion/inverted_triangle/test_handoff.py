import numpy as np
import pytest
import trimesh
from .core import HERE,Robot,HUB,PROX,SPEED,ACCEL
from .handoff_shape import deform,sample,geometry_metrics
from .handoff_entry import EntryRoll,NEUTRAL,SIGNS
from .handoff_probe import initialize

@pytest.fixture(scope='module')
def vertices():return trimesh.load(HERE/'assets/CustomLegFoot.stl',process=True).vertices

def test_zero_shape_is_exact(vertices):assert np.array_equal(vertices,deform(vertices))

def test_base_recess_and_tip_shortening_are_coupled(vertices):
    original=geometry_metrics(vertices,vertices)
    changed=geometry_metrics(vertices,deform(vertices,5,10,0))
    assert changed['central_attachment_max_change_mm']==0
    assert changed['axial_change_mm']==0
    assert changed['local_central_recess_mm']>original['local_central_recess_mm']+2
    assert changed['tip_extent_mm']==pytest.approx(original['tip_extent_mm']-10)

def test_sites_and_mesh_share_identical_deformation(vertices):
    ids=np.arange(0,len(vertices),113)
    assert np.allclose(deform(vertices,3,6,2)[ids],deform(vertices[ids],3,6,2),atol=1e-12)

def test_formation_trials_respect_tip_spread():
    for case in sample(42,100):
        assert np.ptp(case['tip_shortening_mm'])<=10
        assert np.all(np.array(case['support_extension_mm'])>=0)

def test_command_continuity_and_noncanonical_start():
    q=NEUTRAL.copy();q[HUB]-=SIGNS*np.pi;q[PROX]+=[.02,.03,-.01,-.02,.01,.05,-.01,-.03]
    control=EntryRoll(q,settle_seconds=3);previous=q.copy();v=np.zeros(12);phases=set()
    for _ in range(520*35):
        next_q=control.step(q,v,1/520);next_v=(next_q-previous)*520
        assert np.max(abs(next_v)/SPEED)<=1.000001
        assert np.max(abs(next_v-v)*520/ACCEL)<=1.000001
        phases.add(control.phase);q=next_q.copy();previous=next_q.copy();v=next_v
        if control.phase=='done':break
    assert phases>={'entry','roll','settle','done'}
    assert np.max(abs(control.command-control.goal))<.01

def test_stuck_entry_reports_failure():
    q=NEUTRAL.copy();q[HUB]-=SIGNS*np.pi;control=EntryRoll(q,settle_seconds=3)
    for _ in range(520*14):control.step(q,np.zeros(12),1/520)
    assert control.phase=='failed' and control.failure=='entry_did_not_settle'

def test_coupled_floor_parts_belong_to_correct_leg():
    r,boundary=initialize({'formation':sample(9)[0]})
    assert len(r.contact_owners)==8
    assert [list(r.contact_owners.values()).count(i) for i in range(4)]==[2]*4
    assert abs(r.bottoms().min())<1e-9
    state=r.snapshot(r.initial[7:]);other=Robot(formation=r.manifest['formation'])
    other.restore(state)
    assert np.allclose(other.d.qpos,r.d.qpos)
    assert not boundary['actual_alignment_replay']

def test_failed_alignment_cannot_be_called_a_handoff():
    with pytest.raises(ValueError):initialize({},dict(schema_version=1,completed_mask=7))


def test_terminal_preserves_velocity_and_previous_proximal_target():
    r=Robot();q=r.initial.copy();q[7:]=NEUTRAL;q[7+HUB]=np.pi
    velocity=np.linspace(-.001,.001,18)
    target=NEUTRAL[PROX]+.01
    record=dict(schema_version=1,source_kind='unit_test_fixture',source_commit='fixture',
        model_sha256='fixture',target_model_sha256=r.manifest['model_sha256'],
        joint_names=['leg_'+leg+'_'+str(j) for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)],
        alignment_passed=True,completed_mask=15,failed_mask=0,lowering_and_settling_finished=True,
        proximal_frame_verified=True,qpos=q.tolist(),qvel=velocity.tolist(),wheel_home=[0]*4,
        last_proximal_position_target=target.tolist())
    out,boundary=initialize({},record)
    np.testing.assert_array_equal(out.d.qvel,velocity)
    np.testing.assert_array_equal(np.array(boundary['entry_initial_command'])[PROX],target)
    np.testing.assert_allclose(out.d.qpos[7+HUB],NEUTRAL[HUB]-SIGNS*np.pi)
    record['target_model_sha256']='wrong'
    with pytest.raises(ValueError,match='current nominal'):initialize({},record)
