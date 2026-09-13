import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('triangle_reference',ROOT/'scripts/inverted_triangle_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
plan=json.loads((ROOT/'ros2_ws/src/neural_controller/launch/triangle_roll_plan.json').read_text())
cal=dict(calibration_id='fixture',encoder_session_id='session',wheel_home=[1,2,3,4])

def test_forward_roll_maps_each_side_without_wrapping_or_recalibration():
    measured=plan['initial'].copy()
    for i in [2,5,8,11]:measured[i]+=17.2
    mapping=reference.make_mapping(plan,cal,measured)
    for i in [2,5,8,11]:
        delta=plan['goal'][i]+mapping['model_to_encoder_offset'][i]-measured[i]
        assert delta==pytest.approx(-3.141592653589793 if i in [2,8] else 3.141592653589793)
    assert mapping['wheel_home']==cal['wheel_home']

def test_mapping_rejects_proximal_pose_guess_and_does_not_match_old_plan():
    measured=plan['initial'].copy();measured[0]+=.16
    with pytest.raises(ValueError):reference.make_mapping(plan,cal,measured)
    old=json.loads((ROOT/'ros2_ws/src/neural_controller/launch/inverted_triangle_plan.json').read_text())
    assert old['plan_sha256']!=plan['plan_sha256']


def test_near_alignment_entry_preserves_proximal_encoder_frame():
    measured=plan["initial"].copy();measured[0]+=.04;measured[1]=.035
    result=reference.make_mapping(plan,cal,measured)
    assert result["schema_version"]==2
    assert result["captured_q"]==measured
    assert result["model_to_encoder_offset"][0:2]==[0.,0.]


def test_calibrated_hanging_hips_are_accepted_without_offset_guessing():
    measured=plan['initial'].copy()
    measured[1::3]=[-.179972,.179973,-.179972,.179973]
    result=reference.make_mapping(plan,cal,measured)
    assert result['model_to_encoder_offset'][1::3]==[0.]*4
    measured[1]=-.201
    with pytest.raises(ValueError):reference.make_mapping(plan,cal,measured)
