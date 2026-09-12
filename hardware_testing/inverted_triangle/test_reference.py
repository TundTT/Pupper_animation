import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('triangle_reference', ROOT/'scripts/inverted_triangle_reference.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
plan = json.loads((ROOT/'ros2_ws/src/neural_controller/launch/inverted_triangle_plan.json').read_text())
cal = dict(calibration_id='fixture', encoder_session_id='session', wheel_home=[1,2,3,4])


def test_reference_preserves_calibration_and_unwrapped_turns():
    q = plan['initial'].copy()
    for i in (2,5,8,11):
        q[i] += 37.123
    result = reference.make_mapping(plan, cal, q)
    assert result['wheel_home'] == cal['wheel_home'] == [1,2,3,4]
    for i in range(12):
        assert result['model_to_encoder_offset'][i] == pytest.approx(37.123 if i%3==2 else 0)


@pytest.mark.parametrize('index,value', [(0,.04), (2,float('nan')), (5,float('inf'))])
def test_bad_pose_is_not_calibrated_away(index, value):
    q = plan['initial'].copy()
    q[index] += value
    with pytest.raises(ValueError):
        reference.make_mapping(plan, cal, q)
