import importlib.util
from pathlib import Path
import pytest
path=Path(__file__).resolve().parents[2]/'scripts/check_upper_encoder_persistence.py'
spec=importlib.util.spec_from_file_location('persistence',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_reconstructs_raw_instead_of_comparing_rezeroed_coordinates():
    names=[f'leg_{leg}_{j}' for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)]
    log='ros2_control_node-2: process started with pid [42]\n'+''.join(f'Homing {n} at raw=0.4\n' for n in names)
    urdf='<robot><ros2_control>'+''.join(f'<joint name="{n}"><param name="homed_position">1</param></joint>' for n in names)+'</ros2_control></robot>'
    raw,offset=m.raw_positions(log,urdf,{n:1.2 for n in m.UPPER},42)
    assert all(abs(v-.6)<1e-12 for v in raw.values())
    with pytest.raises(ValueError):m.raw_positions(log,urdf,{n:1.2 for n in m.UPPER},43)
def test_no_wrapping_or_false_same_boot_acceptance():
    a={'boot_id':'a','pose_label':'A','raw_joint_positions':dict.fromkeys(m.UPPER,0.),'position_ranges':dict.fromkeys(m.UPPER,0.)}
    with pytest.raises(ValueError):m.compare(a,a)
    b=dict(a,boot_id='b',raw_joint_positions=dict.fromkeys(m.UPPER,2*m.math.pi))
    assert abs(m.compare(a,b)['max_abs_difference_deg']-360)<1e-9
    assert m.compare(a,b)['persistent_homing_approved'] is False
