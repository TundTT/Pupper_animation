"""Offline comparison of saved HOLD raw coordinates and disabled post-boot SPI."""
import json,math,statistics,xml.etree.ElementTree as ET
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[2]
before=json.loads((here/'settled_tips_down.json').read_text())
after=json.loads((root/'hardware_testing/start_pose/encoder_checks/disabled_4e6b3aa4.json').read_text())
assert before['boot_id']!=after['boot_id']
samples=after['rows'][-50:]
assert all(not b['all_zero_packet'] for s in samples for b in s['boards'])
rows={}
for j in ET.parse(here/'actual_model.urdf').findall('./ros2_control/joint'):
    n=j.get('name');p={x.get('name'):x.text for x in j.findall('param')}
    channel=int(p['can_channel'])-1;can_id=int(p['can_id'])-1
    values=[-s['boards'][channel//2]['q'][2*can_id+channel%2] for s in samples]
    current=statistics.median(values);delta=current-before['raw_joint_positions'][n]
    rows[n]={'before_raw_rad':before['raw_joint_positions'][n],'after_raw_rad':current,
             'difference_rad':delta,'difference_deg':math.degrees(delta),
             'post_boot_range_rad':max(values)-min(values)}
result={'before_boot':before['boot_id'],'after_boot':after['boot_id'],'joints':rows,
        'max_upper_difference_deg':max(abs(v['difference_deg']) for n,v in rows.items() if not n.endswith('_3')),
        'physical_configuration':'Operator reported tips remained down through power-off/on.',
        'motor_stack_started_after_reboot':False,'persistent_homing_approved':False,
        'limitations':'Before capture was during HOLD, about minutes before power-off; includes possible physical settling. Disabled packets have no per-motor freshness counters. Do not infer a fixed hub correction.'}
(here/'reboot_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
