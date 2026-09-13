"""Offline comparison; does not establish persistent homing or touch hardware."""
import json,math,re,statistics,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(__file__).resolve().parents[3]
checks=root/'hardware_testing/start_pose/encoder_checks'
log=(checks/'before_3cedd58c.startup.log').read_text()
raw_home={n:float(v) for n,v in re.findall(r'Homing (leg_\w+_[123]) at raw=([-+0-9.eE]+)',log)}
assert len(raw_home)==12
description=ET.parse(checks/'before_3cedd58c.urdf')
report=json.loads((root/'hardware_testing/joint_pose/runs/fd313581a9734c0b9253dcb1168477e9/report.json').read_text())
after=json.loads((checks/'disabled_4c724137.json').read_text())
# Discard startup packets. Compare the last 50 valid replies over about one second.
samples=after['rows'][-50:]
assert all(not b['all_zero_packet'] for s in samples for b in s['boards'])
rows={}
for j in description.findall('./ros2_control/joint'):
    n=j.attrib['name'];p={x.attrib['name']:x.text for x in j.findall('param')}
    channel=int(p['can_channel'])-1;can_id=int(p['can_id'])-1
    before=report['positions'][n]+raw_home[n]-float(p['homed_position'])
    # rt_spi.cpp applies -1 to every incoming joint channel before session offset.
    values=[-s['boards'][channel//2]['q'][2*can_id+channel%2] for s in samples]
    current=statistics.median(values)
    rows[n]={'before_sign_corrected_raw':before,'after_sign_corrected_raw':current,
             'difference_rad':current-before,'difference_deg':math.degrees(current-before),
             'after_range_rad':max(values)-min(values)}
result={'before_boot':'3cedd58c-4293-4826-85ae-72a90093b497','after_boot':after['boot_id'],
        'joint_results':rows,'operator_reported_no_physical_joint_movement':True,
        'max_upper_difference_deg':max(abs(v['difference_deg']) for n,v in rows.items() if not n.endswith('_3')),
        'persistent_homing_approved':False,
        'limitations':['Before snapshot is at successful HOLD entry, not immediately at power loss.',
                       'Includes possible settling after final snapshot, manual measurement uncertainty and raw log rounding.',
                       'Disabled SPI replies changed from zero to nonzero, but per-motor timestamps/sequence counters are absent.',
                       'One unchanged-pose restart cannot establish behavior when joints move while unpowered.']}
Path(__file__).with_name('encoder_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
