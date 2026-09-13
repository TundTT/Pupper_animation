#!/usr/bin/env python3
"""Read-only cross-boot encoder diagnostic. Never commands or calibrates motors."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
UPPER=[f'leg_{leg}_{j}' for leg in ('front_r','front_l','back_r','back_l') for j in (1,2)]

def raw_positions(log, urdf, positions, owner_pid):
    if not re.search(r'ros2_control_node[^\n]*process started with pid \['+str(int(owner_pid))+r'\]',log):
        raise ValueError('Log does not belong to the live hardware-manager PID')
    matches=re.findall(r'Homing (leg_\w+_[123]) at raw=([-+0-9.eE]+)',log)
    if len(matches)!=12 or len(dict(matches))!=12:
        raise ValueError('Need exactly one complete raw startup record')
    raw=dict((n,float(v)) for n,v in matches)
    joints=ET.fromstring(urdf).findall('./ros2_control/joint')
    home={j.attrib['name']:float(j.find("./param[@name='homed_position']").text) for j in joints}
    out={n:positions[n]+raw[n]-home[n] for n in UPPER}
    if not all(math.isfinite(x) for x in out.values()):raise ValueError('Nonfinite raw reading')
    return out,{n:raw[n]-home[n] for n in UPPER}

def compare(before,after):
    if before['boot_id']==after['boot_id']:raise ValueError('A full Pi reboot has not occurred')
    if before['pose_label']!=after['pose_label']:raise ValueError('Different physical pose labels')
    result={}
    for n in UPPER:
        delta=after['raw_joint_positions'][n]-before['raw_joint_positions'][n]
        result[n]={'difference_rad':delta,'difference_deg':math.degrees(delta),
            'observed_stationary_range_deg':math.degrees(max(before['position_ranges'][n],after['position_ranges'][n]))}
    return {'joint_differences':result,'max_abs_difference_deg':max(abs(v['difference_deg']) for v in result.values()),
        'log_rounding_difference_bound_deg':math.degrees(.0001),
        'note':'Unwrapped differences. Includes any actual joint movement, backlash and measurement noise. One same-pose test does not prove absolute position or persistent homing.',
        'persistent_homing_approved':False}

def capture(args):
    sys.path.insert(0,str(ROOT/'ros2_ws/src/robot_calibration'))
    from robot_calibration.storage import current_session,directory
    import rclpy
    from sensor_msgs.msg import JointState
    from rcl_interfaces.srv import GetParameters
    from controller_manager_msgs.srv import ListControllers
    session_id=current_session()
    session=json.loads((directory()/'encoder-session.json').read_text())
    log=args.log.read_text()
    # The current PID could be reused after a reboot; also bind log timestamps to this boot.
    boot_wall=time.time()-float(Path('/proc/uptime').read_text().split()[0])
    raw_times=re.findall(r'\[INFO\] \[([0-9.]+)\].*Homing leg_',log)
    if len(raw_times)!=12 or any(float(t)<boot_wall for t in raw_times):raise ValueError('Raw log predates this boot')
    rclpy.init();node=rclpy.create_node('upper_encoder_persistence_observer')
    def call(kind,name,request):
        c=node.create_client(kind,name)
        if not c.wait_for_service(timeout_sec=3):raise RuntimeError('Missing service '+name)
        f=c.call_async(request);rclpy.spin_until_future_complete(node,f,timeout_sec=4)
        if not f.done() or f.result() is None:raise RuntimeError('Service timeout '+name)
        return f.result()
    def inactive():
        response=call(ListControllers,'/controller_manager/list_controllers',ListControllers.Request())
        if any(c.state=='active' and c.claimed_interfaces for c in response.controller):
            raise ValueError('Deactivate motion controllers before diagnostic recording')
    try:
        inactive()
        req=GetParameters.Request();req.names=['robot_description']
        urdf=call(GetParameters,'/robot_state_publisher/get_parameters',req).values[0].string_value
        rows={n:[] for n in UPPER};last_stamp=None;last_receive=None
        def receive(m):
            nonlocal last_stamp,last_receive,rows
            stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;now=time.monotonic()
            if abs(time.time()-stamp)>.2 or (last_stamp is not None and stamp<=last_stamp):return
            q=dict(zip(m.name,m.position))
            if len(q)!=len(m.name) or any(n not in q or not math.isfinite(q[n]) for n in UPPER):return
            if last_receive is not None and now-last_receive>.2:rows={n:[] for n in UPPER}
            last_stamp=stamp;last_receive=now
            for n in UPPER:rows[n].append(q[n])
        sub=node.create_subscription(JointState,'/joint_states',receive,20)
        until=time.monotonic()+3
        while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.05)
        if last_receive is None or time.monotonic()-last_receive>.2 or min(map(len,rows.values()))<100:
            raise ValueError('Insufficient fresh encoder data')
        ranges={n:max(v)-min(v) for n,v in rows.items()}
        if max(ranges.values())>.002:raise ValueError('Upper joints moved during recording; hold the same physical pose and retry')
        medians={n:statistics.median(v) for n,v in rows.items()}
        raw,offsets=raw_positions(log,urdf,medians,session['owner_pid'])
        inactive()
        if current_session()!=session_id:raise ValueError('Encoder session changed during capture')
        record={'schema_version':1,'purpose':'encoder_persistence_diagnostic_only','boot_id':session['boot_id'],
            'encoder_session_id':session_id,'owner_pid':session['owner_pid'],'pose_label':args.pose_label,
            'time_unix':time.time(),'raw_joint_positions':raw,'position_ranges':ranges,
            'joint_positions':medians,'reconstructed_startup_offsets':offsets,
            'raw_log_rounding_bound_rad':.00005,'calibration_modified':False,'motion_commanded':False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.exists():raise ValueError('Refusing to overwrite diagnostic evidence')
        args.output.write_text(json.dumps(record,indent=2)+'\n')
        args.output.with_suffix('.startup.log').write_text(log)
        args.output.with_suffix('.urdf').write_text(urdf)
        print(json.dumps(record,indent=2))
    finally:node.destroy_node();rclpy.shutdown()

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    c=s.add_parser('capture');c.add_argument('--log',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    c.add_argument('--pose-label',required=True)
    c=s.add_parser('compare');c.add_argument('before',type=Path);c.add_argument('after',type=Path)
    a=p.parse_args()
    if a.mode=='capture':capture(a)
    else:print(json.dumps(compare(json.loads(a.before.read_text()),json.loads(a.after.read_text())),indent=2))
if __name__=='__main__':main()
