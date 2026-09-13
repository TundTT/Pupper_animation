#!/usr/bin/env python3
"""Prepare or execute a supported joint-position request in the live calibrated frame."""
import argparse,json,math,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ros2_ws/src/robot_calibration'))
from robot_calibration import load_current,JOINT_NAMES
from robot_calibration.storage import directory,atomic_json

def targets(q,preset,joints,calibration):
    result=dict(q)
    if preset:
        profile=json.loads((ROOT/'hardware_testing/start_pose/approved_upper_pose.json').read_text())
        result.update(profile['joint_positions'])
    if preset=='tips-up':
        mapping=json.loads((directory()/'triangle-roll-map.json').read_text())
        if mapping['calibration_id']!=calibration['calibration_id']:raise ValueError('Tips-up reference belongs to another calibration')
        for i,n in enumerate(JOINT_NAMES):
            if i%3==2:
                base=mapping['captured_q'][i]
                result[n]=base+2*math.pi*round((q[n]-base)/(2*math.pi))
    for item in joints:
        n,value=item.split('=',1)
        if n not in JOINT_NAMES:raise ValueError('Unknown joint '+n)
        result[n]=float(value)
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('Finite targets required')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',choices=['upper-home','tips-up'])
    p.add_argument('--joint',action='append',default=[],help='Named joint position in radians: leg_front_r_1=1.0')
    p.add_argument('--execute',action='store_true')
    p.add_argument('--stream-telemetry',action='store_true',help='Also stream each new controller sample to stdout for an SSH log copy')
    p.add_argument('--operator-confirmed-supported',action='store_true')
    a=p.parse_args()
    if not a.preset and not a.joint:p.error('Provide a preset or named joint targets')
    if a.execute and not a.operator_confirmed_supported:p.error('Actual operator confirmation of supported, clear limbs required')
    cal=load_current()
    import rclpy
    from sensor_msgs.msg import JointState,Joy
    from std_msgs.msg import Float64MultiArray,Empty
    from controller_manager_msgs.srv import ListControllers
    rclpy.init();node=rclpy.create_node('joint_pose_request')
    q={};received=0;joy_time=0;ps=True;status=None;status_time=0
    def joint(m):
        nonlocal q,received
        if len(m.name)!=len(set(m.name)) or len(m.name)!=len(m.position):return
        q=dict(zip(m.name,m.position));received=time.monotonic()
    def joy(m):
        nonlocal joy_time,ps
        joy_time=time.monotonic();ps=len(m.buttons)<=10 or bool(m.buttons[10])
    def observe(m):
        nonlocal status,status_time
        status=list(m.data);status_time=time.monotonic()
    node.create_subscription(JointState,'/joint_states',joint,10)
    node.create_subscription(Joy,'/joy',joy,10)
    stop=node.create_publisher(Empty,'/emergency_stop',10)
    active=False;report={};evidence=None
    try:
        client=node.create_client(ListControllers,'/controller_manager/list_controllers')
        if not client.wait_for_service(timeout_sec=3):raise RuntimeError('No controller manager')
        f=client.call_async(ListControllers.Request());rclpy.spin_until_future_complete(node,f,timeout_sec=3)
        if not f.done() or f.result() is None:raise RuntimeError('Controller query failed')
        if any(c.state=='active' and c.claimed_interfaces for c in f.result().controller):
            raise ValueError('Motion controller already active; stop it before a new supported pose request')
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.05)
            if all(n in q for n in JOINT_NAMES) and time.monotonic()-received<.2 and time.monotonic()-joy_time<.3 and not ps:break
        if time.monotonic()-received>.2 or time.monotonic()-joy_time>.3 or ps:raise ValueError('Fresh encoders and gamepad required')
        q={n:q[n] for n in JOINT_NAMES};goal=targets(q,a.preset,a.joint,cal)
        request={'request_id':uuid.uuid4().hex,'calibration_id':cal['calibration_id'],
            'operator_confirmed_supported':a.operator_confirmed_supported,'scope':'supported_joint_pose',
            'joint_names':JOINT_NAMES,'target_positions':[goal[n] for n in JOINT_NAMES],
            'captured_positions':[q[n] for n in JOINT_NAMES],'prepared_at_unix':time.time()}
        print(json.dumps({'targets':goal,'moves_rad':{n:goal[n]-q[n] for n in JOINT_NAMES},'execute':a.execute}),flush=True)
        if not a.execute:return
        evidence=ROOT/'hardware_testing/joint_pose/runs'/request['request_id'];evidence.mkdir(parents=True)
        atomic_json(evidence/'request.json',request);atomic_json(directory()/'joint-pose-request.json',request)
        subprocess.run(['ros2','control','switch_controllers','--activate','neural_controller_joint_pose','--strict'],check=True)
        active=True;started=time.monotonic();previous=None;last_streamed=None
        # Subscribe after activation: volatile QoS avoids queued inactive telemetry
        # collected while the blocking controller-switch CLI was running.
        node.create_subscription(Float64MultiArray,'/neural_controller_joint_pose/status',observe,10)
        with (evidence/'telemetry.jsonl').open('w',buffering=1) as log:
            while time.monotonic()-started<65:
                rclpy.spin_once(node,timeout_sec=.05)
                if status_time<started:
                    if time.monotonic()-started>3:raise RuntimeError('No status after activation')
                    continue
                if time.monotonic()-status_time>.4:raise RuntimeError('Status stale')
                if len(status)!=65 or not all(math.isfinite(v) for v in status):raise RuntimeError('Invalid status')
                log.write(json.dumps({'time':time.time(),'status':status,'joints':q})+'\n')
                if a.stream_telemetry and status_time!=last_streamed:
                    print(json.dumps({'time':time.time(),'status':status,'joints':q}),flush=True)
                    last_streamed=status_time
                if int(status[0])!=previous:
                    previous=int(status[0]);print(json.dumps({'phase':previous,'fault':status[1],'elapsed':status[2],'error':status[3]}),flush=True)
                if status[0]==4:raise RuntimeError('Joint controller fault '+str(int(status[1])))
                if status[0]==3:
                    report={'completed':True,'status':status[:5],'positions':q,'request_id':request['request_id']}
                    atomic_json(evidence/'report.json',report);print(json.dumps(report),flush=True);return
            raise RuntimeError('Joint move timeout')
    except Exception as e:
        if active:
            for _ in range(3):stop.publish(Empty());rclpy.spin_once(node,timeout_sec=.05)
        if evidence:atomic_json(evidence/'report.json',{'completed':False,'error':str(e),'status':status})
        raise
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
