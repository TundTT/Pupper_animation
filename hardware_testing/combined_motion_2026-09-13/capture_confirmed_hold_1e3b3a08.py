"""Read-only, operator-confirmed tips-up mapping during stationary JointPose HOLD."""
import hashlib,json,math,sys,time
from pathlib import Path
root=Path('/home/pi/robot-code-leglift')
sys.path[:0]=[str(root/'scripts'),str(root/'ros2_ws/src/robot_calibration')]
from robot_calibration.storage import JOINT_NAMES,StationarySample,capture_lock,directory,load_current,atomic_json
from inverted_triangle_reference import make_mapping
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import ListControllers
from rclpy.qos import qos_profile_sensor_data
rclpy.init();node=rclpy.create_node('confirmed_tip_up_hold_capture')
sample=StationarySample();status=None;status_time=0;ready=False;fixed=None
def check_status():
    if status is None or len(status)!=65 or time.monotonic()-status_time>.2:
        raise ValueError('Need fresh joint hold telemetry')
    if not all(math.isfinite(v) for v in status) or status[0]!=3 or status[1]!=0 or status[4]!=1:
        raise ValueError('Need completed HOLD without fault and fresh gamepad')
    for i in range(12):
        p,v,t,kp,kd=status[5+5*i:10+5*i]
        if v!=0 or t!=0 or not 0<kp<=10 or not 0<kd<=1:
            raise ValueError('Expected pure position hold')
    return status[5::5]
def observe_status(m):
    global status,status_time
    status=list(m.data);status_time=time.monotonic()
def observe_joints(m):
    global ready,fixed
    try:
        targets=check_status()
        if fixed is not None and any(abs(a-b)>1e-9 for a,b in zip(fixed,targets)):
            raise ValueError('Hold targets changed')
        q=dict(zip(m.name,m.position))
        if len(q)!=len(m.name) or any(abs(q[n]-targets[i])>.04 for i,n in enumerate(JOINT_NAMES)):
            raise ValueError('Not settled at fixed targets')
        fixed=targets
        ready=sample.observe(list(m.name),list(m.position),list(m.velocity),m.header.stamp.sec+m.header.stamp.nanosec*1e-9,node.get_clock().now().nanoseconds*1e-9,time.monotonic())
    except (ValueError,KeyError):
        sample.reset();fixed=None;ready=False
node.create_subscription(JointState,'/joint_states',observe_joints,qos_profile_sensor_data)
node.create_subscription(Float64MultiArray,'/neural_controller_joint_pose/status',observe_status,10)
client=node.create_client(ListControllers,'/controller_manager/list_controllers')
def ownership():
    if not client.wait_for_service(timeout_sec=4):raise ValueError('No manager')
    f=client.call_async(ListControllers.Request());rclpy.spin_until_future_complete(node,f,timeout_sec=4)
    if not f.done() or f.result() is None:raise ValueError('Ownership query failed')
    owners=[c for c in f.result().controller if c.state=='active' and c.claimed_interfaces]
    expected={f'{n}/{field}' for n in JOINT_NAMES for field in ('position','velocity','effort','kp','kd')}
    if len(owners)!=1 or owners[0].name!='neural_controller_joint_pose' or owners[0].type!='neural_controller/JointPoseController' or set(owners[0].claimed_interfaces)!=expected:
        raise ValueError('Unexpected command owner')
try:
 with capture_lock():
    cal=load_current()
    if cal['calibration_id']!='581797b3a1d9479ba1a1b91e9a4ad674':raise ValueError('Wrong confirmed session')
    ownership();sample.reset();ready=False;fixed=None
    end=time.monotonic()+20
    while time.monotonic()<end:
        rclpy.spin_once(node,timeout_sec=.02)
        if ready:
            ownership();check_status()
            if ready and sample.last_receipt is not None and time.monotonic()-sample.last_receipt<.2:break
    else:raise ValueError('No stationary HOLD sample')
    plan_path=root/'ros2_ws/src/neural_controller/launch/triangle_roll_plan.json'
    raw=plan_path.read_bytes().replace(b'\r\n',b'\n')
    provenance=json.loads((root/'hardware_testing/triangle_roll/export_provenance.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=provenance['export_sha256']:raise ValueError('Unreviewed plan')
    record=make_mapping(json.loads(raw),cal,sample.positions)
    record['convention']='Operator visually confirmed rigid tips-up after supported half-turn; stationary JointPose HOLD; upper frame unchanged.'
    record['hold_capture_evidence']={'controller':'neural_controller_joint_pose','status':status,'stationarity_check':StationarySample.POLICY,'read_only':True}
    if load_current()['calibration_id']!=cal['calibration_id']:raise ValueError('Calibration changed')
    path=directory()/'triangle-roll-map.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('calibration_id')==cal['calibration_id']:raise ValueError('Current-session map already exists; review before replacement')
    atomic_json(directory()/'history'/('inverted-'+record['mapping_id']+'.json'),record)
    atomic_json(path,record)
    atomic_json(root/'hardware_testing/combined_motion_2026-09-13/tips_up_reference_1e3b3a08.json',record)
    print(json.dumps({'mapping_id':record['mapping_id'],'captured_q':record['captured_q'],'calibration_unchanged':True,'motors_commanded':False}))
finally:
 node.destroy_node();rclpy.shutdown()
