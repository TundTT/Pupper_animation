#!/usr/bin/env python3
"""Read-only tip-up map capture during audited keyframe HOLD. Never commands motors."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws/src/robot_calibration"))
from robot_calibration.storage import (JOINT_NAMES, StationarySample, atomic_json,
    capture_lock, current_session, directory, finite_vector, load_current)

CONTROLLER = "neural_controller_keyframe_align"
FIELDS = ("position", "velocity", "effort", "kp", "kd")


def check_owners(controllers):
    owners = [c for c in controllers if c.state == "active" and c.claimed_interfaces]
    expected = {f"{joint}/{field}" for joint in JOINT_NAMES for field in FIELDS}
    if (len(owners) != 1 or owners[0].name != CONTROLLER or
            owners[0].type != "neural_controller/KeyframeController" or
            len(owners[0].claimed_interfaces) != 60 or
            set(owners[0].claimed_interfaces) != expected):
        raise ValueError("Only the audited keyframe controller may own the 60 command interfaces")


def check_hold(status, commands, command, receipts, now, q):
    status = finite_vector(status, 28)
    commands = finite_vector(commands, 60)
    q = finite_vector(q, 12)
    if len(receipts) != 3 or any(not math.isfinite(t) or not 0 <= now-t <= .2 for t in receipts):
        raise ValueError("Need fresh HOLD status, motor telemetry and command-0 heartbeat")
    if command != 0 or any(status[i] != value for i, value in
                          ((12,6),(13,0),(14,0),(16,-1),(23,1),(24,0),(25,0))):
        raise ValueError("Keyframe must remain command 0, stationary HOLD, no failures or stop")
    if not 0 < status[26] <= .04 or not 0 <= status[27] <= .1:
        raise ValueError("Invalid controller update or IMU age")
    neutral = [1.,0.,-1.,0.,1.,0.,-1.,0.]
    if any(abs(a-b) > .015 for a,b in zip(status[:8], neutral)):
        raise ValueError("Upper command is not neutral")
    targets = []
    for i in range(12):
        target, velocity, effort, kp, kd = commands[5*i:5*i+5]
        if velocity != 0 or effort != 0 or not 0 < kp <= 10 or not 0 < kd <= 1:
            raise ValueError("Expected position hold without velocity or extra effort")
        if i % 3 != 2 and abs(target-status[2*(i//3)+i%3]) > .001:
            raise ValueError("Motor telemetry disagrees with proximal HOLD status")
        if abs(q[i]-target) > (.03 if i%3 == 2 else .1):
            raise ValueError("Measured joint has not settled near its held target")
        targets.append(target)
    return targets


def make_map(plan, calibration, q, evidence):
    q = finite_vector(q, 12)
    if plan.get("schema_version") != 3 or plan.get("joint_names") != JOINT_NAMES or plan.get("axial_gap_m") != .009:
        raise ValueError("This helper requires the reviewed measured-roll schema-3 plan")
    initial = finite_vector(plan["initial"], 12)
    offset = [0.] * 12
    for i in range(12):
        if i%3 == 2:
            offset[i] = q[i]-initial[i]
        elif abs(q[i]-initial[i]) > (.20 if i%3 == 1 else .15):
            raise ValueError("Measured upper joint is outside the prepared roll entry region")
    return dict(schema_version=2, mapping_id=uuid.uuid4().hex,
        calibration_id=calibration["calibration_id"], encoder_session_id=calibration["encoder_session_id"],
        wheel_home=calibration["wheel_home"], joint_names=JOINT_NAMES,
        operator_confirmed_inverted_start=True, axial_gap_m=.009,
        plan_sha256=plan["plan_sha256"], captured_q=q, model_to_encoder_offset=offset,
        convention="Confirmed rigid tips-up triangles; proximal frame unchanged; read-only stationary keyframe HOLD capture",
        hold_capture_evidence=evidence)


def sample_hold(expected_session, timeout=15.):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from controller_manager_msgs.srv import ListControllers
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray, Int32
    rclpy.init(args=[])
    node = Node("triangle_readonly_hold_capture")
    sample = StationarySample()
    status = commands = command = None
    receipts = [-math.inf]*3
    ready = False
    initial_targets = None
    last_error = "waiting for fresh inputs"

    def observe_status(msg):
        nonlocal status
        status=list(msg.data); receipts[0]=time.monotonic()

    def observe_commands(msg):
        nonlocal commands
        commands=list(msg.data); receipts[1]=time.monotonic()

    def observe_command(msg):
        nonlocal command
        command=msg.data; receipts[2]=time.monotonic()

    def observe_joints(msg):
        nonlocal ready, initial_targets, last_error
        ready = False
        now=time.monotonic()
        try:
            names=list(msg.name)
            if len(names)!=len(set(names)):
                raise ValueError("Duplicate joint names")
            q=[msg.position[names.index(name)] for name in JOINT_NAMES]
            targets=check_hold(status,commands,command,receipts,now,q)
            if initial_targets is not None and any(abs(a-b)>.001 for a,b in zip(targets,initial_targets)):
                raise ValueError("Hold targets changed during capture")
            initial_targets=targets
            ready=sample.observe(names,list(msg.position),list(msg.velocity),
                msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                node.get_clock().now().nanoseconds*1e-9,now)
        except (ValueError,TypeError,IndexError) as exc:
            sample.reset(); initial_targets=None; last_error=str(exc)

    node.create_subscription(Float64MultiArray,f"/{CONTROLLER}/alignment_status",observe_status,10)
    node.create_subscription(Float64MultiArray,f"/{CONTROLLER}/motor_commands",observe_commands,10)
    node.create_subscription(Int32,"/keyframe_align_command_index",observe_command,10)
    node.create_subscription(JointState,"/joint_states",observe_joints,qos_profile_sensor_data)
    client=node.create_client(ListControllers,"/controller_manager/list_controllers")
    deadline=time.monotonic()+timeout

    def ownership():
        if not client.wait_for_service(timeout_sec=max(0.,deadline-time.monotonic())):
            raise ValueError("Controller manager unavailable")
        future=client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(node,future,timeout_sec=max(0.,deadline-time.monotonic()))
        if not future.done() or future.result() is None:
            raise ValueError("Controller ownership query timed out")
        check_owners(future.result().controller)

    try:
        ownership(); sample.reset(); ready=False; initial_targets=None
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            if ready:
                ownership()
                if not ready or sample.last_receipt is None or time.monotonic()-sample.last_receipt>.2:
                    continue
                targets=check_hold(status,commands,command,receipts,time.monotonic(),sample.positions)
                if current_session()!=expected_session:
                    raise ValueError("Encoder session changed")
                return sample.positions,dict(controller=CONTROLLER,phase="HOLD",command=0,
                    status=status,motor_commands=commands,held_targets=targets,
                    stationarity_check=StationarySample.POLICY,read_only=True)
        raise ValueError("No stationary audited HOLD sample: "+last_error)
    finally:
        node.destroy_node(); rclpy.shutdown()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator-confirmed-inverted-start",action="store_true")
    p.add_argument("--replace",action="store_true")
    args=p.parse_args()
    if not args.operator_confirmed_inverted_start:
        raise ValueError("Actual operator confirmation of rigid tips-up triangles and 9 mm gaps is required")
    plan_path=ROOT/"ros2_ws/src/neural_controller/launch/triangle_roll_plan.json"
    provenance=json.loads((ROOT/"hardware_testing/triangle_roll/export_provenance.json").read_text())
    raw=plan_path.read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n",b"\n")).hexdigest()!=provenance["export_sha256"]:
        raise ValueError("Unreviewed exported plan")
    plan=json.loads(raw)
    path=directory()/"triangle-roll-map.json"
    with capture_lock():
        calibration=load_current()
        if path.exists() and not args.replace:
            raise ValueError("Mapping exists; explicitly --replace only after reconfirmation")
        q,evidence=sample_hold(calibration["encoder_session_id"])
        if load_current()["calibration_id"]!=calibration["calibration_id"]:
            raise ValueError("Calibration changed")
        record=make_map(plan,calibration,q,evidence)
        atomic_json(directory()/"history"/("inverted-"+record["mapping_id"]+".json"),record)
        if current_session()!=calibration["encoder_session_id"]:
            raise ValueError("Encoder session changed before saving map")
        atomic_json(path,record)
    print(json.dumps(record,indent=2))
    print(f"Mapping: {path}. Startup calibration unchanged. No motor commands sent.")


if __name__=="__main__":
    try:
        main()
    except (ValueError,OSError,KeyError,ImportError) as exc:
        print("TRIANGLE HOLD REFERENCE NOT READY: "+str(exc),file=sys.stderr)
        raise SystemExit(1)
