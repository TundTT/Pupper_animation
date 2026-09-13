#!/usr/bin/env python3
"""Exclusive X roll / Triangle walk / Circle wheel dispatch. Never starts hardware."""
import json
import math
import time
from pathlib import Path

ROLL = 'neural_controller_triangle_roll'
WALK = 'neural_controller_walk_v2'
WHEEL = 'neural_controller_wheel'
POSE = 'neural_controller_joint_pose'
OWNERS = {ROLL, WALK, WHEEL, POSE}
BUTTONS = {0: ROLL, 2: WALK, 1: WHEEL}


def acquire_instance(folder):
    """A second dispatcher must exit before creating any ROS publishers."""
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'motion-buttons.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another motion button handler is already running')
    return lock


def choose_button(previous, buttons, busy):
    """Consume edges even while busy: a held button cannot become a queued action."""
    if len(buttons) <= 10 or buttons[10] or busy:
        return None
    edges = [mode for index, mode in BUTTONS.items()
             if buttons[index] and not previous[index]]
    return edges[0] if len(edges) == 1 else None


def validate_entry(mode, q, cal, mapping, roll_status=None):
    names = cal['joint_names']
    if len(q) != 12 or any(n not in q for n in names):
        raise ValueError('Fresh feedback for all 12 joints required')
    if any(not math.isfinite(x) for v in q.values() for x in v):
        raise ValueError('Nonfinite joint feedback')
    if any(abs(q[n][1]) > .15 for n in names):
        raise ValueError('Wait for stationary joints before switching')
    if roll_status is not None and (len(roll_status) != 16 or
            roll_status[0] != 4 or roll_status[1] != 1 or roll_status[4] != 0):
        raise ValueError('Roll must finish without a fault before changing modes')
    if mode == ROLL:
        if mapping is None or mapping['calibration_id'] != cal['calibration_id']:
            raise ValueError('Capture the calibrated tips-up reference first')
        for i, n in enumerate(names):
            difference = q[n][0] - mapping['captured_q'][i]
            if i % 3 == 2:
                difference = math.remainder(difference, 2 * math.pi)
            if abs(difference) > (.35 if i % 3 == 2 else .20):
                raise ValueError('X requires the prepared tips-up starting pose')
    if mode == WALK:
        home = [1, 0, -1, -1, 0, 1] * 2
        for i, n in enumerate(names):
            reference = 0
            if i % 3 == 2:
                reference = (mapping['model_to_encoder_offset'][i] if mapping
                             else cal['wheel_home'][i // 3] - home[i])
                difference = math.remainder(q[n][0] - reference - home[i], 2 * math.pi)
            else:
                difference = q[n][0] - home[i]
            if abs(difference) > .30:
                raise ValueError('Triangle requires a near-standing tips-down pose')


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from controller_manager_msgs.srv import ListControllers, SwitchController
    from sensor_msgs.msg import Joy, JointState, Imu
    from std_msgs.msg import Empty, Int32, Float64MultiArray
    from robot_calibration import load_current
    from robot_calibration.storage import directory, atomic_json

    class Buttons(Node):
        def __init__(self):
            super().__init__('quadmorph_motion_buttons')
            self.previous = [1] * 13  # release before the first actionable edge
            self.axes = []
            self.joy_at = self.joints_at = self.imu_at = self.status_at = self.commands_at = 0
            self.tilt = math.inf
            self.joints = {}
            self.status = self.commands = None
            self.busy = False
            self.pending_start = False
            self.stopped = False
            self.epoch = 0
            self.stop_pub = self.create_publisher(Empty, '/emergency_stop', 10)
            self.start_pub = self.create_publisher(Int32, '/triangle_roll/command', 1)
            self.list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
            self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
            self.create_subscription(Joy, '/joy', self.joy, qos_profile_sensor_data)
            self.create_subscription(JointState, '/joint_states', self.joint, qos_profile_sensor_data)
            self.create_subscription(Imu, '/imu_sensor_broadcaster/imu', self.imu, qos_profile_sensor_data)
            self.create_subscription(Float64MultiArray, '/neural_controller_triangle_roll/status', self.roll, 1)
            self.create_subscription(Float64MultiArray, '/neural_controller_triangle_roll/motor_commands', self.motor, 1)
            self.create_timer(.05, self.watch)
            self.get_logger().info('X: roll and hold; Triangle: walk; Circle: wheels; PS: stop. All initially inactive.')

        def stop(self, reason):
            self.epoch += 1
            self.pending_start = False
            self.busy = False
            self.stop_pub.publish(Empty())
            if not self.stopped:
                self.get_logger().error(reason)
            self.stopped = True

        def joint(self, m):
            self.joints = dict(zip(m.name, zip(m.position, m.velocity)))
            self.joints_at = time.monotonic()

        def imu(self, m):
            q = m.orientation
            norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
            self.tilt = math.acos(max(-1., min(1., 1-2*(q.x*q.x+q.y*q.y)))) if abs(norm-1)<.02 else math.inf
            self.imu_at = time.monotonic()

        def roll(self, m):
            self.status, self.status_at = list(m.data), time.monotonic()

        def motor(self, m):
            self.commands, self.commands_at = list(m.data), time.monotonic()

        def fresh(self):
            now = time.monotonic()
            if (now-self.joy_at > .3 or len(self.previous)<=10 or self.previous[10] or
                    now-self.joints_at > .2 or now-self.imu_at > .1 or self.tilt >= math.radians(8)):
                raise ValueError('Fresh controller, encoders and level torso required for mode entry')
            if len(self.axes)<4 or any(not math.isfinite(self.axes[i]) or abs(self.axes[i])>.15 for i in (0,1,3)):
                raise ValueError('Release the drive sticks before changing modes')

        def joy(self, m):
            self.joy_at = time.monotonic()
            mode = choose_button(self.previous, m.buttons, self.busy)
            self.previous = list(m.buttons)
            self.axes = list(m.axes)
            if len(m.buttons)<=10 or m.buttons[10]:
                self.stop('PS stop or incomplete gamepad input')
                return
            if mode:
                self.request(mode)

        def request(self, mode):
            try:
                self.fresh()
                load_current()
                if not self.list_client.service_is_ready() or not self.switch_client.service_is_ready():
                    raise ValueError('Controller manager not ready')
                self.busy = True
                self.started_at = time.monotonic()
                epoch = self.epoch
                f = self.list_client.call_async(ListControllers.Request())
                f.add_done_callback(lambda future: self.listed(future, mode, epoch))
            except Exception as e:
                self.busy = False
                self.get_logger().warning(str(e))

        def listed(self, future, mode, epoch):
            if epoch != self.epoch:
                return
            try:
                self.fresh()
                cal = load_current()
                states = future.result().controller
                active = [c.name for c in states if c.state=='active' and c.claimed_interfaces]
                if any(n not in OWNERS for n in active):
                    raise ValueError('Another command owner is active')
                if mode in active:
                    self.busy = False  # repeated button does not restart a live policy
                    return
                available = {c.name: c.state for c in states}
                if available.get(mode) != 'inactive':
                    raise ValueError('Requested controller is not loaded inactive')
                mapping_path = directory() / 'triangle-roll-map.json'
                mapping = json.loads(mapping_path.read_text()) if mode != WHEEL and mapping_path.exists() else None
                if mapping and mapping['calibration_id'] != cal['calibration_id']:
                    raise ValueError('Roll reference belongs to an old calibration')
                if mapping:
                    from ament_index_python.packages import get_package_share_directory
                    plan = json.loads((Path(get_package_share_directory('neural_controller')) /
                                       'launch/triangle_roll_plan.json').read_text())
                    if (mapping.get('schema_version') != 2 or mapping.get('axial_gap_m') != .009 or
                            mapping.get('operator_confirmed_inverted_start') is not True or
                            mapping.get('plan_sha256') != plan['plan_sha256'] or
                            mapping.get('joint_names') != cal['joint_names'] or
                            mapping.get('wheel_home') != cal['wheel_home']):
                        raise ValueError('Roll reference contract mismatch')
                    for i in range(12):
                        offset = mapping['model_to_encoder_offset'][i]
                        captured = mapping['captured_q'][i]
                        if not math.isfinite(offset) or not math.isfinite(captured) or (i % 3 != 2 and offset != 0):
                            raise ValueError('Invalid roll coordinate reference')
                        if i % 3 == 2 and abs(captured-plan['initial'][i]-offset) > 1e-9:
                            raise ValueError('Inconsistent roll coordinate reference')
                status = None
                if ROLL in active:
                    if time.monotonic()-self.status_at > .2:
                        raise ValueError('Roll status is stale')
                    status = self.status
                validate_entry(mode, self.joints, cal, mapping, status)
                handoff_path = directory() / 'walking-handoff.json'
                if mode == WALK and ROLL in active:
                    if time.monotonic()-self.commands_at > .2 or len(self.commands or []) != 60:
                        raise ValueError('Fresh roll commands required for smooth walking handoff')
                    for i, name in enumerate(cal['joint_names']):
                        position, _, _, kp, kd = self.commands[5*i:5*i+5]
                        if (not all(math.isfinite(v) for v in (position, kp, kd)) or
                                abs(position-self.joints[name][0])>.30 or not 0<=kp<=5 or not 0<=kd<=.5):
                            raise ValueError('Roll commands outside walking handoff envelope')
                    atomic_json(handoff_path, {'calibration_id': cal['calibration_id'],
                        'time_unix': time.time(), 'source': 'completed_triangle_roll',
                        'joint_names': cal['joint_names'], 'positions': self.commands[0::5],
                        'kp': self.commands[3::5], 'kd': self.commands[4::5]})
                elif mode == WALK:
                    handoff_path.unlink(missing_ok=True)
                req = SwitchController.Request()
                req.activate_controllers = [mode]
                req.deactivate_controllers = active
                req.strictness = SwitchController.Request.STRICT
                req.timeout.sec = 2
                f = self.switch_client.call_async(req)
                f.add_done_callback(lambda future: self.switched(future, mode, epoch))
            except Exception as e:
                self.busy = False
                self.get_logger().warning('Mode unchanged: '+str(e))

        def switched(self, future, mode, epoch):
            if epoch != self.epoch:
                self.stop_pub.publish(Empty())
                return
            try:
                if not future.result().ok:
                    raise ValueError('Controller switch rejected; inspect logs before retrying')
                self.stopped = False
                self.pending_start = mode == ROLL
                self.busy = self.pending_start
                self.activated_at = time.monotonic()
                self.get_logger().info('Activated '+mode)
            except Exception as e:
                self.stop(str(e))

        def watch(self):
            now = time.monotonic()
            if now-self.joy_at > .5:
                self.stop('Gamepad input lost')
            if self.busy and now-self.started_at > 12:
                self.stop('Mode entry timed out')
            if self.pending_start and self.status_at > self.activated_at:
                try:
                    self.fresh()
                    if len(self.status)!=16 or self.status[0]==5:
                        raise ValueError('Roll activation fault')
                    if self.status[0]==1 and self.status[9]==1 and self.start_pub.get_subscription_count():
                        self.start_pub.publish(Int32(data=1))
                        self.pending_start = self.busy = False
                        self.get_logger().info('X START sent once; completion holds and waits for Triangle')
                except Exception as e:
                    self.stop(str(e))

    instance_lock = acquire_instance(directory())
    rclpy.init()
    node = Buttons()
    try:
        rclpy.spin(node)
    finally:
        node.stop('Motion button process exiting')
        node.destroy_node()
        rclpy.shutdown()
        instance_lock.close()


if __name__ == '__main__':
    main()
