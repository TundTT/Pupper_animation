"""Motor-free ROS joystick transport check on a separate DDS domain."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import rclpy
from sensor_msgs.msg import Joy

assert os.environ.get('ROS_DOMAIN_ID') == '96'
rclpy.init()
node = rclpy.create_node('triangle_input_preflight')
times = []
dimensions = set()
def receive(msg):
    times.append(time.monotonic())
    dimensions.add((len(msg.axes), len(msg.buttons)))
node.create_subscription(Joy, '/triangle_preflight/joy', receive, 10)
log = open('/home/pi/triangle-joy-node.log', 'w')
child = subprocess.Popen(['/opt/ros/jazzy/lib/joy_linux/joy_linux_node', '--ros-args',
    '-p', 'dev:=/dev/input/js0', '-p', 'autorepeat_rate:=20.0',
    '-r', 'joy:=/triangle_preflight/joy'], stdout=log, stderr=subprocess.STDOUT)
try:
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError('joy_linux exited')
        rclpy.spin_once(node, timeout_sec=.1)
    gaps = [b-a for a,b in zip(times, times[1:])]
    report = {'message_count': len(times), 'dimensions': sorted(dimensions),
              'max_receipt_gap_seconds': max(gaps) if gaps else None,
              'mean_hz': (len(times)-1)/(times[-1]-times[0]) if gaps else None,
              'motor_stack_started': False, 'dds_domain': 96}
    Path('/home/pi/triangle-joy-stream.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
    assert len(times) >= 150 and dimensions == {(8,13)} and max(gaps) < .5
finally:
    child.send_signal(signal.SIGINT)
    child.wait(timeout=5)
    log.close()
    node.destroy_node()
    rclpy.shutdown()
