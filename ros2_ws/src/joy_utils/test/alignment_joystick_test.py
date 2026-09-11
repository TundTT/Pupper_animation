"""Exercise real joystick dispatch against a fake controller manager; no hardware."""
import os
from pathlib import Path
import subprocess
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import Joy
from std_msgs.msg import Empty, Int32


def test_alignment_activation_failure_pending_press_and_estop(tmp_path):
    executable = os.environ["ALIGN_JOY_TEST_EXE"]
    domain = 93
    context = rclpy.context.Context()
    rclpy.init(context=context, domain_id=domain)
    node = rclpy.create_node("fake_alignment_manager", context=context)
    requests, commands, stops = [], [], []
    allow = [False]
    delay = [0.0]

    def switch(request, response):
        requests.append(list(request.activate_controllers))
        time.sleep(delay[0])
        response.ok = allow[0]
        return response

    node.create_service(SwitchController, "/controller_manager/switch_controller", switch)
    node.create_subscription(Int32, "/wheel_align_hybrid_command_index", lambda m: commands.append(m.data), 10)
    node.create_subscription(Empty, "/emergency_stop", lambda m: stops.append(True), 10)
    pub = node.create_publisher(Joy, "/joy", 10)
    executor = MultiThreadedExecutor(context=context)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    config = tmp_path / "joy.yaml"
    config.write_text('''/**:
  ros__parameters:
    calibration_required: false
    default_controller_name: neural_controller_wheel_align_hybrid
    controller_names: [neural_controller_wheel_align_hybrid]
    switch_button_indices: [-1]
    leg_lift_button_index: -1
    wheel_align_hybrid_button_index: 0
    wheel_align_hybrid_controller_name: neural_controller_wheel_align_hybrid
    estop_index: 12
    estop_release_index: 9
''')
    log = (tmp_path / "joy.log").open("w")
    process = subprocess.Popen([executable, "--ros-args", "--params-file", str(config)],
                               env={**os.environ, "ROS_DOMAIN_ID": str(domain)},
                               stdout=log, stderr=subprocess.STDOUT)

    def wait(predicate):
        deadline = time.monotonic()+8
        while not predicate():
            assert process.poll() is None, (tmp_path / "joy.log").read_text()
            assert time.monotonic() < deadline, (tmp_path / "joy.log").read_text()
            time.sleep(.03)

    def press(*buttons):
        msg = Joy()
        msg.buttons = [int(i in buttons) for i in range(13)]
        pub.publish(msg)
        time.sleep(.12)
        msg.buttons = [0]*13
        pub.publish(msg)
        time.sleep(.12)

    try:
        wait(lambda: pub.get_subscription_count() > 0)
        press(0)
        wait(lambda: len(requests) == 1)
        assert commands == []  # Failed activation did not dispatch a leg.
        allow[0] = True
        delay[0] = .8
        press(0)
        wait(lambda: len(requests) == 2)
        press(0)  # Activation pending: cannot skip to FR or issue FL early.
        assert commands == [] and len(requests) == 2
        time.sleep(.9)
        delay[0] = 0
        press(0)
        wait(lambda: commands == [1])
        press(0)
        wait(lambda: commands == [1, 2])
        press(12, 0)  # Stop takes precedence over a simultaneous X press.
        wait(lambda: stops and requests[-1] == [])
        assert commands == [1, 2]
        press(9)
        wait(lambda: len(requests) == 4)
        press(0)
        wait(lambda: commands == [1, 2, 1])  # Reentry resets cycle, not calibration.
    finally:
        process.terminate()
        process.wait(timeout=5)
        log.close()
        executor.shutdown()
        thread.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown(context=context)
