#!/usr/bin/env python3
"""Watch the PS5 controller for a D-pad-up press and (re)launch the neural_controller
stack. Runs standalone (not a ROS node) since joy_linux_node -- the thing that normally
reads the controller -- is itself part of the launch this triggers, so nothing ROS-side
can be listening before the stack exists.

Reads /dev/input/js0 directly via the kernel joystick API (evdev isn't installed on this
Pi image). D-pad up is axis 7, value -32767 (0 = released); confirmed by capturing raw
events with the controller connected.

Before every launch, kills any processes left over from a previous launch first -- this
is not optional. Orphaned robot_state_publisher/estop_controller/etc. instances serve a
stale, latched /robot_description to a fresh ros2_control_node, which silently defeats
the homing verification check (see control_board_hardware_interface.cpp) without any
error -- this exact failure mode cost an hour to diagnose by hand once already. Kills by
explicit PID, never `pkill -f`, since -f matches this script's own command line too.
"""
import errno
import os
import struct
import subprocess
import time

JS_DEVICE = "/dev/input/js0"
DPAD_UP_AXIS = 7
DPAD_UP_VALUE = -32767
DEBOUNCE_SECONDS = 3.0
LAUNCH_LOG = "/tmp/launch_out.log"
ROBOT_CODE_DIR = os.path.expanduser("~/robot-code-leglift")

KILL_PATTERNS = [
    "robot-code-leglift",
    "ros2_control_node",
    "robot_state_publisher",
    "ros2 launch neural_controller",
]


def find_pids_to_kill():
    pids = set()
    try:
        out = subprocess.run(["pgrep", "-f", "|".join(KILL_PATTERNS)],
                              capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return pids
    my_pid = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pid = int(line)
            if pid != my_pid:
                pids.add(pid)
    return pids


def kill_leftover_processes():
    pids = find_pids_to_kill()
    if not pids:
        print("[dpad_launch_trigger] no leftover processes, clean start")
        return
    print(f"[dpad_launch_trigger] killing {len(pids)} leftover process(es): {sorted(pids)}")
    for pid in pids:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
    time.sleep(2)
    remaining = find_pids_to_kill()
    if remaining:
        print(f"[dpad_launch_trigger] WARNING: {remaining} still alive after kill -9, "
              "launching anyway")


def launch_stack():
    kill_leftover_processes()
    print("[dpad_launch_trigger] launching neural_controller stack")
    cmd = (
        "source /opt/ros/jazzy/setup.bash && "
        "source ~/robot-code-leglift/ros2_ws/install/setup.bash && "
        "exec ros2 launch neural_controller launch.py"
    )
    with open(LAUNCH_LOG, "w") as log:
        subprocess.Popen(
            ["/bin/bash", "-c", cmd],
            cwd=ROBOT_CODE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def main():
    print(f"[dpad_launch_trigger] watching {JS_DEVICE} for D-pad up (axis {DPAD_UP_AXIS} "
          f"== {DPAD_UP_VALUE})")
    while True:
        try:
            fd = os.open(JS_DEVICE, os.O_RDONLY | os.O_NONBLOCK)
        except FileNotFoundError:
            print(f"[dpad_launch_trigger] {JS_DEVICE} not found, retrying in 5s "
                  "(controller not connected?)")
            time.sleep(5)
            continue

        last_trigger = 0.0
        axis_was_up = False  # edge-triggered: the controller re-sends the held axis
        # value periodically over Bluetooth even with no physical change, so triggering
        # on "value == DPAD_UP_VALUE" alone re-fired every ~3s for as long as the button
        # stayed down, causing a kill/relaunch loop instead of a single launch. Only fire
        # on the transition into the pressed value; DEBOUNCE_SECONDS remains as a backstop
        # against a literal double-press.
        try:
            while True:
                try:
                    data = os.read(fd, 8)
                except OSError as e:
                    if e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        time.sleep(0.02)
                        continue
                    raise  # a real error (e.g. ENODEV on disconnect)
                if not data:
                    time.sleep(0.02)
                    continue
                _, value, ev_type, number = struct.unpack("IhBB", data)
                if ev_type & 0x80:
                    continue  # startup sync event, not a real press
                if not ((ev_type & 0x02) and number == DPAD_UP_AXIS):
                    continue
                is_up = value == DPAD_UP_VALUE
                if is_up and not axis_was_up:
                    now = time.time()
                    if now - last_trigger >= DEBOUNCE_SECONDS:
                        last_trigger = now
                        print("[dpad_launch_trigger] D-pad up detected")
                        launch_stack()
                axis_was_up = is_up
        except OSError:
            pass
        finally:
            os.close(fd)
        print(f"[dpad_launch_trigger] {JS_DEVICE} lost, waiting for reconnect")
        time.sleep(2)


if __name__ == "__main__":
    main()
