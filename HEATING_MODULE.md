# Manual heating module

Imported unchanged from `robot-code` at
`6b07745ab4c7ab59e6da7766a17896f118d5198d`. The introducing commit, `42c322f`,
identifies this Arduino sketch and ROS node as the heating-module implementation.
`HEATING_IMPORT.json` records the exact source hashes and extracted settings.

## Included components

- `ros2_ws/src/dpad_serial_bridge/`: complete ROS Python package, dependency
  declarations, package registration and original README.
- `robot/arduino/dpad_pin_driver/dpad_pin_driver.ino`: Nano firmware.
- `config/heating.yaml`: original `dpad_serial_bridge` parameter section,
  extracted from the source robot configuration.

The ROS node converts D-pad press/release messages into serial GPIO commands.
It drives Up → D3, Right → D4, Down → D5, Left → D6. Pins are HIGH while
pressed and LOW on release; pins initialize LOW at firmware startup.
The code does not specify which physical leg each output heats.

The source settings use `/dev/ttyUSB0`, 115200 baud, horizontal/vertical axes 6/7,
horizontal sign -1 and vertical sign +1. The imported package README records
previous hardware checks; these settings have not been revalidated in this branch.

## Dependencies and explicit use

The package declares `rclpy`, `sensor_msgs` and `python3-serial` (pyserial).
Build it with the rest of this six-package ROS workspace, or build only it:

```sh
colcon build --base-paths ros2_ws/src --packages-select dpad_serial_bridge
source install/setup.bash
```

The firmware also requires the Arduino Nano board toolchain for compilation and
upload. Firmware upload and robot deployment have not been performed.

The heating bridge is separate from `gravity_runtime.launch.py`. Once the
separate heating controller is publishing Joy messages, the following explicit
command uses that controller's topic (replace `/heater/joy` with its actual topic):

```sh
ros2 run dpad_serial_bridge dpad_serial_bridge_node --ros-args \
  -r __node:=dpad_serial_bridge \
  --params-file config/heating.yaml \
  -r /joy:=/heater/joy
```

Run from the repository root in the selected overlay. The node name override
matches the original YAML parameter section. The example does not create or
configure a second gamepad publisher. The new leg-to-wheel sequencer never sends
heating commands; the operator controls the separate pad.

## Existing behavior and limitations

The original implementation sends changes only and resends the current four
states after serial reconnection. It has no temperature feedback, heat timer,
communication watchdog or `/emergency_stop` subscription. An output can remain
HIGH if a release message is lost or communication stops. Closing the ROS node
does not explicitly send all-off. The motion controller's PS stop does not stop
these heater outputs. These behaviors are preserved by this source-only import.

Validation here is limited to source hashes, Python/package build and isolated
mock serial checks. No real serial port, heating hardware or robot was accessed.
