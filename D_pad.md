# Task: D-pad → Arduino Nano GPIO bridge

## Context
This repo (Pupper V3 monorepo, robot-code branch) controls a quadruped robot on a
Raspberry Pi 5 via ROS 2 (Jazzy). Controller input comes from a PS5 DualSense read by
the standard `joy_linux` driver, which publishes `sensor_msgs/msg/Joy` on `/joy` at 50 Hz.

An Arduino Nano is plugged into a USB-A port on the Pi. Goal: pressing each of the four
D-pad directions on the PS5 controller should drive one of the Nano's digital output pins
(D3/D4/D5/D6) HIGH while held and LOW when released, to trigger external hardware.
Mapping: Up→D3, Right→D4, Down→D5, Left→D6.

This should start automatically on boot alongside the rest of the robot stack.

## Findings from exploring the original repo

- No existing PS5/gamepad-reading code of its own — the repo relies entirely on the
  stock ROS 2 `joy_linux` node (`ros2_ws/src/neural_controller/launch/launch.py`,
  `joy_linux_node`), configured in
  `ros2_ws/src/neural_controller/launch/config.yaml` (`dev: "/dev/input/js0"`).
- Multiple independent nodes subscribe to `/joy` and each owns a slice of the button
  mapping — no central controller manager class. Two good patterns to imitate:
  - `ros2_ws/src/joy_utils/src/estop_controller.cpp` (C++) — button→action mapping with
    edge detection (`prev_*_state_` members), ROS params for button indices declared in
    `config.yaml` under `joy_util_node:`.
  - `ros2_ws/src/bag_recorder/bag_recorder/bag_recorder_node.py` (Python,
    `ament_python` package) — simplest full example of a `/joy`-subscriber node: declares
    button-index params, tracks previous button state for edge detection, subscribes to
    `/joy` in `__init__`, does its thing in `joy_callback`. **This is the best template
    to copy for the new node.**
- **No D-pad handling exists anywhere in the repo** (zero hits for `dpad`, `hat`,
  `ABS_HAT`). The D-pad's exact representation in the `/joy` message on this hardware
  (two hat-style axes with values -1/0/1, vs. four discrete `buttons[]` indices) is
  **unverified** — must be checked on the physical robot.
- **No existing serial/Arduino code** — zero hits for `pyserial`, `import serial`,
  `ttyACM`, `ttyUSB` anywhere in the repo, and `pyserial` is not a dependency anywhere.
  No `.ino` files or `arduino/` directory exist either — this is greenfield.
- Startup is via systemd: `robot/utils/robot.service` runs `robot/utils/robot.sh`, which
  does a single `ros2 launch neural_controller launch.py` that starts ~20 nodes together
  (full list in `launch.py`'s `nodes = [...]`, around line 319). Adding a new node to that
  launch file's node list is enough to get it auto-started on boot — no new systemd unit
  needed. (For contrast, `robot/utils/battery_monitor.service` is the pattern for a fully
  standalone non-ROS systemd service, but that's not needed here.)
- Dependencies: ROS Python nodes declare deps in `package.xml` (`<depend>` tags mapped to
  apt packages via `rosdep`), not pip/requirements.txt. `install_dev_dependencies.sh` runs
  `rosdep install --from-paths src -y --ignore-src` which will auto-install whatever apt
  package a new `<depend>` resolves to — so adding `<depend>python3-serial</depend>` to a
  new package's `package.xml` is enough to get `pyserial` installed on `rosdep install`.
- Serial device permissions: `infra/pupper_image_builder/user-data` puts the `pi` user in
  the `dialout` group on fresh Pi images (needed for `/dev/ttyACM0` access). Since this
  robot is already deployed (not a fresh image), verify with `groups pi` on the actual
  Pi and `sudo usermod -aG dialout pi` (+ reboot/relogin) if missing.

## Decisions already made with the user (do not re-ask)
- Pin behavior: **hold-while-pressed** (pin HIGH while D-pad direction held, LOW on release).
- Integration: **new ROS 2 node added to the existing launch tree**, not a standalone
  systemd service.
- Mapping: Up→D3, Right→D4, Down→D5, Left→D6.

## Step 0 (must happen first, on the physical robot)
Run `ros2 topic echo /joy` while the full stack (or at least `joy_linux_node`) is running,
press each D-pad direction one at a time, and record whether it appears as two hat-style
axes (commonly `axes[6]`=left/right, `axes[7]`=up/down, values -1/0/1) or as four discrete
`buttons[]` indices, and the exact indices/signs. Use these real values in `config.yaml`
below — do not guess/hardcode unverified indices as final.

## Plan

### 1. New ROS 2 package `ros2_ws/src/dpad_serial_bridge/`
Model directly on `ros2_ws/src/bag_recorder/` (`ament_python`, single node):
- `package.xml`: `<depend>rclpy</depend>`, `<depend>sensor_msgs</depend>`,
  `<depend>python3-serial</depend>`.
- `setup.py` / `setup.cfg` / `resource/dpad_serial_bridge`: copy `bag_recorder`'s
  boilerplate; console_scripts entry point
  `dpad_serial_bridge_node = dpad_serial_bridge.dpad_serial_bridge_node:main`.
- `dpad_serial_bridge/dpad_serial_bridge_node.py`:
  - Declare params (same idiom as `estop_index`/`record_start_button`):
    `serial_port` (default `/dev/ttyACM0`), `baud_rate` (default `115200`), `dpad_mode`
    (`"axes"` or `"buttons"`, set per Step 0), and the relevant index/sign params for
    each direction.
  - Open the serial port in `__init__` via `pyserial`; if it fails (Arduino unplugged),
    log a warning and retry on a timer — do not crash the node, since the rest of the
    robot must keep working with the Arduino disconnected.
  - `joy_callback`: read the four direction states from `msg.axes`/`msg.buttons` per
    `dpad_mode`, compare against last-sent state per direction (edge detection like
    `bag_recorder_node.py`), and for each direction that changed, write one line to
    serial: `"<U|R|D|L><0|1>\n"` (e.g. `"U1\n"` = up pressed, `"D0\n"` = down released).
    Only send on change, not continuously.
  - On (re)establishing the serial connection, wait ~2s (Arduino Nano resets on DTR
    toggle when the port opens) then re-send current state of all four directions so the
    Nano's pins stay in sync after a reconnect.
- `README.md` matching the style of `ros2_ws/src/bag_recorder/README.md`.

### 2. Launch/config wiring
- `ros2_ws/src/neural_controller/launch/launch.py`: add
  `dpad_serial_bridge_node = Node(package="dpad_serial_bridge", executable="dpad_serial_bridge_node", parameters=[node_parameters], output="both", name="dpad_serial_bridge")`
  near `joy_util_node`/`bag_recorder_node`, and append it to the `nodes = [...]` list.
- `ros2_ws/src/neural_controller/launch/config.yaml`: add a `dpad_serial_bridge:`
  `ros__parameters:` block (next to `joy_util_node:`) with `serial_port`, `baud_rate`,
  `dpad_mode`, and the index/sign params filled in from Step 0's real findings. Comment
  noting Up=D3/Right=D4/Down=D5/Left=D6.

### 3. Arduino sketch (new, greenfield)
- New file, e.g. `robot/arduino/dpad_pin_driver/dpad_pin_driver.ino`.
- `setup()`: `Serial.begin(115200)` (must match `baud_rate` param),
  `pinMode(3..6, OUTPUT)`, all pins initialized LOW.
- `loop()`: read newline-terminated lines from `Serial`; parse first char (`U`/`R`/`D`/`L`)
  → pin (3/4/5/6), second char (`0`/`1`) → `digitalWrite(pin, LOW/HIGH)`. Ignore malformed
  lines.

### 4. Verification
1. Do Step 0 on hardware first, lock in real D-pad indices in `config.yaml`.
2. Bench-test the Arduino sketch standalone via Arduino IDE Serial Monitor (send `U1`,
   `U0`, etc. by hand) before wiring to ROS.
3. `colcon build` the workspace (`ros2_ws/build.sh`), source `install/setup.bash`.
4. Run the full stack, press each D-pad direction, confirm the correct pin goes
   HIGH while held / LOW on release (multimeter/LED/logic analyzer, or the downstream
   device itself). Confirm diagonal presses (if the hardware reports them) drive both
   relevant pins.
5. Unplug/replug the Arduino while the stack is running — confirm the node logs a clear
   warning/reconnect instead of crashing, and the rest of the robot stack is unaffected.
6. Reboot the Pi, confirm `robot.service` still starts cleanly with the new node included,
   both with the Arduino plugged in and unplugged.