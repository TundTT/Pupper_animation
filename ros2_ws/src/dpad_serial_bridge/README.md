# D-pad Serial Bridge

A ROS2 node that reads the PS5 controller's D-pad from `/joy` and drives one digital
output pin on an Arduino Nano HIGH while each direction is held, LOW on release.

## Pin mapping

| D-pad direction | Nano pin |
|---|---|
| Up | D3 |
| Right | D4 |
| Down | D5 |
| Left | D6 |

The Nano runs the sketch in `robot/arduino/dpad_pin_driver/dpad_pin_driver.ino`.

## D-pad representation on `/joy`

Hardware-verified 2026-09-03 via `ros2 topic echo /joy` (joy_linux_node, PS5
DualSense): the D-pad reports as two hat-style axes, not discrete buttons.

- Up -> `axes[7] = +1.0`, Down -> `axes[7] = -1.0`
- Left -> `axes[6] = +1.0`, Right -> `axes[6] = -1.0` (inverted from the typical
  convention, hence `dpad_axis_horizontal_sign: -1` in config.yaml)

`dpad_mode` defaults to `"axes"` with `dpad_axis_horizontal`/`dpad_axis_vertical` = 6/7
and the signs above set in `neural_controller`'s `config.yaml`. A `"buttons"` mode
(four discrete `buttons[]` indices via `dpad_button_up`/`_right`/`_down`/`_left`) is
also supported in case a different controller/driver combination needs it, but is not
what this hardware uses.

## Serial protocol

One line per state change, `"\n"`-terminated: `<U|R|D|L><0|1>`, e.g. `"U1\n"` = up
pressed, `"D0\n"` = down released. Sent only on change, not continuously. On (re)opening
the serial port the node waits `reset_settle_sec` (default 2s, to cover the Nano's
DTR-reset-triggered reboot) then re-sends the current state of all four directions so
the Nano's pins stay in sync after a reconnect.

If the Arduino is unplugged, the node logs a warning and keeps retrying the connection
on `reconnect_period_sec` (default 2s) rather than crashing -- the rest of the robot
stack must keep working regardless of Arduino connectivity.

## Parameters

- `serial_port` (string, default `/dev/ttyUSB0` -- this Nano uses a CH340 USB-serial
  chip, confirmed via `lsusb`/`dmesg` 2026-09-04; a Nano with native USB, e.g. Nano
  Every/33 IoT, would instead show up under `/dev/ttyACM0`)
- `baud_rate` (int, default 115200) -- must match `Serial.begin(...)` in the `.ino` sketch
- `dpad_mode` (string, `"axes"` or `"buttons"`, default `"axes"`)
- `dpad_axis_horizontal` / `dpad_axis_vertical` (int, default 6/7)
- `dpad_axis_horizontal_sign` / `dpad_axis_vertical_sign` (int, default 1/1)
- `dpad_button_up` / `dpad_button_right` / `dpad_button_down` / `dpad_button_left` (int, default -1 = unset)
- `reconnect_period_sec` (double, default 2.0)
- `reset_settle_sec` (double, default 2.0)

## Requirements

- ROS2 (Jazzy or later)
- `sensor_msgs`, `python3-serial` (`pyserial`)
- The Pi user must be in the `dialout` group for `/dev/ttyUSB0` access
  (`sudo usermod -aG dialout $USER`, then reboot/relogin)
