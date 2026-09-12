"""Read only the Linux joystick device; never publish ROS or command hardware."""
import array
import fcntl
import json
import os
from pathlib import Path
import select
import struct
import time

fd = os.open('/dev/input/js0', os.O_RDONLY | os.O_NONBLOCK)
buttons = array.array('B', [0])
axes = array.array('B', [0])
mapping = array.array('H', [0] * 512)
fcntl.ioctl(fd, 0x80016a12, buttons, True)
fcntl.ioctl(fd, 0x80016a11, axes, True)
fcntl.ioctl(fd, 0x84006a34, mapping, True)
result = {'device': '/dev/input/js0',
          'name': Path('/sys/class/input/js0/device/name').read_text().strip(),
          'axes': axes[0], 'buttons': buttons[0],
          'button_map': list(mapping[:buttons[0]]), 'events': [],
          'motor_stack_started': False, 'ros_published': False}
assert buttons[0] > 10 and mapping[10] == 316, 'Expected PS/BTN_MODE at index 10'
print('Listening for physical PS press and release at index 10.', flush=True)
deadline = time.monotonic() + 90
pressed = released = False
while time.monotonic() < deadline and not released:
    if not select.select([fd], [], [], 0.2)[0]:
        continue
    data = os.read(fd, 8)
    if len(data) != 8:
        continue
    timestamp, value, kind, number = struct.unpack('IhBB', data)
    if kind == 1:
        event = {'device_ms': timestamp, 'index': number, 'value': value}
        result['events'].append(event)
        print(json.dumps(event), flush=True)
        if number == 10:
            if value:
                pressed = True
            elif pressed:
                released = True
os.close(fd)
result['ps_press_release_verified'] = pressed and released
Path('/home/pi/triangle-gamepad-check.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['ps_press_release_verified'] else 1)
