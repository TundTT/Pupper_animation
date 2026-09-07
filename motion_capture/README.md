# Motion capture — hand-demonstrated keyframes

Captures labeled joint-angle poses while a human physically moves the robot's joints
by hand, for use as reference keyframes when training a policy. Branch-agnostic: this
lives on `flip-data`, built from `robot-code`, and doesn't depend on any particular
policy being deployed.

## Every launch: confirm home pose first

The hardware homing step no longer cross-checks against a fixed reference (see the
`control_board_hardware_interface.cpp` change on this branch) — it trusts whatever
pose the robot is physically in in at launch. So **every time the stack isn't already
running**, before launching:

1. Ask the operator to physically set the robot to its home pose (gravity-drop pose,
   by hand) — nothing is powered/commanding the joints at this point, so this is safe.
2. Get explicit confirmation it's set correctly (adjust as needed).
3. Only then launch.

## Capture pipeline

**Safety: robot on a stand, legs clear, confirmed each session — same as always.**

1. SSH in and launch (real hardware, no `sim:=True`):
   ```sh
   ssh pi@10.140.55.163
   source /opt/ros/jazzy/setup.bash
   cd ~/robot-code-leglift/ros2_ws        # or wherever this branch is checked out
   source install/local_setup.bash
   ros2 launch neural_controller launch.py
   ```
   Watch for `Finished homing!` and no errors. **No controller auto-activates** unless
   a joystick is connected and used to activate one — only `joint_state_broadcaster`
   and `imu_sensor_broadcaster` come up active by default, so the joints stay
   backdrivable with nothing extra needed (no e-stop, no joystick required).

2. Move the joints by hand through the motion you want to capture.

3. On cue, capture a keyframe (run from the directory you want `keyframes.json` to
   land in — e.g. `~/robot-code-leglift/motion_capture/`):
   ```sh
   python3 capture_keyframe.py --label <short_name_for_this_pose>
   ```
   Repeat with a new `--label` for each pose you want marked. Every call appends to
   the same `keyframes.json` rather than overwriting it.

4. Pull `keyframes.json` back and commit it to this branch:
   ```sh
   scp pi@10.140.55.163:~/robot-code-leglift/motion_capture/keyframes.json \
       motion_capture/data/<capture_session_name>.json
   ```
   Add a row to `data/MANIFEST.md` (session name, date, what poses/motion it covers).

## Keyframe JSON format

```jsonc
[
  {
    "label": "flip_start",
    "captured_at": 1788825600.12,
    "joint_names": ["leg_front_r_1", "leg_front_r_2", ...],  // as reported by the robot
    "position": [...],
    "velocity": [...]
  },
  ...
]
```
