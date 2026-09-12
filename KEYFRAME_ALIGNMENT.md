# Deterministic lift and alignment development

Work on `codex/align-motion-v2`; the branch name is retained. The new implementation
is in `motion/keyframe_align/`. Core revision **a110b09** passes the 22-scenario
simulation matrix. It is not deployed on hardware. No network, checkpoint, PPO, JAX, or optimizer
is used by its controller or simulation runner. Existing training/controller code
is preserved for historical replay. Do not launch this branch's older hardware
stack or merge it wholesale into robot-code.

See [the current validation record](hardware_testing/keyframe_align/VALIDATION_a110b09.md)
and [the original failed prototype](hardware_testing/keyframe_align/VALIDATION_8499fa5.md).
The robot adapter is prepared separately on `codex/keyframe-robot-integration`,
based on robot-code e3e1d97. Read that branch's `KEYFRAME_LAB.md` before deployment.

## What changed

The portable C++ controller drives ENTRY → SHIFT → LIFT → ROTATE → LOWER →
RECENTER → HOLD. All eight proximal joints follow coordinated keyframes with
quintic interpolation and bounded command velocity/acceleration. Timing and the
four complete poses are editable in [config.json](motion/keyframe_align/config.json).
Requested durations are minimum reference durations; filters and measured-state
conditions may extend them. Settings use 1.5 seconds for shift, 1.5 for rise, three
for landing and 1.5 for recentering. Filtered shift plus lift takes about 3.7
seconds; the complete nominal sequence takes 20.4–20.7 seconds per wheel including
rotation and final verification. Wheel rotation speed remains 0.5 rad/s.

The four wheel hubs receive velocity commands from angle feedback. The active
wheel tracks `wrap(startup_home + pi)` through a ramped reference and bounded PI-D
control. The integral contribution is enabled only near the final angle, uses
conditional anti-windup, and resets on gate loss, interruption, timeout, reset and
lowering. Default maximum velocity is 0.5 rad/s, acceleration 1.2 rad/s², integral
contribution 0.4 rad/s. Setting `wheel_ki` to zero disables integration. Completion
requires measured angle error <0.025 rad and speed <0.08 rad/s continuously for
0.5 seconds. Successful lowering also requires final error <0.035 rad and speed
<0.08 rad/s for 0.5 seconds. These are software acceptance tolerances, not a
user-confirmed morphing tolerance. Small holding corrections follow the calibrated
target through lowering and subsequent wheel moves. A >0.10 rad disturbance
invalidates success and does not trigger a grounded realignment. A final-angle
failure sets a failure bit and uses the retry latch; it cannot silently retry.

Rotation still requires >10 mm estimated floor clearance, >10 mm wheel spacing,
>5 mm body spacing, tilt <0.12 rad, angular speed <0.3 rad/s and a qualifying
interval of 0.2 seconds. Diagnostics expose a bitmask: trajectory=1, floor=2,
wheel=4, body=8, tilt=16, angular velocity=32. Gate loss immediately suppresses
active-wheel rotation; it is not permission to advance a blind animation.

Changing the request first lowers/recenters the old leg. A 48-second attempt
timeout does the same and blocks automatic retry of that command. Stand or a
different request clears the timeout latch. Stop/invalid inputs remove command
authority and require reset; a future hardware adapter must map that signal to
the reviewed robot-code stop behavior. Heating is manual and absent here.

## What the lab evidence established

On robot-code `e3e1d97`, front-left never passed the estimated floor check; its
median estimate was −4.1 mm. Front-right passed intermittently (median 9.5 mm)
and ended about 171° short. Back-right/back-left passed clearance but plateaued
3.4°/7.4° short while receiving nonzero wheel commands. Friction or deadband is
a hypothesis, not a measured motor parameter. Synthetic friction tests exercise
the new integral correction; they do not prove the physical cause or fix.

Original recordings are on the lab laptop at
`C:/Users/tundt/Desktop/quadmorph-lab-recordings/rotation-diagnostic-4160e3bd` and
`rotation-comparison-4160e3bd`, including analysis JSON and raw MCAP. The Pi copies
are under `/home/pi/align-v5-setup-backup-c1ef4d2/`. Keep live calibration records
out of git and never reuse those historical homes after a fresh encoder session.

The default front poses remain the previous physics-tested coordinated poses.
The rear hip apex increased from 1.05 to 1.2 radians in magnitude. The new floor
estimate uses the lowest support-wheel bottom, with separate conservative radius
bounds, rather than treating the highest (possibly unloaded) support wheel as the
floor. It assumes rigid, level ground and nominal modeled wheels: this is not a
terrain/contact sensor. The 10 mm floor threshold is unchanged. An independent
MuJoCo geometry test checks the bound over 150 poses/orientations for all wheels.
Static fitting attempts that failed balance/clearance were rejected.
The optional `fit_reference.py` is an offline candidate generator, not an
acceptance test; a low objective or solver success must never replace full motion
and perturbation checks. Do not copy its unverified output onto the robot.

## Run without hardware or training

Use a separate development checkout; keep historical training checkouts frozen.
On Linux/WSL with Python 3.11/3.12, CMake and a C++17 compiler:

```bash
python3 -m venv .venv-keyframes
.venv-keyframes/bin/pip install -r motion/keyframe_align/requirements.txt
cmake -S motion/keyframe_align -B /tmp/quadmorph-keyframe-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/quadmorph-keyframe-build -j2
export KEYFRAME_ALIGN_LIBRARY=/tmp/quadmorph-keyframe-build/libkeyframe_align.so
.venv-keyframes/bin/python -m pytest motion/keyframe_align/test_controller.py -q

# Use a new output directory for each run. Native CPU simulation, no training.
MUJOCO_GL=egl .venv-keyframes/bin/python -m motion.keyframe_align.simulate \
  --output /tmp/keyframes-nominal --video
.venv-keyframes/bin/python -m motion.keyframe_align.simulate \
  --output /tmp/keyframes-friction --friction .06 --seed 1
.venv-keyframes/bin/python -m motion.keyframe_align.simulate \
  --output /tmp/keyframes-interrupted --interrupt
.venv-keyframes/bin/python -m motion.keyframe_align.simulate \
  --output /tmp/keyframes-imu-bias --imu-roll .04 --imu-pitch .04
```

The CLI defaults to online W&B logging to the user's existing QuadMorph project.
Use existing credentials/`wandb login`; never paste keys into commands or chat.
Video is a real deterministic-controller rollout logged as `wandb.Video`, labeled
simulation/no neural checkpoint. `--wandb offline` retains an offline SDK run;
`--wandb disabled` explicitly keeps intermediate development checks local. Logging
errors do not erase local `audit.json`, `config.json`, or `trace.csv`. A URL alone
is not proof that upload finished; verify the remote media before reporting it.

Do not modify the source/library/config while an audit is running. Reports record
the source commit, dirty state and hashes. Each failed audit must remain visible.
The trace columns are time, requested command, then the 25-value C ABI v2 output:
8 proximal positions, 4 wheel velocities, phase, active command, completed mask,
blocked mask, timeout/failure retry command, 3 margins, target error, up-time,
integral, authority, failed-wheel mask.
The simulator's actual floor check uses ground truth only for auditing; the
controller never receives simulation-only contacts or body position.

## Required before a robot-code port

1. Preserve the completed simulation acceptance and video. Any core/configuration
   change requires the acceptance matrix again. Physical ground/geometry and
   ring-reference assumptions still require a supervised lab check.
2. The isolated integration branch keeps the core/configuration as a standalone
   behavior. Its ROS lifecycle/joystick adapter uses robot-code's current
   calibration package, joint mapping, sensor-freshness checks, inactive startup,
   stop behavior and logging. `Controller.reset(q, home)` accepts externally
   validated homes; it does not validate a real encoder-session file itself.
3. Complete the [robot-info pre-lab checks](https://github.com/TundTT/Pupper_animation/blob/e9b0417/robot_info/PRE_LAB.md)
   against exact Pi package versions. An x86 standalone build is not an ARM64 or
   ROS integration pass. Never build over a running hardware controller.
4. Deploy only when authorized. Read robot-code's `STARTUP_CALIBRATION.md`, obtain
   the fresh physical pose confirmation, home, capture and verify calibration.
   Reuse calibration only within the same live session. Then run a supervised,
   recorded trial. No new robot deployment was performed during this development.

## Source distinctions

- User-approved design: deterministic whole-body lift/lower, encoder-based wheel
  rotation, manual heating, and development on the existing alignment branch.
- Hardware mapping and mode/gain limits: [robot-info HARDWARE.md](https://github.com/TundTT/Pupper_animation/blob/b311700/robot_info/HARDWARE.md),
  cross-checked against [robot-code description](https://github.com/TundTT/Pupper_animation/blob/e3e1d97/ros2_ws/src/pupper_v3_description/description/components.xacro).
- Geometry and prior keyframes: `training/wheel_align/geometry.json`, `model.xml`
  and `configs.py` at training source `e663e03`; this is modeled geometry, not a
  new physical measurement. The previous third-joint leg stops are not applicable
  to the continuous wheel profile.
- Calibration target convention: the user's reshaping notes, enforced through
  robot-code's live-session calibration workflow for any future hardware port.
