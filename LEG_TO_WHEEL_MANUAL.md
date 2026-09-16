# Manual leg-to-wheel control

The operator decides when lifting, heating and lowering are satisfactory. Heating
stays on the separate controller. This adapter has no heater output, foot-contact
input, floor-height estimate or automatic transition timer.

## Button sequence

With a current gravity calibration and the robot in a stationary, near-standing,
tips-down pose, **Square** selects leg-to-wheel and requests the first lift.
All motion controllers are inactive at startup. As with the other mode buttons,
initial selection waits for centered sticks before activating the controller.

| Square press | Requested action | Operator decision before the next press |
| --- | --- | --- |
| 1 | Lift/hold front left | Verify lift, heat separately, decide when to lower |
| 2 | Lower front left / stand | Verify support before lifting the next leg |
| 3 | Lift/hold front right | Verify lift, heat separately, decide when to lower |
| 4 | Lower front right / stand | Verify support before lifting the next leg |
| 5 | Lift/hold back right | Verify lift, heat separately, decide when to lower |
| 6 | Lower back right / stand | Verify support before lifting the next leg |
| 7 | Lift/hold back left | Verify lift, heat separately, decide when to lower |
| 8 | Lower back left / stand | Verify all four legs before selecting wheels |

The policy continues running during each wait. No press means no sequence change.
After press 8 it keeps the stand command; additional Square presses do not restart
or automatically enter driving. A deliberate deactivation/reactivation starts over.
Pressing a different mode after a lower request is the operator's confirmation
that support is satisfactory; switching while lift/hold is requested is refused.
Existing entry checks for the destination mode still apply. PS remains emergency stop.
R2 retains the separate wheel-blind lift-and-PD-alignment behavior.

## Implementation contract

- Plugin: `neural_controller/LegToWheelController`; instance:
  `neural_controller_leg_to_wheel`; config: `launch/leg_to_wheel_config.yaml`.
- Uses the selected, unchanged `policy_leg_to_wheel.json`: 140 inputs (four
  35-value frames, newest first), 12 position actions, 50 Hz inference. Actual
  elapsed controller periods schedule inference within the 520 Hz manager loop;
  commands are held between actor ticks. Previous-action observations use raw
  network output, as trained.
- `ManualSequencer` supplies commands `FL, stand, FR, stand, BR, stand, BL, stand`.
  It does not infer touchdown or mark a physical conversion successful.
- `step_manual` uses the export's 8 rad/s initial hip descent easing over 0.2 s,
  without its clearance-dependent fade. This is a deliberate adapter change:
  no fabricated clearance is passed to the original sensor-based API. The rate
  constraint eases away over that interval; it is not a permanent speed cap.
- Shared calibration is mandatory, including direct ROS lifecycle activation.
  Proximal joints retain the hardware's gravity frame. Hub coordinates map the
  calibrated ring marks to the policy's tips-down home, choosing the nearest full
  encoder turn once at activation. The same offset maps observations and targets.
  Entry requires all joints within 0.30 rad of this stand, all joint speeds at
  most 0.15 rad/s, and torso tilt below 8 degrees. No wheel-to-manual automatic
  stance transition is added.
- Runtime rejects nonfinite feedback, invalid/stale IMU data (over 0.1 s), tilt
  over 0.6 rad, clock gaps over 0.01 s, invalid output and out-of-range continuous
  hub targets. Faults and emergency stop latch zero command fields and gains.
  The driver supplies no individual encoder sample-age counter; finite encoder
  values alone do not prove fresh per-motor measurements.

### ROS events and status

`/leg_to_wheel/advance` uses volatile `std_msgs/Int32` messages containing the
**absolute next step number**, not an unnumbered pulse. Activation is step 1;
subsequent requests are 2 through 8. Only the exact next number is accepted.
Duplicates, stale numbers, jumps and requests after 8 cannot advance the sequence.
Requests are consumed at actor ticks. The dispatcher derives the next number from
fresh controller status, so a fast repeated press before acknowledgment does not
queue an extra action. Release and press again after acknowledgment to advance.

`/neural_controller_leg_to_wheel/manual_status` is a `Float64MultiArray`:
`[step, actor_command, leg_command_index, lower_requested, all_requests_sent,
fault, accepted_requests, rejected_requests, imu_age_seconds, manager_dt]`.
Leg command indices are FL=1, FR=2, BR=3, BL=4. `all_requests_sent` means eight
requests were issued, not that the robot reached the floor. Fault codes are
1 clock, 2 sensors, 3 tilt, 4 execution, 5 emergency stop, 6 activation.
`motor_commands` publishes position, velocity, effort, kp, kd for each joint in
FR/FL/BR/BL order. `observation` and `policy_output` expose the actual actor data.

## Validation scope

Local tests cover recorded network inputs, both old and manual easing paths,
eight-step sequencing, duplicate/held buttons, calibrated multi-turn targets,
real plugin lifecycle, stale/nonfinite feedback, timing faults, tilt and stop.
The ROS manager test uses fixed synthetic encoders and an upright synthetic IMU;
it verifies commands and messages, not lifting, balance or touchdown.

Physical performance with manual time-only easing remains untested. Follow
[STARTUP_CALIBRATION.md](STARTUP_CALIBRATION.md) before any hardware use.
