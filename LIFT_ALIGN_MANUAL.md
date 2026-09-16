# R2: operator-verified lift and align

R2 uses visual operator decisions instead of inferred support, wheel-height
matching or clearance gates. Policy weights, gains and alignment targets are unchanged.

| Press | Action |
| --- | --- |
| 1 | Lift/hold front right |
| 2 | Rotate front-right hub after visual clearance confirmation |
| 3 | Lower front right after visual alignment acceptance |
| 4–6 | Lift, rotate, lower front left |
| 7–9 | Lift, rotate, lower back right |
| 10–12 | Lift, rotate, lower back left |

There is no separate pose press or automatic advance. Rotation retains the
calibrated half-turn target (nearest equivalent winding), 0.15 rad/s reference
speed limit and encoder-based settling/stop. A half-turn can take over 20 seconds.
A lower press may accept partial rotation: it stops advancing the reference and
holds the measured hub angle during descent. Visually verify lowering before
advancing or selecting another mode. After step 12, presses do not restart.
Heating remains separate and manual.

`/wheel_lift/advance` is Int32 with the absolute next step (2 through 12;
activation is step 1). Old clients repeatedly publishing 1 must be updated.
Duplicate, delayed or skipped steps cannot advance. The dispatcher uses fresh
acknowledged status; held buttons do not repeat.

The 24-value `lift_status` layout is retained. Index 7 now includes activation
and is the accepted sequence step. Support/clearance estimates (3/4), legacy
qualification time (12), and geometry margins (14–16) are diagnostic only.
Index 5 reports encoder settling, but is not required to lower. Index 8 counts
rejected requests. Leg indices remain FR, FL, BR, BL.

PS/gamepad-loss stop, current-session calibration, fresh/finite sensors,
stationary near-home activation, position/PD limits, clock checks, alignment
timeout and the existing 0.6-rad excessive-tilt stop remain. R2's old 8-degree
dispatcher entry check uses that same excessive-tilt limit instead. Other modes
retain their own entry checks. No calibration or joint-limit gate is disabled.

Regression coverage includes inaccurate height estimates, manual phases, partial
alignment lowering, automatic target settling, duplicate/stale steps, four-leg
order, PS, sensor/clock faults and mock controller-manager execution.
Hardware test result: **NOT WORKING**, as reported by the operator. Testing was
stopped; see HARDWARE_TEST_STATUS.md. Software checks are not hardware validation.
