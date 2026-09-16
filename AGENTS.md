# Selected robot runtime

Read README.md and STARTUP_CALIBRATION.md before hardware startup, restart or testing.
Before each fresh activation, ask the operator to support the robot in the gravity
pose and position the wheel rings; wait for explicit physical confirmation.
Do not self-confirm, disable calibration gates, substitute old session offsets,
or launch a duplicate hardware stack. Ordinary source edits and local mock tests
do not require physical setup. Heating is manual. Calibration is not authorization
for deployment or additional motion.

Preserve selected exports in policies/latest.json and their hashes. Do not use
Stanford's policy weights or torque-threshold homing. The Stanford infrastructure
baseline and adaptations are recorded in STANFORD_IMPORT.json. Distinguish software
tests from hardware validation. Leg-to-wheel uses operator-verified Square presses,
manual heating and time-only lowering easing; see LEG_TO_WHEEL_MANUAL.md. Do not
reintroduce contact/clearance gating into that manual sequence.
