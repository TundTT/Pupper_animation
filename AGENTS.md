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
tests from hardware validation. The leg-to-wheel ROS adapter is explicitly deferred
to the next task until its clearance/contact feedback contract is resolved.
