# Preparing software for this robot

This is the `robot-info` reference branch. Read `ROBOT_INFO.md`,
`robot_info/QUADMORPH.md` for the user's physical configuration and calibration distinctions,
`robot_info/PRE_LAB.md`, and `robot_info/LAB_READY.md` before preparing a policy or
robot-code handoff. The user wants compatibility problems found before traveling
to the lab. Complete every applicable hardware-free check, record its evidence,
and explicitly list unavailable checks; do not call a laptop-only build
target-verified. Keep live deployment and experiment status on the implementation
branch, not in this reference branch.

Match the target's exact ROS/package versions as well as its architecture. Check
real launch argument parsing, generated headers, installed overlays, model hashes,
controller lifecycle, calibration gating, joystick transitions, and stop paths.
Use the preparation procedure in `robot_info/PRE_LAB.md`; preserve target-local
edits and the other robot's checkout. Do not start training, deploy, or move the
robot merely to complete a checklist without authorization for that action.

Before any actual hardware startup/restart or policy use, read
`STARTUP_CALIBRATION.md` from the selected deployment checkout. If inspecting it
from this reference branch, use `git show origin/robot-code:STARTUP_CALIBRATION.md`
and verify the deployment revision. A fresh startup requires the user's explicit
confirmation of the supported encoder-homing pose and marked-ring reference,
then a live-session calibration capture before policy activation. Never infer
that physical confirmation from a request to test. Reuse valid calibration within
the same encoder session; never bypass the gate. Heating remains manual.

Cite hardware claims to measurements, user-confirmed facts, or specific source
revisions. Distinguish implementation facts and legacy comments from physical
verification. Surface conflicts instead of silently trusting an older contract.
