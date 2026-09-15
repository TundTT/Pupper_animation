# Selected policies in this branch

`policies/latest.json` pins the user-selected heating-backpack + 9 mm walking and
wheel exports. Their bytes and hashes are unchanged from `robot-code` at
`6b07745ab4c7ab59e6da7766a17896f118d5198d`. The wheel policy uses 0.65 rad home
abduction; its controller config retains that exact stance.

The other selected components are triangle roll-to-stand (X), the newest shipped
wheel-blind lift policy with PD hub alignment (R2), and the leg-to-wheel C++ policy
and sequencer. Leg-to-wheel's ROS adapter is explicitly deferred to the next task.
The wheel-to-walk-ready export is an existing deterministic stance transition.

Historical installation/evidence fields in imported manifests describe their
source release. They do not mean this Stanford/gravity runtime is deployed or
physically validated. See README.md and VALIDATION.md for this branch's status,
and STARTUP_CALIBRATION.md before hardware use.
