# Latest trained leg and wheel policies

Use these September 13, 2026 heating-backpack + 9 mm policies. They supersede the
September 11 gap-only policies. `policies/latest.json` is the authoritative machine-readable
selection and hash record. Both exports replace the files already loaded by the existing
controllers, so no new controller or joystick wiring is needed in source.

| Mode | Controller | Button in locomotion trial | Current policy file | Steps |
| --- | --- | --- | --- | --- |
| Leg walking | neural_controller_walk_v2 | Triangle (2) | ros2_ws/src/neural_controller/launch/policy_walk_v2.json | 21,299,200 |
| Wheels | neural_controller_wheel | Circle (1) | ros2_ws/src/neural_controller/launch/policy_wheel.json | 201,850,880 |

`launch/config.yaml`, `launch/locomotion_trial.launch.py`, and the full `launch/launch.py`
load those existing controller instances. The leg contract remains 144 observations,
12 position actions, four history frames, kp=5/kd=.25, the established joint homes,
scales and command bounds. Wheel remains 132 observations, four history frames and
mixed position/velocity actions, with wheel kp=0/kd=.35 and the existing direction signs.
The 520 Hz manager / repeat_action=10 timing is unchanged (52 Hz nominal versus 50 Hz
training); check the measured observation rate during later hardware validation.

**Wired in the repository; not installed or tested on the physical robot.**
A later hardware agent should fetch robot-code, run `python3 scripts/check_locomotion_policies.py`
and the existing preparation/build procedure, verify the installed package with that check's
`--package-share` argument, and follow STARTUP_CALIBRATION.md before any physical startup.
Existing controllers still start inactive and retain calibration requirements.
Do not interpret this shipment as evidence that an installed or running robot is already
using the new weights. `policy_latest.json` is a separate legacy controller policy and
is not the latest leg policy despite its historical filename.

Training checkpoints, model snapshots, metrics and rollout videos are in
`trained_policies/backpack_2026-09-13` on `codex/backpack-leg` and `codex/backpack-wheel`.
The manifest records exact checkpoint and export hashes and the training source commits.
The wheel backpack retains its fixed 0.60191707499 kg mass/inertia and floor collisions;
backpack self-collisions are excluded per the user's instruction because MJX lacks
cylinder-box contact. No fixed-body fusion is used.

The logged old-versus-new comparison evaluates the old weights on the backpack model:
leg speed error improved 2%, turning error 9% and tilt 9%, while action changes rose 9%
and ring-side contact rose from .19% to .40%. Wheel RMS speed/turn errors improved 6%/9%
and action-change penalty fell 32%; tilt stayed essentially unchanged. These are training
evaluations, not independent holdout or hardware results. Logs were offline.
Evidence is in hardware_testing/backpack_policies_2026-09-13; previous policy files and
historical validation records remain recoverable from Git history.

Validation for this shipment: both C++ RTNeural inference tests pass on 128 JAX
checkpoint fixtures each (leg max absolute action error 2.40654e-6, wheel 3.12924e-7);
controller source/hash checks pass; six startup-launch tests pass. No ROS controller-manager
integration run or physical robot test was performed for these new weights.
