# Latest trained leg and wheel policies

For the R2 wheel-blind lift / slow position-align sequence now selected in the
combined launch, see [WHEEL_LIFT_POLICY.md](WHEEL_LIFT_POLICY.md). Local ROS and
full C++ native-physics validation passed; Pi installation is pending.

For the trained leg-to-wheel conversion policy and gentler lowering source
release, see [LEG_TO_WHEEL_POLICY.md](LEG_TO_WHEEL_POLICY.md). It includes the
RTNeural export and tested C++ runtime; hardware activation integration is pending.

For X roll-to-stand plus Triangle walking and Circle wheels in one session, use
`COMBINED_MOTION_LAB.md` and `combined_motion.launch.py`. That launch adds the
session-bound walking frame required after hub rotation; the legacy ordinary
walking controller does not infer full-turn offsets.

Use these September 13, 2026 heating-backpack + 9 mm policies. They supersede the
September 11 gap-only policies. `policies/latest.json` is the authoritative machine-readable
selection and hash record. Both exports replace the files already loaded by the existing
controllers, so no new controller or joystick wiring is needed in source.

| Mode | Controller | Button in locomotion trial | Current policy file | Steps |
| --- | --- | --- | --- | --- |
| Leg walking | neural_controller_walk_v2 | Triangle (2) | ros2_ws/src/neural_controller/launch/policy_walk_v2.json | 21,299,200 |
| Wheels | neural_controller_wheel | Circle (1) | ros2_ws/src/neural_controller/launch/policy_wheel.json | 100,925,440 |

`launch/config.yaml`, `launch/locomotion_trial.launch.py`, and the full `launch/launch.py`
load those existing controller instances. The leg contract remains 144 observations,
12 position actions, four history frames, kp=5/kd=.25, the established joint homes,
scales and command bounds. Wheel remains 132 observations, four history frames and
mixed position/velocity actions, with wheel kp=0/kd=.35 and the existing direction signs.
The 520 Hz manager / repeat_action=10 timing is unchanged (52 Hz nominal versus 50 Hz
training); check the measured observation rate during later hardware validation.

**Installed in the Pi's `install-combined` overlay; supervised walking and
roll-to-walking tests reported successful on September 13.** The new wheel
policy's installation was checked; this does not establish a physical wheel
driving result. See `COMBINED_MOTION_LAB.md` for the tested entry point.
A later hardware agent should fetch robot-code, run `python3 scripts/check_locomotion_policies.py`
and the existing preparation/build procedure, verify the installed package with that check's
`--package-share` argument, and follow STARTUP_CALIBRATION.md before any physical startup.
Existing controllers still start inactive and retain calibration requirements.
Do not interpret this shipment as evidence that an installed or running robot is already
using the new weights. `policy_latest.json` is a separate legacy controller policy and
is not the latest leg policy despite its historical filename.

**Wheel stance change, 2026-09-14 (this shipment).** The wheeled home pose was
narrowed from +-1.0 rad abduction (57 deg) to +-0.65 rad (37 deg), because the robot
rolled visibly splay-legged. The previous policy was not drifting: measured over
400-step rollouts it tracked its abduction targets to within 0.05 rad, so the splay
was the home-pose constant, not learned behaviour and not backpack mass. A static
settle sweep on the backpack + 9 mm model stays upright on all four wheels down to
0.55 rad and collapses at 0.50, so 0.65 keeps ~0.1 rad of margin.

`config.yaml`'s `neural_controller_wheel` `default_joint_pos` was updated in the same
commit and MUST stay equal to the exported `default_joint_pos` — the controller applies
actions as offsets from it. `neural_controller_wheel_align_hybrid` and
`neural_controller_keyframe_align` deliberately keep +-1.0 rad, so switching between
wheel mode and either align mode now crosses a stance change; each controller ramps to
its own home over its `init_duration`, but watch that transition on the first hardware run.
Leg/walking policy and config are untouched by this shipment.

Training checkpoints, model snapshots, metrics and rollout videos are in
`trained_policies/backpack_2026-09-13` on `codex/backpack-leg` and `codex/backpack-wheel`.
The 2026-09-14 wheel run is `wheel_2026-09-14_04-59-56` on branch `wheel-stance`
(W&B `QuadMorph/pupper-wheel/runs/y534r02t`), warm-started from the previous wheel
checkpoint; its export was checked against the trained policy over 128 random-observation
fixtures at max absolute action error 0.000e+00.
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
