# QuadMorph: physical meaning and reusable control knowledge

This reference records user-confirmed hardware facts and implementation lessons
from the September 2026 lab sessions. Deployment state and outstanding experiments
belong in [robot-code's handoff](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/LAB_HANDOFF.md).
Do not interpret historical calibration files or a successful video as permission
to activate hardware.

## What the robot actually does

**User-confirmed physical facts:** the same appendages serve as wheels and legs.
The shape-memory polymer remembers the wheel shape. Heating while applying load,
principally body weight, forms the leg; unloaded heating restores the wheel.
For the roll-to-stand task, formation and cooling have already finished: all four
appendages are rigid, with tips pointing upward. Heating is manually controlled.
The motion should roll the four appendages together onto their downward-facing
tips and finish near the leg walking policy's standing pose.

Motor 3 is the distal hub, and selecting a different spoke changes its physical
reference. Motors 1 and 2 position the upper limb. The current hardware retains
continuous hub rotation in both wheel and cold triangle configurations. The old
limited CustomLegFoot profile is not evidence of mechanical stops on these hubs.
The [selected robot-code description](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/ros2_ws/src/pupper_v3_description/description/components.xacro)
uses wide third-joint position bounds and omits the old hard-limit overrides.
This does not remove collision or cable-routing constraints.

**Spacing:** the user confirmed an additional 9 mm outward displacement of the
entire wheel/shin assembly along the hub axis, and explicitly confirmed the XML
was correct after briefly questioning this. Preserve that geometry; do not add
9 mm again. See the [roll contract](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/ros2_ws/src/neural_controller/launch/triangle_roll_plan.json)
and its geometry checks. Keep the actual backpack/payload in the selected model.

## Measured cold shapes, not ideal flat triangles

The user measured blue hub center to upper tip and lower outside surface in the
cooled tip-up orientation. Measurement order was not mapped to named robot legs:

| Measurement order | Center to top (mm) | Center to bottom (mm) |
| --- | ---: | ---: |
| 1 | 55 | 37 |
| 2 | 57–58 | 34 |
| 3 | 60 | 32 |
| 4 | 56 | 37 |

Source: user measurements and photographs supplied September 13, 2026. They also
reported approximately half an inch (12.7 mm) motor-housing clearance in the
photographed supported floor pose. This is an approximate measurement of that
pose, not a guaranteed clearance throughout a rotation. The photos establish
cooled deformation despite the outer tread retaining a rounded outline.

Incomplete compression changes both the base below the hub and the tip above it;
do not model all variation as an independently shortened tip on an ideal flat
base. The user expects roughly 10 mm total length variation between appendages
and small angle variations, with rigid material after cooling. The
[measured-shape simulation source](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/hardware_testing/triangle_roll/measured_360ffcd09344487d/source.json)
uses 57 mm above / 35 mm below as a nominal approximation, not four exact measured
legs. Preserve motor/body collision checks when reconciling model and hardware.

## A home target is different from an encoder reference

The approved upper-joint target, in calibrated policy coordinates, is:

| Leg | Motor 1 (rad) | Motor 2 (rad) |
| --- | ---: | ---: |
| Front right | 1 | 0 |
| Front left | -1 | 0 |
| Back right | 1 | 0 |
| Back left | -1 | 0 |

The user visually approved this pose and its static floor support. The
[shared target](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/ros2_ws/src/neural_controller/include/neural_controller/policy_home.hpp)
and [saved approval](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/hardware_testing/start_pose/approved_upper_pose.json)
record it separately from the hanging startup reference. These values must not
be used as raw encoder zeroes.

The intended workflow is persistent upper-joint referencing plus calibration of
the selected hub/spoke each boot. Saving the target alone does not implement that
workflow. At the pinned implementation, startup still establishes new session
offsets; follow [STARTUP_CALIBRATION.md](https://github.com/TundTT/Pupper_animation/blob/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/STARTUP_CALIBRATION.md)
and its required physical confirmation/capture. Within a valid live encoder
session, reuse calibration across controllers instead of recapturing it.

Two reboot comparisons with no reported movement found small upper differences
but large, nonuniform hub differences. The second included approximately -180°,
-144°, +180°, +180° in FR/FL/BR/BL order. This supports per-boot hub calibration;
it does not establish a universal 180° correction or prove upper persistence for
all poses. See [raw observations and comparisons](https://github.com/TundTT/Pupper_animation/tree/f9f1c48c505af2d076170e0aa3e1eeb5a43cfdc9/hardware_testing/pi_blackout/repro_4c724137).
The motor firmware cause is unconfirmed. A gear/encoder explanation remains a
hypothesis, not a hardware fact. Deeper diagnosis need not block calibrated tests.

## Controller transfer and practical testing

- Maintain canonical FR/FL/BR/BL joint order from HARDWARE.md. Apply signs and
  offsets once. A continuous hub can reach the same visible orientation at angles
  differing by a full turn, while a position-observing policy sees different input.
- Use one session-bound hub-to-model mapping in both observations and commands.
  Check winding, initial action/history, target continuity and gains before
  handing off from roll to walking. A good final pose does not validate the
  walking interface. The pinned roll does not automatically activate walking.
- A supported joint-position move is useful for setup and sweep inspection; it
  is not a floor contact planner. A suspended success establishes neither balance
  nor foot support. Test roll-and-hold on the floor before adding walking.
- The software stop can release stiffness on tilt or stale inputs. Plan support
  before carrying a stiff robot. A black Pi screen/SSH loss does not by itself
  identify low battery, a software crash, or a motor fault. Preserve logs; do not
  label an unreproduced blackout fixed.
- The operator requests a heads-up and confirmation before new motor motion and
  has a physical power e-stop. No startup, calibration or read-only inspection
  implicitly authorizes a subsequent motion.
- Prefer targeted checks for the next physical question over a large robustness
  sweep. Preserve failed trials and actual rollout videos. GPU training is an
  available option on the user's RTX 6000 PC; laptop capacity is not a reason to
  reject it, but do not request more training without a concrete need.

Implementation entry points and evidence are indexed in LAB_HANDOFF.md on
robot-code. Keep changing process IDs, boot IDs, IP addresses, experiment status
and deployment commands there rather than treating them as permanent facts here.
