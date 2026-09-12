# PC agent prompt: validate the simultaneous roll into walking

**Formation update:** read `FORMATION_ROBUSTNESS.md` and reproduce
`formation_cases.json` before proceeding. The user reports rigid legs with about
10 mm total longest-to-shortest variation and slight angle errors. Use -5/+5 mm
as the centered length trial and label +/-5 degrees as an assumed angle range.
The direct motion fails four of seven development cases' non-CAD checks; the
optional settling prototype is also rejected. Keep all failures. Use the new
derived geometry mechanism, preserving the pinned nominal model and 9 mm gap.
Optimize bounded joint/IMU feedback or, if necessary, train a bounded residual
policy over independent per-leg geometry variations. No true lengths or contact
forces in deployed actor observations. Cover leg permutations, common length
offsets separately from spread, fresh held-out combinations and sensor/dynamics
errors. Do not claim robustness by testing only symmetric/equal-length shapes.
Do not change the final-pose gates just to pass; report any demonstrated
geometric infeasibility and the required stance adjustment. Validate continuous
walking handoff on the imperfect geometry, not only the nominal robot.

Continue on `codex/triangle-roll-to-stand` in an isolated checkout. Read
`AGENTS.md`, `motion/inverted_triangle/ROLL_TO_STAND.md` and
`motion/inverted_triangle/results/roll_to_stand_v1/README.md` first. Do not use
the old sequential endpoint as the task goal and do not touch the physical robot.

The user wants all four post-cooled rigid, point-up limbs to roll forward together
and finish in or near the selected leg-walking policy's starting stance. Heating
is manual and outside the control problem. The simplest 12-second all-forward
quintic interpolation has passed one nominal probe and finishes within 0.0302
rad of the exact policy defaults. Start by validating that candidate; a new RL
policy is not required merely to reproduce the nominal result.

Preserve original tip length, 9 mm additional gap, canonical motor order and
the new backpack mass/inertia. The candidate model SHA256 is
`c274c1b3ddba73b58c89e8dded10d2d2d8e0fda37af77f631d4bedefa2d177e2`.
The walking export SHA256 is
`854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8`.
Use the pinned requirements lock. If this checkout inherits sparse settings,
restore only the additional training/runtime source needed, or disable sparse
checkout on the PC. Never substitute a new model hash into an old accepted
candidate to bypass its provenance checks.

1. Freeze/reproduce `all_forward_direct` from `roll_to_stand.py` and archive an
   actual video. Audit dynamically replayed CAD more densely around minima and
   contact changes, including all front/rear shin pairs, motor housings, body and
   backpack. Track impulses, slip and speeds as contact rolls onto each tip.
   This ground-roll task deliberately uses ground contact during rotation; the
   old airborne-leg clearance and three-support rules are not applicable gates.
   Keep the new endpoint and existing relevant torque/tilt/collision limits.
2. Extend the new runner with explicit friction, mass/COM, gain, available torque,
   command delay and initial-state scenarios. Reuse definitions from the original
   robustness suite with the NEW model; its old passes do not transfer. Run the
   original families plus predeclared fresh held-out combinations. Keep all
   failures and distinguish tuning cases from held-out validation.
3. Validate the actual walking handoff in simulation. Inspect the exact selected
   policy's observation history, gravity/orientation convention, action scales,
   gains, control period and joint frames from training and runtime source.
   Do not assume pose closeness proves safe activation. Continue from the actual
   transition's final positions AND velocities, with the backpack included.
   Verify zero commanded walking velocity first, then gentle forward walking.
   Check contact-model differences between the transition CAD-floor envelopes
   and the walking `rigid_flush` model. Report whether inference/physics transfer
   was modeled faithfully; do not teleport or reset the base/joints at handoff.
4. Preserve a single unwrapped coordinate frame. The new simulation chooses the
   left initial hub winding once before integration, so the final targets are
   exactly +1 rad on the left and -1 rad on the right. This is physically the
   same inverted orientation as the old keyframe, but is NOT a recipe for changing
   physical encoder zeros. Plan how both runtime controllers will share the
   measured model-to-encoder mapping; do not alter calibration to force a pass.
5. If the direct candidate fails robustness/handoff, first optimize a small
   number of coordinated proximal waypoints/timings or add a bounded settling
   phase. Keep the final walking stance a hard requirement. Use more expensive
   optimization or training only where the failures justify it. A held-out pass
   requires an unchanged candidate and gates fixed before the final test batch.

Log online to entity `QuadMorph`, project `wheel-leg lift and align triangle base`
with the existing `training/wandb_logging.py`. Put real simulated rollout videos
in Media, clearly label failures/early termination and checkpoint identity (or
no checkpoint for trajectories), and verify cloud bytes and committed artifacts.
Store full configurations, exact source/model/export hashes, actual integration
states, metrics and source revisions. Do not commit credentials.

Commit and push the completed work on this branch. Return the commit, concise
pass/fail tables, final continuous video including walking handoff, remaining
limitations and any concrete hardware-executor changes needed. No hardware
deployment, activation, heating, robot SSH operations or motor motion.
