# QuadMorph reshaping process — full outline (v2)

Scope: this covers only the wheel↔leg reshaping maneuver. `π_wheel` and `π_leg`
(RL) remain unchanged and handle normal locomotion; the reshape sequence is
fully scripted/open-loop, triggered manually, structured into two phases.

## Per-leg state enum

| State | Meaning |
|---|---|
| `Wheel` | Locked, circular, load-bearing |
| `Aligning` | Phase 1: unweight + rotate to fixed base target + reweight |
| `Heating` | Phase 2: operator holds heat button, watching for deformation |
| `Locking` | Heater off, cooling, holding position until rigid |
| `Repositioning` | Lift + 180° rotate to home, then lower (both directions) |
| `Leg` | Locked, elongated, load-bearing |

## Startup calibration (once, per leg, at power-on)

For each leg, rotate its wheel to a pose where one specific ring is touching
the ground, and store that hub encoder angle as `home_i`. This ring is now
permanently that leg's committed "point ring" — the same physical ring
becomes the leg tip every cycle, rather than whichever ring happens to end
up opposite the base.

Trade-off worth knowing: because the point-forming mechanism (TPU ring
constant-perimeter constraint) doesn't actually care *which* ring ends up
opposite the base, you could instead let any of the 3 serve as the point and
always rotate to the *nearest* valid base position (max ≤60° rotation, via
`theta mod 120`). Pinning one specific ring via calibration is simpler to
implement (one fixed absolute target, no modulo/nearest logic) but can
require up to ~180° of rotation to reach it, depending on where the wheel
happens to be sitting after driving. Not a problem, just budget torque/time
for a potential half-rotation, not a small nudge.

## Global prerequisites

- **Coast to stop.** Operator releases RC input; existing locomotion policy
  tracks `v_cmd → 0` normally. Operator presses "begin transform" only once
  stationary — this hands control from the RL policy to the reshape FSM.
- **Heat confirmation is manual/visual.** Operator holds a dedicated "heat"
  button; current flows to that shin's liquid-metal channel while held;
  operator releases once they visually confirm the shin has deformed into a
  point. No timer or thermistor needed for this step.
- **Cool confirmation — still to decide.** Heating is now clearly
  operator-gated, but cooling (waiting for rigidity before loading the new
  shape) isn't yet specified the same way. Worth deciding explicitly:
  same hold-a-button pattern, a fixed wait, or a visual go/no-go check
  before the operator manually advances to the next step.
- **Support-polygon check** — split cleanly by phase now:
  - *Phase 1* is the easy case: every unweighted lift happens while the
    other three legs are still identical, uniform wheels — symmetric stance,
    predictable margin.
  - *Phase 2* is where it actually matters: as legs convert one by one, the
    support set becomes an increasingly uneven mix of wheels and already-
    planted legs. Confirm margin in sim leg-by-leg, not just once.

## Phase 1 — Align all 4 legs to base position (still wheels throughout)

Diagonal order (e.g. FL → RR → FR → RL). For each leg i:

1. **Unweight** — slight hip/knee raise, shifting load to the other three,
   below the bench-determined free-spin threshold.
2. **Rotate to align** — command hub to the fixed target `home_i + 180°`
   (shortest direction around the circle). This is the step that can take
   up to ~180° per the trade-off above.
3. **Reweight** — lower back down. Leg is now aligned (two rings on the
   ground, the committed point-ring at top) but still fully a wheel —
   nothing has been heated yet.

Repeat for all 4 legs before moving to Phase 2.

## Phase 2 — Heat, lock, flip, plant (per leg, diagonal order)

For leg i (already aligned from Phase 1):

1. **Heat** — operator holds the heat button. ~1/4 body weight through the
   two base segments compresses them; the TPU ring's constant-perimeter
   behavior pulls the top (point-ring) segment into a point. Operator
   releases once visually satisfied.
2. **Cool & lock** — hold position until rigid (method TBD, see above)
   before loading the new shape.
3. **Lift** — raise the now-rigid, pointed shin fully clear of the ground.
4. **Rotate 180°** — target is simply `home_i` (no new computation): the
   point-ring is currently at `home_i + 180°`, so another 180° rotation
   lands it back at the calibrated home angle, now pointing down.
5. **Lower & plant** — point contacts the ground. Leg done.

Repeat for remaining 3 legs, diagonal order. Once all 4 done: hand control
to `π_leg`.

## Leg → Wheel (unchanged from prior discussion)

Per leg, diagonal order: lift (unloaded) → heat unloaded (same
operator-hold pattern) → passive recovery to circular → cool & lock →
lower as wheel. No alignment step needed in this direction — a wheel is
rotationally symmetric, so there's no target angle to return to.

One thing worth deciding: whether `home_i` needs to be re-validated after a
return to wheel mode, in case of any encoder slip during the drive session —
see open items below.

## Open items

- **Cool/lock confirmation method** — pick one (hold-button, fixed wait, or
  visual go/no-go) and make it explicit in the FSM, same as heating now is.
- **Rotation budget for Phase 1** — since align can require up to ~180° per
  leg (not the smaller correction possible with a "nearest ring" approach),
  confirm this is acceptable for cycle time and doesn't stress the hub motor
  more than expected.
- **Encoder drift over a drive session** — the whole scheme depends on
  `home_i` staying accurate from a single power-on calibration. Worth
  deciding whether/how to detect or correct for slip (e.g. a quick re-zero
  against a hard stop) before trusting `home_i` on a long-running robot.

(Abort/fault handling intentionally out of scope per current direction.)

---

# Addendum — operator control scheme (v3)

Every confirmation in this process is visual (operator's eyes), and every
transition is a button press. No timers, thermistors, or deformation
detection needed in firmware — it only needs to execute the right scripted
motion on command.

## Balance during heating

Not a separate program — a background behavior tied to the heat control
itself. While heat is active: current flows to the active leg's heater, and
all 4 legs' hip/knee joints hold their current commanded positions (stiff
PD position-hold). No active balancing; stability relies on (a) the
material only deforming inside the wheel structure, not via any leg joint
motion, and (b) Phase 1 having already made the base symmetric, so
compression under load should be roughly even. Confirm in sim that the body
doesn't noticeably sag/tilt during this step before trusting it on hardware.

## Software modules

| Program | Behavior | Used in |
|---|---|---|
| `align_and_advance()` | unweight → rotate to `home_i+180` → reweight → advance leg pointer | Phase 1 (×4) |
| `heat_hold()` | current on while held + hold-pose active; current off on release | Phase 2 heating; leg→wheel heating |
| `flip_and_advance()` | lift (using converted-leg clearance) → rotate to `home_i` → lower to `q_leg^0` → advance leg pointer | Phase 2 (×4, after each heat+cool) |
| `lift_leg()` / `lower_wheel()` / `advance_leg()` | three atomic steps tapped in sequence | leg→wheel (×3 per leg, with `heat_hold()` between lift and lower) |

These collapse into one generic `advance_step()` dispatcher keyed on
internal state (direction, phase, active leg) rather than four hand-written
routines — same FSM as before, just exposed one tap at a time.

**Open assumption:** lift → flip → lower (wheel→leg) is treated as one
atomic NEXT press, not three separate confirms. Flag if a manual gate
before ground contact is wanted instead.

## Control mapping

| Control | Type | Function |
|---|---|---|
| NEXT | momentary tap | Advances the FSM one step; meaning depends on current phase/direction |
| HEAT | momentary hold | Current on while held + hold-pose; off on release. Same for both directions |
| MODE | 2-position switch (or 2 buttons) | Selects wheel→leg vs. leg→wheel once; NEXT's behavior follows from this |

3 physical inputs total, not one per action.

---

# Addendum v4 — PS5 controller mapping + lock-pose semantics

## Why the corner leans, and why that's correct

The lean during heating is not a stability failure — it's the compressive
mechanism working as intended. The hip/knee joints stay fixed (lock-pose is
active); the shortening happens entirely inside the wheel structure
(l1/l2/l3 segments compressing under load), a separate DOF from the leg's
kinematic joints. Actively leveling the body across the other 3 legs would
remove the very load the material needs to deform. The TPU ring's fixed
perimeter bounds how far this can go, so the lean should settle at a
repeatable angle rather than progress indefinitely — worth confirming that
settled angle stays within an acceptable margin (bench or sim), since the
other 3 legs now carry more than an even 1/4 share while this happens.

## Lock pose / unlock pose

`Lock pose` is a one-time transition (not per-leg, not per-heat-cycle):
exits the running RL locomotion policy, holds all 4 legs' current joint
targets, and stays in effect for the whole reshape session. Needs a
symmetric `Unlock pose` to hand control back to `π_wheel`/`π_leg` when
done — this wasn't explicitly assigned yet.

## Button mapping

| Control | Function |
|---|---|
| Face button A | **Next leg** — advances shared pointer 1→2→3→4→1; rumble count (1–4 pulses) confirms selection. General-purpose status/override; buttons B/C/D below don't require it for normal sequential use since each self-advances |
| Face button B | **Leg→wheel procedure** — self-advancing: tap=lift, tap=lower(now wheel), tap=lift(next leg)... |
| Face button C | **Wheel→leg flip procedure** — self-advancing: tap=lift→rotate 180°→plant; next tap=next leg |
| Face button D | **Wheel-aligning procedure** — self-advancing: tap=unweight→rotate to base→reweight; next tap=next leg |
| D-pad (hold, 4 directions) | **Heat** — each direction wired to one specific leg's heater; current on while held, off on release |
| L1 (suggested) | **Lock pose** — one-time: exit RL control, hold joint targets, enter scripted mode |
| R1 (suggested) | **Unlock pose** — one-time: hand control back to the running RL policy |

**Open assumption:** buttons B, C, D each maintain their own internal leg
pointer and self-advance independently (per your description of programs 3
and 4). Button A is a general check/override, not a required step in the
normal sequence. Flag if button A was meant to gate every action instead.

---

# Final control scheme (v5 — supersedes trigger suggestions in v4)

Circle and Triangle are pre-existing locomotion-policy selectors
(`π_wheel` and `π_leg` respectively). They do double duty: also used to
exit morph mode by resuming whichever policy matches the robot's current
physical configuration. No separate "unlock" button needed. This relies on
the operator visually confirming full completion (all 4 wheels or all 4
legs) before pressing — consistent with the rest of this design, which has
no software gate against handing off mid-transform into a stance neither
RL policy has seen. Worth being disciplined about in practice, since
there's nothing stopping a mis-press.

| Control | Function |
|---|---|
| X | **Lock/detach** — exits `π_wheel`/`π_leg`, hands off to the morph controller(s) |
| L2 | **Wheel-aligning procedure** — Phase 1 of wheel→leg, self-advancing |
| L1 | **Wheel→leg flip procedure** — Phase 2 of wheel→leg, self-advancing |
| R2 | **Leg→wheel procedure** — self-advancing |
| R1 | **Next-leg selector** — morphing leg pointer; rumble count (1–4) confirms selection |
| D-pad (hold, 4 directions) | **Heat** — one direction per leg; current on while held, off on release |
| Circle | Resume `π_wheel` — also functions as unlock when robot is in wheel config |
| Triangle | Resume `π_leg` — also functions as unlock when robot is in leg config |

Left shoulder pair (L1/L2) covers the forward transform end-to-end; right
shoulder pair (R1/R2) covers the reverse procedure plus the leg selector.
Swappable if a different physical grouping feels more natural once tested
on the actual controller.
