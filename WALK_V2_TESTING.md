# New leg-walking policy — testing on the real robot

Everything you need to bring this up on the Pupper and bench/hardware-test the new
**leg-walking** policy from the `leg` branch. Self-contained on purpose: `robot-code`
strips the training side, so this is the doc that travels with the robot code. The
training-side story (reward design, domain randomization, the two Opus policy reviews
that shaped this policy) lives on the `leg` branch in
`mujoco_playground/workspace/WALKING.md`.

> **Status: SIM-VALIDATED ONLY. This policy has never run on real hardware.**
> Treat every step below as a first hardware test, not a routine bring-up. Put the
> robot on a stand with all four feet clear of any surface before the first
> activation, and keep the e-stop hand ready.

## What changed in this branch update

1. **Added `neural_controller_walk_v2`** as a new controller instance — a pure
   addition. `neural_controller` (locomotion), `neural_controller_three_legged`,
   `neural_controller_wheel`, and `neural_controller_leg_lift` are untouched: same
   config blocks, same button bindings (X/Square/Triangle/O), same `model_path`s.
2. **Restored `pupper_v3_description`'s leg-mode hardware/sim description**, which
   had silently stayed in the wheel-testing branch's configuration:
   - `components.xacro`: the four `_3` (knee) joints were left with wheel-mode
     `homing_velocity`/`homing_kp`/`homing_torque_threshold` (all zeroed, homing
     disarmed) and `position_min`/`position_max` widened to ±1000 from the wheel
     port. Restored to the pre-wheel leg values (real position limits, active
     homing against a mechanical hard stop). **This affects every leg-based
     controller, not just the new one** — if the robot has been reassembled with
     legs (confirmed as of this update), `neural_controller`/`three_legged`/
     `leg_lift` were *also* homing against the wrong parameters before this fix.
   - `mujoco_xml/pupper_v3_complete.xml` and `.backlash.xml` (used by
     `pupperv3_mujoco_sim` for bench-testing without hardware): restored from
     wheel meshes/collision geometry/joint back to the leg model, including the
     leg home keyframe. Sim-testing on this branch was simulating a wheeled
     robot until this restore.
   - The wheel branch's own values are still recoverable from git history
     (commit `acdfdcd` on `robot-code`, "Port the wheeled policy...") if the
     robot goes back to wheels later — nothing was deleted, just reverted.
   - `neural_controller_wheel`'s own config block is untouched and still points
     at its own policy; it just won't be *mechanically* runnable until someone
     re-applies the wheel-mode xacro (same as before this branch had wheels
     ported at all).
3. Added `neural_controller_walk_v2` to `animation_controller.py`'s
   `neural_controllers` deactivation list — the same fix `leg_lift`/`wheel`
   needed (the animation system's forward_* controllers claim the same command
   interfaces, so any active neural controller left out of that list would
   block the switch).

## What's deployed

| | |
|---|---|
| Policy | `ros2_ws/src/neural_controller/launch/policy_walk_v2.json`, exported from `leg` branch run `walk_2026-09-05_13-30-55` (`best_params`, step 201,850,880, eval reward 36.475) |
| Controller instance | `neural_controller_walk_v2` in `config.yaml`, spawned inactive by `launch.py` |
| Activated by | **L2 (joy button 6) — unverified guess, check against this joystick's actual `/joy` mapping before testing** |
| Exit | **X** returns to `neural_controller` (locomotion) |
| Commanded by | `/cmd_vel`, same as locomotion |

**This policy's home pose is different from every other instance in this file:**
`[1.0, 0.0, -1.0]` rad per leg (hip, abduction, knee), versus `[0.26, 0.0, -0.52]` for
locomotion/three_legged/leg_lift. `kp`/`kd` are baked into the policy JSON (5.0/0.25)
and read from there — `config.yaml` deliberately does not override them, matching how
`neural_controller`/`three_legged`/`leg_lift` already work.

Trained envelope: `vx` ±0.35 m/s, `vy` ±0.15 m/s, `yaw` ±0.8 rad/s. This is a from-scratch
walking policy (not derived from the Stanford `pupperv3-mjx` training this repo's other
locomotion policies used), trained via `mujoco_playground/workspace/train_walk.py` with
its own reward design (a rubber-ring ground-clearance proxy, foot-swing-height shaping,
leg-length/contact-compliance domain randomization — see `WALKING.md` for the full
design history and two rounds of review that shaped the reward weights).

## Sim numbers (from the training-side evaluation — not a hardware promise)

- Reward converged cleanly to ~36.5 with no terminal PPO collapse (an earlier retrain of
  this same pipeline did collapse right at the end of training; this checkpoint's KL was
  stable at ~0.046 through the last eval, not 0.4+).
- Forward/backward tracking to within a few percent of commanded speed; velocity error
  ~0.05 m/s at the eval checkpoint.
- Peak foot swing clearance 10-17mm forward, up to ~17mm on the best-lifting foot in
  reverse (asymmetric — front feet lift less than rear in reverse, worth watching).
- No falls in the deterministic showcase rollout (stop/forward/arcs/spin/reverse/stop).
- **Standing still is not perfectly settled.** A `stance_feet` reward term (added to
  fix an earlier checkpoint that consistently held one foot, back-right, hovering
  ~95-100% of the time) *is* in the training run this policy came from, and that
  specific bug is gone -- no single foot is a consistent outlier anymore. But a
  stand-still check on `best_params` (6 seeds, 200 steps each, deterministic policy)
  shows most seeds cycling all four feet between ~65-85% ground contact rather than
  holding a clean 100%; only 1 of 6 seeds settled to a rock-solid four-foot stance.
  Expect some visible fidgeting/weight-shifting on a "stand still" command, not
  outright limping on three legs. Worth watching on hardware, not necessarily a
  stop-ship issue.

## Test procedure

```sh
git checkout robot-code && git pull

cd ros2_ws && source build.sh          # rebuild after the config/xacro changes above
ros2 launch neural_controller launch.py
```

**Git LFS is required** for the policy `.json` files (`*.json filter=lfs` in
`.gitattributes`). Confirm the file actually pulled:

```sh
wc -c ros2_ws/src/neural_controller/launch/policy_walk_v2.json   # want ~1.7 MB, not ~130 B
head -c 40 ros2_ws/src/neural_controller/launch/policy_walk_v2.json  # must NOT say "version https://git-lfs..."
```

### Confirm the policy loaded correctly

On launch, `neural_controller_walk_v2` should log the values it read from the JSON.
Worth checking:

- `observation_history=4` and an input shape of **144** (= 4 × 36).
- `default_joint_pos` in the log matches `[1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0,
  -1.0, -1.0, 0.0, 1.0]` — **not** the locomotion pose. If it doesn't, the wrong
  `model_path`/config got loaded.

### Bring-up order

Since this is a genuinely new, never-hardware-tested policy (unlike wheel, which had a
sim-verified direction convention to check but a known-working leg mechanism), be more
conservative than the wheel bring-up was:

1. **On a stand, all feet clear.** Press L2 (button 6, or whatever it resolves to on
   this joystick — check first). Watch the init phase: it should smoothly move each leg
   to the `[1.0, 0.0, -1.0]`-style home pose over `init_duration` (2s), not snap or jerk.
   If any joint moves sharply or hits a mechanical limit audibly, e-stop immediately —
   that likely means `components.xacro`'s restored position limits above don't match
   this specific robot's calibration, or the homing at boot didn't complete correctly.
2. **Still on the stand**, command a small `vx` and confirm the legs move in a
   walking-like pattern (not wheel-spinning, not locomotion's pose) and don't hit any
   joint limit.
3. **On the ground, held or spotted, zero command.** Check whether all four feet
   settle, or fidget/shift weight continuously (see the standing-stability caveat
   above and below).
4. **Small `vx` only**, no yaw. Check it drives roughly straight.
5. **Add yaw and reverse.** The reverse gait was specifically reworked over two review
   rounds on the training side; check whether it still looks asymmetric/uneven the way
   the sim numbers suggest, or the mismatch is worse/better on hardware.

## Known issues

- **Button 6 is an unverified guess.** Confirm the actual `/joy` index for L2 (or
  whatever button you want) on this specific joystick before relying on it, and update
  `switch_button_indices` in `config.yaml` if it's wrong. This only affects this new
  controller's activation — no other button binding was touched.
- **Never run on real hardware.** Everything above the "Sim numbers" section is a
  training-side simulation result, not a hardware guarantee. Treat the whole bring-up as
  exploratory.
- **The `components.xacro`/`mujoco_xml` restore affects the other leg-based policies
  too** (see "What changed" above) — if the robot was actually still relying on the
  wheel-mode homing/limits for some other reason, flag that before merging this further;
  this update assumes the robot is currently legged, per hardware confirmation at the
  time of this addition.
- **Standing still fidgets rather than settling completely** in most sim seeds — see
  "Sim numbers" above. The specific one-foot-always-up bug from an earlier checkpoint is
  fixed; this is a milder residual instability.

## Test log

*(empty — no hardware test has been run yet)*

## Reporting back

Useful things to capture: whether the init-phase move to home pose was smooth, whether
all four feet plant evenly at a stand-still command, forward/backward/turning quality
and any asymmetry (especially reverse), any joint straining or unusual sounds, and
motor temperature after a few minutes. Videos help far more than descriptions,
especially for anything involving gait asymmetry or foot placement.
