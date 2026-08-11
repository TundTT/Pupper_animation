# Leg-lift policy — testing on the real robot

Everything you need to pull this branch onto the Pupper and test the leg-lift policy.
Self-contained on purpose: the `robot-code` branch strips the training side, so this is
the doc that travels with the robot code. The training-side story (why the reward looks
the way it does) lives on `master` in `mujoco_playground/workspace/README.md` and
`.notes/swapping_new_policies.md`.

> **Safety first, every time.** Put the robot on a stand with the legs clear before
> launching anything. Re-check even if it was on a stand five minutes ago.

## What's deployed

| | |
|---|---|
| Policy | `leg_lift_2026-08-11_01-35-11` (wandb `t0e8qg3f`) |
| File | `ros2_ws/src/neural_controller/launch/policy_leg_lift.json` |
| Controller | `neural_controller_leg_lift` (already in `config.yaml` + `launch.py`) |

Measured in sim over 256 domain-randomized robots, 12 s five-lift sequence:

- raises **all four** legs, foot clearance **0.123 – 0.132 m**
- torso stays at standing height (0.155 – 0.160 m), tilt ≤ ~4.7° mean
- horizontal drift ≤ ~0.04 m
- **0.0% actually fell over** (no tilt-out, no sitting down)
- rotates about **-21° over the full 12 s sequence** — see Known issues

## Test procedure

```sh
ssh pi@pupper.local                       # ask for the current password; it changes

# first time on this robot:
git clone --depth 1 --branch robot-code \
  https://github.com/TundTT/Pupper_animation.git ~/robot-code-leglift
# already have it:
cd ~/robot-code-leglift && git pull

source /opt/ros/jazzy/setup.bash
cd ~/robot-code-leglift/ros2_ws
source build.sh                            # `source`, NOT ./build.sh (exec bit often unset)

source install/local_setup.bash
ros2 launch neural_controller launch.py    # no sim:=True -> real hardware
```

`--depth 1` matters: a full clone drags every training artifact ever committed on other
branches.

### Confirm the policy loaded correctly

**This is the one new thing to check on this policy.** `action_scale` is now a
**12-element array** (per-joint; the hip needs a wide range to actually raise the leg).
`neural_controller.cpp` supports arrays via `set_param_from_json_mixed`, but **no policy
before this one used the array form**, so this code path has never run on hardware.
Look for this line in the launch log:

```
From JSON, setting action_scale vector element-by-element
```

- See `setting action_scale[:]=...` instead → the JSON has a **scalar**. The lift will be
  far too small; a 0.3 rad hip physically cannot raise the leg. Stop and re-check the JSON.
- See `Invalid size for action_scale` → the JSON and `kActionSize` disagree. Stop.

Otherwise wait for `neural_controller_leg_lift` to report configure success and for the
actuators to finish homing.

### Buttons

| Button | Action |
|---|---|
| **O** (Circle, idx 1) | 1st press: activate leg-lift, command `front_l`. Each press after: `front_l → front_r → back_r → back_l → …` |
| **X** (Cross, idx 0) | Back to locomotion, exits leg-lift |
| **PS** (idx 12) | E-stop, deactivates all controllers |
| **Options** (idx 9) | Release e-stop, reactivate last controller |

Watch the commanded state from another shell:
```sh
ros2 topic echo /leg_lift_command_index
```

Hold duration is entirely operator-timed — the policy holds whatever leg is commanded
until you press O again, so take as long as you need to apply heat.

## What good looks like

The commanded leg swings up and holds steady, foot roughly 13 cm off the ground, while
the body stays level and at full standing height on the other three feet. It should
**not** lean back, sit down, or set a knee or spare foot on the ground to free the
commanded leg. Earlier policies did all of those; if you see them, something regressed.

## Known issues

- **Slow rotation (~21° over a 12 s, five-lift cycle).** Real and expected in this
  build. The policy's observation has angular *velocity* but no absolute heading, so it
  can damp rotation but cannot steer back to a heading it cannot sense. Fixing it needs
  a new observation input (a `neural_controller` change, not a JSON swap). Not a safety
  issue on a stand; note how bad it looks in practice and report back.
- **A body tilt instead of a clean lift is NOT expected any more.** Older notes said a
  tilt was a known sim-to-real actuator gap and could be waved off — that was wrong; the
  tilt was the trained behavior of a policy whose leg physically could not reach the
  target. If this policy tilts instead of lifting, that is a real finding: capture the
  log and report it.
- **Actuator model gap is still unmodeled.** Training uses an idealized position
  actuator; the robot uses a torque motor with backlash. Widened kp/kd randomization is
  a proxy, not a substitute. The lift transient briefly touches the 3 Nm motor ceiling
  on one leg (~20 ms) — if the raise looks sluggish or the motor complains, that is the
  first thing to suspect.
- `hailo_detection` / `person_follower` dying with `ModuleNotFoundError: vision_msgs`
  is expected and harmless — unrelated vision nodes, they don't affect leg-lift.
- If `ros2 launch` says a package (`foxglove_bridge`, `camera_ros`, `topic_tools`) isn't
  found, comment its entry out of the `nodes = [...]` list in
  `ros2_ws/src/neural_controller/launch/launch.py`. Local throwaway fix; don't upstream it.

## Reporting back

Worth capturing: whether the `action_scale` vector line appeared, per-leg behavior for
all four legs, how pronounced the rotation is, and whether it was a clean session or
ended in an e-stop.
