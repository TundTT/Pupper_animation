# Wheeled locomotion policy — testing on the real robot

Everything you need to pull this branch onto the Pupper and test the **wheeled**
velocity-command policy. Self-contained on purpose: the `robot-code` branch strips the
training side, so this is the doc that travels with the robot code. The training-side
story (the reward, the domain randomization, why the wheel numbers are what they are)
lives on the `wheel` branch in `mujoco_playground/workspace/README.md`.

> **Safety first, every time.** Put the robot on a stand with the wheels clear of the
> ground before launching anything. Re-check even if it was on a stand five minutes ago.
> Wheels carry rotational momentum that legs never did — this is exactly why the
> e-stop gain had to be raised (see Known issues).

## What's deployed

| | |
|---|---|
| Policy | `ros2_ws/src/neural_controller/launch/policy_wheel.json` (run 3, `wheel_2026-09-02_00-21-44`) |
| Controller instance | `neural_controller_wheel` in `config.yaml`, spawned inactive by `launch.py` |
| Activated by | **Triangle** (joy button 2) |
| Commanded by | `/cmd_vel` (`geometry_msgs/Twist`) — the same topic locomotion uses |
| Exit | **X** returns to `neural_controller` (locomotion). See the warning below. |

The robot is a **four-wheeled** machine on this branch. Each leg's `_3` joint — the old
knee — *is* the wheel: a continuous joint driven in **velocity** mode. The `_1`
(abduction) and `_2` (hip) joints are still position-controlled and hold a splayed
stance that puts the wheels on the ground.

What the policy can actually do: **forward/back (`vx`) and yaw**. It cannot strafe —
these are fixed, non-steerable wheels. `linear.y` is deliberately ignored by the wheel
behavior (see Known issues).

Trained envelope: `vx` −0.6…0.8 m/s, `yaw` ±2.0 rad/s, wheel speed capped at 1.0 m/s.
Measured achievable in sim: 0.958 m/s and 4.65 rad/s.

## Test procedure

```sh
# first time on this robot:
git clone -b robot-code https://github.com/TundTT/Pupper_animation.git
# already have it:
git checkout robot-code && git pull

cd ros2_ws && source build.sh          # the wheel C++ must be compiled at least once
ros2 launch neural_controller launch.py
```

**Git LFS is required.** The policy `.json` files are stored in LFS (`*.json filter=lfs`
in `.gitattributes`). Without `git-lfs` the clone succeeds but leaves a ~130-byte
pointer stub in place of the policy and `neural_controller` fails to parse it.

```sh
git lfs version || sudo apt install git-lfs      # then: git lfs install
wc -c ros2_ws/src/neural_controller/launch/policy_wheel.json   # want ~1.7 MB, not ~130 B
head -c 40 ros2_ws/src/neural_controller/launch/policy_wheel.json  # must NOT say "version https://git-lfs..."
```

If you got stubs, `git lfs install && git lfs pull` fixes it in place.

### Confirm the policy loaded correctly

On launch, `neural_controller_wheel` should log the values it read from the JSON. Worth
eyeballing:

- `observation_history=4` and an input shape of **132** (= 4 × 33). If the shape check
  fails the controller refuses to activate rather than running a misaligned observation.
- No error about a wheel row whose `action_type` is not `"velocity"`. The wheel branch
  hard-errors on that rather than silently producing a wrong observation.

### Bring-up order

Do these in order. Steps 1–2 are the cheap places to catch the failure modes that matter.

1. **Wheels off the ground, on a stand.** Press **Triangle**. Push the left stick
   forward a little. **All four wheels must turn the same direction.** If one side spins
   opposite the other, stop — the wheel direction convention is wrong somewhere and the
   robot will fight itself and not move (this is a real failure mode that was caught in
   sim; the correction is folded into the policy's exported `action_scale` and must not
   be applied a second time anywhere else).
2. **Still on the stand**, try yaw: the two sides should spin opposite each other.
   Then let go — the wheels should stop, not coast indefinitely.
3. **On the ground, low speed.** Small `vx` only, no yaw. Check it drives straight and
   stops when commanded to zero.
4. **Add yaw.** Arcs, then spin in place.
5. **Watch for oscillation** and check motor temperature after a few minutes of driving
   — see Known issues.

## What good looks like

- Drives straight at commanded speed with no visible weave.
- Stops promptly and stays stopped at zero command (the policy has a dedicated
  stand-still term).
- Spins in place smoothly on a pure yaw command.
- Torso stays level; the wheels stay on the ground.

In sim the policy tracks commands to within a few percent, holds ~0.7° of tilt, and
never lifts a wheel. Sim numbers are not a promise about hardware, but a large
divergence from this is worth investigating rather than tuning around.

## Known issues

- **X exits into the LOCOMOTION policy, which was trained for feet.** On a wheeled robot
  that policy is running far outside anything it ever saw. On a stand it just looks odd;
  on the ground it could be violent. For now, prefer the **e-stop** as the way out of
  wheel mode, not X. This is a wiring choice that can be changed if it proves annoying.
- **E-stop on wheels needed a stronger gain.** `estop_kd` for this controller is 1.0
  (the joint's `kd_max`), raised from the leg-derived 0.3 after a hardware session found
  it too weak to arrest a spinning wheel. If e-stop still feels slow to stop the robot,
  say so — that is a safety issue, not a nicety.
- **Lateral (`vy`) commands are ignored on purpose.** Fixed wheels cannot strafe, and
  the policy was trained with `vy` pinned to zero, so it had never seen a nonzero value
  there. The joystick still publishes `linear.y` for the legged policies, and feeding
  that through made the wheeled policy behave erratically when the stick was pushed
  sideways. The wheel behavior now forces that observation slot to zero. Pushing the
  stick sideways should simply do **nothing**.
- **Torque is up ~40% versus the previous policy** (0.918 Nm peak in sim, against a 3 Nm
  ceiling and a ~1.32 Nm back-EMF limit at top speed). Inside the envelope, but worth
  checking motor temperature after sustained driving.
- **Wheel joints do not home.** A free-spinning joint has no hard stop to push against,
  so the torque-threshold homing routine can never complete on it — the threshold is set
  to 0.0 so those joints are marked homed immediately, without moving. This is
  deliberate and load-bearing: with the stock threshold the robot hangs at startup and
  never activates. It also means the wheel joints' absolute angle is meaningless, which
  is fine — the policy observes wheel *speed*, not angle.

## Test log

### 2026-09-03 — `wheel_2026-09-02_00-21-44` (run 3), first test of the new policy

- **Result: clean pass. Policy is stable, drives well, no issues found. Shippable.**
- Went through the staged bring-up (stand, then ground) without any problems reported.
- No oscillation, no direction-convention issues, no erratic behavior.
- This is the first fully clean wheeled-driving hardware session after the prior
  session's two fixes (raised `estop_kd` to `kd_max`, and forcing the `vy` observation
  slot to zero for the wheel behavior) — both appear to have resolved what they were
  meant to.

**Status: `wheel_2026-09-02_00-21-44` (run 3) is the recommended, hardware-validated
wheeled policy going forward.**

## Reporting back

Useful things to capture: what the robot actually did versus what was commanded, any
oscillation (and at what speed it starts), motor temperature after a few minutes, and
whether the e-stop arrests the wheels quickly enough. Videos help more than
descriptions for anything involving oscillation.
