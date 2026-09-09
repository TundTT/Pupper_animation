# QuadMorph Phase 1: eight position joints plus wheel PD

This branch trains abduction and hip for FR, FL, BR, BL, in that order. Action
rows map to full actuator rows `[0,1,3,4,6,7,9,10]`; there are exactly eight
learned outputs. Position targets are `default + [0.5,1.6]*4 * action`, clipped
to the physical joint ranges. No learned action affects wheel target or speed.
The model remains this branch's continuous-wheel XML. The existing corrected
comment about motor position/velocity capabilities is preserved.

## Verified model and calibration

MuJoCo reports 19 qpos, 18 qvel, 12 actuators. All `_1` and `_2` joints are
bounded, position controlled (kp 5, damping 0.25 in this environment). Each
`_3` joint is unlimited and its velocity actuator has gain 0.35, zero position
bias, velocity bias -0.35, control range ±21 rad/s, force limit ±3 Nm. The
mixed-actuator randomizer preserves those semantics and independently jitters
wheel radii by up to 2.5 mm. Settled nominal torso height is 0.13128 m.

`home[i]` is stored calibration, independent of encoder phase accumulated in
driving. `target[i] = atan2(sin(home[i]+pi), cos(home[i]+pi))` is persistent across
selection, interruption and retry. Reset randomizes calibration and driving
phase independently over a full circle. There is no `current + pi` retargeting.

Every wheel is always actively controlled by
`velocity = clip(2*wrap(reference - encoder_angle) - 0.35*encoder_velocity, ±2)`.
Nonactive wheels hold a locked encoder snapshot. While lifted, the active
wheel reference slews toward its persistent calibrated target at 0.25 rad/s.
When lowering begins, its current encoder angle becomes a locked hold snapshot.
The calibrated target remains separate and unchanged.

The isolated physical slew test measured peak reaction torque 0.00185 Nm at
0.25 rad/s versus 0.00926 Nm at 1 rad/s; a half-turn settled within 0.035 rad
and 0.08 rad/s in 13.64 seconds. This is an actuator experiment with constrained
chassis/leg coordinates, not proof of learned balance. Post-training audits
measure clearance, balance, held-angle error and drift in the free robot.

## Phase and command contract

An external command index is one-hot encoded with the leg-lift convention:
`stand, front_l, front_r, back_r, back_l`. `select_command` accepts changes;
the environment never chooses the next leg. Training uses a separate operator
wrapper that selects a random leg for 20 seconds and then commands stand.
Evaluation externally selects all four, with both front-first orders.

The supervisor progresses lift → rotate while lifted → verify settled → lower
→ hold. During the first three phases the actor receives the selected one-hot;
during lower/hold it receives stand. A new command interrupts into lower,
then starts the requested leg only after lowering finishes. Completion requires
25 consecutive samples (0.5 s) within 0.035 rad / 0.08 rad/s, followed by
lowering. A retry never redefines calibration. There is no rotation-error,
reaching or completion term in the RL reward.

The current lift guard uses encoder/IMU signals (the first attempt used a 0.25 rad abduction bound): signed hip above 1.1 rad, abduction
within 0.35 rad of home, tilt below 0.12 rad, and angular speed below 0.3 rad/s,
held for 10 samples. Loss of the guard freezes and actively holds an encoder
snapshot. This is a conservative pose-based guard, not a contact sensor. True
cylinder clearance is audited independently; unsafe rotation disqualifies a
sequence. It must be validated on the resulting policy before deployment.

RL learns lift height, smooth lift/lower motion and three-leg balance, using
leg-lift's soft stance mechanism. Body drift is free inside 35 mm while lifting
or lowering and 20 mm while idle. Stance joints have a soft home-pose reward,
not fixed targets imposed on the action. Actual cylinder clearance accounts
for wheel tilt, half-width and randomized radius. Geometry and body translation
are used only by reward/evaluation, never as policy/controller inputs.

## Deployment interface

The actor is an ELU MLP with eight direct joint-position outputs under the same
scale/offset convention as leg-lift/walk. Its 51 scalar observations are:

| Indices | Signal |
|---|---|
| 0–2 | body-frame IMU angular velocity |
| 3–5 | projected gravity |
| 6–10 | effective one-hot command |
| 11–22 | 12 encoder positions; position rows relative to home, wheels wrapped |
| 23–34 | 12 encoder velocities × 0.1 |
| 35–42 | previous eight raw policy actions |
| 43–46 | sin(calibrated target − wheel angle), FR/FL/BR/BL |
| 47–50 | cos(calibrated target − wheel angle), FR/FL/BR/BL |

No simulation-only observations or privileged critic are used. Normalization
must be folded into export just as in the existing deployment convention.
The small self-contained deployment helper must own the four wheel PD loops,
locked snapshots, persistent targets, slew, and phase/settling bookkeeping.
It also routes the effective leg/stand command to the actor. It does not
synthesize any of the eight leg-position targets or use IK. Encoder/IMU lift
and lower guards above must be mirrored rather than replaced with simulation
contact. This first attempt does not modify robot C++ or deploy to hardware.

The selected hip's position command is rate-limited during lift and lower,
as in the leg-lift training principle. Deployment must either mirror this
small clamp or establish experimentally that the raw policy respects it;
training alone does not establish that parity.

## Reproduction

Use the existing environment's Python directly; do not run uv sync. Set
`PYTHONPATH=mujoco_playground`, `CUDA_VISIBLE_DEVICES=0`, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` from this worktree. The available interpreter
is `/home/theerawit/Pupper_animation-combine/mujoco_playground/.venv/bin/python`;
this only supplies installed libraries, and imports the new code/model here.

```bash
python -m workspace.hybrid_smoke
python -m workspace.hybrid_train --steps 50000000 --envs 4096 --abduction_guard .35 --wandb --out mujoco_playground/workspace/hybrid_results/NEW_EMPTY_RUN_DIRECTORY
python -m workspace.hybrid_evaluate --params mujoco_playground/workspace/hybrid_results/attempt2_guard/mjx_params --out mujoco_playground/workspace/hybrid_results/attempt2_guard/audit64.json
```

The full-sequence audit runs 90 simulated seconds: 20 per selected leg, with
2 seconds stand before/between/after. It does not autoreset. Success uses the historical final thresholds: all four completed, final calibrated
errors below 0.1 rad, speeds below 0.25 rad/s, and no termination. A separate
stricter count requires 0.06 rad and no rotation below 5 mm actual clearance.
Drift quantiles include all samples through first termination. Historical
held-angle error is against calibrated targets on completed wheels only; error
against nonactive locked snapshots is a separate metric. Thresholds
and timing are explicit; comparison with saved historical aggregate metrics
is indicative rather than an assertion of bit-identical evaluation protocols.

Training uses TaskAutoResetWrapper to restore phase, command, wheel snapshots,
action history and task timers alongside the reset pose. Brax episode metrics
and truncation remain untouched, and noise RNG continues across resets. The
smoke test crosses two episode boundaries and checks both task-state parity
and timeout flags. An earlier invalid launch is retained in attempt1 with an
explicit INVALID_RUN.md; no parameters were reused from it.

The approved second attempt changes only the abduction readiness bound, from
0.25 to 0.35 rad. Offline validation covers 76,840 newly admitted recorded poses;
with a 10 mm torso drop, largest randomized radius, ±5 mrad joint and ±10 mrad
roll/pitch perturbations, minimum clearance was 13.7 mm. The 5 mm physical audit
margin and 0.25 rad/s slew remain unchanged. See hybrid_results/attempt2_guard/
for the validation data and the separately retrained checkpoint. Evaluation
reads the guard from the checkpoint directory config; old configs without this
field retain 0.25 rad. Use --abduction_guard only for an explicit sensitivity check.
