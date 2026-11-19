# Supported wheel alignment: diagnosis and corrective experiment

Front-leg lifts are feasible with active stance compensation. The failed test
was a **fixed, uncompensated home stance**, not a limit on what a policy or the
robot can do. This distinction matters: earlier fixed-torso forward kinematics
also overstated clearance by preventing the body from tipping toward a lifted
front wheel. The tests below include free-body dynamics and compensating motion.

## What the hardware-verified leg-lift implementation actually does

Read through `git show master:mujoco_playground/workspace/leg_lift_env.py`,
`configs.py`, `README.md`, and `master:CLAUDE.md` from the combine worktree.
No other worktree files or processes were accessed.

- All **12 position actuators** are policy-controlled. The nine stance-leg joints
  remain available for balance; they are not frozen at home. Per-leg action ranges
  are abduction ±0.5 rad, hip ±1.6 rad, knee ±1 rad around home. Those are allowed
  ranges, not a claim about exact successful-policy excursions.
- `_get_reward` applies a **soft** home-pose reward to stance joints:
  `exp(-sum(stance_joint_error²)/0.15)`, weight 3.5. It excludes the active leg.
  Stance-foot contact (1), upright body (2), torso height (1.5), and heading (2)
  preserve support while allowing adjustment.
- There is **no explicit diagonal-shift target, CoM-direction reward, or scripted
  12 mm translation**. A radial **35 mm drift deadband** permits the necessary
  motion. Beyond it, `exp(-drift_excess²/0.0025)` loses reward (weight 2.5).
  The history measures ~12 mm outside the front-leg support triangle, versus
  ~8 mm inside for back legs. The policy learns coordinated joint offsets that
  move the torso/CoM onto the remaining contacts.
- Dense hip progress (3) and actual foot height (8) provide the incentive to do
  this. Command one-hot, IMU angular velocity/gravity, joint offsets, and previous
  actions provide feedback. There is no direct CoM observation.
- Only the lifting/lowering hip is rate-limited; stance joints retain balance
  authority. Its real knee provides a third posture joint per leg. In wheel form
  that actuator instead holds hub angle, leaving abduction and hip for posture.

Thus “hold home pose” is a regularized stance with active compensation. It is not
an exact, mechanically locked posture. The same principle works in the wheel model.

## Evidence from the 12M pilot

Run `xsqrtn5j`, output `wheel-align_2026-09-07_18-02-52`, finished at 12,124,160
steps. Its final reward was +275, but the full-sequence audit had 0/16 successes
and 0.1875 completed legs per episode. Holding error conditional on actually held
samples was already small (0.01094 rad in PPO evaluation). The large historical
`hold_err` mixed failed LOWER states with true holds and was misleading.

The final checkpoint was replayed on the same 16 randomized models, with a trace
of command, phase, true clearance, load, wheel error/speed, and requested/actual
hip lift. `output/align_diagnosis/pilot_stages.json` records:

| Leg | Selected time in LIFT | Clearance p95 | Load median | Verified rotations / 16 | Completions / 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| front_r | 99.03% | 0.136 mm | 10.33 N | 0 | 0 |
| front_l | 99.43% | 0.540 mm | 11.04 N | 0 | 0 |
| back_r | 90.04% | 2.619 mm | 5.83 N | 1 | 1 |
| back_l | 88.61% | 2.029 mm | 8.72 N | 2 | 2 |

The main bottleneck was **unloading and sustaining clearance**, before angular
settlement. Front-left's median requested hip offset was actually -0.435 rad in
the lifting-sign convention: the unconstrained policy often pushed downward.
Mirrored mechanical tests successfully lift both fronts, ruling out an inherent
front-left incapability or a necessary sign reversal.

The old align drift penalty already had a 20 mm deadband. It did **not** penalize
all movement. However, the eight independent posture actions had to discover a
coordinated shift while competing with home-pose shaping and a poor task incentive:
the final reward included **+327.4 from holding**, largely pre-completed reset legs,
versus **+0.0225 from new completions**. The completion bonus was incorrectly weak
for an event because it was multiplied by dt: weight 8 meant only 0.16 per success.
The policy learned preservation and drift reduction without learning the sequence.

The single 3 mm threshold also made intermittent unloads fragile: dipping below
it reset rotation. This was secondary to the near-zero front clearance. PPO KL
spiked severely (including 1.04e10 in the first interval, 62.5 in another); simply
running this optimization longer was not a sound convergence strategy.

## Feasibility with compensation included

Native MuJoCo, free torso, actual wheel geometry, position PD, gradual hip motion:

| Front-leg test (same 0.7 rad lift command) | Wheel clearance | Wheel normal load | CoM margin inside actual stance-contact support area |
| --- | ---: | ---: | ---: |
| front_r, home stance unchanged | -0.577 mm | 3.384 N | -10.933 mm |
| front_r, coordinated stance compensation | 5.807 mm | 0 N | +4.048 mm |
| front_l, home stance unchanged | -0.576 mm | 3.384 N | -10.933 mm |
| front_l, coordinated stance compensation | 5.807 mm | 0 N | +4.048 mm |

Compensated torso displacement was about (-17.56, ±9.44) mm, magnitude 19.9 mm.
This is within the reference leg-lift's 35 mm allowance. Back legs cleared
4.560 mm with a 0.5 rad hip command and no preparatory offset. See
`output/align_diagnosis/compensated_statics.json` and `support_candidates.json`.
The simpler geometry-center triangle calculation at home gave ±9.22 mm margins;
the table above uses actual force-bearing contact points after motion, rather
than conflating those two measurements.

A complete nominal **MJX** sequence, with arbitrary wheel phases, actual command
latency, and zero learned correction, then completed **4/4**, no falls, ~4–5 mm
peak active clearance, drift p95 **21.7 mm**, max **25.4 mm**. Nominal smoke disables
pushes; the separate randomized test below restores the full disturbance settings.

## Adaptation to alignment

The learned policy now has **three outputs**: clearance trim ±1 mm around a 5 mm
gap, plus two stance corrections ±12 mm. Named wheel-geometry Jacobians map the
latter into coordinated abduction/hip offsets about wheel-driving home. Front
leg reference shifts were measured in the wheel model: front_r (-10,-10) mm,
front_l (-10,+10) mm in its nominal body frame; back legs use zero baseline.
These are reference coordinates for the differential kinematic map, not a claim
that the torso follows those numbers exactly. They were tested mechanically,
not copied as a nonexistent diagonal target from leg-lift.

The support adjustment ramps before/during lifting. A clearance feedback loop
adds a bounded, gradual reference to **only the selected hip**. It regulates the
actual randomized cylinder gap, not a fixed joint height. References lower to
home after verified rotation, retaining angle PD. Balance corrections remain
available during lowering/reweighting. Other joints have only the coordinated
stance adjustment; they no longer explore eight unrelated offsets.

Body drift is free within **35 mm while lifting/returning**, and **20 mm in idle**,
always relative to the original episode origin. No button re-anchors it. The
stance tracking cost follows the permitted compensated reference, so it does not
oppose the very support adjustment needed. Idle returns to home. Raw drift and
outside-20-mm diagnostics remain logged for comparison.

Holding now has an error cost and no large positive pre-completion bonus. New
completions earn **5 each without dt scaling**. Eleven simple running terms cover
survival, lift/lower tracking, holding error, scrubbing, support, compensated
stance, tilt, drift, action changes and ground proximity. No reward requests a
12 cm lift. Rotation enters at 3 mm clearance and stays enabled down to 1 mm only
while contact-free, providing hysteresis without permitting grounded rotation.

Wheel angle goals, one-time calibration, wrapped PD, mixed-drive isolation,
randomized radii, full external-button sequences, partial starts, idle holding,
torque limits, broad friction/mass/CoM/gain randomization and pushes remain.

## Verification and training decision

On 16 randomized robots with **all production pushes/randomization enabled**,
the zero-correction controller achieved:

- 5/16 full four-leg successes, versus 0/16 for the trained pilot;
- 3.25 completed legs per episode, versus 0.1875;
- front_r 11/16 and front_l 12/16 completions;
- 16/16 survival, held-error p95 0.0191 rad, drift p95 42.5 mm.

This supports the diagnosis that support preparation, action structure, and the
task incentive were the main problems. It does not establish full robustness:
some rotations still fail under randomization and drift remains above target.
The `reached_rotation_*` fields in the first standalone randomized JSON were
produced before correcting that metric to exclude aborted LOWER states; use its
completion/verified counts above. The training evaluator has the corrected metric.

CPU smoke tests pass for random phases, finite rewards, 376-observation/3-action
shapes, selected-hip routing, gradual return, active/idle drift bands, stage guards,
actual randomized radii, PD torque through ±50 turns/seam crossings, full task
resets, and wheel-drive regression. GPU evaluator/EGL rendering pass. A separate
test verifies zero-mean actor outputs and initial std 0.2 for arbitrary inputs.

Training starts from this tested zero-mean correction policy rather than random
mean offsets. It remains ordinary tanh-Gaussian Brax PPO with 128×128 networks;
only final-layer initialization changes. The **8M-step refinement budget**, 4096
envs, lr 5e-5, two updates per batch and five evaluations is smaller because the
mechanism already works before RL. Smaller exploration, lower-magnitude rewards,
and gentler updates address the observed KL instability. Every evaluated and best
checkpoint remains available. Earlier runs are preserved.

Reproduction tools: `diagnose_wheel_align.py` replays the frozen pilot source;
`probe_align_support.py` and `search_align_support.py` derive/test support offsets;
`smoke_wheel_align_supported.py` verifies the complete compensated nominal sequence.

The refinement run **finished cleanly at 8,192,000 steps**. Launch PID was
**2999063**, GPU 1, detached with PPID 1; log
`mujoco_playground/train_wheel_align_supported.log`; output
`workspace/output/wheel-align_2026-09-07_19-04-12/`; W&B
[9xcraanb](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/9xcraanb).
The exact launch command and source snapshot are saved in that output directory.

The initial policy reproduced 5/16 randomized full successes and a 4/4 nominal
showcase. After an initial dip at 2M, the selected **4,096,000-step best checkpoint**
reached **6/16 full successes**, **3.3125 completed legs**, **16/16 survival**, front_r
11/16 and front_l 13/16. Held-error p95 was 0.0205 rad and drift p95 50.2 mm.
Final 8.192M also reached 6/16 full successes (front_r 12/16, front_l 13/16), with
3.1875 completed legs and 49.7 mm drift p95. Both final nominal showcases finished
4/4; the final-video seed had 28.3 mm drift p95 and 30.0 mm max.

The best parameters are in `mjx_params_best`, with scores in `best_checkpoint.json`.
The first PPO interval still had a large transient KL spike; later intervals
settled to ~0.004–0.006 and the final value was 0.00561. This warrants attention
in future initialization/normalization work, but did not persist through the run.
Reported PPO training wall time was 78.6 seconds; end-to-end execution was about
11 minutes including initialization, evaluation and six rollout videos.

The dominant front-leg failure is corrected and actual full sequences now succeed.
The small fixed-seed audit does **not** establish robust completion across domain
randomization: remaining failures still include unloading/rotation under difficult
support conditions, and drift exceeds the desired band in some episodes. No
additional training was launched beyond this bounded experiment.
