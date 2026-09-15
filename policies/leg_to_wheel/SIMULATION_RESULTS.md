# Leg-to-wheel results

## Gentler lowering update

The current demo uses the same trained checkpoint with an inference-time easing
step. It limits only the lowering hip's downward target motion, starting at
8 rad/s and smoothly releasing over 0.2 seconds. The limit also fades between
40 and 20 mm capsule clearance and is fully released below 20 mm. Abduction,
knee/hub, support joints, and upward hip corrections remain responsive. Raw
policy actions remain in observation history; smoothing is applied at actuation.
Commands remain one-hot and the heating hold remains operator-paced.

| Demo measurement | Previous | Updated |
|---|---:|---:|
| Peak downward foot speed | 0.676 m/s | 0.498 m/s |
| Peak touchdown speed | 0.0378 m/s | 0.0378 m/s |
| Front-lift tilt | 6.43° | 6.43° |
| Height sag | 8.23 mm | 8.23 mm |

Peak descent speed decreased by 26.2%. The full sequence
completes with all demo gates passing and converted mask 15.

**Current validation: 195/195 nominal cases pass; 179/195 randomized cases pass.**
The previous checkpoint without easing passed 182/195 randomized cases. The
update has 11 soft-touchdown failures (previously 8), 2 front-tilt failures,
2 stance-contact failures and 1 sag failure. This tradeoff is retained in the
reports; the randomized acceptance target is still unmet. There are zero falls
in either updated suite. Two new motion-limit tests pass; training code and
weights were unchanged in this update.

Updated evidence in `training_runs/leg_to_wheel/`:

- `gentle_hip_nominal.json` and `gentle_hip_physics.json`: complete 195-case suites.
- `gentle_lowering_comparison.json`: previous/updated demo metrics.
- `gentle_hip_demo.mp4` and `.json`: the updated render and its measurements.
- `leg_to_wheel_before_gentle.mp4` and `.json`: preserved previous demo.

The default renderer reads easing settings from `leg_to_wheel_demo_run.json`.
Use `--lowering_joint_speed 0` to reproduce the previous motion. For evaluation,
pass `--lowering_joint_speed 8 --lowering_ease_seconds .2` to reproduce the current
inference settings. All thresholds and evaluation schedules are unchanged.

## Original checkpoint report (before easing)

### Original result and remaining gap

The implementation and GPU-trained policy are complete. **Full randomized
acceptance is not complete.** The selected checkpoint passes all 195 nominal
cases and 182 of 195 randomized-physics cases. All 390 cases have zero falls.
All 24 long-hold cases (60 seconds lifted per case) pass every gate.

The final `leg_to_wheel_demo.mp4` is a 26-second learned-policy rollout in the
requested **FL → FR → BR → BL** order. The policy drives all 12 joints. The
sequencer enforces actual 15 mm capsule clearance, waits for heating confirmation
(automatically supplied by this simulation demo after the hold), and waits for
stable touchdown before advancing. All four converted bits are set at completion.
Upstream reach decreases by 7.5 mm after each unloaded hold; thermal deformation
is not simulated.

| Measurement | Demo | Nominal suite worst | Randomized suite worst | Target |
|---|---:|---:|---:|---:|
| Front-lift peak tilt | 6.43° | 9.82° | 10.46° | <10° |
| Height sag | 8.23 mm | 8.23 mm | 11.64 mm | <10 mm |
| Touchdown speed | 0.038 m/s | 0.091 m/s | 0.204 m/s | <0.10 m/s |
| Drift | 39.0 mm | 39.0 mm | 49.1 mm | <60 mm |
| Falls | 0 | 0/195 | 0/195 | 0 |

The randomized failures are 8 touchdown-speed cases, 2 front-tilt cases,
2 stance-contact cases, and 1 full-cycle sag case. These are distinct failed
cases. Clearance and drift gates pass throughout. The nominal suite passes all
3 complete cycles, all 12 long holds, and all 180 mixed-stance cases. Randomized
physics passes 2/3 complete cycles, 12/12 holds, and 168/180 mixed-stance cases.

## Protocol and evidence

Both suites use seeds 0, 1, and 2. Each seed includes one progressive four-leg
cycle, four 60-second holds, and all support subsets of 0–3 converted limbs at
5 and 10 mm shortening. These seeds were used during refinement and are
validation samples, not an independent generalization test. Dynamics randomization
reuses walking friction, gains, mass/inertia/CoM, contact, torque and armature
variation. Stationary acceptance disables repeated external pushes, latency and
sensor noise, while retaining seeded reset perturbations. Training retains them.
Repeated-push robustness is not certified by these final reports.

Maneuver sag, drift and touchdown speed begin after the initial two-second
settling stand, matching the baseline; setup impacts are recorded separately.
Clearance and stance are checked after a one-second transition allowance.
The real touchdown threshold was never relaxed to obtain a pass.

Native MuJoCo and MJX were cross-checked on a failing randomized landing:
0.20381 versus 0.20404 m/s, both at 5.08 seconds. The remaining hard landing
is a policy limitation, not a disagreement between evaluators.

Evidence under `training_runs/leg_to_wheel/`:

- `refine0_nominal.json`: all 195 nominal cases, full per-case metrics.
- `refine0_physics.json`: all 195 randomized cases, including failures.
- `nominal_accepted_demo.json`: metrics for the delivered video.
- `scripted_baseline.json`: both scripted orders across three seeds.
- `mjx_landing_comparison.json`: native/MJX landing check.
- `verified-tests.log`: 12 passed, 1 optional integration test skipped.
- `verified-smoke.log`: separate CPU PPO integration run completed.

The CPU smoke uses only 64 steps and tiny normalization statistics; its large KL
warning is not a policy-quality measurement. Full GPU refinements had stable PPO
updates. The scripted front-first baseline reaches about 33° tilt and 52.5 mm sag,
with its last two legs timing out and clearance gates disabled.

## Checkpoint and reproduction

`leg_to_wheel_demo_run.json` points to the immutable selected weights and both
acceptance reports. The selected run is
`final_refine_0/leg_to_wheel_2026-09-14_21-41-08/params_000093716480`.
Its `run.json` records the exact config, model and ring hashes, runtime versions,
and parent weights; `sources/` preserves the training code. Its balance weights
are upright 20, height -8, support contact -5, touchdown -100, and drift -3.
Later margin/approach experiments remain archived and are not the selected policy.
The current training defaults include those later reward refinements; loading the
selected policy uses its recorded configuration.

From the worktree root:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONPATH=mujoco_playground   .venv/bin/python -m workspace.render_leg_to_wheel_demo
```

The original scripted video is preserved as
`training_runs/leg_to_wheel/scripted_demo.mp4`. Model XML and heating mesh changes
were pre-existing and were not edited or committed as part of this work. ROS2,
deployment export and hardware remain outside scope.
