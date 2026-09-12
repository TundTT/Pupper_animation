**Review of v3 alignment run 17d412eac8004407**

Run: [align-motion-v3-seed0-20260910T211747Z](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/17d412eac8004407). Source commit: `8be7fd378c05b0dba95aabe1dea6ea981de721e7`. Finished at 50,790,400 environment steps. The checkpoint, all three audits, training history, final video and rollout CSV were downloaded and reviewed. The first 64 seconds were also replayed on CPU with the recorded checkpoint, locked dependencies and all 37 source hashes verified. No learning or robot deployment occurred. The shared robot-code checkout was left unchanged.

**Result: the rear wheels work much better, but the front support/lift behavior remains unreliable.**

| Audit (64 environments each) | FL completed | FR completed | BR completed | BL completed | All four completed | Worst contact approach speed |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 0 | 2 | 64 | 64 | 0 | 0.219 m/s |
| Randomized | 4 | 8 | 47 | 61 | 1 | 0.269 m/s |
| Interrupted | 2 | 8 | 50 | 63 | 0 | 0.256 m/s |

All three audits fail. The speed limit is 0.10 m/s. Nominal minimum conservative wheel gap is 2.54 mm against a >5 mm audit requirement; randomized minimum is -0.082 mm, indicating overlap of the conservative envelopes, not proof of physical wheel penetration. Nominal and interrupted trials all survived; one randomized trial terminated. No audit reported a fall or unsafe rotation, but that indicator covers enabled rotation only and does not establish safety during lifting/lowering.

**What the actual final video/CSV shows**

- 2-32 s: front-left stays in LIFT, never enables rotation, and never gets even one gate count. The next timed command cancels it and lowers it during 32-36 s.
- About 40.1-59.2 s: front-right rotates intermittently. Rotation is enabled in 127 of 249 recorded ROTATE samples, approximately 51% duty at the CSV's 13 Hz sampling rate.
- About 59.2-64 s: front-right enters VERIFY but rotation is enabled in 0 of 63 recorded VERIFY samples. It briefly reaches an error below 0.035 rad, then loses clearance/settling conditions. The timed stand command at 64 s cancels it; it is not a verified completion.
- Back-right subsequently completes at about 78.5 s; back-left completes at about 118.5 s. The video ends with two completed wheels.

The nominal audit spends a mean 43.0 seconds with the floor gate blocked, versus 4.3 seconds blocked on stability and 0.12 seconds blocked on wheel spacing. Reasons may overlap. The front-right nominal median final error is only 0.0416 rad despite 2/64 completions: getting near the angle is not the same as staying lifted through verification and completing the lower.

**Additional CPU replay: why the front gates fail**

The source-matched CPU replay reproduces the same front-leg failure pattern. CPU/GPU physics is not bitwise identical: for example, the front-right angle error at 63 s is -0.055 rad on CPU versus -0.062 rad in the original GPU CSV. The following geometric measurements are from the diagnostic CPU replay, not retrospective measurements from the original video.

During the front-left stall (10-30 s), median actual front-left wheel clearance is -0.58 mm, consistent with contact and simulation penetration tolerance. The back-right wheel is instead about 4.2 mm above the floor. The conservative encoder/IMU floor estimate is -4.64 mm. Wheel spacing is 18.7 mm and body tilt is 0.081 rad, so neither wheel spacing nor the 0.12-rad tilt gate explains this stable stall. It is the wrong support distribution: the requested front wheel remains loaded.

The policy also reduces the intended front-left lift. Its median active-hip action is +0.899; with nominal -0.85 rad and a +0.12-rad residual scale, this commands approximately -0.742 rad. The measured joint settles near -0.755 rad. The previous larger residual range made a feasible pose available, but also allowed the policy to partially undo the lift.

During front-right VERIFY (60-64 s), median actual front-right clearance falls back to -0.58 mm while the back-left wheel reaches about 13.2 mm clearance. The median estimated active floor margin is -13.7 mm. This is not just a conservative-estimator false alarm: the requested wheel really comes back to the ground. The actor changes its balance outputs across the ROTATE/VERIFY phase transition even though both phases need essentially the same lifted support posture.

**Specific design weaknesses**

1. The nominal reference moves mainly the active hip while PPO must discover the coordinated support shift. A single feasible front-left static-action test did not establish a robust nominal trajectory for all four wheels. The prior fix addressed missing task rewards but left this difficult discovery problem largely intact.
2. The training support-contact reward uses `clear < .006`, so a wheel hovering 4.2 mm above the floor can still earn contact credit. That tolerance rewards the stalled front-left pose as if all intended supports were planted. Actual simulator contact/load should distinguish support from hovering; this can remain a training-only signal. [env.py](https://github.com/TundTT/Pupper_animation/blob/8be7fd378c05b0dba95aabe1dea6ea981de721e7/training/wheel_align/env.py)
3. ROTATE and VERIFY are distinct actor observation states. The reference remains lifted, but the residual policy can redistribute support when the phase changes. Verification also requires continuously enabled rotation and settling; losing clearance prevents completion. This is consistent with the observed front-right failure, although the phase encoding's causal contribution still needs an ablation. [contract.py](https://github.com/TundTT/Pupper_animation/blob/8be7fd378c05b0dba95aabe1dea6ea981de721e7/training/wheel_align/contract.py)
4. Fixed 32-second slots cancel incomplete operations. Gate chatter stretches front-right's rotation, then the timer cancels verification. A longer slot is a useful diagnostic, but cannot solve front-left's static ground contact. [evaluate.py](https://github.com/TundTT/Pupper_animation/blob/8be7fd378c05b0dba95aabe1dea6ea981de721e7/training/wheel_align/evaluate.py)
5. Slower joint commands alone do not bound wheel touchdown speed when the whole body transfers load. The impact cost is integrated over control time, while completion gives a one-time 60 reward bonus; brief hard contacts remain an affordable trade. At 0.22 m/s, the impact term costs about 0.56 in one control interval, before other reward terms. The strongest sampled impact in this video occurs during back-right lowering at about 75.1 s; another occurs during front-right VERIFY. Not every impact is the explicitly lowered wheel. [env.py](https://github.com/TundTT/Pupper_animation/blob/8be7fd378c05b0dba95aabe1dea6ea981de721e7/training/wheel_align/env.py), [rewards.py](https://github.com/TundTT/Pupper_animation/blob/8be7fd378c05b0dba95aabe1dea6ea981de721e7/training/wheel_align/rewards.py)

**Recommended next revision**

First build an explicit, smooth support-shift -> unload -> lift -> hold -> lower reference for each wheel. Validate dynamic trajectories, not only static poses, with three intended supports planted and useful clearance reserve. Search for 15-20 mm clearance where feasible, then test manufacturing/dynamics variation; keep existing safety thresholds. Use the confirmed current XML, not an assumed mirrored mechanism. The policy should provide bounded balance corrections around those references, with active corrections unable to cancel the required lift. This retains the current eight proximal commands and four hub servos.

Train the hard parts separately before long sequences: single-wheel unload/lift/hold, sustained holding while the hub turns, verification and gentle lowering, then interrupted/full sequences. Balance leg sampling and require per-leg success so the rear legs cannot conceal front failure. Include simulator contact/load and individual clearance deficits in rewards. Keep onboard inputs encoder/IMU/supervisor based; no new force sensor is implied.

Maintain a continuous support posture through ROTATE -> VERIFY, for example by sharing the policy's holding-state representation and blending residual commands across supervisor transitions. Do not bypass clearance checks just to permit verification. Make normal evaluation completion-driven with a generous timeout, and test user interruption separately. Temporarily extending only the front-right timeout can distinguish slow completion from a permanent verification stall without retraining.

Treat touchdown as a separate controlled phase: slow the final wheel approach in Cartesian clearance, preserve the support stance through load transfer, and penalize/report peak impact events per wheel and phase. Then gradually release residuals/return to stand. This addresses body-induced impact that joint-rate caps missed.

Training was still improving near the end: mean completion events rose from 1.06 at 40.6M steps to 1.69 at 45.7M and 2.00 at 50.8M. More training may help, but another unchanged full run would retain the contact-reward and support-transition weaknesses. Prefer the targeted revision and staged validation first. Do not flip the pi calibration convention: rear wheels converge correctly, front-left never rotates, and front-right reaches the target vicinity.

The existing encoder/IMU interfaces and actuator arrangement are documented in the [robot-info hardware contract](https://github.com/TundTT/Pupper_animation/blob/b3117008170c9fef7b3d0874be78b7b3b6bd50de/robot_info/HARDWARE.md) and current [robot-code hardware description](https://github.com/TundTT/Pupper_animation/blob/582fd88ccce51d922e38cfd938d26bb5181bb276/ros2_ws/src/pupper_v3_description/description/components.xacro). The user confirmed the current geometry for now and requested manual heating. These recommendations concern motion only.

W&B's `training_status` and `handoff_status` still contain the old text "audits pending", but `simulation_audit_status=failed` and all three uploaded audit JSON files are present. The pending text is a stale logging label, not missing audits. `eval/episode_completed=7227` is an accumulated count over timesteps, not 7,227 successes; `completed_event=2` and the audit completion counts are the meaningful success measures.

Local evidence: `artifact/audit-*.json`, `artifact/videos/policy-50790400.trace.csv`, `front-diagnostics.csv`, `history.json`, and `rollout-contact-sheet.jpg`. The original final MP4 is under `media/videos/policy/`. No source changes or new training run were made during this review.
