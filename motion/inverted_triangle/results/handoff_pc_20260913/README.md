# PC handoff: entry timeout corrected, full acceptance fails

[W&B review: complete acceptance matrix, videos and artifacts](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/c4318b8974414955). All 78 experiment runs and the separate review run are independently verified online.

The saved diagonal 10 mm **entry timeout is corrected in a diagnostic configuration**. The motor-floor failure is not corrected: nominal cold geometry is infeasible at the recorded posture, and every tested handoff still loads a motor housing. Diagonal and combined formation also fail final four-tip support. No candidate is accepted for walking or hardware. No robot connection, heating, motor action, deployment, or calibration change occurred.

The requested branch was fetched at exact commit `4eb8260790b42829c960a9c47032b9c6b3b88c07`. Original assets, original nominal tip, approved additional **9 mm axial spacing**, all historical failures, and all existing acceptance thresholds are preserved. A new explicit cold-boundary geometry gate adds a diagnosis; none of the old gates was relaxed. The ordinary default entry remains the historical configuration; the failed diagnostic candidate is explicit in `frozen-candidate.json`.

[Continuous actual-endpoint entry/roll/settle video and dense integration artifact](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/6919a38d0564467b) shows the selected diagnostic, **not a passing motion**. Its 47.68 simulated seconds contain 24,794 physics steps, all independently replayed with maximum state error zero. Dense CAD checks 6,449 integrated states and reports a 3.160 mm minimum separation, without intersections. The CAD sampling is 52 Hz, refined to 520 Hz around contact changes and small gaps; it is not continuous-collision certification. The downloaded W&B video is byte-for-byte identical to the local file (SHA256 `d02ea96e83e813f8dbae793d20b99563fd14d9c310c361b0b47fc761caf01a25`).

**Saved failures reproduced before changes.** Nominal reproduced 25,548 steps and 68.105525 N motor-floor load. Diagonal reproduced the 6,239-step entry timeout and 5.900936 N load. Both were rendered and uploaded:

- [Unchanged nominal motor-floor failure](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/dec65a3f8445466c)
- [Unchanged diagonal entry failure](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/e29cd4d5c0e44c4c)

These first two reproductions and the parameter/contact screens explicitly omitted CAD and cannot count as engineering passes. The subsequent development, conditional validation, and parent regression runs include dense CAD and independent integration replay. Every failed trial remains indexed and has actual video in W&B Media plus configuration/source/integration artifacts.

**Cold geometry prevents a software-only solution from the nominal endpoint.** Detailed lower-housing CAD is 1.118 mm below the intended foot plane at ideal neutral full compression, and 1.898 mm below it at the saved actual alignment endpoint. A common root-Z translation cannot change this ordering: moving the housing above the floor also lifts every intended foot. This disagrees with the user's earlier approximately 5 mm physical clearance estimate. No dimension, mounting gap, motor contact, force threshold, or post-initialization state was changed to hide it.

A static pre-formation sweep finds a sampled 0.09 rad symmetric splay with at least 1 mm geometric clearance for the nominal saved proximal posture. This is only a geometric prerequisite, not a dynamically supported or measured endpoint. A supported posture must be established before manual formation, or the physical geometry discrepancy must be resolved. The present task does not authorize a powered check. Partially formed cases can clear the lowest-foot plane initially yet load another housing as the body settles; positive initial clearance alone is not proof of stable support.

**Entry comparison and limits.** Eight duration/integral trials using the old 0.29 rad splay still time out. The rear-right hip reaches its preserved −0.32 rad command limit and remains about 5.9 degrees from the reference; adding integral action cannot overcome that saturation. Direct 12 s and 20 s rolls finish their phases but fail floor load and final support. Three smaller-splay trials finish measured entry. The selected diagnostic uses 2 s nominal entry duration, 0.10 rad splay, and proximal-only integral gain 0.25 with a 0.15 rad cap. It first appears in the sampled roll phase at 3.702 s. Hub angles are held during entry, and the shared velocity/acceleration filter remains active across phases. The 0.1 rad / 0.12 rad/s entry gate, timeout, and all final gates remain unchanged.

The selected diagonal run still peaks at **5.799877 N** unintended floor force, and its rear-right tip ends **9.783 mm above the floor with zero load**. Final maximum pose error is 0.085675 rad, showing why pose tolerance is not a substitute for support. Static numerical final-stance searches retained all 24 attempts; best found tip-height spreads were 3.417 mm (diagonal) and 2.667 mm (combined) inside the 0.1 rad joint box. These are numerical diagnostics, not global infeasibility proofs or dynamic support evidence.

**Alignment provenance is a blocking dependency.** The intended historical deterministic keyframe source is `a7e3bb1e67f73b4fb14ee6658d66d0263fd2bfd8`; origin rejects fetching it with `upload-pack: not our ref`. Historical W&B artifacts contain endpoints/integration evidence but not the missing source. The published backpack source `dde1f961b519f53379d80fc4e77eab797b38be73` differs in controller, config, geometry, and model; differences are not merely Windows line endings. It was replayed unchanged under nominal seed 0, seed 1 with the existing 0.06 Nm synthetic hub-friction stress, and combined seed 5/friction/IMU bias. All three fail alignment with completion mask zero, and the handoff rejects every record. Their actual videos and terminal artifacts are indexed. They were never substituted as successful endpoints. The legacy v5 neural policy remains a separate, unvalidated upstream implementation.

Consequently, the new shape/friction/sensor/dynamics cases are **fresh conditional trials on one historical actual endpoint**, not fresh successful alignment endpoints. Calibrated phase errors are within the original 0.035 rad tolerance; terminal velocity and previous proximal position commands are exactly preserved. Hub torque/velocity outputs are never treated as position targets. The one-time post-cooling model boundary moves root Z to the lowest intended foot and preserves root XY/orientation, joint coordinates and velocities. This is not continuous wheel-to-cooled-polymer physics.

| Audit scope | Accepted | Interpretation |
|---|---:|---|
| Development matrix, actual historical endpoint | **0/8** | All complete entry/roll and dense CAD; all fail motor-floor load; diagonal/combined also fail four-tip support. |
| Fresh shape/friction/sensor/dynamics conditions | **0/8** | Conditional on the single historical endpoint; all fail motor-floor load. |
| Sensor-free baseline repeats | **0/2** | Exact dynamic parity with prior nominal/diagonal runs; not held-out cases. |
| Parent friction/seed regression | **20/20** | Parent fixed keyframe, not an actual-alignment handoff. |
| Parent dynamics/start stress regression | **16/16** | Parent fixed keyframe, not an actual-alignment handoff. |
| Published-source fresh alignment replay | **0/3** | All failed/rejected; unavailable historical-source endpoint distribution remains unvalidated. |
| Walking activation/history/fade, interface parity | **Not run** | Prerequisite entry-to-standing acceptance fails. |

| Development case | Peak motor/body floor load (N) | Final three-second tip support |
|---|---:|---|
| [combined_trial](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/f93978fc99234c0e) | 1.298160 | Fail |
| [diagonal_10mm_spread](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/6919a38d0564467b) | 5.799877 | Fail |
| [front_short_10mm](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/3043aeb06d5f4fa7) | 46.288063 | Pass |
| [left_short_10mm](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/574dabad94be4813) | 60.824678 | Pass |
| [nominal_endpoint](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/237c46235e9d438f) | 68.105526 | Pass |
| [pose_error_2deg](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/93563aed49894dc7) | 66.274847 | Pass |
| [rear_short_10mm](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/449dccfea36e466e) | 54.605864 | Pass |
| [uniform_undercompression](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/1d30ce2a75f0418e) | 48.033237 | Pass |

Across **78 experiments**, **1,991,186 environment steps** were recorded. W&B independently confirms finished runs, nonempty Media videos and committed artifacts for **78/78** experiment runs; the review run is verified separately. No failed runs were discarded.

Contact comparisons use the exact same deformed visual/CAD mesh, with 2, 4 and 8 convex floor-contact slabs, plus a zero-variation control against the original nominal hull. All preserve the failure outcomes. Maximum diagonal joint-trajectory differences reach about 0.0216 rad relative to two parts, so general contact convergence is **not established**. Global mass/inertia, COM, gains and delay are stressed in the conditional and parent suites; per-limb polymer mass redistribution remains unmeasured and unvalidated. There is no accepted result whose success can be claimed independent of those assumptions.

Sensor instrumentation was isolated in a second worktree while the original audits ran. Explicit sensor trials provide sample-held 52 Hz joint/IMU measurements to the 520 Hz controller, with noise, bias and delay. Contacts, actual shape and root position are audit-only. The historical ideal-sensor branch remains unchanged: nominal and diagonal baseline repeats are exactly equal in commands, gains, qpos, qvel and loads. The focused model/formation/entry/roll tests pass **37/37**.

The frozen walking-policy SHA256 is `854ac8ba4ffc305079b7f6f7b52187a211413c3cdb18f0de016dd819ff2450a8`. It was not activated because entry-to-standing acceptance failed. The real endpoint's right hubs retain an additional −2π winding; physical pose similarity does not establish the policy's raw observation/history mapping. Activation/history/fade, command/gain continuity at the upstream and walking boundaries, portable C++ parity, lifecycle/stop handling, and the reviewed continuous-coordinate robot adapter remain unvalidated. No motor-limit override or hardware readiness is implied.

Results and provenance are in `case-index.json`, `summary.json`, `cloud-verification.json`, and `all-audits-configs-and-provenance.zip`. Full integration arrays, dense samples, traces and videos remain in each indexed W&B run/artifact and locally under `/tmp/handoff-evidence-20260913`. The archive contains lightweight audits/configs/source records; it intentionally does not duplicate the large arrays. `drivers/` retains the exact orchestration and diagnostic scripts. Alignment used MuJoCo 3.6.0 / NumPy 2.4.6; handoff used MuJoCo 3.3.7 / NumPy 2.2.6. No neural training occurred. Dynamic search was a retained 13-trial CPU grid; separate static least-squares searches record evaluation counts and are not rollout acceptance.

To reproduce the selected diagnostic after fetching assets and installing `motion/inverted_triangle/requirements.txt`:

```bash
MUJOCO_GL=egl python -m motion.inverted_triangle.handoff_probe \
  --terminal motion/inverted_triangle/results/handoff_local_20260913/alignment-terminal-0.json \
  --config motion/inverted_triangle/results/handoff_pc_20260913/selected-diagonal-config.json \
  --output /ABS/NEW-RESULTS/selected-diagonal
```

Exit status 1 is the expected preserved failure. To resume actual endpoint-distribution validation, first make the historical alignment commit fetchable. Before any hardware work, resolve the cold geometry and pre-formation support prerequisite, then obtain a passing continuous entry/standing/walking simulation and reviewed interface parity. Physical geometry can be checked without powering motors.
