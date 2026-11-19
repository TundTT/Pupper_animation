# seq_d walking policy — provenance

The walking policy shipped to `robot-code` as **63820b0**
(`ros2_ws/src/neural_controller/launch/policy_walk_v2.json`, sha256
`ddbe02391a53ce0249b340c15f1324cb33ec85aacbb05c2d384da4193bc049a2`).

| | |
|---|---|
| run | `training_runs/reference_2026-09-14_06-51-00` |
| W&B | https://wandb.ai/QuadMorph/Walk/runs/g8ndfhn3 |
| checkpoint | `mjx_params` here, sha256 `73a91d1846f93f127adeaa6cfbd8fe2cec0d05450d09548992d23a1aadf97ba5` |
| warm-started from | seq_c, `training_runs/reference_2026-09-14_05-28-37` (itself from FIN_A, `reference_2026-09-14_03-53-58`) |
| model | `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml` at this commit (backpack pre-fused into `base_link`, 9 mm hub gap) |
| steps | 200M |

## Reproduce the shipped export

Re-exporting from the checkpoint in this directory reproduces the shipped JSON
byte-for-byte (verified 2026-09-14):

```sh
cd mujoco_playground
.venv/bin/python -m workspace.export_reference_policy \
  --params ../trained_policies/walk_seq_d_2026-09-14/mjx_params \
  --out policy_walk_v2.json \
  --vx_range -0.35 0.35 --vy_range 0.0 0.0 --wz_range -2.0 2.0 \
  --source_run reference_2026-09-14_06-51-00
# -> sha256 ddbe02391a53ce0249b340c15f1324cb33ec85aacbb05c2d384da4193bc049a2
```

`--vx_range` **must** be given explicitly as `-0.35 0.35`. `train_reference.py`'s
`lin_vel_x_range` default was retuned to ±0.18 after this run, so exporting without
the flag would stamp a command envelope the policy never trained on.

## Retrain

```sh
cd mujoco_playground
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m workspace.train_reference \
  --reward_shaping shaped \
  --warm_start_params_path ../training_runs/reference_2026-09-14_05-28-37/mjx_params_2026-09-14_05-28-37 \
  --num_timesteps 200000000 --learning_rate 5e-5 \
  --w_knee_clearance 2.0 --w_foot_clearance 0.6 \
  --w_drag -0.3 --w_tip_vertical -0.3 --w_touchdown_land -0.3 \
  --w_multi_swing -0.6 --w_ring_clash -1.5 --ring_clearance_m 0.100 \
  --w_over_lift -40.0 --over_lift_m 0.020 \
  --target_clearance_m 0.018 --target_knee_height_m 0.068
```

Two caveats for an exact replay:

- seq_d trained with `lin_vel_x_range` ±0.35 and `tracking_sigma` 0.25, which are no
  longer the defaults. Pass `--tracking_sigma 0.25` and set `lin_vel_x_range` back to
  ±0.35 in `build_config`, or the run will not match.
- seq_d trained with foot-capsule DR **56–61 mm**; the default is now 56–60 mm, which
  is the spec. The 1 mm difference is why this is recorded rather than silently fixed.

## Measured behaviour (`workspace/evaluate_gait.py --env reference`, 10 commands × 5 seeds)

See `eval_seq_d.json`. Summary, against the previous best FIN_A:

| metric | FIN_A | seq_d |
|---|---|---|
| foot clearance | 29.5 mm | 15.3 mm |
| knee height | 83.3 mm | 76.5 mm |
| touchdown speed | 510 mm/s | 334 mm/s |
| foot slip | 53 mm/s | 49 mm/s |
| front/rear gap | 132 mm | 74 mm |
| stride frequency | 1.31 Hz | 2.28 Hz |
| duty | 0.572 | 0.579 |
| falls | 0 | 0 |

## Known defects (not fixed)

- **Ring interpenetration ~24 mm.** 74 mm measured vs the 98 mm at which the 48.8 mm
  TPU rings touch. Simulation cannot see this: the foot collision geom is a 12 mm
  capsule and the ring is a visual mesh with no collision geometry, so rings pass
  through each other in sim while they would strike on hardware. Governed by
  `gap ≈ 150.5mm − v·duty/f`, i.e. bounded by commanded speed, not by reward tuning.
- **Clockwise turns bound** (3.14 Hz) where counter-clockwise walks (1.55 Hz).
  Tracking is correct in both directions.
- **Forward/back read as two-legs-together**, not a true single-support walk.
  `duty` 0.58 where a walk needs ≥0.75. Raising duty worsens ring clash at constant
  speed, so a true walk needs a speed cap near 0.15 m/s — and `tracking_sigma` must be
  rescaled to ~0.04 when doing that, or the velocity reward flattens and PPO freezes
  (this cost two full runs, seq_f and seq_g).

## Artifacts NOT in git

`training_runs/reference_2026-09-14_06-51-00/` is 38 MB (Orbax checkpoints for every
20 M steps, plus per-checkpoint rollout videos). Only the final `mjx_params` is
committed here. The rest stays on the training host and in the W&B run above.
