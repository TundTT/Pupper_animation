# Pupper V3 leg-lift — RL training

Training pipeline for the Pupper leg-lift behavior, built on this repo's MJX / brax
RL stack. One policy learns to raise a single commanded leg, hold it up while
balancing on the other three, and lower it when the command changes.

This `workspace/` is **our** code; the rest of `mujoco_playground/` is the upstream
library we build on (brax PPO, MJX env patterns; go1 `getup.py` is the closest
reference task). The trained policy deploys to the robot exactly like the locomotion
policy — an exported JSON MLP loaded by `neural_controller` (see the monorepo).

## Design

- **One policy, command-conditioned.** The policy observes a 5-way one-hot command
  = which leg is up (`stand`, `front_l`, `front_r`, `back_r`, `back_l`).
- **Stay in the standing pose; raise the commanded leg as high as it can.** The
  three planted legs track the home pose, the torso stays upright, at full standing
  height, and roughly where it started; only the raised leg is free, and it is
  rewarded on a ramp in foot clearance (higher = better, up to a cap) rather than
  against a fixed target pose. See "Reward design" below — this replaced an earlier
  formulation that taught the robot to lean and sit.
- **Hold is operator-timed, not baked in.** "Hold" is just the command staying
  constant, so the hold duration is however long the operator waits between button
  presses — no fixed duration in the policy, no retrain to change it. During
  training the command is held for a random window then switched, which teaches
  smooth raise/hold/lower transitions.
- **O-button state machine lives on the robot, not in the policy.** Each press of O
  advances a clockwise sequence (`stand → front_l → front_r → back_r → back_l → …`);
  the press lowers the current leg and raises the next by changing the command fed
  to the policy. The policy itself is order-agnostic — it only ever sees "which leg
  is up now."

## Reward design (and the bug it fixes)

Earlier runs produced a policy that *did* raise the commanded leg, but got there by
shifting the body backwards, sitting down, or setting another limb on the ground.
That was not a tuning problem — the reward was asking for something the robot could
not physically do, so leaning was the genuine optimum:

- The policy head is `tanh`-squashed, so actions are in (-1, 1) and the reachable
  joint range is exactly `DEFAULT_POSE ± action_scale`. With the old uniform
  `action_scale = 0.3`, **every joint was capped at 0.3 rad from home.**
- `target_foot_height = 0.08 m` needs roughly **1.2 rad** of hip rotation. Not
  reachable by leg motion — only by moving the body.
- `target_knee_height = 0.18 m` is **above the torso** (which stands at 0.156 m).
  Not reachable at all, by any means. That term (weight 2.0) paid the policy to
  pitch the whole robot over.
- `tracking_pose` simultaneously pulled the leg toward a fixed `LIFT_DELTAS` pose
  worth only 0.036 m of clearance, directly fighting the clearance term.
- `torso_height` targeted 0.14 m while the robot actually stands at 0.1556 m, so
  it was *rewarding sitting down*.

The fix has two halves:

1. **Actuation** — `configs.ACTION_SCALE` is now per-joint: abduction 0.5, hip 2.0,
   knee 1.0. The hip can now actually swing the leg up (~0.19 m of clearance at the
   limit, verified against the model), while abduction/knee stay tight so the leg
   lifts in its own sagittal plane and the stance legs keep fine control near home.
   The deployment side already accepts a 12-element `action_scale` array
   (`neural_controller.cpp`'s `set_param_from_json_mixed`), so this needs **no C++
   change** — `export_policy.py` writes the vector into the JSON.
2. **Reward** — nothing competes with anything else any more. The raised leg is
   driven *only* by a linear ramp in foot clearance saturating at
   `target_lift_height = 0.15 m` (constant positive gradient, so "as high as
   possible", with no incentive to contort past a useful height) plus a weak prior
   keeping its abduction/knee near home. Everything else describes *"still standing
   where it was"*: `stance_pose` (the planted legs hold the home pose),
   `stance_feet_contact`, `orientation`, `torso_height` (at the measured 0.1556 m),
   and `body_drift`. `ground_contact` penalizes any knee nearing the floor.

Posture is additionally protected by **termination**, not just weights: tilt > 0.4 rad,
torso below 0.10 m, any knee touching the floor, or the torso wandering more than
0.09 m from where it started ends the episode. That caps the payoff of cheating
regardless of how the weights are tuned.

### Two local optima this task falls into (both hit during tuning)

Worth knowing before changing weights, because both score *well* on
`eval/episode_reward` and are invisible unless you look at the diagnostics:

1. **"Stands beautifully, never lifts."** The posture terms are earned whether or not
   a leg goes up, so with `lift_height` alone the policy banked ~102 of a ~144 max at
   zero risk and never raised a foot (`lifted_foot_height` = -0.001 m, tilt 0.8°,
   drift 17 mm — a flawless statue). The cause was a **dead zone**: `lift_height`
   clips to 0 for every configuration where the foot still touches down, so nothing
   rewarded the first few degrees of hip rotation. Fixed by `lift_progress`, which is
   measured on the hip *angle* and therefore pays from the first degree.
2. **"Lifts well, but walks."** With the lift learned, the policy held the stance legs
   ~9° off home and translated the torso ~0.14 m (~13× the ~12 mm the physics needs).
   It would not come back from this on shaping alone — `body_drift` had already
   bottomed out, leaving no gradient to pull it in. Fixed with `terminal_body_drift`
   plus heavier `stance_pose` / `body_drift` and a tighter `stance_pose_sigma`.

3. **"Lifts three legs and quietly gives up on the fourth."** Adding the `heading`
   reward from scratch produced a policy that never lifted `front_r` at all (foot
   clearance -0.002 m, i.e. still on the floor) while lifting the other three fine.
   `front_r` generates the most yaw, so under heading pressure *not lifting it* was
   cheaper than lifting it — it kept all the posture reward and forfeited only that one
   command's lift. Fixed by **warm-starting** (`--init_params`) from a policy that
   already lifted all four, so heading control is refined onto an existing skill rather
   than competing with acquiring it. Watch per-leg foot clearance in
   `workspace/evaluate.py`, never the aggregate — the aggregate barely moves when one
   leg out of four is dropped.

The general lesson: **reward weights shape behavior inside the feasible set;
termination is what removes a strategy.** Every posture problem here was ultimately
solved by making the bad strategy end the episode, not by out-weighting it.

### Known residual limitation: yaw

The shipping policy still rotates roughly **-21 deg net over a 12 s, five-lift
showcase** (down from -44 deg before the `heading` term, but not zero). Back-leg lifts
are the main contributors. This is structural, not a tuning miss:

**The policy has no absolute heading in its observation.** It sees
`[ang_vel, gravity, command_one_hot, joint_pos, last_action]` — angular *velocity* is
there, so it can damp rotation, but accumulated *heading* is not, so it cannot steer
back to a reference it cannot perceive. The same is true of position, which is why
`body_drift` is a deadband. `heading` can only push the policy toward motions whose yaw
reactions cancel; it cannot close the loop.

Closing it properly means adding a heading signal to the observation, which is a
deployment-side change too (obs size 35 -> 36, so `neural_controller` would need a real
edit rather than a JSON swap) and needs an on-robot heading source — IMU yaw integration
drifts, so this deserves thought before anyone attempts it. **Do not "fix" this by
raising the `heading` weight**: that is what produced local optimum 3 above.

Note also that `evaluate.py`'s headline termination number counts the yaw limit, and a
yaw excursion is *not* a fall. Read the by-cause breakdown: the shipping policy is
**0.0% actual falls** (tilt / sat-down) across 256 randomized robots.

Two measured facts worth knowing before re-tuning:

- Lifting a **front** leg leaves the CoM ~12 mm *outside* the triangle of the
  remaining three feet, so a small body shift is **physically mandatory** — this is
  why `body_drift` is a deadband (free within `allowed_body_drift = 0.035 m`) rather
  than a pure penalty. Back-leg lifts are statically stable (+8 mm margin).
- The model has **no self-collision** (every robot geom is `contype=0`, so only
  robot↔floor collides). Sim will happily fold a leg through the body, which is why
  the lift is capped in joint space instead of letting PPO chase the 0.25 m the
  kinematics allow.

Episodes also now **start already standing** (the env settles the home pose once at
construction) instead of dropping the robot from the model keyframe's z = 0.28 and
letting it bounce — that landing slide alone moved the torso ~25 mm, eating most of
the drift deadband before the policy did anything.

## Files

| File | Purpose |
|---|---|
| `configs.py` | Joint order/limits, home pose, per-joint action scale, reward weights, PPO hyperparameters, model path. **Single source of truth.** |
| `leg_lift_env.py` | `PupperLegLiftEnv` (brax `PipelineEnv`, MJX): reset/step/obs/reward, command sampling. |
| `train.py` | brax PPO training entry; saves brax params to `output/<run>/mjx_params`. Optional W&B logging + rollout videos. |
| `visualize.py` | Rolls the policy out through the O-button sequence and renders a `tracking_cam` video — what you watch to judge the policy. |
| `export_policy.py` | Converts brax params → `neural_controller` JSON (normalization folded into layer 0). |

The Pupper MJX model is referenced in place from the `pupper_v3_description` checkout
(`pupper_v3_complete.mjx.position.xml`); nothing is copied across repos.

## Setup & run — on a CUDA workstation

Training needs a GPU. **Check `nvidia-smi` first rather than assuming which machine
you're on** — this has been run on more than one. The original RTX 5090 workstation
and the current host (2× RTX PRO 6000 Blackwell, 96 GB each) are both Blackwell
(sm_120), so either way use a **recent** JAX + CUDA 12 build.

```sh
cd mujoco_playground
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras            # installs the playground + brax
uv pip install wandb                        # optional, for --use_wandb
python -c "import jax; print(jax.default_backend())"   # -> gpu

# smoke test first (JITs the env, a few PPO iters, ~1 min) before a long run
python -m workspace.train --num_timesteps 200000 --num_envs 1024

# full run (run from the mujoco_playground/ dir so `workspace` is importable)
wandb login                                 # once, if using W&B
CUDA_VISIBLE_DEVICES=0 python -m workspace.train --use_wandb
# export the trained policy to the robot's JSON format
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```

`num_timesteps` defaults to 200M and `num_envs` to 8192 (fits the 5090's 32 GB with
room to spare on bigger cards); lower `--num_envs` if VRAM is tight. Pin
`CUDA_VISIBLE_DEVICES` on a multi-GPU host — brax would otherwise pmap across every
visible device, which brings its own `num_envs`/`batch_size` divisibility constraints
for no gain at this scale.

### Watching the policy

Every eval, training renders a rollout that steps the command through the O-button
sequence (`stand → FL → FR → BR → BL`) and logs it to W&B as `eval/video` (plus a
final `eval/video_final`). Videos are also written to `workspace/output/<run>/*.mp4`
regardless of W&B. Rendering is headless via EGL (`MUJOCO_GL=egl`, set in
`workspace/__init__.py` — it has to be set before anything imports `mujoco`, which is
too early for `train.py`; getting this wrong aborts the process rather than raising).
Flags: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--no_eval_videos` (skip the
per-eval video if it slows things down — the final video still renders).

Alongside the reward terms, these diagnostic metrics are logged so you can tell *how*
a policy is earning its reward rather than just how much:
`lifted_foot_height` (how high the commanded foot actually gets), `body_drift_dist`,
`torso_z`, and `tilt_deg`. **brax reports these as per-episode sums**, so divide by
`eval/avg_episode_length` to read them back as averages.

## Status / what still needs doing

- **The reward and action scale were redesigned (2026-08-10) so the robot lifts
  with its LEG instead of its whole body.** The previous policies raised the
  commanded leg by shifting backwards, sitting, or planting another limb — the
  genuine optimum of a reward that asked for clearances no joint could reach.
  See "Reward design (and the bug it fixes)" above for the full diagnosis and the
  measurements behind the new numbers. `LIFT_DELTAS` / `lifted_pose_for()` are
  **gone**: there is no fixed lifted-pose target any more, so there is nothing left
  to "tune" there, and the reconstructed-from-wandb `knee_clearance` term that came
  with them is gone too. Consequently the recovered weights from the lost run
  `leg_lift_2026-06-24_20-04-18` are no longer in use, and `eval/episode_reward` is
  **not comparable** across this boundary (the old scale topped out near 96/episode,
  the new one near 144).
- **Earlier runs' artifacts are kept as a record, not as a baseline.** Anything
  under `workspace/output/leg_lift_2026-07-22_*/` predates the redesign.
- **Retrained on this checkout, and this time actually compile/run-tested**
  against the real deployment stack (a user-space ROS2 Jazzy install via
  RoboStack, since this workstation has no sudo — see below). First attempt
  (wandb run `leg_lift_2026-07-22_18-07-33`, `activation="swish"`) trained
  fine (`eval/episode_reward` ~51) but **would have crashed on the actual
  robot**: the vendored RTNeural in `neural_controller` only implements
  `tanh`/`relu`/`sigmoid`/`softmax`/`elu` activations, and a `"swish"` layer
  silently produces a null layer that segfaults on load. Fixed by switching
  `policy.activation` to `"elu"` (matching the already-deployed locomotion
  policy) and retraining — run `leg_lift_2026-07-22_21-35-05`, same 150M
  timesteps, `eval/episode_reward` ~51.7. Params, exported JSON, and eval/final
  videos for **that** run are committed under
  `workspace/output/leg_lift_2026-07-22_21-35-05/` (and deployed to the
  monorepo's `neural_controller/launch/`); the earlier swish run's artifacts
  are left in place under `.../leg_lift_2026-07-22_18-07-33/` as a record but
  should not be deployed.
- **Compiled and activated successfully** against `pupper_mujoco_sim` (real
  `colcon build`, real RTNeural inference loop, no crash) after also fixing
  two unrelated pre-existing repo bugs: `pupperv3_mujoco_sim`'s vendored
  `libmujoco.so` symlinks (`lib_x86` and `lib_arm`) were committed as plain
  text files containing their target's name instead of real symlinks, which
  broke both the build (missing rpath/install step for the real `.so`) and
  would have broken this on the Pi 5 too.
- **The deployed policy's stable ~26° body tilt was mostly the trained behavior,
  not (only) an actuator gap.** Commanding `front_l` via `/leg_lift_command_index`
  in `pupper_mujoco_sim` converged to a ~26° lean rather than a clean lift, which
  was previously attributed to the sim-to-real actuator-fidelity gap (training uses
  the idealized `pupper_v3_complete.mjx.position.xml`; the sim's hardware interface
  drives the torque-motor `pupper_v3_complete.backlash.xml`). That gap is real, but
  it is no longer the leading explanation: **leaning is precisely what that policy
  was trained to do**, since with `action_scale = 0.3` its leg physically could not
  reach the commanded clearance. Re-measure the gap against a policy from the new
  reward before spending effort on torque-model-aware domain randomization.
- **Still to do:** re-export and re-validate on `pupper_mujoco_sim` against the new
  policy, and confirm the per-joint `action_scale` vector round-trips through
  `neural_controller` (the C++ supports it, but no run has exercised the array form
  yet — every policy shipped so far wrote a scalar).
- **`action_scale` array form confirmed working on real hardware (2026-08-17/18).**
  `leg_lift_2026-08-11_01-35-11` deployed and tested on the physical robot across two
  sessions (see `Stanford/pupperv3-monorepo/LEG_LIFT_TESTING.md`'s test log for the raw
  notes). `front_l`/`front_r`/`back_l` lifted cleanly; `back_r` failed the first session
  from a cracked 3D-printed leg link (hardware fault, not policy), and improved after a
  reprint but still visibly struggles more than the other three legs.
- **Two sim-to-real gaps to address in the next training pass, from watching that
  hardware testing:**
  1. **Real robot's center of mass sits further back than the sim model assumes.**
     The back legs — `back_r` especially — visibly struggle to lift and hold balance on
     hardware in a way the sim rollout doesn't show. This reads as a genuine model
     mismatch, not just noise: **the next policy should train against a model with the
     CoM shifted backward** (or otherwise corrected to match the real robot's actual
     mass distribution) rather than assuming the current MJCF's CoM placement is
     accurate.
  2. **The policy snaps to the commanded lift target immediately and only then seems to
     work on balance**, rather than raising the leg gradually while continuously
     re-balancing. A policy that ramps into the lift instead of snapping to it is
     expected to be more robust sim-to-real. **Worth trying for the next policy:** push
     `action_rate` (or a similar rate/smoothness term) more aggressively, or add an
     explicit curriculum/reward shaping that rewards a slower approach to the target
     pose instead of an instantaneous jump — the goal is "slowly lift while balancing"
     rather than "snap to position, sort out balance after."

## Deployment (monorepo side)

The exported JSON carries `observation_layout`, `command_states`, and
`button_sequence` metadata. To run it on the robot:

1. Copy the JSON into `neural_controller/launch/` and add a third controller instance
   in `config.yaml` with its `model_path` (mirror `neural_controller_three_legged`).
2. Spawn it in `launch.py` and add it to `joy_util_node`'s `controller_names`.
3. **New integration work:** feed the policy its command. Unlike the locomotion
   policy (driven by `cmd_vel`), this one needs a small state machine that, while the
   controller is active, increments the command index on each O press and supplies
   the one-hot to the observation. This is the main on-robot task and does not exist
   yet — `neural_controller`'s observation builder currently has no command-of-this-shape.
