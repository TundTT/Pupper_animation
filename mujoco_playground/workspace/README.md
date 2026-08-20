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
- **Two sim-to-real gaps identified from that hardware testing, and a third
  (whole-body oscillation while a leg is held up) — all three addressed in the
  training run started 2026-08-18, run `leg_lift_2026-08-18_15-09-43`:**
  1. **Real robot's center of mass sits further back than the sim model assumes.**
     The back legs — `back_r` especially — visibly struggle to lift and hold balance on
     hardware in a way the sim rollout doesn't show. Addressed by re-centering and
     widening `randomize.py`'s `body_com_x_shift_range` backward (torso local +x =
     forward) from the old symmetric `(-0.03, 0.03)` to `(-0.07, 0.02)`, plus a modest
     widen on y/z, so training samples a CoM distribution biased toward what the real
     robot appears to have rather than assuming the MJCF's placement is accurate.
  2. **The policy snaps to the commanded lift target immediately and only then seems to
     work on balance**, rather than raising the leg gradually while continuously
     re-balancing. Fixing this with a body-wide action-rate penalty was rejected: it
     would also cap the 9 stance-leg joints, which are exactly what needs fast
     authority to correct balance while a leg is in the air — see point 3 below, and
     the concern the user raised that motivated this design. Instead,
     `configs.LIFT_HIP_MAX_ACTION_DELTA` (in `leg_lift_env.py`'s `step()`) hard-clamps
     the physics consequence of *only* the actively-lifted leg's hip action per
     control step, modeling an actuator max-velocity limit, while leaving all 11 other
     joints (the 9 stance joints plus that leg's own abduction/knee) untouched. The
     clamp is applied only to what reaches `pipeline_step`; `action_rate`/`dof_acc`/
     `last_act` still see the raw action, so a raw action beyond the limit costs
     `action_rate` for zero extra physical effect — the intent is for the policy's raw
     output to converge to already respecting the limit, so it transfers to the
     deployed (unclamped) policy without a `neural_controller.cpp` change. **This is
     not yet verified on hardware** — if the exported policy still snaps on the robot,
     the clamp needs to be replicated in `neural_controller.cpp` directly, since
     nothing there currently rate-limits actions (confirmed: no
     `rate_limit`/`slew`/velocity-limit logic in `neural_controller.cpp` as of
     2026-08-18).
  3. **Whole-body oscillation while a leg is held up** — not confined to the lifted
     leg; the stance legs/body wobble too (per user report from watching training,
     2026-08-18). Because a hard cap on this would blunt balance-correction
     authority (see point 2), this is addressed with **soft** costs instead, applied
     body-wide: `action_rate` raised from -0.05 to -0.1, `dof_acc` raised 4x from
     -2.5e-6 to -1e-5. Both are priced-in costs rather than physical limits, so a
     large fast corrective action to avoid a fall (which risks a much larger
     termination penalty) is still fully available — only *sustained, unnecessary*
     jitter should get trained out. These are starting points, not tuned values;
     check the eval video for residual oscillation and adjust weights if it persists.
  - **Run `leg_lift_2026-08-18_15-09-43` completed** (200M steps, warm-started from
    `leg_lift_2026-08-11_01-35-11`, ~15.5 min wall-clock on the dual RTX PRO 6000
    workstation). `eval/episode_reward` plateaued around 90-104 (final eval 96.4,
    vs. 53.8 at the warm-start baseline eval), tilt held 3.4-4.0°, torso height
    0.158-0.160 m throughout — posture metrics look at least as good as before.
    Exported and shipped to `neural_controller/launch/policy_leg_lift.json` for a
    round of testing.
  - **On testing that run, the user asked for the CoM correction to go further**:
    an additional ~2cm back, ~2cm right (robot-forward frame; axis convention
    verified directly against the model's leg attachment positions, not assumed
    — see the code comment in `randomize.py`). `body_com_x_shift_range` recentered
    from -0.025 to -0.045, `body_com_y_shift_range` from 0 to -0.02. Both ranges
    stay entirely on the back/right side, so every episode gets a consistent
    back+right bias, not just wider noise.
  - **Run `leg_lift_2026-08-19_20-18-08` completed** with that further CoM
    correction (200M steps, warm-started from `leg_lift_2026-08-18_15-09-43`,
    ~17.2 min wall-clock). `eval/episode_reward` plateaued 85-106 (final eval
    92.9), tilt 3.5-4.4°, torso height 0.159-0.161 m — comparable posture to the
    previous run. **Exported and shipped to `neural_controller/launch/
    policy_leg_lift.json` — this is the policy currently deployed, awaiting the
    next round of hardware testing.** None of the three targeted fixes (CoM
    correction, gradual lift vs. snapping, reduced hold-phase oscillation) have
    been visually/hardware confirmed yet; that confirmation is what this round
    of testing is for. `rollout_final.mp4` for this run is at
    `workspace/output/leg_lift_2026-08-19_20-18-08/rollout_final.mp4` if you want
    to sanity-check the sim rollout before/alongside hardware testing.
  - **Process note**: both training runs behind this policy needed a background
    process fully detached from the Claude Code session (`nohup ... &` +
    `disown`) to reliably finish — the session runs over SSH and can drop, which
    killed two earlier in-session attempts before completion (one at 94% done).
    Always launch long training runs detached from here on.
  - **Hardware test of `leg_lift_2026-08-19_20-18-08` (2026-08-19): success —
    all four legs lifted and stabilized.** First fully clean run across all
    four commands. One regression noted: **`front_l` doesn't lift as high as
    before**, suspected to be a side effect of the ~2cm back/~2cm right CoM
    correction above being too aggressive — pushing the assumed CoM further
    right plausibly makes the left-side lift (which needs to shift weight
    rightward onto the stance legs) harder to reach as high. **Next: retry
    with a less extreme CoM correction — 1cm back / 1cm right instead of the
    current 2cm/2cm** — i.e. `body_com_x_shift_range` recentered to -0.035
    (not -0.045) and `body_com_y_shift_range` recentered to -0.01 (not -0.02),
    keeping the same half-widths as the current range.
  - **`randomize.py`'s committed default backed off to the 1cm/1cm correction**
    (commit `e847e9d`). `train.py` also gained `--com_x_shift_range`/
    `--com_y_shift_range` CLI overrides (commit `1072a9e`), so a run can use a
    different CoM range than the committed default without editing the shared
    file — used to run a 1.5cm/1.5cm variant alongside the 1cm default as an
    extra data point, in case 1cm undercorrects.
  - **Two runs completed in parallel** (2026-08-19, both warm-started from
    `leg_lift_2026-08-19_20-18-08`, both ~16.5 min wall-clock, GPUs 0 and 1):
    - `leg_lift_2026-08-19_21-12-29` — **1cm back / 1cm right** (committed
      default). `eval/episode_reward` plateaued 80-103, final eval 94.3, tilt
      3.5-3.9°. **Exported and shipped to `neural_controller/launch/
      policy_leg_lift.json` — this is the policy currently deployed, ready for
      the next round of hardware testing. Watch `front_l` specifically** (the
      leg that regressed under the 2cm/2cm correction) to see if it recovers.
    - `leg_lift_2026-08-19_21-48-33` — **1.5cm back / 1.5cm right** (CLI
      override, NOT the committed default). `eval/episode_reward` plateaued
      86-106, final eval 91.3, tilt 3.3-3.9° — comparable posture to the 1cm
      run. Exported to `workspace/output/leg_lift_2026-08-19_21-48-33/
      policy_leg_lift.json` but **not deployed** — kept as a ready-to-swap-in
      backup in case the 1cm correction under-corrects and `front_l` (or
      back-leg balance) doesn't fully recover on hardware.
    - Aggregate reward/tilt/torso-height numbers look similar across both and
      don't distinguish them — as with every round so far, the thing that
      actually matters (per-leg lift height, specifically `front_l`) needs
      hardware testing or at least `evaluate.py`'s per-leg breakdown, not the
      aggregate eval metrics, to tell them apart.
  - **Hardware test of both 1cm and 1.5cm (2026-08-20): neither beat the
    original 2cm/2cm correction overall.** Both ran cleanly on hardware (no
    e-stop, no fall), but the user's verdict after comparing all three:
    **2cm/2cm is still the best CoM correction so far**, despite its known
    `front_l` shortfall — 1cm was "too central" (undercorrected) and 1.5cm
    didn't improve on 2cm either. **Recommendation for the next training
    pass: go back to the 2cm/2cm correction** (`body_com_x_shift_range`
    recentered to -0.045, `body_com_y_shift_range` to -0.02, i.e. revert
    `e847e9d`) as the base to build the two fixes below on top of, rather
    than continuing to back off the correction.
  - **Two new issues identified in this same round of testing, both to fix in
    the next training pass alongside reverting to 2cm/2cm:**
    1. **Lowering snaps just as hard as lifting does.** When the O-button
       command switches (leg-lift cycling to the next leg), the
       currently-lifted leg drops straight down into the ground abruptly
       rather than lowering smoothly. The existing fix
       (`configs.LIFT_HIP_MAX_ACTION_DELTA` in `leg_lift_env.py`'s `step()`,
       from the 2026-08-18 training pass) only rate-limits the hip of the
       **actively-lifted** leg — it does not cover the transition where a
       leg stops being commanded and needs to come back down. **Next:**
       extend the rate limit (or add an equivalent one) to cover the
       lowering transition too, so the commanded leg raises AND lowers
       gradually through the O-button cycle, not just on the way up.
    2. **`front_l` still doesn't lift as high as the other three legs** —
       persists across 2cm, 1cm, and 1.5cm CoM corrections alike, so this is
       a separate issue from the CoM-correction magnitude, not something the
       CoM tuning alone will fix. Needs its own investigation next pass
       (e.g. per-leg lift-height reward/cap asymmetry, or something specific
       to the front-left leg/joint limits) rather than more CoM sweeps.
  - **Net priorities for the next training pass, per user feedback
    (2026-08-20): (1) revert to the 2cm/2cm CoM correction, (2) add a
    gradual-lowering rate limit to match the existing gradual-raise one,
    (3) fix `front_l`'s lift height specifically.** Overall the 2cm-based
    policy family is considered a real success on hardware — command
    routing, balance, and 3-of-4 leg lift quality are solid; these three
    items are refinements, not a redesign.

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
