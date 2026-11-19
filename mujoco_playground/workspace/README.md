# Pupper V3 wheeled locomotion — RL training

Training pipeline for a new **wheeled locomotion** policy, forked from this repo's
leg-lift training pipeline (see `master` branch) to reuse its MJX / brax RL stack,
project structure, and deployment path rather than starting from zero.

This `workspace/` is **our** code; the rest of `mujoco_playground/` is the upstream
library we build on (brax PPO, MJX env patterns; go1 `getup.py` is the closest
upstream reference task). A trained policy deploys to the robot the same way the
existing locomotion/leg-lift policies do — an exported JSON MLP loaded by
`neural_controller` (see the monorepo).

For the new **12-output wheel-align policy**, see [WHEEL_ALIGN_DIRECT.md](WHEEL_ALIGN_DIRECT.md)
for its encoder/IMU observation contract, teacher demonstrations, cloning, rewards,
training results and the remaining observation/startup integration edits. This
variant uses position control on all twelve joints. The existing three-output
align teacher and velocity-controlled wheel-driving task remain available.

## The wheeled robot

The MJCF
(`Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`)
is now a **four-wheeled** robot, converted in place from the quadruped:

- Each leg's **`_3` joint IS the wheel** — the old knee joint, repurposed as the
  wheel's continuous spin joint (`limited="false"`), with the wheel's mesh, mass and
  collision cylinder attached directly to that body. The old shin/foot geometry is
  gone. **Still 12 joints, not 16.**
- So each leg is `_1` abduction (position) + `_2` hip (position) + `_3` wheel
  (**velocity**: ctrl is a target speed in rad/s, not an angle). That **mixed
  actuation** is the thing to be careful about — see below.
- Wheel mass/inertia come from real scale measurements, not CAD estimates — see
  `Stanford/training/pupper_v3_description/WHEEL_MASS_LOG.md`.
- The `home` keyframe is the **splayed stance** (abduction at ±1 rad) that puts all
  four wheels on the ground; `configs.DEFAULT_POSE` matches it and `wheel_env.py`
  asserts the two agree.

### The mixed-actuation trap

Both the providers' `pupperv3_mjx.environment.PupperV3Env` and this repo's
`leg_lift_env.py` stamp **one position-PD gain set over every actuator row**:

```python
sys.replace(actuator_gainprm=...at[:, 0].set(kp),
            actuator_biasprm=...at[:, 1].set(-kp).at[:, 2].set(-kd))
```

On this robot that silently overwrites the wheels' velocity gains with position
gains — the model still loads and still trains, just against wrong physics.
`wheel_env.py` and `randomize.domain_randomize_wheeled` override the two groups
separately, indexed by `configs.POSITION_ACTUATOR_ROWS` / `WHEEL_ACTUATOR_ROWS`.
Anything new that touches actuator gains must do the same.

Two more consequences of the split, both handled in `wheel_env.py`:
- The joint **observation** reports angle for the position rows but **speed** for the
  wheel rows (a free-spinning joint's angle is unbounded and wraps), and
  `stance_pose`/`dof_pos_limits` score the position rows only.
- `configs.WHEEL_FORWARD_SIGN` flips the wheel commands per side. The left and right
  legs' frames are **mirrored**, so the two sides' wheels spin about opposite world
  axes; without this the sides fight each other and the robot **does not move at all**
  (measured: <3 cm in 2 s with wheels at ~14 rad/s). The real motors need the same
  convention.

The full leg-lift design rationale (reward-shaping history, local optima hit during
tuning, hardware test results) lives on `master`'s copy of this README — worth reading
for the lessons that generalize (e.g. "reward weights shape behavior inside the
feasible set; termination is what removes a strategy").

## Files

| File | Purpose |
|---|---|
| `configs.py` | Joint order/limits, default pose, per-joint action scale, actuator row splits, wheel geometry, reward weights, PPO hyperparameters, model path. **Single source of truth.** `get_wheel_config()` is the live one; `get_config()` is legacy leg-lift. |
| `wheel_env.py` | `PupperWheelEnv` (brax `PipelineEnv`, MJX): the wheeled task — reset/step/obs/reward, velocity-command sampling. |
| `train.py` | brax PPO training entry for the wheeled policy; saves brax params to `output/<run>/mjx_params`. Optional W&B logging + rollout videos. |
| `wheel_visualize.py` | Rolls the policy through a fixed command showcase (stop / forward / arcs / spin / reverse) and renders a `tracking_cam` video. |
| `randomize.py` | Physics domain randomization. Use `domain_randomize_wheeled`; the plain `domain_randomize` is the leg-lift (all-position-actuator) version. |
| `export_policy.py` | Converts brax params → `neural_controller` JSON (normalization folded into layer 0). |
| `leg_lift_env.py`, `visualize.py` | **Legacy leg-lift reference, not used by the wheeled pipeline.** Kept untouched; the live leg-lift task is on `master`. |

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

# smoke test first (JITs the env, a few PPO iters) before a long run
.venv/bin/python -m workspace.train --num_timesteps 300000 --num_envs 512

# full run (run from the mujoco_playground/ dir so `workspace` is importable)
wandb login                                 # once, if using W&B
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m workspace.train --use_wandb \
    > train.log 2>&1 & disown
# export the trained policy to the robot's JSON format
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```

> **Do NOT use `uv run` in this project — use `.venv/bin/python`.** `uv run` re-syncs
> jax/jaxlib to `uv.lock`'s 0.6.2, which mismatches the installed
> `jax_cuda12_plugin` 0.5.0: that silently disables both GPUs and makes
> `brax.io.mjcf.load` **segfault** (exit 139). If you hit that, re-pin with
> `uv pip install "jax==0.5.0" "jaxlib==0.5.0" "orbax-checkpoint==0.11.1" "flax==0.10.4" "optax==0.2.4" "numpy==2.2.6"`
> and check `jax.devices()` lists two `CudaDevice`s. (`uv pip install` is fine; it is
> `uv run`'s implicit sync that breaks things.) Long runs should be detached with
> `nohup ... & disown` so an SSH drop cannot kill them.

`num_timesteps` defaults to 200M and `num_envs` to 8192 (fits the 5090's 32 GB with
room to spare on bigger cards); lower `--num_envs` if VRAM is tight. Pin
`CUDA_VISIBLE_DEVICES` on a multi-GPU host — brax would otherwise pmap across every
visible device, which brings its own `num_envs`/`batch_size` divisibility constraints
for no gain at this scale.

### Watching the policy

Every eval, training renders a rollout through a fixed velocity-command showcase
(stop → forward → forward fast → arc left → arc right → spin in place → reverse →
stop; see `wheel_visualize._SHOWCASE`) and logs it to W&B as `eval/video` (plus a
final `eval/video_final`). The schedule is fixed so runs are comparable
frame-for-frame. Videos are also written to `workspace/output/<run>/*.mp4`
regardless of W&B. Rendering is headless via EGL (`MUJOCO_GL=egl`, set in
`workspace/__init__.py` — it has to be set before anything imports `mujoco`, which is
too early for `train.py`; getting this wrong aborts the process rather than raising).
Flags: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--no_eval_videos` (skip the
per-eval video if it slows things down — the final video still renders).

Alongside the reward terms, these diagnostic metrics are logged so you can tell *how*
a policy is earning its reward rather than just how much: `lin_vel_error` and
`ang_vel_error` (is it actually tracking the command, or just parking and banking the
posture terms?), `torso_z`, `tilt_deg`, and `wheel_contacts`. **brax reports these as
per-episode sums**, so divide by `eval/avg_episode_length` to read them back as
averages — `train.py`'s progress line already does this.

## Status / what still needs doing

**First wheeled policy trained and sim-verified (2026-08-31):** run
`wheel_2026-08-31_18-53-25`
([W&B](https://wandb.ai/QuadMorph/pupper-wheel/runs/h6yhfsxe)), 200M steps in
**14m27s** on one Blackwell GPU (~230k steps/s), params at
`workspace/output/wheel_2026-08-31_18-53-25/mjx_params`.

Final eval: reward 79.67, episode length 600/600 (never falls), tilt 0.29°, all four
wheels in contact, `vel_err` 0.0097, `yaw_err` 0.0965. It converged by ~43M steps and
then drifted up slowly; the remaining ~150M bought about +0.5 reward.

Measured on the trained policy (deterministic, steady state) — commanded vs achieved:

| commanded | achieved vx | achieved yaw | tilt |
|---|---|---|---|
| stop | −0.003 | +0.009 | 0.1° |
| vx 0.40 | +0.387 | −0.002 | 0.0° |
| vx 0.80 | +0.783 | −0.047 | 0.6° |
| vx −0.40 | −0.416 | +0.011 | 0.0° |
| yaw +1.0 | −0.020 | +0.931 | 0.2° |
| yaw +2.0 | −0.029 | +1.879 | 0.3° |
| yaw −2.0 | −0.032 | −1.885 | 0.4° |
| vx 0.5 + yaw 1.0 | +0.512 | +1.000 | 0.0° |

Within 2–7% on every axis with no meaningful cross-coupling.

**Run 2 — `wheel_2026-08-31_19-17-45`**
([W&B](https://wandb.ai/QuadMorph/pupper-wheel/runs/xoegcor0)), the current
recommended policy. Same task, two sim-to-real changes: `observation_history`
1 → 4 and `action_rate` −0.05 → −0.1. Final eval reward 79.56 vs run 1's 79.67,
i.e. inside eval noise despite double the smoothness penalty.

Head-to-head on the trained policies (probes include the env's obs noise and
action latency):

| | run 1 (hist=1, ar=−0.05) | run 2 (hist=4, ar=−0.10) |
|---|---|---|
| mean tracking error | **0.0517** | 0.0644 |
| mean \|Δaction\| | 0.0313 | **0.0272** (−13%) |
| max wheel torque | 1.070 Nm | **0.734 Nm** (−31%) |

A genuine trade, not a clean win. Run 2 is preferred for DEPLOYMENT because its
advantages are the ones an outer loop cannot fix: a ~6% steady-state speed error
(run 2's worst case, vx 0.8 → 0.748) is trivially corrected by an operator or a
higher-level controller, whereas oscillation and actuator saturation are not —
and 31% more torque headroom matters against a real motor with current limits and
backlash the sim does not model. 4-frame history is also what the already-deployed
locomotion policy uses, so it has precedent on this robot. Run 1 is kept as a
fallback; if hardware shows run 2 tracking too loosely, `action_rate` −0.075 with
history 4 is the middle option.

**Run 3 — `wheel_2026-09-02_00-21-44`**
([W&B](https://wandb.ai/QuadMorph/pupper-wheel/runs/3vzecsvs)), **the current
recommended policy**, trained after the first hardware session. Adds wheel-diameter
randomization (+-5mm, independent per wheel), slippery ground (friction 0.25-1.5),
IMU latency, and start randomization. CoM deliberately untouched.

Final eval reward 78.72 vs run 2's 79.56 -- but eval reward is the wrong yardstick
here, because run 3's evals run under harder randomization. Measured head to head on
the same probes:

| policy | nominal tracking err | slippery (0.30) + mismatched wheels | mean \|dact\| | max wheel torque |
|---|---|---|---|---|
| run 2 | 0.0632 | 0.1058 | 0.0270 | 0.657 Nm |
| **run 3** | **0.0308** | **0.0615** | 0.0270 | 0.918 Nm |

Run 3 is ~2x more accurate nominally AND ~1.7x more accurate under the perturbed
conditions, at identical smoothness. It draws ~40% more wheel torque (0.918 Nm), still
well inside the 3 Nm ceiling and the ~1.32 Nm back-EMF limit at top speed. The extra
randomization acted as a regularizer rather than a tax.

**Nothing here has touched hardware**, and the reward weights / gains are still a
first pass, not tuned. The run-2 preference is a judgement about untested
transfer, not a measured hardware result.

Built and verified in sim (2026-08-31):
- Wheeled MJCF: four wheels on repurposed knee joints, real measured mass/inertia,
  splayed home keyframe.
- `configs.get_wheel_config()`, `wheel_env.PupperWheelEnv`,
  `randomize.domain_randomize_wheeled`, `wheel_visualize`, and `train.py` wired
  together.
- Sim checks: settles upright on all four wheels; drives straight forward and reverse
  (0.694 m/s measured vs. 0.72 theoretical at that command — minimal slip); spins in
  place on a differential command; random-action rollouts give finite rewards with no
  spurious terminations.

Known open items:
- **No intermediate checkpointing.** `train.py` saves params only after `ppo.train`
  returns, and no `checkpoint_logdir` is set — killing a run mid-flight loses the
  weights entirely. Worth adding before any run long enough that you'd want to
  early-stop it.
- Yaw is the weaker axis (`trk_ang` plateaued ~0.80 of 1.0 vs `trk_lin` 0.97). If
  yaw accuracy matters, the lever is raising `tracking_ang_vel` (1.0) toward
  `tracking_lin_vel` (2.0) — linear tracking has margin to give up.
- **Wheel speed is capped at 1 m/s** (`configs.WHEEL_MAX_LINEAR_SPEED`). At the
  earlier 1.44 m/s cap the robot **flipped** when all four wheels were commanded to
  full scale. Command ranges are kept strictly inside the cap so the policy has
  authority left to steer and correct.
- `wheel_kv` (0.35), the reward weights, and the DR ranges are untuned starting
  points. The wheeled DR CoM ranges are symmetric placeholders — the leg-lift ones
  encode quadruped-on-feet hardware findings that do not transfer.
- **Deployment is not done.** `export_policy.py` still emits leg-lift metadata
  (`command_states`/`button_sequence`), and the robot side has no velocity-command
  path or wheel velocity-control mode. The on-robot motors also need
  `WHEEL_FORWARD_SIGN` applied.
- `evaluate.py` still references the leg-lift env.

## Deployment (monorepo side)

Status: the policy is exported and its artifact has been validated OFFLINE against
the wheeled MuJoCo model (see below), but **nothing has run in `pupperv3_mujoco_sim`
or on the robot yet** — that needs a ROS2 Humble machine (the Pi, or x86 Ubuntu
22.04); the training workstation has no ROS2 installed.

Export:

```sh
.venv/bin/python -m workspace.export_policy \
    --params workspace/output/wheel_2026-08-31_19-17-45/mjx_params
# -> workspace/output/.../policy_wheel.json   (behavior="wheel", in_shape [None,132])
```

**Offline validation already done.** `policy_wheel.json` was used to drive the
wheeled MJCF with the observation assembled and the MLP evaluated the way
`neural_controller.cpp` does (plain numpy, no brax). It reproduces the training-side
behaviour: vx 0.8 → 0.748, vx −0.4 → −0.416, yaw 2.0 → 1.825, combined 0.5/1.0 →
0.491/0.932. So the weights, the folded normalization, the action scaling and the
gain mapping are all correct in the exported artifact.

### What the robot still needs

**1. Two small `neural_controller.cpp` patches** (neither exists yet):

  a. *A `behavior == "wheel"` branch.* `single_observation_size` must be 33
     (`ang_vel 3 + gravity 3 + cmd_xyyaw 3 + joints 12 + last_action 12`), i.e.
     `joint_position_idx_ = 3 + 3 + 3`. The existing `locomotion` branch is 36 —
     it carries an extra `desired_world_z[3]` this policy was not trained with, so
     reusing it would mis-align every downstream index. (The in_shape check would
     catch it loudly: 132 ≠ 4×36.)

  b. *Populate the wheel rows of the joint block.* Today the observation loop only
     writes a slot `if (action_types[i] == "position")`, so velocity-type joints
     keep a permanent 0. This policy expects
     `wheel_joint_velocity / wheel_velocity_normalizer * wheel_forward_sign` there
     (all three values are in the JSON). Left unpatched the policy flies blind on
     wheel speed — it will still move, but not as trained.

**2. A controller instance in `config.yaml`**, mirroring the locomotion one but
with the JSON's per-joint values — `action_types` = `velocity` on rows 2/5/8/11 and
`position` elsewhere, `kps` = **0 on the wheel rows**, `kds` = 0.35 there,
`default_joint_pos` = the splayed ±1 rad stance, and `model_path` pointing at
`policy_wheel.json`. Spawn it in `launch.py` and add it to `joy_util_node`'s
`controller_names`.

**3. Nothing to build for the command path.** The controller already subscribes to
`/cmd_vel` (`geometry_msgs/Twist`) for the locomotion behavior; the wheeled policy
uses the same three numbers (vx, vy, yaw). Note `vy` is ignored by design — these
are fixed wheels.

**4. Do NOT re-apply `WHEEL_FORWARD_SIGN` on the robot.** It is already folded into
the exported per-joint `action_scale` (negative on the right-hand wheels), because
the controller computes `action * action_scale` for velocity joints with no sign
handling of its own. Applying it twice cancels it out and the two sides will fight
each other — the failure mode where the robot spins its wheels and does not move.

**5. Why `kp = 0` on the wheels matters.** `control_board_hardware_interface`
computes `torque = (pos_cmd − pos_state)·kp + (vel_cmd − vel_state)·kd`. With
`kp = 0` that reduces to `torque = kd·(vel_cmd − vel_state)`, algebraically the same
law as MuJoCo's `<velocity kv>` actuator, so `kd` here IS `configs.wheel_kv`. Any
nonzero `kp` fights the free-spinning wheel against a stale position target.

### Suggested bring-up order

1. `pupperv3_mujoco_sim` first, on a ROS2 box — it exercises the same controller,
   config and JSON without risking hardware, and the wheel-sign and mixed-actuation
   traps above would show up immediately there.
2. Robot **on a stand, wheels off the ground**, `/cmd_vel` at low vx only. Confirm
   all four wheels turn the same direction for +vx. This is the sign check.
3. Ground test at reduced command. Measured envelope is 0.958 m/s and 4.65 rad/s;
   the policy is trained for vx ≤ 0.8 and |yaw| ≤ 2.0.
4. Watch for oscillation. Run 2 was chosen partly for smoothness/torque headroom
   (0.734 Nm peak of a 3.0 Nm ceiling); if it still oscillates, `wheel_kv` is the
   first thing to lower.

### Not yet addressed

* The robot's own `pupper_v3_description` copy is still the QUADRUPED model. The
  wheeled MJCF/URDF changes on this branch were made to the `Stanford/training`
  copy only; the monorepo copy (and the ros2_control joint definitions, which must
  declare the wheel joints continuous and expose a velocity command interface)
  has not been updated.
* Homing: startup must be able to reach the ±1 rad abduction stance safely.

## QuadMorph Phase 1: wheel to leg, lift and rotate (`combine`)

`PupperWheelAlignEnv` implements one-leg-at-a-time alignment while the robot
remains in wheel form. The current controller is described and diagnosed in
[ALIGNMENT_DIAGNOSIS.md](ALIGNMENT_DIAGNOSIS.md); [ALIGNMENT.md](ALIGNMENT.md)
retains the superseded pilot history.

Run from this worktree's `mujoco_playground/` with its existing venv:

```sh
JAX_PLATFORMS=cpu .venv/bin/python -m workspace.smoke_wheel_align
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python -m workspace.smoke_wheel_align_supported
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  nohup setsid .venv/bin/python -u -m workspace.train --task wheel-align \
  > train_wheel_align_supported.log 2>&1 < /dev/null &
train_pid=$!
echo "$train_pid" > train_wheel_align_supported.pid
disown
sleep 2
```

Check GPU availability before choosing it. Alignment enables the existing W&B
metrics/video path with entity `QuadMorph`, project
`wheel-leg lift and align triangle base`. Defaults: 8M steps (rounded by PPO),
4096 environments, 128×128 ELU actor/critic, learning rate 5e-5, two PPO updates,
five evaluations. The actor starts with zero mean corrections and std 0.2.
Every evaluated checkpoint, best checkpoint, config, metrics.jsonl and videos
are saved in `workspace/output/wheel-align_<timestamp>/`.

The policy has **three bounded actions**: clearance trim ±1 mm around a 5 mm
collision gap, plus two coordinated stance corrections ±12 mm in the nominal
body frame. The latter map through measured wheel-geom Jacobians into the eight
abduction/hip joints around the wheel-driving home pose. They are coupled
corrections, not eight independent joint actions. Front-leg lifts receive a
measured preparatory stance adjustment; the policy can refine it during lifting
and lowering/reweighting. Only the selected hip gets an additional lift reference.
That reference follows measured clearance and returns gradually to zero; the
preceding reference returns before another leg can rise. Idle returns to home.

The reward allows 35 mm of body movement while supporting a lift/return, following
leg-lift's permitted weight-shift mechanism, and 20 mm while idle. Excess drift
is a soft quadratic cost relative to the original episode origin. That origin is
never reset at a button press. Raw drift and outside-20-mm diagnostics remain
comparable to previous runs. Holding has an angle/speed error cost, with no large
bonus for pre-completed legs. Actual new completions earn an event bonus of 5,
which is not multiplied by the control timestep. Total reward remains signed.

External commands remain `stand/front_l/front_r/back_r/back_l`; set
`automatic_commands=False` and call `env.set_command(state, index)` before
inference. The command decides which leg is active; measured intra-leg phases
handle lift, rotate, lower, hold. Enter rotation after three contact-free steps
with ≥3 mm clearance; use a 1 mm hysteresis threshold while rotating, always
requiring <0.5 N load. Settlement requires <0.10 rad angle error and <0.25 rad/s
speed for five steps. Completion additionally requires <1 mm clearance and ≥2 N
load for five steps. An interrupted, unverified rotation cannot claim completion.

Each hub goal is its one-time calibrated `home + pi`, independent of its arbitrary
current encoder phase after driving. Reset samples the reference and phase over
the full circle. Targets persist across later commands and idle. Position PD
(kp 2, kd 0.35, torque cap ±3 Nm) and a bounded 3 rad/s reference slew handle the
wheel; every physics substep uses `current_qpos + wrap(reference-current_qpos)`.
The wheel-driving env stays velocity-actuated. Both hardware modes are verified.

The 36-second training episodes contain all four activations and idle periods.
85% of resets force zero completed legs; the rest sample 0–4 pre-completed legs.
The full maneuver state resets together with physics on termination. Broad
`domain_randomize_wheeled` geometry/friction/mass/CoM/gain/latency variation and
pushes are retained through the align PD adapter. Collision clearance always
uses actual randomized per-wheel cylinder radius, width and orientation.

Observations are **376 values: four 94-value frames**. Each frame contains IMU
angular velocity/gravity (6), mixed joint-position/wheel-speed proprioception
(12), last policy actions (3), command (5), target-error sin/cos (8), per-leg
phase (20), tracked/completed masks (8), clearance/load (8), xy drift/body velocity
(4), heading sin/cos (2), servo-reference error sin/cos (8), lift references (4),
stance shift (2), and phase age (4). No raw unbounded wheel angle is observed.
Old 340/12 or 356/8 checkpoints and the legacy drive exporter are incompatible.
Controller state and clearance/load estimates must accompany eventual deployment;
this training task does not implement the ROS integration.

Each evaluation audits 16 randomized robots starting with zero completed legs,
without auto-reset after falls. It reports per-leg rotation/verification/completion,
all-four success, survival, conditional held-angle error with sample count, drift,
and scrubbing. A fixed all-four video is also logged. A small fixed-seed audit is
an iteration diagnostic; eventual evaluation should include unseen seeds.

The CPU smoke tests cover routing, random phases, actual geometry, PD, wrapped
angles, lowering, drift allowances, fresh episode resets and wheel-drive regression.
`smoke_wheel_align_supported.py` additionally verifies a complete nominal sequence
with zero learned corrections; `smoke_wheel_align_evaluate.py` checks the randomized
evaluator and GPU/EGL rendering. Use the pinned existing `.venv`; never `uv run`.

Robustness refinement continues from the supported controller's best checkpoint:
direct unloading feedback and a stronger drift cost outside the same support
allowance, through a 24M unloading pass and a 12M drift pass. The selected policy
is the second pass's best 4M checkpoint. See [ALIGNMENT_ROBUSTNESS.md](ALIGNMENT_ROBUSTNESS.md)
for the failed-lift traces, unchanged evaluation conditions, unseen-seed comparison,
and run details. `audit_align_robustness.py` evaluates saved checkpoints with a
chosen cohort size/seed and can save full stage traces with `--trace`.
