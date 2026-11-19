# Direct-joint QuadMorph Phase 1

The 3-output teacher and the committed `robot-code` export are retained. The new
`wheel_align_direct_env.py` learns every joint command, including all four hubs.
Its inherited task phases are **training/evaluation bookkeeping only**. Neither
phase, clearance, contact load, torso displacement, the pose reference bank, nor
the teacher is an input to deployed inference or to the action mapping.

The September 8 evening refinement is a separate `wheel_align_refine_env.py`
variant. It preserves the sensor-only actor but adds the small encoder-only
settled-wheel latch described below. The original direct variant remains intact.

## Control and deployment contract

Current supervised actor: **224 -> 512 ELU -> 512 ELU -> 512 ELU -> 12 tanh**. Legacy 220-input checkpoints remain loadable. PPO's Gaussian scale
head and separate critic are training-only. All 12 native joint rows use position
commands: `ctrl[i] = default[i] + action_scale[i] * action[i]`, subject to the
ordinary joint-position limits. There is no wheel reference governor, selected
hip clamp, clearance controller, coordinated stance mapping, or high-rate wrapped
servo in this action path. The standard motor position loops remain, as for
leg-lift: abduction/hip kp=5, kd=.25; hubs kp=2, kd=.35, with existing torque caps.
Scales per leg are (.5, 1.6, 3.5) rad (abduction, hip, hub). This uses the proven
leg-lift abduction/hip ranges and gives continuous hubs full-circle authority.

Canonical joint/leg order is FR, FL, BR, BL. Commands are stand, FL, FR, BR, BL.
At align-mode entry, given startup calibration `home_i` and current raw wheel
encoder `q_i`, latch once:

```
wheel_default_i = q_i + wrap(home_i + pi - q_i)
wheel_entry_i = q_i
```

`wheel_default` is a local representative of the persistent calibrated target,
not a new calibration. It remains fixed throughout all button presses. This
allows the *same affine position action mapping* after any number of previous
driving revolutions. There is no current-angle addition or target wrapping in
the per-step actuator path. The policy decides the wheel command each tick.

Each current frame has 56 values; four frames are newest first:

| Offset | Count | Meaning |
| --- | --- | --- |
| 0 | 3 | Body-frame IMU angular velocity |
| 3 | 3 | Projected gravity |
| 6 | 5 | External command one-hot |
| 11 | 12 | Joint angles relative to defaults, in rad; wheels relative to their latched calibrated defaults |
| 23 | 12 | Previous normalized policy outputs |
| 35 | 4 | Native wheel encoder speeds / 20.8333333 rad/s |
| 39 | 4 | sin(wrapped calibrated target error) |
| 43 | 4 | cos(wrapped calibrated target error) |
| 47 | 4 | Ever-selected bits, latched from button commands; no contact-based completion estimate |
| 51 | 4 | Fixed wheel-entry encoder offsets relative to calibrated defaults, divided by hub action scale |
| 55 | 1 | Seconds since button-command change, capped at one second; derived from command events and control ticks |

The first **35 entries match leg-lift's layout and the output/action shape is
the same 12-position convention**. The complete 224-input model is *not* a
drop-in replacement for the existing 35-per-frame `leg_lift` observation builder.
`robot-code`'s `neural_controller.cpp` (leg-lift layout near line 155 and affine
position output near line 759) still needs a small encoder/command observation
extension and mode-entry initialization. It needs no new clearance, load, drift,
or ground-plane estimator and no hand-ported maneuver controller.

That initialization must seed previous wheel outputs to hold the entry angles,
retain the four startup calibrations, select the local defaults, set native wheel
position limits appropriately, and avoid the legacy startup ramp/fade commanding
the calibrated wheel goals before the first policy inference. Seed history by
repeating the first frame, as the deployed controller does. Apply one history
shift per inference tick, including button changes. These are concrete remaining
integration edits, not a claim that the current C++ can already load this model.
No `robot-code` files or its exported teacher policy were changed here.

## Encoder-hold refinement and reference resets (rounds 32–34)

The paired 64-sequence wheel-only intervention measured:

| Wheel control | Full successes | Falls | Drift p95 | Held error p95 |
| --- | ---: | ---: | ---: | ---: |
| Original learned wheel targets | 12/64 | 0 | 5.052 cm | .05757 rad |
| Encoder settled latch | 11/64 | 0 | 5.086 cm | .02368 rad |
| Diagnostic simulated unloading gate | 19/64 | 0 | 4.671 cm | .04309 rad |
| Hip encoder gate at .25 rad | 14/64 | 0 | 4.704 cm | .02217 rad |
| Hip encoder gate at .45 rad | 10/64 | 0 | 4.778 cm | .02285 rad |

These are interventions on the same fixed clone, not retrained hybrid policies.
They show that holding can be simplified successfully; they do not establish
that a fixed hip threshold reliably indicates unloading. The diagnostic oracle
reads contact/clearance and is explicitly excluded from deployable code.

The adopted refinement latches a wheel after five consecutive 20 ms samples
with encoder error below .08 rad and speed below .25 rad/s while selected.
Thereafter that wheel's position target is `q + wrap(calibrated_goal - q)`.
It keeps restoring the target while other legs are selected. All eight stance /
lift outputs remain direct position commands; unlatched wheel rotation remains
learned. The same 224 inputs and twelve network outputs are retained, including
previous raw network outputs. Current checkpoints use history taps 0, 1, 10, 50
(0, 20, 200, 1000 ms), rather than four adjacent frames.

`deployment/wheel_align_hold.h` is the small standalone C++ equivalent to insert
after the existing affine action mapping and before motor command writes. It
requires encoders, four calibrated goals, the selected leg and four dwell/latch
states. It does not introduce a load/clearance/displacement estimator. The standalone C++17 test passes (including multi-revolution targets, button
changes, dwell and speed gating). A local Zig/Clang toolchain was installed under
`output/build_tools` without changing the training venv. Python physical/control
tests also pass. No robot-code integration
or hardware validation is claimed.

Round 33 additionally uses 50% training-only reference-state resets. A new set
of 128 randomized teacher sequences yielded 82 full successes. The recorder
selects 128 snapshots per leg and lift/rotate/lower phase, including actual
position/velocity, calibrated references and 51 real sensor-history frames.
PPO samples these states alongside normal starts, without adding their phase or
contact fields to actor observations. Earlier completions and settled bonuses
are marked already credited. The remaining demonstration episode duration is
used only as a training timeout. Full audits and videos explicitly disable
reference resets, use zero precompleted legs, and start at the ordinary stance.
The existing physical settle/touchdown success gates and drift tolerances remain.

The next-run results are recorded in `output/direct_iterations.jsonl`; these
changes are experimental and have not yet reached the teacher's 45/64 benchmark.

## Demonstration guidance

`record_align_direct.py` rolls out the selected teacher checkpoint
`wheel-align_2026-09-07_20-41-57/mjx_params_best` under full wheeled DR and
independent full-circle initial wheel phases. Of 256 episodes, 180 completed all
four legs. The retained dataset contains 319,080 state/action examples and every
rotation-delta octant for every leg, including both front legs. `manifest.json`
records the counts. No absolute trajectory time or absolute wheel angle is an actor feature. The current observation adds only button-event age capped at one second, to expose the short preparation interval; rotation still depends on encoder feedback.

Wheel labels are the teacher's actual desired position commands expressed in
the student's fixed, episode-local calibrated coordinates. They vary with the
current wheel phase and are not a canned 180-degree clip. Eight abduction/hip
labels transfer the successful weight shifts. Behavior cloning samples command /
measured task-phase groups equally, so idle frames and back legs cannot swamp
front-leg lift/rotate/lower examples. Validation splits whole episodes.

During PPO, a light negative pose-distance cost compares the eight abduction/hip
positions with a bank of 32 diverse successful poses for the current command and
measured task phase. Farthest-point selection covers a family of poses within
each phase; no nearest-time or absolute-angle trajectory is used. Hubs are not
constrained by this pose bank. The reference path is required whenever this cost
is enabled. It is training-only and does not ship with the actor.

The second PPO trial uses **unloaded ROTATE poses as the LIFT goal** and raises
the pose cost weight from .25 to .75. The original LIFT bank included the home
stance (minimum position-offset norm .018–.019 rad for the front legs), so a
nearest-pose objective could be satisfied without lifting. The demonstrated
front-leg unloaded banks have minimum norms .584–.616 rad across the eight
position joints. This change gives learning a concrete support-shift/lift goal
while leaving the path and every joint target under policy control.

The first cloned policy fit teacher targets closely but failed in closed loop.
The initial 204-input variant also lacked a fixed reference for unselected wheels;
the final 220-input contract adds their latched entry encoders. The 8,000-update
clone achieved held-out stance/lift target RMSE .005–.012 rad and wheel RMSE
.023–.030 rad, but closed-loop errors still compounded. This is not reported as
a working distilled policy.

`clone_align_direct.py --augment-holds` adds disturbed inactive/held encoder
positions, speeds, and previous outputs while preserving their known position
goals. `aggregate_align_direct.py` additionally queries the teacher on states
visited by a student/expert action mixture. Its first collection used expert
actions with probability .7 and produced 230,400 recovery samples from 128
episodes; 78 full successes and per-leg completions [98,111,120,124] describe that
*assisted collection*, not independent student performance. These recovery
samples augment the original demonstrations; the pose bank remains the original
successful reference family. There is no teacher query at deployment.

That first aggregation retained counterfactual teacher controller references and
did not improve unassisted success. The revised collector reconstructs lift,
stance and wheel references from the student's actual previous joint commands
before querying the teacher. A pure-student collection added 220,092 recovery
examples, for 539,172 total. This collection itself achieved 0/128 full successes;
it supplies correction labels on failure states, not successful demonstrations.
The original 180 fully successful teacher episodes remain in the dataset.
After another 8,000 cloning updates, the independent 16-episode audit still had
zero completions, but p95 drift improved from 20.83 to 12.07 cm and both front
legs reached ROTATE in 3/16 episodes. This is an initialization for PPO, not a
deployment-ready replacement for the teacher.

A separate label/control sanity check executed the queried teacher's twelve
targets through the direct environment at every step: **21/32 full successes**,
32/32 survival, per-leg completions FR=24, FL=31, BR=29, BL=30. This checks the
position-control contract and reconstruction, not student inference accuracy.
Results are in `output/align_direct_label_control_check/manifest.json`.

## Rewards and checks

- Running costs cover small clearance around 5 mm, unloading, rotation error
  after unloading, lowering/reweighting, held-angle error, unselected-wheel
  position, active-wheel scrubbing, support, imitation, orientation and drift.
- A **+2 one-time settled-rotation event** requires the existing unloaded phase
  guard, target error <.10 rad and speed <.25 rad/s for five steps. A separate
  **+3 touchdown event** requires clearance <1 mm, load >=2 N and settled angle /
  speed for five steps. Four new legs can earn at most +20 total. Falls cost -50.
  Initial completed legs and repeated settling cannot earn these events.
- Signed reward is retained. Contact uses actual randomized per-wheel geometry.
  Drift keeps 35 mm while lifting / during a 75-step return allowance, then 20 mm
  idle, measured from the fixed episode origin. Full DR/pushes and complete
  multi-leg/partial-start episodes remain.
- `smoke_wheel_align_direct.py` verifies 12 independent affine outputs, 220-input
  history/reset shape (224 for the current variant), encoder whole-turn invariance, absence of privileged actor
  inputs, finite random rollouts, event gating/no double credit, and full task
  state reset under batched DR/PPO. It caught and fixed a metric dtype mismatch.

## Runs

First PPO pilot: `wheel-align-direct_2026-09-08_13-03-49`, W&B
[ahxgcg5h](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/ahxgcg5h),
PID 3485755, GPU 1, `../train_wheel_align_direct.log`. It completed 12,042,240 steps
and is preserved as a failed baseline. Full-sequence audit stayed at 0/16 full
successes; p95 drift fell from 22.73 cm at the cloned initialization to 12.00 cm,
but lifting behavior deteriorated. This motivated recovery-data aggregation,
rather than committing blindly to the hardware leg-lift policy's 200M budget.

Second PPO trial: `wheel-align-direct_2026-09-08_13-41-10`, W&B
[mp4scn5x](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/mp4scn5x),
PID **3526311**, GPU 1, `../train_wheel_align_direct_recovery.log`. It starts from
`output/align_direct_recovery/cloned_params`, with 24M additional environment
steps, 4,096 environments, learning rate 3e-5 and audits every approximately 4M
steps. This bounded second trial tests the corrected phase guidance and recovery
initialization; it does not assume that a full 12-output transfer should train
as quickly as the three-output teacher. CPU smoke checks passed after the reward
change; the recovery checkpoint also passed all smoke and finite full-sequence
audit checks on GPU.

At the first trained checkpoint, **4,014,080 steps**, the fixed 16-episode
full-DR audit reports 16/16 survival, **0/16 full successes, 0 mean completed
legs**, and p95 drift **9.065 cm** (initialization: 12.100 cm). Neither front leg
reached ROTATE in this checkpoint, compared with 3/16 each at initialization;
back-right reached it in 5/16 and back-left in 4/16. Thus drift is improving but
the student has not retained the teacher's lifting behavior. That run finished at 24,084,480 steps with 0/16 full successes, zero mean completed legs, and 6.51cm p95 drift. It is preserved as a failed baseline.

Run from `mujoco_playground/` (the exact detached command is also saved in the run):

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 XLA_PYTHON_CLIENT_PREALLOCATE=false nohup setsid .venv/bin/python -u -m workspace.train --task wheel-align-direct --num_timesteps 24000000 --num_envs 4096 --seed 32 --init_params workspace/output/align_direct_recovery/cloned_params > train_wheel_align_direct_recovery.log 2>&1 < /dev/null &
quadmorph_direct_recovery_pid=$!
echo "$quadmorph_direct_recovery_pid" > train_wheel_align_direct_recovery.pid
disown
```

Sources, configuration, reference-bank copies, hashes and exact detached launch
commands are stored in each run directory. All training uses the existing venv,
Brax PPO, existing W&B/video path, and the original QuadMorph entity/project.

## Protected transfer and supervised iterations — 2026-09-08

The stall diagnosis in `output/direct_stall_diagnosis/REPORT.md` demonstrated a
normalization-only KL jump of 134.16 with zero weight updates, and an already
non-working clone before PPO. `supervised_align_ppo.py` now freezes the cloned
normalizer for actor, critic, supervision and checkpoints. It backtracks or
rejects optimizer proposals against the collected behavior distribution with a
mean-KL budget of .02; a final check across the entire rollout rejects the whole
batch if necessary. It does not merely log excessive KL. The twelve-target
supervision loss uses calibrated local wheel coordinates, weighs physical errors
by (.01,.015,.05) rad per leg, and now anneals its weight from .25 to .025 over4M steps (the earlier protected pilots used5 to .5).
The existing training callback must first establish nonzero full-sequence success
and no falls before this loop spends any PPO steps. Synthetic rejection, wheel
gradient, zero-baseline refusal and normalization checks passed.

`clone_align_recurrent.py` trains through self-fed previous-action observations
for 16–64 steps. This is training-time backpropagation through the actor's existing
action feedback; it adds no runtime RNN state or controller. Its optional auxiliary
phase classifier receives only actor features and is discarded on save.
`clone_align_direct.py --action-memory-noise` tests denoising previous target
observations. `aggregate_align_direct.py` supports corrective teacher/student
mixtures and teacher-only lift-margin curricula. Neither teacher phase nor contact
measurements are added to the deployed actor.

Every supervised round and checkpoint audit is recorded in
`output/direct_iterations.jsonl`; each run also retains `config.json`,
`cloning.json`, `snapshot_audits.jsonl`, `audit_best.json` and W&B metadata.
The current exploration includes an 8mm teacher lift target (student reward
remains 5mm), corrective visitation, holding disturbances, and a paired .01
sensor-noise trial based on walking. The lower-noise trial was not adopted after
later snapshots fell. Full physical DR, drift tolerances and settle gates remain
unchanged. The later precision clones passed the readiness gate; protected PPO results follow below.

A broad audit of `direct_round09_margin_feedback/cloned_params_audit_best`
produced 1/256 full successes, 253/256 survival, 7.75cm p95 drift and .0807rad
p95 held error. This establishes rare task completion but is still a fragile
initialization, far below the teacher. The best denoising snapshot holds at
.0303rad p95 but still has zero full successes in 16. These partial results must
not be described as a deployment-ready replacement.

### Precision cloning and protected PPO pilots

The best dense-history clone reached 5/64 full successes, no falls, 5.24cm
p95 drift and .0854rad p95 held error. The longer-history clone reached 5/64,
no falls, 4.78cm and .0617rad. Both are still well below the teacher. Its four
frames sample ticks 0,1,10,50 (20ms control), instead of four adjacent frames.
This uses a sensor-history buffer, with no maneuver estimator or adapter.

The decisive cloning change was measuring stance/lift command errors on
.01/.015rad scales: teacher correction steps are only about .0035rad, while
the earlier actor's stance error was about .018rad. A stronger wheel-goal label
dominated the old loss and made lifting worse; current supervision retains
the teacher's actual slew commands in calibrated local coordinates.

An initial protected 2.048M-step PPO pilot preserved normalization bit for bit
and enforced its .02 mean-KL budget, but regressed from 5/64 to 1/64 full
successes. At4096 environments it covered only500 ticks per environment,
short of one1800-tick sequence. The next paired pilots use512 environments
so2M steps cover more than two sequences per environment.

PPO now also checks KL on a phase-balanced reference bank, and rejects the
entire parameter/optimizer update if either full-rollout or full-reference
mean KL exceeds .02. Logged interval maxima verify the bound. Explicit
all12 action supervision runs through self-fed previous-action histories
(16 ticks for dense history,64 for sparse history), matching the successful
cloning objective. Unit smoke tests exercise excessive-KL rejection, a
reference-only jump, all-wheel gradients and sparse history reconstruction.
The new runs are 1ddmicao (GPU0,dense) and7zi5vuj0 (GPU1,sparse), both in
QuadMorph/wheel-leg lift and align triangle base. All iterations are recorded
in output/direct_iterations.jsonl.

### Airborne dwell and measured loss balance

The one-second-history clone's failure trace contained123 failed leg attempts:
96 ended near the target angle;48 of55 failures that had reached ROTATE lost
unloading before verification. The teacher-only recording curriculum now tests
20 settled airborne ticks before lowering, instead of5. Student physics, reward
gates and audit criteria retain5 ticks. The teacher completed91/128 sequences;
90 whole valid sequences were retained, including all starting-phase octants
for both front legs.

Fine-tuning the existing student on these phase-balanced demonstrations produced
**13/64 full successes,64/64 survival,2.65625 mean completed legs,5.05cm p95
drift and.05736rad p95 held error**, checkpoint
`output/direct_round25_dwell_feedback/cloned_params_audit_best` (snapshot10000).
Its small selection audit was4/16 full successes and2.9375 mean legs. Expanding
the four taps to0,1,50,200 was worse (best2/16,1.75 mean legs), so current history
remains0,1,10,50. The200-tick history implementation passed full CPU smoke but
was rejected on behavior. This checkpoint is improved, still below the teacher.

Both512-env2M PPO pilots failed to improve success (final2/64 dense,3/64 sparse),
while retaining exactly the clone normalization and applied mean KL below.02.
The next diagnosis measured actor gradients separately. On actual32x20 PPO
minibatches and16x64 supervision sequences, the old weight5 made supervision
5.6--9.5 times larger than PPO; cosine similarity ranged-.068 to+.052.
`probe_align_gradients.py` and `output/direct_gradient_balance_minibatch.json`
record this without updating any weights. Weight.25 reduces that measured
ratio to.28--.47, retaining explicit all12 supervision while letting PPO learn
from task rewards. It anneals to.025 over4M steps.

Protected PPO round27 starts from the13/64 clone with an8M-step cap,512envs and
a64-sequence audit every1M. W&B run`bqlxjv2g`, PID4105723, GPU0, log
`../train_wheel_align_direct_balanced27.log`. Round28 independently collects
corrections on states visited by the improved student using50% teacher actions;
those assisted results must never be reported as autonomous student success.

### Exploration calibration

The paired stochastic audit kept every deployed mean weight and normalization
statistic fixed. Reducing Gaussian standard deviations per leg from approximately
(.03,.03,.01) to(.015,.0075,.005) normalized units increased full success from
5/64 to11/64 and mean completed legs from1.921875 to2.609375; drift fell from
6.11cm to5.07cm, with no falls. These are stochastic training-policy audits,
not a new deterministic actor. Physical target deviations become
(.0075,.012,.0175)rad per leg.

`calibrate_align_exploration.py` reached that training-only scale head through
35 guarded stages, maximum applied full-reference mean KL.019751. It asserts
bit-identical deployed mean weights and retains the normalizer unchanged.
`output/direct_round29_precision_exploration/cloned_params` stores the calibrated
initialization. PPO still requires a fresh nonzero64-sequence deterministic
readiness audit, and retains its frozen normalization and per-update KL guard.
Round29 uses the same8M/512env/seed84 setup as round27 on GPU1; PID4167400,
log`../train_wheel_align_direct_precision29.log`.

The corrective-data round28 clone audited at11/64 full successes, no falls,
4.58cm p95 drift and.07392rad p95 held error. This is below the dwell clone
overall, so it was retained as a comparison rather than adopted.
