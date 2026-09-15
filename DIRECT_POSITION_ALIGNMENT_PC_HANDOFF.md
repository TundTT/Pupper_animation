# PC agent handoff: direct-position RL for wheel lift and alignment

Prepared September 14, 2026. Repository: https://github.com/TundTT/Pupper_animation

## Task and agreed design

Implement, train, evaluate and export a new wheel-alignment policy on the user's NVIDIA GPU PC. The robot should gently lift one wheel, remain balanced while that wheel rotates to the calibrated heating-preparation orientation, lower gently, and repeat for the other wheels.

The user has selected this architecture:

- RL chooses meaningful absolute position targets for all eight proximal joints as `nominal_pose + action_scale * normalized_action`.
- A deterministic position controller chooses the four hub targets. Supporting hubs hold; the selected hub follows its alignment reference while elevated.
- Low-level joint PD tracks all twelve position targets. The existing bounded near-target hub integral correction may be retained with its exact state/reset semantics.
- RL runs continuously during lifting, alignment settling, holding and lowering. Do not freeze its targets during wheel verification or fade away its authority when balance/clearance is poor.
- A small state machine manages requests, desired clearance, rotation permission, completion and cancellation. It does not prescribe the eight-joint lift trajectory.
- Reuse the robot's existing ROS 2, RTNeural, calibration, position-command and joystick infrastructure. No MPC, trajectory optimizer, new middleware or separate force-based whole-body controller is required for this task.
- Heating is manual and outside scope. This task ends with all four wheels aligned and load-bearing, ready for the operator's heating procedure. Do not implement polymer deformation, leg-to-wheel recovery or roll-to-walk here.

This handoff is an implementation-and-training task when the user gives it to the PC agent. The authoring task on the laptop has not started training. PC work may prepare and test a runtime adapter in an isolated branch, but must not deploy to, start, calibrate or move the physical robot, or change the user's locomotion-policy selection. Hardware integration/testing is a subsequent task.

## 1. Read these sources and create an isolated checkout

The source versions below were checked on September 14. Inspect current remote history first and report material changes; do not silently combine a different model, checkpoint or controller revision.

| Purpose | Available reference |
| --- | --- |
| Current robot runtime and calibration integration | `origin/robot-code`, inspected at `085a3f0a620a9942459fab24eea7fa507786ec7b` |
| Remotely available alignment trainer, logging helper and backpack/9 mm model | `origin/codex/backpack-align`, inspected at `dde1f961b519f53379d80fc4e77eab797b38be73` |
| Historical v5 and deterministic-controller development instructions | `origin/codex/align-motion-v2`, inspected at `05299b3a61bf45bb5e89a371fa3313643be2014e` |
| More recent LOCAL backpack-alignment work | Laptop branch `codex/backpack-align` at `a7e3bb1`; includes `18b8919`. These commits were not present on the inspected remote branch. Do not depend on them being available on the PC. |

Start a NEW `codex/align-direct-position` branch/worktree from the inspected robot-code revision. If that name already exists, inspect and resume only if it is this task; otherwise use a distinct `codex/` name. Preserve existing training checkouts, dirty changes, jobs and checkpoint provenance. Use Linux or WSL2, with the checkout on the Linux filesystem and a persistent tmux session. Check running jobs before using the GPU; do not interrupt another experiment.

Read the following before editing:

1. `AGENTS.md`, `LATEST_POLICIES.md`, `policies/latest.json`, `STARTUP_CALIBRATION.md` and `LAB_HANDOFF.md` from robot-code. Reading these does not authorize hardware startup.
2. `AGENT_TRAINING_HANDOFF.md`, `WANDB_LOGGING.md`, `ALIGN_MOTION_V2.md`, `ALIGN_MOTION_V5.md` and `KEYFRAME_ALIGNMENT.md` on origin/codex/align-motion-v2. Their old v5 training commands and checkpoint selection are historical references, not the implementation requested here. This user-selected direct-position RL task supersedes their deterministic-only development direction for this new branch.
3. Runtime sources in section 12 below, especially the actual position-mode hub controller and saved configuration rather than earlier velocity-mode prose.
4. `models/heating_module/README.md`, `spec.json`, and `training/wheel_align/model.xml` from the pinned backpack-alignment reference.

Suggested organization: a new `training/align_direct_position/` package, a small versioned C++ command/state helper within the existing neural-controller package, and a separate candidate JSON export. Reuse/copy the existing PPO, audit, exporter, rendering and W&B utilities where appropriate. Record every imported file's source commit. Do not merge the whole training branch into robot-code, overwrite historical v5 sources, or replace current locomotion exports/default launch selections.

The remote backpack reference contains `training/wheel_align/pyproject.toml`, `uv.lock`, assets and `training/wandb_logging.py`. Reuse its locked Linux/CUDA environment initially; it pins MuJoCo/MJX 3.6.0, JAX 0.8.2 and Brax 0.14.2. Native historical audits used other versions, so validate the selected model under this exact training environment rather than assuming equivalence. Resolve LFS files before loading models/configurations. Record any necessary dependency change in a new lockfile; do not alter shared GPU drivers or the user's existing environments to satisfy the task.

## 2. W&B destination: explicit user override

Project URL: https://wandb.ai/QuadMorph/Align%20triangle

Use these literal SDK values:

```python
entity = "QuadMorph"
project = "Align triangle"
mode = "online"
```

This destination replaces the older default `wheel-leg lift and align triangle base` FOR THIS TASK. Do not pass the URL or `%20` as the project name. Pass the override through training, evaluation, checkpoint selection and video workers. Leave unrelated projects' defaults unchanged.

Reuse `training/wandb_logging.py:ExperimentLogger` with explicit `entity`, `project` and `mode`; extend its artifact/audit collection if the new package uses new filenames/stage names. Its existing audit scopes assume the old curriculum and must not mislabel new results. The project URL was supplied by the user; access was not verified by the laptop author. Authenticate using existing PC credentials or local `wandb login`. Never put an API key in commands, source, logs or chat. Verify authenticated access before a long training run. If access fails, continue implementation/local preflight, but report the logging blocker instead of silently launching an offline long run. Provide explicit offline support for later user selection.

Log all of the following:

- Positive environment steps, individual reward terms, task success, phase durations, clearance, target tracking, body motion, saturation and failure reasons.
- Configuration, seed, selected curriculum stage, source commits, dirty patch if any, dependency lock hash, model/geometry hashes, gain and action-processing settings, and observation/action schema.
- Periodic and final checkpoints, selected checkpoint, exact export and their SHA-256 hashes; full failed and successful audit records.
- Genuine policy rollout MP4s as `wandb.Video` through `ExperimentLogger.video`, so they appear in the run's Media view. Uploading an artifact containing an MP4 is not sufficient.
- A video for each stage's selected checkpoint; periodic videos approximately every 5 million steps; final and selected full-sequence videos; at least one failed-case video if audits fail. A smaller periodic subset is acceptable, but label its scope.
- Captions/overlays: simulation, seed, checkpoint step, active leg, phase, desired/actual clearance, rotation permission, target angle error, completion and early-termination reason. Retain local MP4 and synchronized trace CSV/JSON.

Use distinct, linked run IDs for stages/changed experiments and a shared group such as `align-direct-position-YYYYMMDD-seed0`. Preserve run IDs for upload resumption; W&B logging resumption is not optimizer-state resumption. Verify remote scalar history and actual Media files after upload/finish; a local `logged` flag is not cloud verification. Report offline data as not uploaded. Never label a high reward or attractive video a task pass.

Official media reference: https://docs.wandb.ai/models/track/log/media

## 3. Robot/model contract

Canonical motor order is FR, FL, BR, BL, each with motors 1, 2, 3. Proximal action rows: `[0,1,3,4,6,7,9,10]`. Hub rows: `[2,5,8,11]`. External command indices remain stand=0, FL=1, FR=2, BR=3, BL=4; do not confuse command order with motor order.

Use the cold WHEEL geometry with the heating backpack and the confirmed additional 9 mm outward assembly translation. The remote backpack model already includes the spacing: do not add 9 mm again. Its modeled backpack mass is 0.60191707499 kg. Preserve the source inertia tensor and mounting; that numeric model value is not a new measurement by this task.

Maintain continuous hubs, physical proximal limits, wheel-floor contacts, wheel-wheel and wheel-body clearance checks. The historical robot-info leg-profile hub hard stops do not apply. Retain documented conservative collision proxies required by the selected MJX backend. Backpack self-collision exclusions already explicitly requested by the user remain; do not use that exception to disable other robot collisions. Validate any modified collision arrangement in both native MuJoCo and MJX.

Proximal nominal pose, in eight-action order:

```text
[1, 0, -1, 0, 1, 0, -1, 0] radians
```

Proximal initial gains: kp=5, kd=0.25, following current alignment/locomotion references. For hubs, reuse robot-code `launch/keyframe_config.json` and `keyframe_align/controller.hpp`: position kp=4, kd=0.15; near-target integral ki=0.5 Nm/(rad*s), cap=0.10 Nm, window=0.10 rad, with its anti-windup/reset conditions. These are recorded baseline controller settings, not guaranteed optimal settings for the new policy. Keep identical equations/gains in simulation and prospective runtime. Do not introduce a second outer velocity PD on top of the hub position controller.

Hardware source declares position/velocity/effort/kp/kd command interfaces and position/velocity state interfaces. Treat any effort reconstructed from PD error as an estimate, not a force sensor. Preserve configured effort/gain limits (source declares 3 Nm effort, kp ceiling 10 and kd ceiling 1); verify what the hardware actually clamps instead of assuming the feedforward-effort bound alone guarantees a total physical torque limit. Model final PD plus feedforward torque saturation consistently in simulation.

Convert the imported model's four velocity actuators to the agreed hub POSITION semantics. Update its actuator randomizer as well: the old wheel randomizer deliberately forces position stiffness to zero and is incompatible with this task. Account for joint refs/bias terms explicitly; do not accidentally double-count q0 in the actuator or exporter.

## 4. Direct policy actions and execution

Start with eight tanh actor outputs and the following mapping:

```text
q_target = clip(q_nominal + action_scale * action, proximal_lower, proximal_upper)
```

Initial simulation action scales: `[0.5,1.6]` repeated four times, in radians. These are design starting values with substantially more authority than v5's 0.02-rad active correction. Verify they contain feasible all-four-leg lift/balance poses within the current hardware limits. Do not expand physical limits to make a pose reachable. Log clipping rates. Change a scale only with documented reachability/training evidence and regenerate the contract fixtures.

These are absolute targets around q_nominal, NOT accumulated increments, NOT residuals on a scripted lift, and NOT the whole-body walking policy's weights. The policy can move all eight joints in every normal phase. Do not retain v5's residual fade-to-zero, positive-only active-hip residual restriction, scripted eight-joint apex target, or VERIFY target freeze.

Use a nominal 520 Hz command update and 52 Hz actor rate (`repeat_action=10`, control_dt=10/520 s), matching current v5 timing. Simulate ten command/physics substeps per actor inference. Keep measured dt semantics explicit in C++; do not run a 50 Hz command schedule while claiming exact parity.

Reuse bounded velocity/acceleration target execution rather than introducing a heavy global low-pass filter. Initial simulation limits may reuse the current keyframe baseline (abduction 0.45 rad/s, hip 0.65 rad/s, acceleration 2 rad/s^2); these are initial experiment settings, not immutable physical maxima. Instrument when they bind and compare requested, filtered and measured motion. If the policy is persistently command-limited, revise these limits deliberately and retrain under the revised values; do not compensate with untested deployment-only changes. Match previous-action and applied-target state exactly across simulation/export/runtime.

Hub alignment references retain the current 0.5 rad/s and 1.2 rad/s^2 reference limits. Choose the nearest equivalent continuous target to `wrap(home_i + pi)` on activation of a leg request; hold that winding choice through the maneuver. Supporting hubs retain captured/aligned holds. Continuous winding must not become a long rotation when switching from wrapped observations to motor position commands. Do not apply locomotion wheel-forward signs to these angle targets.

At entry, retain valid startup homes and capture fresh holds and current command/measurement state. Initialize policy/filter history for a continuous handoff from supported stand. During normal operation the actor remains active even when hub rotation is gated off. An emergency stop retains the existing robot stop semantics; this handoff does not design a new physical stop behavior.

## 5. Observation and state-machine specification

Use only robot-available actor inputs. Ground-truth world height, contact force, base velocity or contact flags may be used for rewards/audits, not silently supplied to the actor. Start with one frame and the explicit software state below; no recurrent-network runtime is needed. If history is later added, version it and retrain.

Proposed first contract: 91 observations, 8 actions, ID `quadmorph-align-direct-position-v1`. Define exact offsets and transforms in one machine-readable schema and generate/check C++ and Python against it before training.

| Offset | Width | Signal |
| ---: | ---: | --- |
| 0 | 3 | Body angular velocity |
| 3 | 3 | Projected gravity, existing body convention |
| 6 | 5 | Active command one-hot: stand, FL, FR, BR, BL |
| 11 | 5 | Motion phase: IDLE, LIFT, ROTATE, LOWER, HOLD; alignment verification remains in ROTATE |
| 16 | 8 | Proximal joint positions minus q_nominal |
| 24 | 8 | Sine then cosine of four hub angles relative to startup home, canonical order |
| 32 | 12 | Measured joint velocities, fixed documented scaling |
| 44 | 8 | Previous raw normalized actor output |
| 52 | 8 | Last applied proximal position targets minus q_nominal |
| 60 | 8 | Applied proximal target velocities |
| 68 | 1 | Desired wheel-bottom clearance above ground |
| 69 | 1 | Desired clearance rate |
| 70 | 1 | Whether hub alignment motion is actually enabled for the upcoming interval |
| 71 | 4 | Commanded hub reference velocities for the upcoming interval |
| 75 | 8 | Sine then cosine of four instantaneous hub reference-minus-measurement errors |
| 83 | 8 | Sine then cosine of four final goal-minus-measurement errors |

Compute phase/gates/reference commands from the current measured state before actor inference; execute those same reference commands over the next interval. Do not provide future measurements. Supporting final goals are their holds, not arbitrary startup targets. Document reset values for every block and normalization folding in the export. The pending requested leg does not replace the active leg until lowering completes.

State-machine behavior:

1. IDLE/HOLD: all-wheel support, desired clearance zero; actor remains active. Weak nominal-pose regularization may help here.
2. LIFT: select one wheel; ramp requested clearance smoothly from zero to 30 mm over approximately four seconds (initial design value). Hold the requested clearance afterward, allowing time for the policy to shift support. No forced hip-angle threshold or scripted body trajectory.
3. ROTATE: enable alignment only after continuously qualified model-based clearance, spacing and attitude conditions. Start with >10 mm conservative floor clearance, >10 mm wheel spacing, >5 mm body spacing, tilt <0.12 rad and angular speed <0.3 rad/s for 0.2 seconds. These are candidate simulation gates, not claims of physical calibration. The saved keyframe hardware trial used a 5 mm floor gate; this new candidate intentionally starts at 10 mm and must be trained/evaluated with that choice. Do not silently lower it to obtain a pass.
4. While rotating/settling: maintain requested clearance and keep the RL actor active. If the gate is lost, stop advancing the hub reference with the existing bounded hold behavior; do not continue a blind spin or disable proximal balance corrections. Audit any residual wheel motion near contact.
5. Alignment complete: actual wrapped angle error <0.025 rad and actual hub speed <0.08 rad/s for 0.5 s, following the current position-control baseline.
6. LOWER: smoothly ramp requested clearance to zero over about four seconds; actor balances throughout. Hold the verified hub target. Finish only after motion settles and the hardware-available geometry/tracking checks agree; verify true contact separately in simulation. Do not declare touchdown merely because a timer expired or a particular hip angle was reached.
7. Mark completion only after final angle <0.035 rad and speed <0.08 rad/s have settled, and lowering/support criteria pass. Maintain aligned hub holds through subsequent legs. Repeated requests cannot spin an already completed wheel again.
8. NEXT/cancellation while elevated: queue the next command and lower the current leg first. Preserve current desired height/rate and filtered target velocity when replanning descent; never reset a moving reference to an unrelated endpoint. Interrupted/unverified legs remain incomplete.
9. Keep the 48 s per-attempt timeout with logged reason, a soft-lower request and retry latch. No unbounded automatic attempts. Evaluate the entire timeout descent/recovery; timeout is not successful alignment. If normal lowering cannot recover, record failure rather than inventing a successful contact.

Use the corrected conservative floor-estimation method in robot-code `keyframe_align/geometry.hpp`, not v5's highest-support-bottom calculation. Port/test it against current backpack geometry and perturbed poses. Its flat, rigid-ground and radius-bound assumptions remain assumptions; compare with independent true wheel-floor geometry in audits.

## 6. Reward and curriculum

Reward the physical task, not a particular lifting hip angle. Use actual cylinder/shape bottom clearance, not a hub height or an old foot-site coordinate. Main terms:

- Selected-wheel clearance tracking, ramped with the command; no reward for excess lift beyond the target band.
- Support on the other three wheels, low support-contact slip and tolerable body displacement. Permit deliberate weight shift; do not reward all eight joints staying at neutral during lifting.
- Stable attitude/angular velocity, with enough tolerance for feasible weight transfer.
- Wheel-wheel/body separation and avoidance of unwanted contacts.
- Small command changes, excessive effort/saturation and contact-approach speed. Regularize smoothness without making immobility the easiest solution.
- Stage completion and maintained alignment/support through lowering and following legs.

The deterministic hub servo does not remove the need to train with its reaction forces. Rotate real simulated hubs during training, including starts, stops, both directions, near-pi targets and gate loss. A `rotation_enabled` observation alone is insufficient. The actor must experience the actual servo dynamics and physical coupling.

Use the existing PPO implementation as the starting point: actor 128/128/128 ELU with tanh-squashed eight actions; critic 256/256/256; normalization enabled. Retain compatible locked PPO defaults initially (the reference uses learning rate 3e-4, discount .999, entropy .01, unroll 20, batch 256, 16 minibatches, four updates). Verify valid batch/environment divisibility and the exact deterministic action transformation in export. Do not load v5 weights into this incompatible observation/action contract.

Proposed FIRST experiment budget: approximately 50 million total environment steps, seed 0, 2048 parallel environments, split below. This is a planning default, not a user-specified exact compute budget or guaranteed sufficient training. Preserve batch rounding and report actual counts. Do not launch a large sweep or repeated 50M attempts without first reporting the failed first experiment and a concrete next hypothesis.

| Stage | Initial budget | Task |
| --- | ---: | --- |
| A: lift/hold/lower | 5M | Randomize selected leg uniformly across all four; ramp lift, hold, lower, stand. Hubs hold. Progress from easier lift heights to the full target. |
| B: lift with rotation | 10M | All four selected-leg cases, random initial phases/home offsets, actual deterministic hub rotation, variable hold duration and canceled requests. |
| C: sequences | 35M | Complete four-leg sequences, retained aligned targets, varied valid order and repeated commands; include early NEXT and timeout recovery. |

Keep the same actor/schema/action scales across these stages. Transfer compatible actor/normalizer state explicitly; document whether optimizer/critic are resumed or reset. Select checkpoints by task audits before transferring. If no Stage A/B candidate demonstrates its task without hard failures, stop the chain, save findings/video and fix the cause; do not burn the remaining budget by blindly transferring failed weights. Keep full-angle/full-sequence diagnostics clearly distinct from each stage's promotion test.

Start with nominal physics and widen uncertainty through the curriculum. Declare numeric ranges in config before long training. Suggested initial final-stage uncertainty: wheel radius +/-2.5 mm independently with collision proxies updated; friction 0.4–1.2; actuator gain multipliers 0.8–1.2; backpack mass +/-10%; total-model COM perturbations up to 10 mm; IMU roll/pitch biases up to 0.04 rad; proximal zero errors up to 0.01 rad; measured/modest latency variation. These are simulation hypotheses, not measured distributions. Preserve consistent inertia scaling and don't accidentally add backpack mass twice. Do not reuse the velocity-hub gain randomizer. Introduce mild external disturbances only after the nominal task works and record their force/duration precisely.

## 7. Preflight and diagnostics before expensive training

Implement a new package-specific preflight. The old v5 preflight passing is not evidence that the new policy interface is valid.

- Model loads in native MuJoCo and MJX; all twelve actuators implement the chosen position PD law; geometry, mass, 9 mm offset, continuous hubs and collision pairs are checked by name/hash.
- Proximal target ranges allow feasible lifts for every leg. A manual/test target may be used for geometry checks, but an animation or supplied support force is not a learned balance result.
- Hand-computed actuator fixtures confirm PD signs, units, joint refs, q0 mapping, velocity feedforward (if any), integral and saturation. Zero action means the expected proximal nominal pose.
- Python and C++ controller helpers agree on synthetic encoder/IMU sequences, wrapping/winding, all phases, cancellation, gates, completion, timeout, reentry and command filtering.
- Native simulation uses the same execution and sensor-derived gates as MJX. Simulator truth is only for reward/audit; no privileged actor leak.
- Single/batched reset, one compiled physics step, a short actual PPO smoke run, nonzero parameter update, checkpoint restore and video rendering all work. Label smoke runs; do not present them as trained success.
- W&B online access and an actual smoke-rollout Media upload work in QuadMorph / Align triangle.

Log `raw_action`, `q_target`, `q_applied`, `q_measured`, target/measured velocities, estimated torque/saturation, all gate components and phase transitions. This distinguishes hesitant policy requests, restrictive filters and physical tracking lag. Do not label stuttering a weak motor or insufficient kp without this comparison.

## 8. Acceptance and checkpoint selection

Freeze acceptance definitions before evaluating final candidates. These are engineering simulation gates, not physical certification. Use fixed seeds disjoint from training/checkpoint-promotion tuning; preserve the exact scenario manifest and all failures.

Initial final acceptance:

- Nominal: 64/64 complete four-wheel sequences, covering both rotation directions, near-pi errors, all leg indices and multiple wheel-selection orders.
- Randomized: at least 60/64 complete sequences on EACH of two held-out seeds. Report per-leg failures; pooled success cannot hide a systematically failing FL.
- Across all accepted audit sets: zero falls, prohibited robot contacts, unsafe rotations or actuator/position-envelope violations. Incomplete randomized cases can only be explicitly reported recoverable task failures; do not average away hard failures.
- Rotation: independently measured true floor clearance >5 mm during material hub motion, wheel surface gap >5 mm, body gap >0; preserve stricter runtime gate values above. Audit each physics substep, including gate-loss deceleration, not only actor samples.
- Final angle/speed/settling meet the section 5 thresholds after lowering and remain valid while subsequent wheels move. Completed mask must represent actual success.
- Ground approach speed <0.10 m/s at touchdown, with a preferred target below 0.03 m/s. Report peak/contact impulse as well as averages. These are provisional engineering targets, not measured damage thresholds.
- No falls during lift-and-hold endurance: each selected leg can hold the full clearance target for 20 s while the servo performs representative alignment motions. Do not command repeated full rotations simply to fill the hold time.
- Interrupted suite: at least 16 cases per active phase (LIFT, ROTATE, LOWER), balanced across legs. Every case must preserve single-leg sequencing, finish the requested supported recovery within the documented timeout/recovery budget, and avoid false completion. Add software tests for rapid NEXT, repeated stand and reentry after multiple hub revolutions.
- Log lift time, full sequence time, actual target tracking and drift. Investigate lift readiness taking more than 10 s after a normal request. Keep the 48 s timeout; don't increase it to hide a stalled policy.

Compare against the current deterministic baseline in the SAME backpack/spacing model where feasible. The September 12 floor successes and older 22-case matrix belong to their original model/configuration and must not be relabeled as new-model results. Read-only hardware records may inform diagnosis, but no robot access is needed for this task.

Select an actual checkpoint satisfying these gates, not the last checkpoint or highest training reward. Record the selected step/hash separately from final training step/hash. If none passes, deliver the best diagnostic candidate clearly marked FAILED, the failed scenarios and a concise explanation. Do not silently relax thresholds, edit audit results or make a failed export the robot default.

## 9. Runtime adapter and export

Use a new behavior/contract ID in the existing neural controller. Old v5 exports must continue to load only under their old ABI; the new eight-output model must not be mistaken for a twelve-output walking policy or old eight-output residual policy.

Keep existing policies and default hardware launches unchanged. An inactive candidate configuration/minimal trial launch can be prepared for later review. Reuse the shared live-session calibration requirement, canonical joint mapping, stop paths and sensor-freshness handling. Do not embed PC/simulation homes or old saved calibration IDs in the export.

Export: RTNeural-compatible ELU hidden layers, correct eight-action tanh transformation, normalization folded consistently, float32 fixtures and explicit named observation/action schema. Include q_nominal, action scales, per-joint limits/gains, position action types for all twelve motor commands, hub servo/state settings, filter settings, control timing, model/source hashes and calibration convention. Eight learned outputs are expanded with four deterministic hub targets; model output count and motor-command count are different fields.

Use the existing exporter as a reference, but replace old v5 dimensions/mapping/provenance checks with checks for this new immutable training source. Never bypass the old checks to pass a new policy through the old ABI. Per-joint kps/kds must be consumed by the candidate runtime configuration rather than assumed effective because they appear in descriptive JSON.

Test at least 128 real trained-network inference fixtures against the deterministic training actor, including phase edges and near-wrap inputs. Require maximum absolute normalized action error <=1e-5 unless a documented numerical analysis supports another tolerance. Check exact position target/servo/phase parity over extended traces. Complete applicable existing controller regression tests and the new lifecycle/update tests where ROS/C++ are available. If the PC cannot run ARM64/ROS checks, explicitly mark them pending and return reproducible commands/fixtures; never substitute a zero-output test model as evidence about the trained export.

The result is a reviewable candidate, not a deployed or physically validated controller. Before any later hardware trial, the separate hardware agent must follow robot-code STARTUP_CALIBRATION.md, check the actual installed overlay and obtain the required operator physical confirmation. That is not a prerequisite for this PC-only implementation/training task.

## 10. Execution deliverables and stop conditions

Create and commit the implementation on the new task branch after focused local tests. Keep historical assets/checkouts unchanged. Return the branch and commit; if it is only local, provide a transferable patch/bundle or clearly state that it has not been pushed. No merge into robot-code or default-policy replacement is requested here.

Add package-specific entry points for `preflight`, `train`, `evaluate`, `select_checkpoint`, `export` and `render/log video`, adapting existing modules where sensible. Write a tested `run_training.sh` (or equivalent) that starts the staged chain, propagates the exact W&B override, checks promotion results and exits on failure. Record its actual invocation, tmux name, absolute checkout/run/log paths and reattachment command.

Those new commands do NOT exist merely because this handoff names them. Implement and verify their `--help`/smoke behavior before publishing runnable instructions. Do not paste the historical v5 training command and claim it trains this design.

Return a final `PC_HANDOFF_RESULT.md` containing:

1. What was implemented, source/branch and any departures from this plan with reasons.
2. GPU/environment identity, lockfile, exact configuration and real steps per stage.
3. All W&B run URLs under QuadMorph / Align triangle; verified scalar and Media upload status.
4. Selected checkpoint/export hashes and local/transferable locations, plus final-versus-selected distinction.
5. Nominal/randomized/interrupted/endurance results, per-leg breakdown and failed cases/videos.
6. Python/native/RTNeural/ROS checks passed, failed or unavailable.
7. Measured candidate limitations and what remains before a supervised hardware trial.

Stop long training and diagnose if the model/controller contract is wrong, W&B access is unavailable without an explicit offline choice, no promotion checkpoint passes, or the initial experiment budget is exhausted without success. Do not silently repeat large runs. If a training process is still running when responding, say RUNNING, preserve tmux/local logs and report actual progress rather than completion or an unscheduled monitoring promise.

## 11. Why this differs from the older attempts

- v5's active-leg correction is only 0.02 rad; support corrections are also small. It can fade corrections to zero when a lift is not comfortable and freeze targets during VERIFY. This new policy has direct meaningful pose authority throughout the maneuver.
- The hub interface here is position control, including the retained bounded integral behavior if used, not the old outer angle-to-velocity PD.
- Earlier collision omissions, outdated geometry and floor-estimator errors must not be reintroduced. Today's FL recording demonstrates a blocked estimate, not proof of a failed hub or weak motor.
- A known-good stand pose is an action reference, not a raw encoder calibration. Startup home is separate from current hold and is shared across controllers within a valid encoder session.
- More GPU steps cannot compensate for an incompatible motor model, missing reaction forces, contradictory rewards or deployment-only action processing.

## 12. Source index and evidence boundaries

All runtime paths below are relative to robot-code commit `085a3f0a620a9942459fab24eea7fa507786ec7b` unless another reference is named. Use `git show <commit>:<path>` without switching an active checkout.

| Claim/reference | Source |
| --- | --- |
| Latest locomotion exports are separate and must be retained | `LATEST_POLICIES.md`, `policies/latest.json` |
| Session calibration, fresh holds, physical startup procedure | `STARTUP_CALIBRATION.md`, `ros2_ws/src/robot_calibration/` |
| Actual motor interfaces, limits and CAN mapping | `ros2_ws/src/pupper_v3_description/description/components.xacro`, `ros2_ws/src/control_board_hardware_interface/src/control_board_hardware_interface.cpp` |
| Approved upper nominal pose | `ros2_ws/src/neural_controller/include/neural_controller/policy_home.hpp`, `hardware_testing/start_pose/approved_upper_pose.json` |
| Direct position-mode hub baseline, including bounded integral and winding | `ros2_ws/src/neural_controller/include/neural_controller/keyframe_align/controller.hpp`, `ros2_ws/src/neural_controller/launch/keyframe_config.json` |
| Operator-reported successful prior floor trials, not new-model certification | `hardware_testing/keyframe_align/LAB_BASELINE_20260912.md` |
| Corrected conservative floor bound | `ros2_ws/src/neural_controller/include/neural_controller/keyframe_align/geometry.hpp` |
| Existing RTNeural loader/observations/actions | `ros2_ws/src/neural_controller/src/neural_controller.cpp` |
| v5's restricted residuals and fade/VERIFY behavior | `ros2_ws/src/neural_controller/include/neural_controller/wheel_align_motion.hpp`, `wheel_align_reference_data.hpp` beside it |
| v5 nominal/randomized audit boundary | `ALIGN_V5_LAB.md` |
| September 14 FL investigation | Laptop-local, untracked `hardware_testing/align_v5_2026-09-14/findings.md`; it may not be on the PC. FL stayed in LIFT below the estimated 10 mm threshold; other legs were advanced before completion. No physical diagnosis is proven by that summary. |
| Physical wheel/leg meaning and confirmed 9 mm spacing | `origin/robot-info:robot_info/QUADMORPH.md`; pin its resolved commit in the task record |
| Remotely available training model, backpack mass/inertia and restrictions | `dde1f961b519f53379d80fc4e77eab797b38be73:models/heating_module/README.md`, `spec.json`, `training/wheel_align/model.xml` |
| Existing PPO/export/audit/media infrastructure | `dde1f961b519f53379d80fc4e77eab797b38be73:training/wheel_align/` and `training/wandb_logging.py` |

Hardware facts, operator reports, implementation behavior and proposed experiment settings are deliberately distinguished above. Use the latest user's physical confirmations to resolve discrepancies; do not promote a legacy comment into hardware truth.
