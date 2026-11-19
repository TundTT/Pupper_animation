# QuadMorph alignment: superseded 8-action pilot history

The current supported-lift redesign is in [ALIGNMENT_DIAGNOSIS.md](ALIGNMENT_DIAGNOSIS.md).
This file records the previous design and its reasoning; it is not the current controller contract.

Work is isolated to the `combine` worktree. References below were read through
`git show` in this worktree; no files or processes in the walking worktree were
accessed. The user confirms all three reference policy families have run on the
robot, and the hub supports both position and velocity control on hardware.

## What the runs actually showed

| Run | Steps examined | Evidence |
| --- | ---: | --- |
| Velocity, `esgvuy4v` | 40.3M | Reward improved from -3366 to -104, but mixed LOWER/HOLD error remained 1.10 rad; fixed showcase completed 2/4. |
| Position PD with exploratory ±pi residual, `3pmz5020` | 25M | At 10M, reward -29 masked episode length 209/1800 and a fallen showcase. At 20M length was 257; at 25M it was 248. Repeated showcases fell. |

The latter is not convergence: partial resets contribute already-completed legs
to `completed_avg`, and terminating a poor trajectory early reduces its accumulated
negative reward. A pure actuator change helped isolated angle restoration but
did not solve the task. In actual MuJoCo free-space physics, a 0.3 rad wheel
perturbation decayed to 0.01021 rad in 0.6 s with PD; velocity-zero remained at
0.30000 rad. Allowing the policy to offset the PD goal by ±pi undermined the
reason for introducing a spring toward the calibrated goal.

The PD-residual trainer PID 2865076 was stopped after this audit. Its checkpoints,
W&B run, log, and a `pd_residual_baseline_source/` snapshot under its output
directory are retained. No old checkpoints are deleted.

## Concrete reference patterns

Reference revisions: wheel `a9ef276e50366b598b05c70ae0e4bda53359b20c`, master
`b933b5c889e65106ad7a00a957ebb5675658ca23`, leg
`6b55e30ff224193a2a21c8dc3dbb88e029a417c8`.

| Reference | Concrete values/history | Alignment decision |
| --- | --- | --- |
| `master:.../workspace/leg_lift_env.py`, `configs.py`, `README.md` | Small uniform action scale 0.3 prevented the required lift; dead-zone shaping produced a stationary policy. Mature config: hip scale 1.6, dense lift progress 3, height reward 8, stance 3.5, support 1, orientation 2, torso height 1.5. | Measure wheel-form reachability instead of importing its 12 cm lift. Add progress 3 and clearance 8; use these stance/support/posture weights. |
| Same leg-lift history | Selected hip action increment limited to 0.05; lowering cooldown 20 steps; stance joints unrestricted. Hardware notes confirm smoother raising and identify abrupt lowering as a separate problem. | Clamp active and lowering hip increments to 0.05, leaving stance hips unrestricted. Phase determines scope instead of a fixed lowering timer. |
| Same leg-lift history | Aggregate reward hid a failed fourth leg. Warm starts improved refinement stability. | Per-leg and all-four audit, checkpoint every evaluation, retain best by task metrics. Fresh weights now because observation/action dimensions changed; future compatible refinements can warm-start. |
| Wheel `wheel_env.py`, `configs.py`, `randomize.py` | Position kp/kd 5/0.25; wheel velocity kv 0.35; wheel scale 20.83 rad/s. Friction 0.25–1.5, radius 48±2.5 mm, kp multipliers 0.5–1.6, kd 0.5–2, mass/inertia 0.9–1.3, CoM shifts ±20 mm, frequent pushes. | Preserve shared physics, latency, contact geometry and broad DR. Alignment alone uses wheel position kp/kd 2/0.35, with randomized position-gain multipliers and original ±3 Nm torque cap. |
| `leg:.../workspace/walk_config.py` and training history | Swing-clearance weight 6 corrected an old 0.5 reward whose contribution was too small. Signed reward; hip/abd/knee scales 0.5/0.25/0.5; kp/kd 5/0.25. | Ensure the actual lift incentive can compete with posture reward; retain signed totals. Do not transplant walking geometry/action ranges. |
| Walking `walk_train.py` | Saves step checkpoints, metrics.jsonl, best checkpoint; recent W&B iterations used learning rates 1e-4 and 5e-5. | Same persistence pattern, task-success ranking rather than reward alone, gentler learning rate 1e-4. |

`...` above denotes `mujoco_playground`; branch files can be inspected with
`git show master:mujoco_playground/workspace/configs.py`, etc. Leg-lift README
contains historical values; current code, rather than stale prose, supplies the
1.6 rad hip scale. Its robot geometry and balance bias are not copied.

## Current controller

RL outputs eight abduction/hip actions. A measured-phase wheel position reference
advances toward the persistent calibrated `home + pi` only after unloading, at
at most 3 rad/s reference slew. This is a bounded target-position controller,
not a guarantee on actual angular speed. When a rotating active leg touches down,
its pending angular reference is stopped and queued angular commands are flushed
at the next control update. Verified goals retain a restoring spring throughout
lowering, all subsequent activations, and idle. A failed/interrupted rotation
cannot claim completion merely by lowering onto the ground.

Every physics substep uses `ctrl = raw_qpos + wrap(reference - raw_qpos)` after
latency. The wheel-drive env remains velocity-actuated. Observation size is
356 (four 89-value frames), including target and servo-reference errors encoded
as sine/cosine. No raw unbounded angle is observed. This policy is incompatible
with the earlier 340-observation/12-action checkpoints and legacy drive exporter.

Fixed-torso forward kinematics showed that a maximum-radius wheel at 0.4 rad hip
lift had only 2.942 mm clearance, below the 3 mm unload gate. At 0.6 rad it had
8.120 mm (all four pass the regression). Hip authority is now 0.6 rad; actual
clearance reward still prefers 3–7 mm, and excess height is penalized beyond 7 mm.
This provides exploration margin without rewarding a 12 cm leg-lift maneuver.

Hold reward is summed per completed leg (16 each), rather than averaged into a
constant bonus. Fall penalty is -50 total, formerly -5. Extra completion must
produce lasting value, and early failure must not appear successful. The 20 mm
soft drift band, full-circle encoder phases, one-time calibration, external
button sequence, partial-completion starts, idle periods, and broad wheel DR stay.

## Budget and evaluation

The default is a **12M-step pilot**, 4096 environments, five evaluations including
initialization (roughly every 3M), batch 256 × 16 minibatches, four PPO updates,
unroll 20, learning rate 1e-4, entropy 0.005, discount 0.995, and 128×128 ELU
actor/critic. PPO may round the requested step budget up to a complete batch.

This is a trial budget, not a claim that alignment converges in 12M. Evidence:

- Wheel training used 200M (~14m27s), with convergence reported around 43M and
  little benefit from the remaining ~150M.
- Hardware-successful leg-lift refinements used 200M warm starts, 8192 envs,
  128×128×128 ELU, batch 256 × 32, lr 3e-4, gamma 0.97, roughly 15.5–17.2 minutes.
  An earlier initial ELU run used 150M. Its workstation notes recommend 10M trials
  after reward changes before long runs.
- Read-only W&B walking history showed actual 43.2M and 86.4M runs with 8192 envs,
  ~8–12 minute runtimes: `5uf5o2f3` 43.2M/lr1e-4; `mz6ipazu` 86.4M/lr1e-4;
  `zw331stu` 86.4M/lr5e-5. A selected checkpoint in the latter was 21,626,880 steps.
  These are concrete run histories, not a claim every checkpoint was deployed.

4096 is half the proven 8192 reference concurrency and doubles the prior align
batch of environments; it is not an assertion that more environments always
reduce wall time. Long multi-leg episodes and broad DR still make this harder
than a free-space angle servo. Keep the smaller 128×128 network initially.

Each evaluation audits 16 fixed-seed randomized robots starting with **zero**
completed legs for 1800 steps, no reset after failure. It logs every leg's
completion fraction, all-four completion, survival, final all-four settlement,
conditional held-error mean/p95 with sample count, drift p95/max, and scrubbing.
All-four success requires surviving to the end with all completed wheels within
angle/speed tolerance. It does not impose the aspirational drift band as a hard
success condition. Best checkpoint ranks success, number completed, survival,
held error, then drift. This small fixed-seed audit needs a larger unseen-seed
evaluation before any eventual deployment judgment.

At the pilot cap, extend a compatible checkpoint only if true completion and
holding improve; redesign if the policy remains a stable non-lifter or falls.
Reward or partial-start `completed_avg` alone must not justify more compute.

Completed pilot (zero all-four successes at 12.124M): PID **2919741**, GPU **1**, log
`mujoco_playground/train_wheel_align_phase.log`, output
`workspace/output/wheel-align_2026-09-07_18-02-52/`, W&B
[xsqrtn5j](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/xsqrtn5j).
Launched detached with `nohup setsid`, PPID 1, using this worktree's `.venv`.
At 3,031,040 steps, the full-sequence audit improved from 0.0625 to 0.3125
completed legs per episode and drift p95 from 0.374 to 0.138 m. Survival was
14/16, no episode completed all four, and held-error p95 was 0.0303 rad across
4432 held leg-timestep samples. These few successful leg completions support
continuing the 12M pilot, not claiming convergence or committing to a longer run.
