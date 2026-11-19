# Supported-controller refinement, 2026-09-07

This continues the successful supported 3-action controller, with actor, critic,
and running observation statistics restored from
`output/wheel-align_2026-09-07_19-04-12/mjx_params_best` (4,096,000 steps).
The position PD, coherent support shift, clearance feedback, 376 observations,
three policy actions, command timing, and full wheel randomization are preserved.

## Diagnosis

The original 16-sequence audit reproduced 6/16 full successes and 3.3125 completed
legs per episode. Its 11 missed legs break down as follows:

- Five front-right, three front-left, one back-right, and one back-left never
  reached rotation. All ten saturated the 1.2 rad lift reference while still
  loaded. Their final-second mean normal forces were 0.88–12.32 N. Policy balance
  outputs were generally well inside their bounds: missing authority from output
  saturation is not the main explanation.
- The remaining back-right miss verified rotation but stayed 2.82 mm above ground
  during LOWER, with zero load. Its balance correction was not reweighting it.
- Large drift was concentrated in failed lifts (front failure peaks 4.38–7.09 cm).
  Successful full sequences had per-episode p95 drift 2.47–3.53 cm. This is not
  primarily angle-holding failure or uniform idle drift accumulation.

Artifacts: `output/align_diagnosis/supported_best_trace.{json,npz}`. The trace
records commands, phases, clearance/load/angle/speed, lift references, balance
corrections, drift, survival, and completion. Tiny floating-point differences
between compiled audit/trace graphs change some drift samples, not completion.

A separate baseline on 64 unseen full-randomization sequences (seed 1901) achieved
11/64 full successes, 2.8594 completed legs, 100% survival, 4.122 cm p95 drift,
and 0.02046 rad p95 held error. This larger sample shows the original 16 was
optimistic; both cohorts must be compared before claiming general robustness.

The prior PPO evaluation still improved from 2.65 new completions at 4M to 3.10
at 8M, with late KL around 0.0056. Extending a warm start is warranted. The initial
huge KL spike in that run is a reason to retain its learned observation statistics:
Brax restores those statistics along with the actor and critic instead of starting
normalization from zero again.

## Changes

- Add a LIFT-only unloading cost, weight -1: active normal load divided by mean
  supported load (denominator floor 1 N). It supplies a dense weight-transfer
  signal while clearance is nearly constant at ground level. It is zero once
  unloaded and does not reward lifting higher. Existing LOWER contact feedback
  continues to teach reweighting.
- Raise the drift cost magnitude from 0.5 to 1.5, preserving the 35 mm active
  support allowance and 20 mm idle allowance. Required front-leg compensation
  remains permitted, as in the hardware-verified leg-lift pattern.
- Allocate 24M additional steps, 4096 environments, checkpoints every ~4M.
  Preserve 128x128 ELU, learning rate 5e-5, two PPO updates per batch, gamma .995,
  and entropy .005. No changes to disturbance ranges or evaluation difficulty.
- For tied success/completion/survival, choose less drift before tiny differences
  in already-small held-angle error when saving the best checkpoint.
- Add seed selection to the evaluator and a reusable checkpoint audit/trace CLI.

## Verification

`smoke_wheel_align` checks random finite rollouts, full-circle phases, actual
randomized wheel radii, persistent targets, stage guards, full task-state resets,
PD restoring torque across many revolutions, and unchanged wheel-drive behavior.
It now also checks dense unloading progress at fixed clearance, mass invariance,
and no idle unloading reward. `smoke_wheel_align_supported` completes a nominal
four-leg feedback sequence with no falls and 2.175 cm p95 / 2.543 cm max drift.

Reproduce checkpoint metrics using the saved config:

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python \
  -m workspace.audit_align_robustness workspace/output/RUN \
  --count 64 --seed 1901 --output workspace/output/align_diagnosis/RUN_unseen64.json
```

## Refinement run

- W&B: https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/1wpnxywj
- Output: `output/wheel-align_2026-09-07_20-25-49`
- Detached PID: `3049648`, GPU 1; log: `../train_wheel_align_robustness.log`.
- Exact command: output directory's `launch_command.txt`. Training seed 17 draws
  a different training population; the comparison audit seeds stay fixed.
- Sources and config are frozen in the output directory; `warm_start.json`
  records the parent checkpoint and restored state.

| Additional steps | Full successes / 16 | Mean completed | Drift p95, cm |
| --- | --- | --- | --- |
| 0 | 6 | 3.3125 | 5.005 |
| 4,014,080 | 7 | 3.1250 | 6.814 |
| 8,028,160 | 7 | 3.0625 | 6.652 |
| 12,042,240 | 9 | 3.3750 | 6.303 |
| 16,056,320 | 9 | 3.2500 | 6.294 |
| 20,070,400 | 11 | 3.6250 | 5.131 |
| 24,084,480 | 12 | 3.7500 | 4.860 |

All audit checkpoints survived 16/16. Initially the policy overgeneralized a
backward balance correction that solved front-leg unloading but hurt back legs
and drift (`output/align_diagnosis/refine8m_trace.{json,npz}`). Continued training
recovered back-leg reliability and reduced drift without further controller or
reward edits. The final checkpoint is also `mjx_params_best`; front-right,
front-left, back-right, back-left complete in 15/16, 14/16, 15/16, 16/16 respectively.
Conditional held error p95 is 0.0241 rad, still well within the 0.10 rad criterion.
Final fixed showcase completes 4/4 at 3.85 cm p95 drift. The drift improvement on
the fixed audit is modest; success improvement is much larger.

On the independent 64-sequence cohort (seed 1901), full successes rose from 11/64
to 46/64 (17.19% -> 71.88%), mean completion 2.8594 -> 3.6719, survival stayed
100%, and held-error p95 was 0.02470 rad. However, p95 drift rose 4.122 -> 4.840 cm.
Thus the success improvement generalized; the small fixed-cohort drift improvement
did not. Both paired results were added to this run's W&B summary under
`baseline_unseen64/` and `validation_unseen64/`. The run finished cleanly, with
seven checkpoint videos and a final video uploaded; its PID has exited.

## Second pass: retain unloading, emphasize drift

The larger-cohort tradeoff warrants a targeted 12M continuation from the 24M
checkpoint, not a controller redesign. Lower unloading cost magnitude 1 -> .25
and raise drift magnitude 1.5 -> 3. Keep the 35 mm active / 20 mm idle deadbands,
full DR and pushes, action/observation contract, PPO architecture, learning rate,
and checkpoint normalization intact. Training seed stays 17. Evaluate every ~4M.
Use the existing `--no_eval_videos` option to avoid repeated rendering overhead;
the training script still records all metrics/checkpoints and a final rollout
video in the same W&B project. All smoke/regression checks are rerun after the
weight change. The first 24M checkpoint and its run remain available for comparison.

Second run: [olqtuca3](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/olqtuca3),
`output/wheel-align_2026-09-07_20-41-57`, detached PID 3084826, GPU 1,
log `../train_wheel_align_drift.log`. It completed 12,042,240 steps and exited
cleanly. Its exact command and source snapshot are in its output directory.

| Additional steps in second pass | Full successes / 16 | Mean completed | Drift p95, cm |
| --- | --- | --- | --- |
| 0 | 12 | 3.7500 | 4.855 |
| 4,014,080 | 12 | 3.7500 | 4.826 |
| 8,028,160 | 11 | 3.6250 | 4.670 |
| 12,042,240 | 10 | 3.5625 | 4.750 |

Retain `mjx_params_best`, the 4,014,080 checkpoint, not the last `mjx_params`.
The longer drift pass started sacrificing front-leg completion for small drift
reductions. Its separately rendered `rollout_best.mp4` is logged as
`eval/video_best`, with `selected/checkpoint_step=4014080` identifying it. The
ordinary final video corresponds to the last checkpoint, not this selection.

On the same 64-sequence validation cohort, the selected checkpoint achieves
45/64 full successes (70.31%), mean completion 3.6406, 100% survival, p95 drift
4.804 cm, max drift 7.385 cm, and p95 held error 0.02481 rad. This is comparable
to the first refinement's 46/64 and 4.840 cm. Compared with the original baseline,
success is substantially better (11/64 -> 45/64), but drift is worse on this
larger cohort (4.122 -> 4.804 cm), despite the small reduction on the original
16-case audit (5.005 -> 4.826 cm). **Do not claim generalized drift improvement.**
The held-angle error remains far below tolerance, and no validation episode fell.

This iteration used 24M plus 12M additional training steps. The selected policy's
lineage contains 4,096,000 + 24,084,480 + 4,014,080 = 32,194,560 steps; all
intermediate checkpoints and both new W&B runs are preserved. Further blind
training on the stronger drift cost is not supported by the observed decline in
front-leg success. The remaining gap is reliable low-drift support/clearance
under the broad randomized model, not hub angle holding.
