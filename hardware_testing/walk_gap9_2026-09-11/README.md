# Selected walking policy for the 9 mm assembly gap

The active `ros2_ws/src/neural_controller/launch/policy_walk_v2.json` is replaced by step **12,779,520** from `walk_2026-09-11_23-02-17`, W&B run **j3xez9z5**. This is the checkpoint shown in `eval/video_selected`; it is different from the final/best-training-reward weights. The selected export is a drop-in replacement for the existing `neural_controller_walk_v2` / **Square (button 3)** binding.

The active wheel policy is already the September 11 gap retrain at `launch/policy_wheel.json`, selected through `neural_controller_wheel` / **Triangle (button 2)**. Its SHA256 is pinned in `selection.json` alongside the leg release. This update keeps the current controller wiring, home poses, gains, action scales, command limits, startup behavior, calibration and emergency-stop settings.

## Simulation results

The leg checkpoint passes 39/39 eight-second selection trials and 39/39 twelve-second trials using fresh reset seeds and a different floor friction. In the latter comparison with the old policy on the modified XML, yaw-rate error improves from 0.10055 to 0.07404 rad/s (26.4% lower); XY error increases from 0.04095 to 0.04297 m/s (4.9% higher); normalized action changes increase 7.1%.

At forward 0.2 m/s and friction 2, front strides are 90.6/91.2 mm versus 98.8/99.9 mm; cadence is 2.40 versus 2.12 Hz, major swing time is 200 versus 228 ms, and front peaks are 12.4/14.1 versus 11.8/12.7 mm. Both policies pass 3/3 forward gait trials. The selected video covers the full 32-second showcase without early termination. These are finite simulation tests; the new checkpoint has no hardware or obstacle-course validation.

## Local checks

```bash
python3 scripts/check_walk_policy.py
bash scripts/test_walk_policy.sh
bash scripts/test_wheel_policy.sh
```

The preflight pins the new export hash and checks the existing Square mapping, actuator settings and target envelopes. C++ tests compare the controller's RTNeural Eigen backend against independent JAX reference actions for both active policies. `CXX` may select a compiler wrapper. Both 128-case checks passed: maximum action errors 1.73599e-6 (leg) and 4.17233e-7 (wheel), below the 3e-5 tolerance. File/target-envelope preflight passed. The preflight’s stale generic X assertion was corrected to the already-existing dedicated X alignment binding; runtime wiring was not changed. See `preflight.json` and `cpp_tests.log` for results.

[Selected video and run](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/j3xez9z5): use `eval/video_selected`. Source/checkpoint/export hashes and the active wheel hash are in `selection.json`; matched measurements are in `holdout.json` and `stride.json`.

This is a branch artifact update. No robot was connected, started or moved. Later hardware use follows `STARTUP_CALIBRATION.md` and `WALK_V2_TESTING.md`. The previous weights and inference fixtures remain in Git history.
