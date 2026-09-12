# Wheel policy for the 9 mm outward assembly gap

`policy_wheel.json` now contains `wheel_2026-09-11_21-58-55`, retrained from scratch on wheel commit `88f6a88` for 201,850,880 steps. The model moves each wheel assembly 9 mm outward. Controller interface, gains, observation layout, action scales and wheel direction signs match the previous export exactly.

## Validation

Matched simulation tests: 11 fixed commands, three seeds each, eight seconds per trial; no falls in 33/33 trials. Versus the old policy on the original XML, mean XY error is 0.03410 vs 0.03190 m/s (+6.9%), yaw error 0.06565 vs 0.06404 rad/s (+2.5%), and tilt 0.13552 vs 0.15747 degrees. These small differences come from one training seed. The old policy also passes 33/33 on the new XML.

`comparison.json` preserves all cases and per-command metrics; `protocol.json` defines the tests. `manifest.json` records source/checkpoint/export hashes. The C++ test compares 128 independent JAX reference actions against the RTNeural Eigen backend used by the controller and checks mixed-actuator metadata. Run it without ROS or a robot:

```bash
bash scripts/test_wheel_policy.sh
```

The local check passed all 128 cases with maximum absolute action error **4.17233e-7** (tolerance 3e-5); YAML gain/actuator metadata also matched. See `cpp_tests.log`.

It is also registered as `wheel_policy_contract_and_inference` in CTest. `CXX` may select a compiler wrapper.

## Deployment status

This release updates the branch artifact. This checkpoint has not been tested on physical hardware; the previous run 3 retains its historical hardware validation. Follow `STARTUP_CALIBRATION.md` and `WHEEL_TESTING.md` for a later operator-supervised session. No hardware stack was started by this update.

The previous run-3 policy remains recoverable from Git history. Its weights predate the new inference fixtures, so regenerate matching JAX fixtures if restoring that policy.
