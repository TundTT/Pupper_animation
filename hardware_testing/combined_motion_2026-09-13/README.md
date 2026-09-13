# Combined motion software validation

Current hardware status: see `hardware_session_summary.md` for the subsequent
operator-verified floor roll and walking tests, including Triangle's settling
gate. The paragraphs below record earlier software-only stages and their scope.
The tested Pi startup wrapper is `start_combined.sh`. The session-specific
capture helper is preserved as lab evidence, not a reusable calibration record;
it requires actual operator confirmation and rejects a different session.

Later Pi deployment: `pi-deployment.json`, `native-build-and-tests.log`, and
`native-launch.log` record installation into `install-combined`. Seven native
C++ tests, six button tests, the launch-resolution check and installed policy
hash checks passed. The hardware manager was not started and no motors moved.
Physical startup/calibration and roll-to-walking testing remain pending.

Initial software stage: September 13, 2026; laptop WSL Ubuntu / ROS Jazzy, isolated domain 193.
No Pi connection, motor command, hardware restart or physical policy activation
was performed for this integration.

- `test_results.json`: five CTest checks passed. Includes actual
  `NeuralController` activation/update against fake motor/IMU interfaces with the
  latest walking network, a valid roll map, full-turn encoder winding, continuous
  first command and gains, gain blending, mapped observation and motor output,
  stop handling, repeated activation, stale snapshot/map/session rejection.
- `buttons.log`: six button-edge and entry tests passed.
- `test_results.json`: one ROS launch-resolution test passed;
  policy spawners parsed inactive with calibration required, one dispatcher,
  mapped walking enabled, roll tolerance 30 degrees. No launch was executed.
- `policy-check.json`: new backpack leg/wheel export hashes and source wiring pass.
- Both C++ policy inference tests use the new release's 128 checkpoint reference
  cases per policy. The roll core regression also passed.

The first fake-interface test failed because its temporary calibration directory
was not created; that fixture was corrected and the test rerun. Recursive vendor
lint discovery was slow on the Windows mount, so focused controller tests were
configured with `QUADMORPH_FOCUSED_TESTS=ON`. Existing ROS deprecation warnings
remain. This is not a complete lint run or hardware validation.

The CTest and launch results were observed directly in command-tool output. The
original temporary WSL logs/build were unavailable during the subsequent evidence
copy, so `test_results.json` records those observed results rather than claiming
to be the original logs. The Python button and hash checks were saved directly.

Build used `/tmp/quadmorph-combined/build/neural_controller` in WSL. The native Pi
deployment path is separately prepared by `scripts/prepare_combined_motion.sh`;
it has not been run on the robot for this release. The combined stack must be
installed and physically tested before treating the handoff as hardware-proven.
