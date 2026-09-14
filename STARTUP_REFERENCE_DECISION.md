# Current startup requirement

[STARTUP_UPPER_HOME.md](STARTUP_UPPER_HOME.md) is authoritative: every fresh stack activation homes motors 1 and 2 automatically and returns to the saved upper pose. Hubs remain manually aligned and captured. The material below is historical evidence and is superseded where it describes manual upper posing or says automatic return is unavailable.

# Startup reference decision — September 14, 2026

## Latest: native Stanford routine restored for homing-only tests

The operator requested copying Stanford's implementation instead of continuing
the low-effort Python experiment. The native hardware driver now has an opt-in
`stanford_upper_homing_test` mode (`fr1` or `upper`) and a separate launcher:
`ros2_ws/src/neural_controller/launch/stanford_upper_homing_test.launch.py`.
The tested Pi installation is `ros2_ws/install-stanford-homing`. Ordinary startup
is unchanged. Source provenance is commit
`40116f60bd0f01506f5a4b22e95d2af0e4a81daa`, original `do_homing()` and
`pupper_v3_description/description/components.xacro`.

The adaptation retains Stanford's kp=5.5, kd=0.2, speed=1.5 rad/s, filtered
estimated-torque threshold=2, motor-2-before-motor-1 order, stop angles and
return-to-zero ramp. It excludes all motor-3 hubs. Additions are a 2.5 Nm
estimated PD-command bound, eight-second stage deadlines, five-second return
deadline, finite feedback/speed/travel checks, and measured return proximity
within 0.06 rad at low velocity for 20 cycles. It disables motors and invalidates
the runtime calibration afterward. This is NOT an automatic startup calibration
or a restoration of the user's saved home. No hub reference is inferred.

Native live results in boot `79371d4d-6a1d-4bbb-a5e0-bbe0cc22164b`:
- FR1 isolated stops: -2.15667248 and -2.15705395 rad (0.022 degree difference).
  First trial rejected the strict 0.03 rad return check. The second completed
  with a final logged error around +0.04574 rad under the original low gains.
- All eight upper joints completed the algorithm, but FR1's detected endpoint
  was -1.80418873 rad, 20.22 degrees away from its isolated endpoint. This is
  **not a validated mechanical-stop calibration**. A torque threshold can detect
  resistance before the intended stop; the cause here is not established.
- Motors were disabled and the hardware processes stopped. Controller-manager
  also reported an abort during shutdown; this was after the successful test
  completion/disable log, not evidence of a valid full calibration.
- Preserve existing desired home. Do not promote these full-sequence offsets
  into runtime or unattended startup without resolving the FR1 discrepancy.

Logs: `hardware_testing/manual_stop_2026-09-14/stanford_fr1_trial01.log`,
`stanford_fr1_trial02.log`, `stanford_upper_trial01.log`, and
`after_stanford_upper_trial01.json`. The independent C++ helper checks and launch
configuration checks passed, and the separate Pi build succeeded. The Python
feed-forward trial below is superseded, retained only as historical evidence.

## Automatic stop-homing trial (latest work)

The operator requested Stanford-style automatic stop referencing for motors 1
and 2, excluding continuous wheel hubs. `scripts/test_fr1_stop_homing.py` is an
isolated first validation for FR1 near its freshly measured negative stop. It
commands only FR1 feed-forward effort, capped at 0.4 Nm in command units, with
all firmware gains zero. FR leg enable is shared; its other motors receive zero
effort. This is not a verified physical torque limit or firmware watchdog.
It requires a 0.045 rad demonstrated backoff before accepting stable contact
within 0.012 rad of the measured endpoint. It has five-second phase deadlines,
a 14-second process alarm, travel/speed/timing checks, and a finally block that
sends disable packets. It does not save runtime calibration or move to home.
Five offline checks cover packet isolation/checksum, effort bounds, stale/stuck
feedback, travel/speed/timing faults, a synthetic stop, and a false stop.

First live attempt, boot `79371d4d-6a1d-4bbb-a5e0-bbe0cc22164b`, aborted during
zero-effort initialization because the refreshed position differed from the
measured FR1 stop of -2.1349277496 rad. No nonzero effort was sent; disable packets
completed without error. Subsequent disabled feedback read +0.1630811691 rad.
Whether physical repositioning or feedback behavior caused this is unresolved.
Evidence: `hardware_testing/manual_stop_2026-09-14/fr1_auto_trial01.json` and
`fr1_after_auto_trial01.json` on the Pi. Arbitrary-pose automatic homing and all
eight-joint integration remain unvalidated and are not enabled at startup.

## Intended workflow (operator decision)

- Upper joints: motors 1 and 2 on each leg retain the measured mechanical-stop ranges and saved model references. The operator expects startup with these joints somewhere within their allowed ranges, without manually posing all eight at a reference every time. Startup must resolve their model coordinates before motion; range membership alone is not an encoder offset.
- Wheel hubs: motor 3 on each leg has no mechanical stop and can rotate continuously. The operator will manually position all four marked hubs at the agreed physical reference every startup, explicitly confirm, and capture fresh hub homes for that encoder session. Symmetry does not identify the intended marked ring automatically. Upper-joint pose affects the physical hub reference and must be defined consistently.
- Reuse valid hub calibration when switching policies in the same live encoder session. Fresh hardware activation/zeroing requires a new capture. Heating remains manual.
- Capture trusts the operator's explicit physical confirmation and uses fresh finite readings plus position stability. Reported velocity magnitude is not a capture gate, per the operator's September 14 instruction. Motion-controller velocity protections remain unchanged.

## Implementation status — do not confuse intent with deployed behavior

All eight upper joints have endpoint measurements; repeat measurements and historical-range differences are retained. These are reference data, not an installed persistent coordinate mapping. Some measured spans differ from historical model ranges by several degrees; do not rescale encoder readings to force agreement.

Front-left motor 1 showed approximately a 36-degree raw-reference shift across a confirmed reboot, with nearly unchanged stop-to-stop travel. Consequently, simply reusing a raw offset or assuming the joint is within its range does not establish its physical angle. Cross-boot reference resolution remains required before enabling the intended automatic upper-joint startup.

The production hardware startup still assigns upper-joint offsets using its configured hanging pose. Do not launch it from arbitrary in-range upper poses and claim the new workflow is implemented. The dedicated manual-stop test only demonstrated a session-specific front-left motor-1 reference; that stack was stopped. No full upper-joint offset set has been applied to runtime.

Next implementation work: consolidate the measured references, resolve the encoder ambiguity or establish upper references through validated homing, preserve that mapping during startup, then capture the manually positioned hubs. Do not repeat manual upper-joint calibration as the intended permanent workflow; explain any temporary requirement imposed by the current implementation.

## Evidence on the Pi

Checkout: `/home/pi/robot-code-leglift`.
Measurements: `hardware_testing/manual_stop_2026-09-14/` and `hardware_testing/start_pose/encoder_checks/`.
Latest completed measurement: rear-right motor 2 repeat (`br_motor2_stop03.json`).
Historical source: `40116f60bd0f01506f5a4b22e95d2af0e4a81daa`, `ros2_ws/src/pupper_v3_description/description/components.xacro`.

## Saved upper home after cross-boot repeatability tests

The operator selected the current disabled physical pose as the new home for motors 1 and 2. The eight targets are saved in `hardware_testing/start_pose/upper_home.json` and merged into `start_pose.json` on both PC and Pi, with each file's prior wheel-hub targets preserved. Prior complete presets are backed up in the adjacent history directory. PC and Pi previously had different hub presets; this upper-only update deliberately preserves each rather than silently replacing either.

The reference uses median mechanical-stop readings from runs 07-09 in boot `3b48458c-2536-4d57-945f-4afd0a3453e5` and the last 50 stable samples of `upper_home_after_trial09.json`. These three runs had a maximum stop spread of 0.044 degrees and maximum median change of 0.022 degrees compared with runs 04-06 in the preceding boot. Disabled feedback has no independently proven motor timestamp freshness; the saved record states that limitation. No live calibration or automatic return was enabled. The desired home is the measured resting pose after motor disable, not Stanford's commanded zero pose.
