# Measured roll: native Pi preparation

The updated controller was built and installed on the Pi without starting the
motor stack. Dependencies remain in `install-roll`; the new controller is in
`install-measured-roll`. Source changes are in the robot-code checkout.

Validation: four C++ tests (measured motion envelope, old support regression,
ground lifecycle, suspended lifecycle), four mapping tests, and all 24,051
recorded integration steps passed. Peak command difference from the saved
passing simulation was 1.77636e-15 rad. The installed headers, contract, gains,
joint order and continuous-hub hardware limits passed the interface check.
The suspended envelope test includes the documented +/-0.18 rad hanging hips.

The earlier incremental build was rejected by the compiled-contract gate:
archive timestamps had prevented recompilation. Its failure log is preserved.
The update was extracted with current timestamps, rebuilt and retested; the
final log and library/plan hashes are recorded here.

No robot startup, live calibration or motor commands occurred. Physical motion,
ground support, and walking handoff remain untested. Follow the root
TRIANGLE_ROLL_LAB.md and STARTUP_CALIBRATION.md before startup. The last read-only
check found no motor stack and no `/dev/input/js*` gamepad device.
