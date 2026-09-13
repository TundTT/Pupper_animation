# Operator-requested 30-degree hub tracking tolerance

After the first floor attempt hit 0.150316 rad of hub tracking error, the user
explicitly requested 30 or 40 degrees of allowed error, with no gain increase.
The selected retry is 30 degrees (pi/6). The original 0.15 rad default remains
available. Use the explicit retry YAML as documented in TRIANGLE_ROLL_LAB.md.

Native Pi core, ground-controller lifecycle and stand-controller lifecycle tests
all passed. They cover default behavior, the new parameter reaching the real
controller, unchanged gains, rejection above the allowed setting, and the
independent torque guard remaining effective below the new tracking threshold.
The updated library was installed with the hardware manager absent. SHA-256:
`a3ea27a908ac31b82e1826cf8c9a1b42a6c2d1724ca1e0f83717b73800b880fe`.

The final line of native-pi-tests.log has a stale installer message saying
0.25 rad; direct inspection of the installed retry YAML confirmed
`hub_tracking_error_limit: 0.5235987755982988`. The test assertions use 30 degrees.
No 30-degree physical trial has occurred. The motion still releases torque on
other faults; lag-aware reference pausing is not implemented.
