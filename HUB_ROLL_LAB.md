# Suspended hub-only diagnostic

The operator selected this diagnostic on September 13 after confirming rigid
triangles, the existing XML spacing, torso supported on a stand, and tips up.
The hanging upper joints differ from the full roll's required starting pose.
This diagnostic **holds their measured angles** and rotates only the four hubs.
It is not the simulated roll-to-stand trajectory and cannot validate standing.

Startup follows STARTUP_CALIBRATION.md. Reuse the current live calibration;
capture the distinct point-up reference with all motion controllers inactive:

```bash
python3 scripts/capture_hub_roll_reference.py --operator-confirmed-tips-up
```

The flag requires actual physical confirmation. Capture checks stationary
feedback and preserves the shared startup calibration and encoder winding.
The plugin refuses activation if any joint moved more than 0.03 rad since
capture. It commands the measured pose, ramping gains over 2 seconds, then waits
for a fresh `/hub_roll/command` Int32 value 1. Activation is a motor action and
requires the operator's confirmation. Starting the half-turn is a separate
motor action and requires their confirmation too.

After command 1: hold 2 seconds, then a 12-second quintic half-turn with doubled
damping, then hold with base damping. Right hubs change -pi rad and left hubs
change +pi rad, following the same model/hardware sign convention as the pinned
roll. Proximal commanded angles remain exactly constant. Zero velocity and
feed-forward effort commands are used. No adaptation or walking is present.

The new plugin `neural_controller/HubRollController` lives in a **separate**
`libhub_roll_controller.so`; it links the existing neural-controller library
for shared calibration and sensor access. Config: `hub_roll_config.yaml`.
Name: `neural_controller_hub_roll`. Status and motor commands use that namespace.
PS, missing/stale gamepad, stale IMU, excessive tilt, invalid controller timing,
joint tracking/speed/position limits, or >1.5 Nm estimated PD demand latch a
fault and zero all command fields. Releasing PS never resumes the diagnostic.
The PS topic and direct button subscription both apply even if the joystick
utility's controller list still names the older roll controller.

Do not overwrite a library mapped by a running process. Build in separate
build/install folders. A library refresh may unload inactive motion controllers
and sensor broadcasters without restarting the hardware; verify this procedure
against a mock controller manager before using it. Restore sensor broadcasters,
verify unchanged encoder-session/calibration IDs, and load the new motion
controller inactive. Never activate or start motion as part of software setup.
