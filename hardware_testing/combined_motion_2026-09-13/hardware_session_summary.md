# September 13 combined-control hardware observations

Source: operator reports in the lab conversation and SSH outputs observed by
the assistant. This is a summary of those observations, not a raw telemetry file
or an automated acceptance report. Later full Pi logs were not downloaded before
the robot became unreachable during commit preparation. Existing failed trials
and software validation logs remain preserved.

- Operator verified tips-up to tips-down floor motion and reported that the leg
  test and subsequent flip-to-walking test went well.
- In boot `1e3b3a08-08eb-4763-9d22-5e7b80090604`, the log showed roll activation at
  Unix time 1789313118.274, one X START at 1789313120.294, and walking activation
  at 1789313167.923. Controller-manager inspection showed walking active and
  roll inactive. Walking inference continued with fresh IMU readings.
- Triangle presses at 1789313145.778, 1789313150.738 and 1789313158.401 were
  rejected while roll settling was unfinished. Later Triangle activated walking.
  The operator need not obtain chat permission between buttons: centered sticks
  and a fresh press after successful roll completion are the required interface.
- The initial setup remains confirmed tips-down calibration on the stand,
  supported upper-home/180-degree hub reset, then confirmed tips-up mapping.
  X rolls to tips-down and holds; Triangle selects walking; Circle selects wheels;
  PS stops. Heating and geometry changes remain manual.
- Calibration and mappings are per encoder session. Historical IDs or copied
  reference files must not be restored as live calibration after reboot.
- A direct walking test used fresh tips-down calibration after archiving stale
  roll-map and handoff files. It did not require a new tips-up flip first.

Known issues: the duplicate dispatcher caused an earlier X failure; the process
lock fix and tested wrapper prevent a second dispatcher. A later joint-pose
activation succeeded but its ROS service reply was lost; the CLI retried and
reported failure because the controller was already active. Read-only telemetry
confirmed completed HOLD with fault zero; the move was not sent again. Inspect
live ownership/status after an ambiguous switch response before retrying.

Repeated Pi reboots were observed; their cause remains undetermined. Do not
attribute them to a particular power or software fault from these observations.
These lab successes do not certify all formation variations or wheel driving.
