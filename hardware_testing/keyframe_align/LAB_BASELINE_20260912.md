# Successful supported and floor alignment baseline

The operator reported a successful supported trial followed by two successful
floor trials on September 12, 2026. These are operator observations, not a newly
completed quantitative audit of all recorded trajectories.

The tested keyframe controller/config source was c11f5ce. Installed controller
SHA256 was `c3af6613437c1ca8b3f726fe9a0d3134af0347452c7a3e863780df528860dcf4`.
Wheel gains: P=4, D=0.15, near-target I=0.5 Nm/(rad*s), integral cap=0.10 Nm.
Floor clearance gate=0.005 m. Lift/lower trajectories were unchanged.

Pi recordings:

- `/home/pi/keyframe-integral-lGN8Ob`: supported integral trial.
- `/home/pi/keyframe-floor-lwm0zJ`: first successful floor trial after activation recovery.
- `/home/pi/keyframe-floor-repeat-IzzTlX`: successful floor repeat after battery power cycle.
- `/home/pi/keyframe-floor-U6xqC8`: operator rejected the calibration; do not use as the reference.

The repeat trial's original calibration is preserved in
`../start_pose/successful_floor_calibration.json`, ID
`190621cde1094ae3ba05300aead1906c`. It supplies the requested editable permanent
desired reference. Automatic return across power loss remains unfinished; see
`../start_pose/README.md`.

## Startup lesson

Enter stance while the body is supported, then lower the robot to the floor.
Putting it down while limp allowed the left hips outside the controller's initial
position envelope. The first floor attempt rejected activation and became
unconfigured; later button requests did not recover it. Unloading/reloading the
keyframe controller and restarting the button node restored inactive readiness
without rehoming. The earlier blanket claim that the floor test could start from
the limp pose was incorrect. The button-node success reporting after failed
activation remains an issue to fix; no fix is claimed in this baseline commit.

## Exact Pi source differences now preserved

The successful trials used three longstanding local Pi changes, now included in
the repository: log raw homing readings without the legacy fixed-reference warning;
retain four historical raw-reference values in components.xacro; disable three
unavailable optional nodes in the full launch. These are existing deployed changes,
not newly tested behavior. Some surrounding historical hardware comments still
refer to old checks; they do not establish absolute encoder behavior.

## New locomotion policies preserved alongside alignment

The merge retains robot-code commits 25b510a (wheel) and 8f78cc8 (walking), including
their policy exports, reference tests, and selection reports. Training model
commits 27bc668 on `leg` and 88f6a88 on `wheel` specify a **9 mm** outward assembly
translation. The operator mentioned a 6 mm spacer when requesting this merge;
that discrepancy is recorded, not resolved by changing either model.

The alignment geometry has not been updated or revalidated for that new morphology.
The successful alignment trials do not validate these newly merged locomotion
exports on hardware. This merge preserves both developments; it does not retrain
or deploy new policies onto the live Pi.
