# Focused wheel-to-wheel clearance check

No wheel/shin mesh intersections were found in the saved nominal rear-right flip.
This uses the same approved 9 mm axial-gap model and candidate as the video.

| Pair | Smallest surface gap | Time / phase |
| --- | --- | --- |
| Right front / right rear | **21.80 mm** | 4.290 s, beginning of lift |
| Left front / left rear | **50.37 mm** | 2.588 s, weight shift |
| Right front / left rear | 188.34 mm | Initial hold |
| Left front / right rear | 189.64 mm | Initial hold |
| Front / front | 167.96 mm | Initial hold |
| Rear / rear | 146.00 mm | Initial hold |

During the rotation phase alone, the right front/rear gap stays at least 96.45 mm.
The tightest approach happens before the wheel has lifted far out, not during
the 180-degree rotation. `closest_right.png` shows the actual recorded pose, with
front CAD colored blue, rear CAD orange and the shortest gap marked yellow.
The rest of the robot is shown for context; geometry has not been shrunk.

The earlier 3.87 mm global CAD result was the minimum across shin-to-motor,
shin-to-body and shin-to-shin checks. It was **not** the wheel-to-wheel gap.

## Method and evidence

- Replayed the complete 26.752-second motion using the original torque-limited
  physics and saved candidate. The final integration state matched the original
  video's saved state exactly (maximum absolute difference 0).
- Checked all six pairs of original nonconvex wheel/shin STL surfaces with FCL,
  including their XML offsets and rotations. These were not the convex floor
  collision hulls. Validated that nearest-point separation matched each reported
  distance.
- Sampled the complete motion every 25 ms (40 Hz), then checked every physics
  step (1.923 ms / 520 Hz) within 125 ms of both same-side front/rear minima.
  There were 1,312 CAD samples per pair, with no detected intersections.
- `audit.json` records the model/candidate/source hashes, minima by phase,
  closest points, recorded poses and final dynamics state. `distances.csv`
  contains a downsampled distance trace, not every CAD query.
- Model SHA-256: `768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`.
  Candidate SHA-256: `90bbe5d52caff1e70825a9c6da81c12b68df07e16e1daaa8e9dd8ecf5d2f7399`.

This is a sampled nominal rigid-CAD result for **this one flip**. It is not a
continuous-time collision proof, a physical tolerance/compliance measurement,
or validation of the yet-unsolved four-flip sequence. The new audit script was
run in the development checkout based on `46aedb0`; its exact hash is recorded.
The first exhaustive-query attempt was stopped before completion to reduce
query cost; no result is claimed for that attempt. Local geometry checking did
not upload data to W&B or operate the robot.

Reproduce from the repository root:

```sh
python -m motion.inverted_triangle.check_wheel_clearance --output runs/inverted_triangle/wheel-clearance-recheck
```
