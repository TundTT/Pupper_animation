# Local entry-to-roll prototype, September 13

**Not accepted for hardware.** Three preliminary trials reached a settled
walking-style pose, but failed the motor/body floor-force gate during entry.
A fourth, frozen-commit 10 mm diagonal formation trial timed out during entry.
No walking policy or physical robot was activated. No training occurred.

| Trial | Endpoint | Formation | Standing gates | Dense CAD | Motor floor |
|---|---|---|---|---|---|
| neutral-first | Synthetic neutral | Original tip/base | Pass | Not run | Fail, peak 56.73 N |
| undercompressed-13 | Synthetic neutral, +/-2 deg trial error | Coupled partial compression, seed 13 | Pass | Not run | Fail, peak sampled 4.70 N |
| actual-backpack-0 | Actual deterministic alignment replay | Original tip/base | Pass | Pass | Fail, peak 68.11 N |
| actual-diagonal-10mm | Same alignment replay, +/-2 deg trial error | 10 mm diagonal shortening contrast, +/-3 deg bend | Entry timeout | Not run | Fail, peak sampled 5.90 N |

The actual alignment replay uses unchanged backpack-alignment source `a7e3bb1`,
all four completion bits and no failure bits, and final lowering/settling. It is
not the robot-code v5 neural policy. The exported endpoint and alignment report
are included. Coordinate-frame verification error was 8.74e-13. MuJoCo versions
are 3.6.0 for alignment and 3.3.7 for triangle dynamics, with an explicit
post-cooling geometry boundary; no heating/deformation dynamics are claimed.

For actual-backpack-0, 25,548 physics steps were integrated and exactly reproduced
by command replay (maximum state difference zero). The nonconvex CAD audit checked
5,924 states: minimum shin-to-other-part clearance was 2.807 mm; the minimum
front-to-rear shin gap across sampled states was 39.794 mm. This does not cover
other formation errors. No CAD intersections were detected. The three-second
final pose, all-tip support, settled speed, tilt, effort and command-rate gates
passed. The complete task did not pass.

Motor contact occurred from the first simulation step through about 2.25 seconds
of entry. Independent detailed CAD inspection of the synthetic neutral point-up
start places the lower motor housings about 1.118 mm below the intended foot
plane. Contact is therefore not merely an overly conservative collision hull.
This conflicts with the earlier user-reported roughly 5 mm physical clearance;
do not silently alter geometry or remove motor contacts to obtain a pass.

The first three are pre-commit development runs. Source files were being developed during
the local audit, so the original end-of-run source-hash map is not a frozen
release manifest. Exact integration arrays, initial state, CAD audit and actual
video are retained locally under `output/` and should accompany external evidence.
The runner now captures source metadata at the beginning of future runs.
Use a clean pinned commit for PC acceptance, with fresh held-out cases.

The fourth trial used clean source `2c308767062296f94beef56d184cb265c5d07e38`.
It stopped after 6,239 integration steps with `entry_did_not_settle`; the rear-right
hip remained about 7.03 degrees from its entry target (the gate is 0.1 rad,
about 5.73 degrees). This is evidence that a fixed entry target with the current
gains is insufficient for the requested formation contrast, not justification
to loosen the gate. The failed trial is retained as `actual-diagonal-10mm.json`.

Online evidence was verified through W&B's API (Media files and artifacts):

- [Nominal continuous motion, synthetic failure, and full local audit artifacts](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/c6b0e1f36eaa4c31)
- [10 mm diagonal entry failure: actual video and integration artifact](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/1c0b66024a514990)

33 focused model, formation, entry and original roll checks passed locally.
Incomplete-base contact convergence, full endpoint distribution, combined shape
robustness, v5 handoff, walking activation and hardware adapter parity are pending.
See `../../HANDOFF_PC.md` for runnable commands and the ordered PC assignment.
