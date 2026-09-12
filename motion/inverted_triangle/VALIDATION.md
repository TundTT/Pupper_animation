# Reproducible PC validation

The PC work uses the original model and 9 mm gap. The prepared Powell optimizer
was tried first. Its failed front-leg audits motivated a body shift and independent
landing/support targets. `extended_search.py` supports these trajectories and
parallel CPU search. Its objective is never an acceptance gate. The final version
includes shin-to-motor/body CAD pairs in search after a shin-only search missed a
motor intersection; the independent replay caught and preserved that failure.

`simulate.py` accepts optional `pre_shift_pose` and `landing_pose` fields in a
candidate. Both must preserve the commanded hub references. Body shifts take at
least four seconds and independent landings at least four seconds, with all
original speed/acceleration bounds retained. Full integration state and delayed
command history propagate through `sequence.py`.

Install `requirements.lock.txt` in a fresh Python 3.12 environment to reproduce the
PC dependency set. `requirements.txt` remains the original minimum setup.
On headless Linux set `MUJOCO_GL=egl` for the command, not globally on Windows.

```sh
python -m pytest motion/inverted_triangle/test_model.py -q
python -m motion.inverted_triangle.sequence --plan PLAN/plan.json --output runs/new-nominal
python -m motion.inverted_triangle.robustness --plan PLAN/plan.json --output runs/new-grid --workers 12
python -m motion.inverted_triangle.review_replay --replay runs/new-nominal/03-front_r --output runs/new-cad-review
```

Every output directory must be new. The sequence and grid commands log online to
the shared W&B destination, including actual stage videos and a concatenated
continuous video. A failed screening gate is retained as FAILED. A complete SDK
finish is followed by explicit cloud media/artifact verification in the PC report.

The grid tests one fixed plan in 20 friction/initial-perturbation cases and 16
additional uncertainty cases. Its saved configuration explains the ranges. These
are sensitivity tests, not measured hardware distributions. XML/CAD provenance
stays fixed; runtime mass, inertia, trunk COM, gain, torque and command-delay
changes are explicit in each audit and continuation state. Initial height/tilt
offsets are applied only at the original start, never again between stages.

The CAD review reintegrates the saved trajectory and checks its full end state
against the saved audit before using recorded poses for geometry or rendering.
It refines around each phase's sampled CAD minimum and both front/rear wheel
minima at 520 Hz. This remains finite sampling, not continuous collision proof.
Three viewing angles show the original state, maximum lift, halfway rotation and
final state. This work does not validate polymer compliance or spacer mass.

Four-tip standing does not establish direct policy-entry compatibility. The final
joint posture and unwrapped hub coordinates must be compared with the selected
leg-policy export. No calibration changes, hardware executor, deployment or
policy activation are part of this simulation work.
