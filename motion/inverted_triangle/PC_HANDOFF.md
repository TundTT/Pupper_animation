# PC agent: complete and audit the inverted-triangle motion

Work in an isolated checkout of `origin/codex/inverted-triangle` in
`https://github.com/TundTT/Pupper_animation`. Read root AGENTS.md, this directory's
AGENTS.md and README.md. This is authorized offline simulation/search work on the
user's PC. Do not connect to, deploy to, start or move the physical robot. Keep
robot-code, leg, wheel and historical training checkouts unchanged.

## Task in plain language

The robot starts on four **already rigid, point-up triangles**. There is no
heating or deformation to model. With the original tip length and approved
additional **9 mm axial gap**, find a gentle self-supported motion that flips
each triangle through 180 degrees and ends standing on four tips. Keep three
limbs supporting it while one turns. The user confirms lower motor housings are
roughly 5 mm above the floor at the start. The simulated stance approximates that;
its joint coordinates have not been captured on hardware.

The starting implementation is deliberately CPU MuJoCo trajectory optimization,
not RL. Try this first because the controller interface already exists. The GPU
is available if measured search failures justify a policy or faster batched
physics approach; do not assume you must train a neural network to use the PC.

## Setup and first run

Use a new Python 3.11/3.12 virtual environment. Install
`motion/inverted_triangle/requirements.txt`. On headless Linux set
`MUJOCO_GL=egl` for video rendering. Do not set this globally on Windows.

```sh
python -m pytest motion/inverted_triangle/test_model.py -q
python -m motion.inverted_triangle.core
python -m motion.inverted_triangle.experiment --output runs/inverted_triangle/pc-01 --maxiter 20 --method powell --order back_r front_l back_l front_r --candidate motion/inverted_triangle/results/nominal_rear/candidate.json
```

The experiment defaults to online W&B logging, entity `QuadMorph`, project
`wheel-leg lift and align triangle base`. Use existing credentials or `wandb login`.
Do not expose keys. Explicit `--wandb offline` retains uploadable offline data;
`--wandb disabled` is for an intentionally unlogged local development check only.
Each output directory must be new. Preserve failures and actual replay videos in
the run's Media tab using the shared logger's `wandb.Video` method. Search
trajectories are not trained checkpoints; label candidate hash, leg, seed,
simulation status, environment steps and early termination accurately.

## How to progress

1. Inspect the checked-in `results/` status and reproduce its current-model audit.
   The XML/CAD hashes must match. A solver's success or a low objective value is
   never an acceptance result.
2. Optimize whole lift/rotation/landing trajectories with `search_flip.py`, then
   run the independent `simulate.py` CAD and dynamic audit. `search_lift.py` is
   only a fast seed generator; a lift-only result can lose balance during rotation.
3. The experiment carries the full MuJoCo integration state and commanded joint
   targets from each successful flip into the next search. Never reset four
   inverted shins between stages and call that a full sequence. Try multiple
   seeds, rear-first orders and both turn directions if the greedy sequence
   stalls. Keep each attempt's evidence.
4. If the current nine-parameter search cannot pass, extend landing/support
   targets or a short body-shift phase before reaching for RL. Do not loosen
   joint, speed, torque, floor, support or collision gates to manufacture success.
   Maintain the 9 mm gap and original shin tip; no fictitious support or body pose
   pinning. Explicitly model and report any justified dynamics change.
5. Once a full plan passes nominally, replay the *same* plan continuously across
   friction values 0.5, 0.65, 0.8 and 1.0 and at least five initial perturbation
   seeds each. Use `sequence.py`. It propagates the actual preceding end state,
   including velocities; it does not load the nominal state for every stage.
   Extend robustness to mass/COM, actuator delay, lower available torque, gain
   variation and start-height/tilt uncertainty with justified ranges and recorded
   configuration. Retain failures. No robustness claim without actual audits.
6. Review CAD nearest pairs and sample minima; increase temporal resolution near
   marginal gaps. The CAD audit checks detailed shin interfaces, while motor/body
   floor collision is included in physics. Account for finite CAD sampling and
   unmodeled polymer/spacer mass. Render several views of the initial stance,
   maximum lift, halfway through rotation and final stance.
   Report front/rear wheel clearances separately. The focused audit in
   `results/wheel_clearance/` found 21.80 mm right and 50.37 mm left minima for
   the saved rear flip; these numbers do not validate subsequent mixed-stance
   flips. `check_wheel_clearance.py` provides a reproducible focused check.
7. Before calling any result hardware-ready, implement/review a measured-state
   executor using available encoder/IMU data: wait for lift/angle tracking and
   low speed before progressing, bound landing descent, handle lost tracking and
   aborts, preserve hub references and calibration. Ground-truth simulation contact
   forces are audit-only. Do not pretend motor effort estimates are foot load
   sensors. Physical deployment/testing remains a separate task.

## Deliverables

Commit and push source changes on this branch (or a clearly named derivative if
necessary). Include a readable results summary, immutable plan/candidate hashes,
model and source provenance, dependency versions, complete nominal four-flip
rollout, robustness results, failed audits and videos. Report W&B URLs and actual
upload status. Report whether the final stance is compatible with the leg policy;
do not activate that policy as part of this task. If blocked, identify the exact
first failing phase and measured gate and return the best failed replay.

Do not stop at a plan or at four independent first-flip demonstrations. The
requested outcome is a reproducible continuous sequence in simulation, with
honest remaining hardware limitations for the laptop agent to review.
