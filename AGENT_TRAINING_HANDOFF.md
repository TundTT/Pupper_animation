# PC agent handoff: wheel alignment v2

## Task and authorization

Set up and run wheel-alignment v2 training on the user's NVIDIA RTX 6000 PC, then evaluate the resulting checkpoint and prepare an export for review. The user explicitly reserved training for that PC. **Do not train on the laptop.** This file is a handoff for the PC agent; it is not a request to start another run wherever it is read.

Proceed with ordinary environment setup, dependency installation into the isolated project environment, checks, and the initial training run without repeatedly asking for permission. Preserve existing work and running jobs. If GPU access or a required system change blocks setup, report the concrete blocker. Do not silently fall back to CPU training, change system drivers, or replace the project's dependency versions to make an error disappear.

Scope is rigid-wheel lift, rotation, alignment and gentle lowering. Heating is manual. Do not implement heater control, SMP deformation, leg-policy handoff, robot deployment, or changes to joystick bindings in this task. The existing deployment YAML intentionally selects the old policy until a new checkpoint has passed evaluation and physical review.

## Read first

Read [ALIGN_MOTION_V2.md](ALIGN_MOTION_V2.md), including its hardware-source links and calibration convention, then inspect:

- `training/wheel_align/train.py`, `env.py`, `contract.py`, `configs.py`, `randomize.py`.
- `training/wheel_align/evaluate.py`, `export.py`, `preflight.py`, `pyproject.toml` and `uv.lock`.
- `ros2_ws/src/neural_controller/include/neural_controller/wheel_align_motion.hpp` and `wheel_align_hybrid.hpp`.

The implementation was added in `5a1f8f4`; `af1a9f8` fixes storage of the geometry JSON. Use the latest `origin/codex/align-motion-v2`, including this handoff. This is a new 82-observation, eight-action residual policy. **Do not resume or load the old 51-input alignment checkpoint as if it were compatible.**

Hardware facts to preserve: 12 existing motors, canonical joint order FR/FL/BR/BL with three joints each, third joints used as continuous hubs, eight proximal position commands and four encoder-based hub velocity servos. There is no measured foot-contact/force input. Startup home is distinct from the current wheel hold. The marked-ring convention still needs physical verification; a successful simulation is not that verification.

## 1. Establish the checkout and GPU environment

Inspect the PC's OS, repository status and current GPU jobs first. Use Linux or WSL2 with GPU access. Prefer a checkout on the Linux filesystem. If the existing checkout has unrelated edits, use a separate checkout/worktree; do not stash, reset or overwrite the user's work. Do not switch away from a checkout used by another running training job.

For an otherwise available checkout:

```bash
git fetch origin
git switch codex/align-motion-v2
git pull --ff-only
git lfs install
git lfs pull
uv sync --project training/wheel_align --extra cuda --frozen
PY="$PWD/training/wheel_align/.venv/bin/python"
nvidia-smi
"$PY" -c "import jax; print(jax.devices()); assert any(d.platform == 'gpu' for d in jax.devices()), 'JAX cannot access an NVIDIA GPU'"
```

Run Python modules from the repository root. If `uv` or Git LFS is missing, install the missing tool using the PC's normal supported setup method. Use Python 3.11 or 3.12 as required by the project. Do not use an unrelated global Conda environment. Confirm that the mesh files contain mesh data and `geometry.json` parses as JSON, rather than treating Git LFS pointer text as the actual data.

Record the source commit, GPU model/VRAM, driver version, Python version and JAX devices. A successful `nvidia-smi` alone does not establish that JAX can use the GPU. The trainer also checks this and refuses CPU-only execution.

## 2. Pass preflight before launching the long run

```bash
"$PY" -m training.wheel_align.generate_geometry --check
"$PY" -m training.wheel_align.preflight --jit-step
mkdir -p runs/checks
c++ -std=c++17 -O2 \
  -Iros2_ws/src/neural_controller/include \
  -Iros2_ws/src/joy_utils/include \
  ros2_ws/src/neural_controller/test/align_motion_test.cpp \
  -o runs/checks/align_motion_test
export ALIGN_MOTION_TEST_EXE="$PWD/runs/checks/align_motion_test"
"$ALIGN_MOTION_TEST_EXE"
"$PY" -m pytest training/wheel_align/tests -q
```

These checks include nominal clearance, MuJoCo/FK agreement, interrupted lowering, 3,000 Python/C++ control-step comparisons, and exporter regression. The standalone C++ test does not require ROS. The exporter test uses randomly initialized test-only weights; do not confuse those fixtures with a trained policy. Without `ALIGN_EXPORT_TEST_EXE`, Python tests check Brax/NumPy export but do not exercise RTNeural; report that distinction and complete the runtime checks after training.

Preflight reports 82 observations, eight actions, randomized batch shape `[2, 82]`, three compiled control steps, and zero optimizer steps. Optional Warp import messages are not a reason to change physics backends if the selected JAX backend passes. Fix real failures before training. If a code fix is needed, keep it focused, rerun the relevant checks, and commit it before the run so provenance is reproducible.

## 3. Start one initial run and verify progress

Use the default initial budget of 50 million environment steps, seed 0, and 2,048 environments. This is a starting experiment, not a convergence guarantee. Do not launch multiple seeds or an open-ended tuning sweep without a further request.

Run inside a persistent terminal session such as `tmux`, so closing the agent or its tool session does not kill training. Check that the chosen session/run does not already exist before starting another process. In that session, from the repository root:

```bash
set -o pipefail
PY="$PWD/training/wheel_align/.venv/bin/python"
RUN="runs/align-motion-v2-seed0-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p runs
printf '%s\n' "$RUN" > runs/align-motion-v2-active-run.txt
"$PY" -u -m training.wheel_align.train \
  --envs 2048 --steps 50000000 --seed 0 --out "$RUN" \
  2>&1 | tee "${RUN}.console.log"
train_exit=${PIPESTATUS[0]}
printf '%s\n' "$train_exit" > "${RUN}.exit-code.txt"
```

Do **not** create `$RUN` itself beforehand: the trainer intentionally requires a new output directory. The active-run text file is a convenience pointer, not proof that a process is alive. Record the persistent session name, process identity, exact command and absolute run/log paths in your report.

Check for an actual live training process, GPU use, finite metrics, and increasing positive training steps. Initial JAX compilation and the initial evaluation can take time. `config.json`, `params_0`, or a step-zero evaluation alone does not prove learning has begun. After positive progress is observed, tell the user that training is running; do not claim it has finished. If the agent turn must end while the process continues, leave the session alive and give the user an exact reattachment command. Do not promise future monitoring unless a real monitoring mechanism was established.

If GPU memory is exhausted, preserve the failed run and its log, then retry in a new run directory with 1,024 environments, or 512 if needed. Supported values are 256, 512, 1024, 2048 and 4096; arbitrary multiples of 256 are not valid with the fixed PPO batch. Do not change collision masks, contact capacity, motion limits, observations, network dimensions or the physics backend to reduce memory usage.

There is currently **no resume CLI**. A retry with a new directory starts over; do not claim that it resumes a saved checkpoint. Preserve checkpoints after interruptions and report the last saved step. If resumption is necessary, implement and verify explicit compatible checkpoint restoration as a separate change rather than inventing a `--resume` option.

## 4. Evaluate the completed run in its original checkout

Do not pull, edit training sources, regenerate geometry, or change line endings in the running checkout after training starts. The run records source/mesh hashes; evaluation and export deliberately reject a mismatch. Keep experimental changes in a separate checkout. Do not bypass the provenance check.

Confirm a successful process exit and the final `mjx_params` file. Recover the recorded run path, then run the three 64-environment audits:

```bash
PY="$PWD/training/wheel_align/.venv/bin/python"
RUN=$(cat runs/align-motion-v2-active-run.txt)
"$PY" -m training.wheel_align.evaluate --params "$RUN/mjx_params" \
  --nominal --out "$RUN/audit-nominal.json"
"$PY" -m training.wheel_align.evaluate --params "$RUN/mjx_params" \
  --out "$RUN/audit-randomized.json"
"$PY" -m training.wheel_align.evaluate --params "$RUN/mjx_params" \
  --interrupt --seed 20260911 --out "$RUN/audit-interrupted.json"
```

Inspect each audit's `passes_simulation_gate`, completion counts, final angle/speed errors, unsafe rotations, minimum wheel/body gap and maximum contact approach speed. All environments must finish all four alignments and pass the criteria documented in `ALIGN_MOTION_V2.md`. A high PPO reward alone is not acceptance. If the audit fails, report the actual failure and preserve the checkpoint; do not weaken thresholds, omit failed environments, or call the result hardware-ready. Propose a focused next experiment instead of silently spending another full training budget.

## 5. Export and verify the runtime, without deploying

```bash
"$PY" -m training.wheel_align.export --params "$RUN/mjx_params" \
  --out "$RUN/policy_wheel_align_motion_v2.json"
```

This writes the policy JSON and `policy_wheel_align_motion_v2.reference.csv`. Export refuses to overwrite an existing policy output. If an export is made to diagnose a failed audit, label it as a failed-audit artifact.

Follow the ROS Jazzy build, CTest and `align_export_test` commands in [ALIGN_MOTION_V2.md](ALIGN_MOTION_V2.md). Run RTNeural against the **trained export's** CSV, not just the untrained unit-test fixture. If ROS is unavailable on the PC, preserve the export/CSV and identify the runtime verification as pending; GPU training does not require installing the full robot stack. Do not connect to the robot, replace its checkpoint, or alter the selected deployment YAML as part of this training handoff.

## Report and artifacts to return

For a running job, report the source commit, GPU, test results, exact launch command, persistent session/reattachment command, run/log paths, current positive training step and latest saved checkpoint. Clearly state that training is still running.

For a completed job, return:

- Source commit and any setup/code fixes, GPU/dependency details, seed, environment count, actual final step count and wall time.
- `config.json`, `metrics.jsonl`, console log and exit status; final `mjx_params` and `latest.json`.
- All three audit JSON files, with pass/fail and the key measurements summarized.
- Exported policy JSON, reference CSV, checkpoint/export SHA-256 hashes, and RTNeural/ROS test results or explicit pending checks.
- Any remaining issues, especially collision clearance, lowering impact, drift or alignment failures. Simulation success does not verify physical marked-ring calibration or hardware performance.

`runs/` is gitignored. A push of source code does not transfer trained artifacts. Give exact artifact paths and a concrete transfer method; do not claim the checkpoint is available on the laptop merely because this branch is pushed. Keep large training outputs out of normal git commits unless the user requests an artifact publication route. Do not upload model artifacts to an unrelated external service.
