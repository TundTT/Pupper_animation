# Agent runbook — first training run (workstation w/ RTX 5090)

You are an agent on the GPU workstation. Goal: get the **Pupper leg-lift RL policy**
to its first real training run, produce a rollout **video**, and report back what the
policy looks like. Do **not** change the design or rewrite the env unless a step below
tells you to — this is an execution + smoke-test + tuning task, not a redesign.

## 0. Read first
- `CLAUDE.md` (workspace root) — what this project is and the repo layout.
- `mujoco_playground/workspace/README.md` — the training pipeline you are running.
- Convention that applies to any code you touch: **no silent fallbacks** — surface
  failures (warn/raise), don't paper over them with broad try/except or default values.

## 1. Prerequisites / sanity
- All commands run from the `mujoco_playground/` directory (so `workspace` imports as a
  package). `cd .../mujoco_playground`.
- **The Pupper model must be present.** `workspace/configs.py` resolves it at
  `Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml`
  relative to the bundle root. If you only cloned `mujoco_playground` and that path does
  not exist, get the `pupper_v3_description` checkout onto the box and pass its xml via
  `--model_path`. The code raises a clear `FileNotFoundError` if it can't find the model —
  do not work around it by faking a path.

## 2. Environment setup
The RTX 5090 is Blackwell (sm_120); it needs a **recent** JAX + CUDA 12 build, or the first
GPU op fails with a PTX/sm error.
```sh
cd mujoco_playground
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras        # playground + brax
uv pip install wandb                    # for --use_wandb
python -c "import jax; print(jax.default_backend())"   # MUST print: gpu
```
If it prints `cpu`, stop and fix the JAX/CUDA install before training (try
`unset LD_LIBRARY_PATH`; confirm the jax build supports sm_120). Report the exact error.

## 3. Smoke test (do this BEFORE any long run)
Validates that the env JITs and PPO steps end-to-end in ~1–2 minutes.
```sh
python -m workspace.train --num_timesteps 200000 --num_envs 1024
```
Pass criteria:
- It prints eval reward lines and finishes without an exception.
- A video appears at `workspace/output/<run>/rollout_final.mp4`.
If it errors, capture the full traceback and report it — common causes: model path (see §1),
GPU/JAX (see §2), or an env bug. Do not silently retry with different flags.

## 4. W&B login (if using Weights & Biases)
```sh
wandb login          # paste API key when prompted
```

## 5. Full training run
```sh
python -m workspace.train --num_timesteps 150000000 --use_wandb
```
- `num_envs` defaults to 8192 (fits 32 GB). If you hit OOM, drop to `--num_envs 4096`
  (note: 4096 = batch_size 256 × num_minibatches 16, so it stays valid; for other values
  keep `num_envs == batch_size * num_minibatches` or training will assert).
- Videos are logged to W&B as `eval/video` each eval and `eval/video_final` at the end,
  and written to `workspace/output/<run>/*.mp4` regardless of W&B.

## 6. What to watch for in the video
The rollout steps the command through the O-button sequence
`stand → front_l → front_r → back_r → back_l → stand`. A good policy:
- raises the **commanded** leg clearly off the ground and holds it,
- keeps the body upright and the other three feet planted,
- lowers smoothly and transitions to the next leg when the command changes,
- and does all of that **without moving the body**: no shifting backwards, no sitting
  down, no setting a knee or another foot on the ground to free the commanded leg.
  That last point is the whole reason the reward was redesigned — see
  `workspace/README.md` "Reward design (and the bug it fixes)".

## 7. Likely tuning (only if the first run looks wrong)
All knobs are in `workspace/configs.py`. Change one thing at a time and short-train
(`--num_timesteps 10000000`) to check the video before committing to a long run.

Check the diagnostic metrics before touching weights — they tell you *how* the reward is
being earned: `lifted_foot_height`, `body_drift_dist`, `torso_z`, `tilt_deg`. brax logs
these as per-episode SUMS, so divide by `eval/avg_episode_length`.

- **Foot barely leaves the ground** → first check `ACTION_SCALE[hip]`, not the reward. The
  policy head is tanh-squashed, so the reachable range is exactly `DEFAULT_POSE ±
  ACTION_SCALE` — if the hip can't rotate far enough, no reward weight can fix it, and the
  policy will learn to move the body instead. Then consider raising `lift_height` or
  `target_lift_height`.
- **Body tips / shifts / sits when a leg lifts** → raise `orientation`, `torso_height`,
  `body_drift`; or tighten `terminal_body_angle` / `terminal_body_z`. Note that a *front*
  leg lift genuinely requires ~12 mm of CoM shift (the CoM starts outside the remaining
  3-foot triangle), so `allowed_body_drift` cannot go to zero.
- **A knee or spare foot ends up on the ground** → raise `ground_contact` magnitude or
  `knee_ground_margin`; `terminal_knee_clearance` already ends the episode on contact.
- **Jittery / unsafe motion** (bad for the polymer link) → make `action_rate` / `torques`
  more negative, or lower `ACTION_SCALE`.
- **Refuses to lift at all (just stands)** → the posture terms are earned whether or not a
  leg goes up, so `lift_height` is the only term that discriminates; raise it.
- **Ignores the command** → increase `stance_pose`; confirm commands actually switch in the
  video.

## 8. Export the trained policy
```sh
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```
Produces `policy_leg_lift.json` (the `neural_controller` format) with `observation_layout`,
`command_states`, and `button_sequence` metadata. This is the artifact for the robot side.

## 9. Report back
- Did setup → smoke test → full run each succeed? Any tracebacks (verbatim).
- The final eval reward and a link/path to `rollout_final.mp4` (and the W&B run URL).
- Your read of the video against §6, and any `configs.py` changes you made and why.
- The path to the exported `policy_leg_lift.json`.
Do **not** start wiring anything into the `pupperv3-monorepo` (robot) side — deployment is a
separate task.
