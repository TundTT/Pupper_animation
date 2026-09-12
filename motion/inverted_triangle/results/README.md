# September 12 local result

**One nominal rear-right flip passed. The continuous four-flip task and physical
robot are not yet validated.**

The pinned leg geometry retains the approved additional 9 mm outward gap and
original tip. Starting stance: four rigid inverted triangles, proximal joint-2
splay +/-0.29 rad, approximately 4.5 mm settled lower-housing floor clearance.
No heating or material dynamics were simulated.

| Nominal rear-right replay measurement | Result |
| --- | --- |
| Minimum shin/floor gap during rotation | 7.95 mm |
| Minimum sampled detailed shin CAD separation | 3.87 mm; no detected intersections |
| Minimum normal force on any of the three supporting shins during rotation | 2.22 N |
| Maximum body tilt | 2.15 degrees |
| Peak requested joint torque | 0.429 Nm in this model |
| Maximum downward speed near landing | 7.87 mm/s |
| Final load on flipped tip | 2.64 N |
| Motor/body floor support | 0 N |
| Simulated duration | 26.75 seconds / 13,911 physics steps |
| Scope | One nominal flip, friction 0.8, seed 0; 9 tests passed |

`nominal_rear/rollout.mp4` is a real torque-limited MuJoCo replay, not a rendered
kinematic animation. `audit.json` contains all gates, source/model/candidate
hashes, package versions and limitations of temporal sampling. `trace.json` is
the recorded state/command/contact trace. The optimizer's checkpoint is the
candidate JSON; no neural policy was trained.

The audit was made in the development worktree based on robot-code commit
`bb393e4bbce8a8414fffaa71bac8702477831f39`, with `source_dirty=true`. Its individual
source hashes identify the actual tested files. Do not silently relabel it as a
clean-commit run. The PC should reproduce it on the published commit and log that
reproduction. Model SHA-256:
`768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`.

Search/replay logging was explicitly disabled. A separate offline smoke test
successfully stored the actual replay through `wandb.Video`; **nothing from these
local experiments has been uploaded to W&B**. The PC command defaults to online
logging. The sequence CLI's logging support was added after the saved rear audit;
the simulator, CAD model and tested rear-flip behavior were unchanged.

## What failed and why it changed

The earlier lift-only candidate could raise the shin, but lost support as the
shin turned and lowered. The four reports in `failures/` retain those failed
landing replays on the same geometry hash. They used an earlier audit source
revision, as their hashes record, and deliberately skipped the expensive CAD
audit; `cad_checked=false` is a failure, never a pass. Other archived exploratory
runs remain locally under `runs/inverted_triangle/` and are not used as current
model evidence.

The full-flip optimizer now evaluates shifting, lifting, turning, landing and
settling together. It changes the support pose as well as lift and landing depth.
The saved passing candidate is a seed for the next phase, not a deployable robot
configuration. Its ending physics state is provided for continuation searches.

## Reproduce

From the repository root after installing the requirements:

```sh
python -m motion.inverted_triangle.experiment --output runs/inverted_triangle/reproduce-rear --order back_r --candidate motion/inverted_triangle/results/nominal_rear/candidate.json --audit-only
```

This renders the actual replay and logs it online using existing W&B credentials.
Use explicit `--wandb offline` or `--wandb disabled` only when appropriate. No
hardware stack is involved. Continue using [PC_HANDOFF.md](../PC_HANDOFF.md).
