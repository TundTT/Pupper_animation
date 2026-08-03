# Agent runbook — swapping in a new leg-lift policy

You are an agent picking up a **newly trained** leg-lift policy and taking it from
"exported JSON" to "verified working in sim, then on the real robot." This is the
deployment/testing loop, not a training task — see `.notes/workstation_first_run.md`
for training itself. Assume the user gives you a path to a new `mjx_params` dir or
`policy_leg_lift.json` and expects you to run this whole loop with minimal hand-holding.

Do the steps **in order** and **stop between sim and hardware** to let the user watch —
this instruction predates this doc and still applies: never launch the real robot's
`ros2 launch neural_controller launch.py` without explicit just-now confirmation that
the robot is secured (on a stand, legs clear) and the user is present.

## 0. Read first
- `CLAUDE.md` (workspace root) — project layout, the two repos, and the "no silent
  fallbacks" convention.
- `mujoco_playground/workspace/README.md` — training pipeline / export details.
- This doc assumes you're on the `master` branch of this `Pupper_animation` checkout
  (has both `mujoco_playground/` and `Stanford/pupperv3-monorepo/`). There is also a
  lean **`robot-code`** branch (just the monorepo, flattened to repo root, no training
  bloat) meant to be cloned directly on the robot — see §5.

## 1. Get + validate the exported policy

If you only have `mjx_params`, export it first:
```sh
cd mujoco_playground
python -m workspace.export_policy --params workspace/output/<run>/mjx_params
```
This writes `policy_leg_lift.json` next to the params.

**Before deploying it anywhere, check these by hand** (there is no committed validation
script on `master` as of this writing — don't assume one exists, grep first in case a
later session added one):
```sh
python3 - <<'EOF'
import json
d = json.load(open("workspace/output/<run>/policy_leg_lift.json"))
print("in_shape:", d["in_shape"])
print("behavior:", d.get("behavior"))
print("command_states:", d.get("command_states"))
acts = {l.get("activation") for l in d["layers"]} if "layers" in d else None
print("activations seen:", acts)
EOF
```
Checklist:
- **`activation` must be one of `tanh`, `relu`, `sigmoid`, `softmax`, `elu` — never
  `swish`.** The vendored RTNeural in `neural_controller` only implements those five;
  an unrecognized activation string silently produces a null layer that **segfaults on
  load** on real hardware. This has actually happened (see `git log` on
  `mujoco_playground/workspace/configs.py` around the `elu` switch). If `configs.py`'s
  `policy.activation` is `"swish"`, change it to `"elu"` and retrain before exporting —
  don't try to patch the exported JSON after the fact.
- `behavior` should be `"leg_lift"` and `command_states` should be
  `["stand", "front_l", "front_r", "back_r", "back_l"]` (or whatever
  `configs.COMMAND_STATES` currently is — check `mujoco_playground/workspace/configs.py`
  if this doc and the code have drifted).
- `in_shape`'s last dim should be `observation_history * single_obs_size` where
  `single_obs_size` = 3 (ang vel) + 3 (gravity) + `len(command_states)` (one-hot) + 12
  (joint pos) + 12 (last action). For the 5-command setup that's 35 per step.

## 2. Place the policy + check config.yaml

The file `neural_controller` loads is:
```
Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/policy_leg_lift.json
```
Copy your validated JSON there, overwriting the old one. Then check
`Stanford/pupperv3-monorepo/ros2_ws/src/neural_controller/launch/config.yaml`'s
`neural_controller_leg_lift:` block:
- `model_path` should already point at `.../launch/policy_leg_lift.json` — only edit
  this if you intentionally renamed the file.
- `init_kps` / `init_kds` — should match what the policy was trained with
  (`configs.py`'s `PPO`/gain assumptions; currently kp=5.0, kd=0.25). Mismatched gains
  won't crash anything but will make the policy feel off on hardware.
- You normally do **not** need to touch `neural_controller.cpp`/`.hpp` — the observation
  layout (35 vs 36-dim, joint index offsets, etc.) is derived at runtime from the JSON's
  `in_shape`/`behavior` fields. Only touch the C++ if you changed the *shape* of the
  observation itself (e.g. added a new input), not just retrained with the same shape.

## 3. Sim-test on the workstation

### 3a. One-time environment setup (skip if already done — check `which colcon`)
This workstation is Ubuntu 22.04, not the 24.04 the repo's own
`install_dev_dependencies.sh` targets, and that script also runs a blind
`sudo apt upgrade -y` you don't want touching the CUDA/JAX training stack. Use an
isolated RoboStack (conda-forge) ROS2 install instead:
```sh
mkdir -p ~/ros2_env_setup && cd ~/ros2_env_setup
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o micromamba.tar.bz2
tar -xvjf micromamba.tar.bz2 bin/micromamba
export MAMBA_ROOT_PREFIX=~/ros2_env_setup/mamba_root
MM=~/ros2_env_setup/bin/micromamba

$MM create -y -n ros_jazzy -c robostack-jazzy -c conda-forge \
  ros-jazzy-desktop colcon-common-extensions rosdep
$MM install -y -n ros_jazzy -c robostack-jazzy -c conda-forge \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-hardware-interface \
  ros-jazzy-joy-linux ros-jazzy-xacro ros-jazzy-topic-tools ros-jazzy-foxglove-bridge \
  ros-jazzy-ros-testing ros-jazzy-ament-cmake-gtest ros-jazzy-ament-lint-auto \
  ros-jazzy-ament-lint-common compilers cmake make ninja pkg-config
$MM install -y -n ros_jazzy -c conda-forge glfw   # needed by pupper_mujoco_sim's viewer

eval "$($MM shell hook --shell bash)"
micromamba activate ros_jazzy
rosdep init && rosdep update
pip install python-xlib   # only if you'll use the spacebar-joystick trick, §3c
```

### 3b. Build
```sh
eval "$(~/ros2_env_setup/bin/micromamba shell hook --shell bash)"
micromamba activate ros_jazzy
cd Stanford/pupperv3-monorepo/ros2_ws
colcon build --continue-on-error --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_FLAGS="-g0" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```
Known gotchas, both already worked around by the flags above / this order:
- **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5` is required.** The vendored RTNeural's
  `CMakeLists.txt` has a `cmake_minimum_required` too old for RoboStack's modern CMake;
  without this flag the build fails immediately on `neural_controller`.
- **Plain-Python (`ament_python`) packages fail under `--symlink-install`** with a newer
  setuptools (`error: option --editable not recognized`) — this hits
  `animation_controller_py`, `bag_recorder`, `hailo`, `llm_websocket_server`,
  `openai_bridge`, `person_follower`, `pupper_feelings`. If `colcon build --continue-on-error`
  reports those as failed, rebuild just them **without** `--symlink-install`:
  ```sh
  colcon build --continue-on-error --packages-select animation_controller_py bag_recorder \
    hailo llm_websocket_server openai_bridge person_follower pupper_feelings \
    --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_POLICY_VERSION_MINIMUM=3.5
  ```
- If `ros2 launch` later complains a package isn't found even though you just built it,
  you forgot to `source install/local_setup.bash` (and `source
  ~/ros2_env_setup/mamba_root/envs/ros_jazzy/... ` / `micromamba activate ros_jazzy`) in
  *that* shell — each new SSH/bash invocation starts fresh, sourcing doesn't persist.

Confirm the plugin registered before moving on:
```sh
find install/neural_controller -iname '*plugin*'   # should print a controller_interface__pluginlib__plugin file
```

### 3c. Launch in sim
```sh
export DISPLAY=:1   # or whatever the active X session is — check `who` / `loginctl list-sessions`
source install/local_setup.bash
ros2 launch neural_controller launch.py sim:=True
```
A MuJoCo window should appear on the physical display. **Tell the user before running
this** so they can watch — that's the whole point of testing in sim first.

If there's no physical PS5 controller on hand and Bluetooth isn't available, run
`Stanford/pupperv3-monorepo/scripts/spacebar_joy_sim.py` in another shell (same env
sourced) to drive the O-button (leg-lift activate/cycle) from the keyboard instead —
read the script's docstring, it explains the X-grab approach and why it doesn't
double-trigger MuJoCo's own spacebar binding.

Watch `/leg_lift_command_index` to confirm each button press registers:
```sh
ros2 topic echo /leg_lift_command_index
```

**Set expectations before this runs**: a policy converging to a body tilt instead of a
clean lift is a known, previously-documented actuator-fidelity sim-to-real gap (training
uses an idealized direct-position model; `pupper_mujoco_sim` uses a torque-motor model)
— not necessarily a sign the new policy is broken. Judge it, but don't panic over it.

## 4. Only after the user has watched and approved sim — hardware test

**Stop and get explicit confirmation the robot is on a stand / secured before running
anything on real hardware.** Re-confirm even if it was on a stand five minutes ago —
things get moved.

### 4a. Connect
```sh
ssh pi@pupper.local     # or by IP if mDNS is flaky — ask the user to check the robot's
                         # display / `ip a` if pupper.local doesn't resolve or ping fails;
                         # this robot has changed IP/network between sessions before
```
Ask the user for the current password — don't assume last session's still works, and
don't hardcode it anywhere persisted. (Consider suggesting `ssh-copy-id` to set up
passwordless auth as a one-time convenience if the user's open to it.)

### 4b. Check what's already on the robot before touching anything
```sh
ls ~ ; ls ~/pupperv3-monorepo 2>/dev/null && (cd ~/pupperv3-monorepo && git remote -v && git branch)
```
**Do not assume `~/pupperv3-monorepo` is your fork/branch.** It may be cloned from a
completely different remote with its own history (this happened once already — a fork
called `ED1-WELL/pupperv3-monorepo` on branch `leg_lift_policy_tund`, unrelated to this
`Pupper_animation` workspace, with its own uncommitted build artifacts). If it's not
obviously `TundTT/Pupper_animation` on `robot-code`, **don't clone over it or switch its
branch** — ask the user, or just clone into a fresh directory alongside it:
```sh
git clone --branch robot-code --depth 1 https://github.com/TundTT/Pupper_animation.git ~/robot-code-leglift
```
(Use `--depth 1`. A normal clone pulls the *entire* repo history/object store — which
includes every training artifact ever committed on other branches, defeating the point
of the lean branch.)

If you already have a `~/robot-code-leglift` (or equivalent) checkout from a previous
session and just need the new policy: `git pull` inside it, or re-clone if `--depth 1`
history makes pulling awkward.

### 4c. Build
```sh
source /opt/ros/jazzy/setup.bash
cd ~/robot-code-leglift/ros2_ws          # or wherever you cloned it
source build.sh                          # NOT ./build.sh — fresh clones often don't
                                          # have the execute bit set on this file
```
The Pi's ROS2 (`/opt/ros/jazzy`) is baked into the OS image, not apt-installed — there's
no `ros2.list` apt source configured. This means:
- You generally **cannot** `apt install ros-jazzy-<whatever>` for a missing package here.
- `foxglove_bridge`, `camera_ros`, and `topic_tools` have been missing/broken on this
  image before (`camera_ros`'s `share/` dir exists but has no real ament-index entry —
  it's not actually installed despite looking present). None of the three are needed for
  leg-lift. If `ros2 launch` fails with "package X not found" for one of these, comment
  its `Node` out of the `nodes = [...]` list (not the `Node(...)` definition, just its
  entry in the list) in
  `~/robot-code-leglift/ros2_ws/src/neural_controller/launch/launch.py`, rebuild is not
  required for a launch.py-only edit, just relaunch. **This is a local, throwaway edit
  for this robot's environment — don't try to carry it back to `master` as a real fix.**
  If the same gap shows up repeatedly across sessions, that's a signal to actually fix it
  properly in `master` (e.g. a launch arg to disable vision nodes) rather than
  re-discovering and re-patching it by hand each time — flag this to the user rather than
  silently re-patching forever.
- `hailo_detection` and `person_follower` will likely die at startup with
  `ModuleNotFoundError: No module named 'vision_msgs'`. This is expected and harmless —
  those are unrelated vision nodes; a node dying doesn't abort the rest of `ros2 launch`.
  Don't treat it as a leg-lift failure.

### 4d. Launch + test
```sh
source /opt/ros/jazzy/setup.bash
cd ~/robot-code-leglift/ros2_ws
source install/local_setup.bash
ros2 launch neural_controller launch.py       # no sim:=True — real hardware
```
Watch the log for `neural_controller_leg_lift` reaching "configure successful" and for
the actuators finishing homing. Then hand off to the user with the button reference:

| Button | Action |
|---|---|
| **O (Circle, index 1)** | First press: activate `neural_controller_leg_lift`, command `front_l`. Each subsequent press: advance `front_l → front_r → back_r → back_l → front_l → …` |
| **X (Cross, index 0)** | Return to `neural_controller` (locomotion), exits leg-lift mode |
| **PS button (index 12)** | Emergency stop — deactivates all controllers |
| **Options (index 9)** | Release e-stop, reactivate last controller |

Optionally watch `/leg_lift_command_index` over SSH (same as §3c) so you can confirm each
press registered without relying only on what the user reports seeing.

## 5. Keeping `robot-code` in sync

`robot-code` is a **derived** branch — a pruned, flattened snapshot of
`Stanford/pupperv3-monorepo/` from `master`, not a place to accumulate independent
history. After any `master` change that should reach the robot (a new policy, a
`config.yaml` tweak, a real `launch.py` fix — not the throwaway per-robot edits from
§4c), regenerate it rather than trying to merge (the tree layouts differ — flattened vs.
nested — so a plain `git merge` will not apply cleanly):
```sh
git checkout master && git pull
git checkout robot-code
git reset --hard master
git rm -r -q mujoco_playground Stanford/training .notes CLAUDE.md
git rm -q .gitignore
git mv Stanford/pupperv3-monorepo/.gitignore .gitignore
git mv Stanford/pupperv3-monorepo/.gitattributes .gitattributes
git mv Stanford/pupperv3-monorepo/.vscode .vscode
for item in ai analysis bags infra install_dev_dependencies.sh LICENSE llm_logs.sh \
            pupper-rs README.md robot ros2_ws scripts stop_all_services.sh; do
  git mv "Stanford/pupperv3-monorepo/$item" "$item"
done
# re-add the "this is the robot-code branch" note at the top of README.md if git mv
# dropped it (it won't have — the note lives in the file content, not the path — but
# double check `head README.md` looks right)
git add -A
git commit -m "Sync robot-code from master (<one-line reason>)"
git push origin robot-code
```
This clobbers any local-only commits on `robot-code` — that's intentional (see above),
but if you're not sure whether something valuable was committed only there, check
`git log robot-code` and diff against `master` before force-resetting.

## 6. Report back
- Which policy (path to `mjx_params` / JSON, eval reward if known) you deployed.
- Sim result: did it build clean, did the user watch it, what did they say about it.
- Whether you proceeded to hardware, and if so: build result, launch result, what the
  user observed leg-by-leg, and whether it was an e-stop / crash / clean session.
- Anything you had to work around that isn't already covered in this doc's gotchas —
  update this file if you hit something new so the next pass is smoother.
