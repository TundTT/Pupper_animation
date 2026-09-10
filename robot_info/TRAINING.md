# Training For Direct Deployment

## Objective

Train against the real controller contract from the first environment step. A policy that succeeds only after joint remapping, observation conversion, output reinterpretation, activation replacement, or target-side repair is not robot-ready.

## Workflow

### 1. Select The Hardware Profile

Choose `leg_position` or `wheel_velocity` from `robot_info/robot_contract.json`. Copy its joint order and action types exactly. If the requested behavior does not fit either profile, document and verify a new physical profile before implementing the environment.

### 2. Reuse The Closest Verified Environment

Start from the Stanford-derived MJX environment and the closest behavior already in the repository. Keep the verified observation-history, PD-target, RTNeural export, and ROS controller patterns. Change the task, rewards, command distribution, and task-specific observations without replacing the deployment interface.

### 3. Implement Deployment Mathematics First

Before long training runs, write and test the exact deployed functions for:

- Quaternion and projected-gravity convention.
- Canonical joint ordering.
- Position-minus-default observations.
- Wheel-velocity normalization and signs when applicable.
- Previous faded normalized action.
- Newest-first history stacking and reset seeding.
- Position target clipping or velocity scaling.
- Policy-rate action repetition.

Use the same functions for evaluation and export tests wherever practical.

### 4. Configure Timing And Dynamics

The Stanford-derived reference uses a 0.02 second policy environment step and a 0.004 second physics step. The current ROS configuration is nominally 52 Hz. Measure target timing when it matters, represent the controller's action repetition, and randomize around observed latency and jitter rather than assuming ideal periodic execution.

Use the current robot description, actuator limits, default positions, inertial data, contacts, and friction as the nominal model. Record deliberate deviations.

### 5. Randomize Deployment Uncertainty

Retain the baseline categories that improve sim-to-real transfer:

- Observation latency and action latency.
- IMU and joint-state noise.
- Motor proportional and derivative gains.
- Link mass and center of mass.
- Ground friction and contact behavior.
- Initial state and command transitions.
- External pushes or kicks when appropriate.

Ranges should bracket measured hardware without making the nominal robot rare. Randomization does not excuse a wrong interface or calibration.

### 6. Train And Evaluate The Right Contract

Evaluation must load the same normalized input and output representation used by the C++ controller. Include:

- Multiple seeds and command sequences.
- Reset and controller fade-in behavior.
- Joint, velocity, effort, and gain limits.
- Observation and action saturation rates.
- Perturbed timing and dynamics.
- Task-specific failure conditions.

Simulation video is supporting evidence, not a compatibility test.

### 7. Export Once

Fold required normalization into the exported network or the documented controller math. Do not maintain a separate simulation model plus a manually translated robot model. Export float32 RTNeural JSON with a supported activation set, 12 outputs, `tanh` final activation, and all metadata required by `robot_info/POLICY_INTERFACE.md`.

Compute hashes after export and do not modify the model by hand.

### 8. Verify Numerical Parity

Feed recorded or generated observation vectors through both the training framework and RTNeural. Compare all 12 outputs within a documented float32 tolerance. The repository's C++ policy-contract fixtures and existing policy check scripts are useful patterns.

### 9. Exercise ROS Before Hardware

Load the exported model through the real neural controller in a non-motor ROS environment. Confirm:

- Plugin discovery and model parsing.
- Input and output dimensions.
- Embedded metadata agreement.
- Command subscriptions and state transitions.
- Observation history initialization.
- Position and velocity action paths.

### 10. Build For The Target

Run a clean ARM64/Jazzy build on an equivalent environment or the robot before the lab session. Resolve dependencies and LFS files before declaring the artifact ready. Follow `robot_info/LAB_READY.md` for the handoff.

## Artifact Layout

A robot-ready policy change should make these items easy to locate:

- Training configuration.
- Environment and reward implementation.
- Evaluation command and summarized results.
- Export command.
- Self-describing policy JSON.
- Model SHA-256 hash.
- Offline validation output.
- Training-versus-RTNeural parity result.
- Target build result.
- Hardware test instructions and safe starting conditions.

## Anti-Patterns

Do not:

- Train in an arbitrary joint order and remap at deployment.
- Observe values the physical controller cannot provide.
- Use wheel angle as a bounded state.
- Train one action meaning and reinterpret it on the robot.
- Apply wheel direction signs in both training and deployment.
- Export an activation unsupported by RTNeural.
- Rely on desktop-only packages or x86 binaries.
- Call an artifact ready because it produces a convincing simulation video.
- Leave build, dependency, model-format, or launch integration work for the lab.
