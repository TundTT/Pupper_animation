# Latest wheel policy: heating backpack + 9 mm gap

Use `policy_wheel.json` for controller inference and `mjx_params` for a training warm start.
Trained on GPU 0 for 201,850,880 additional environment steps at learning rate 5e-5.
The warm start is the latest selected/completed 9 mm policy, identified in launch_job.json.
`run.json` and launch_job.json preserve the original PC paths for provenance; use this
checkout's model and assets when starting a new run. The saved XML uses the repository's
Stanford/training/pupper_v3_description/description/meshes/stl assets.

`rollout.mp4` is a real final checkpoint simulation rollout. Logs were recorded offline;
no W&B upload was performed. comparison.json compares the old checkpoint on the backpack
model (step zero) with this fine-tune, not the old robot without its backpack. No independent
holdout or physical hardware validation was performed.

The robot-code branch uses this export at `ros2_ws/src/neural_controller/launch/policy_wheel.json`
through `neural_controller_wheel` (Circle). Read robot-code's LATEST_POLICIES.md and policies/latest.json.
