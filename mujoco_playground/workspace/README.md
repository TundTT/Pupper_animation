# Active training on this branch

This branch now trains capsule-foot **walking**, with ring-clearance penalties
and a stable default/home pose. See [WALKING.md](WALKING.md) for model conventions,
remote RTX 6000 setup, training, evaluation, and export.

The files `leg_lift_env.py`, `configs.py`, `train_leg_lift.py`, and the old evaluation
and visualization helpers are retained as legacy references. Their leg-lift pose,
rewards and model-import assumptions do not apply to this walking model.
