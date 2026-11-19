> Latest steering: use a flush rigid capsule with no ring-bottom compression reward. See [RIGID_CAPSULE.md](RIGID_CAPSULE.md).

# Hardware front-foot drag follow-up, September 7

Hardware feedback: selected robot-code policy b455cbf looks reasonable overall,
but front feet drag during forward walking; rear feet and reverse drag less.
Surface has not been specified. No new hardware deployment is part of this test.

First test: identical deployed checkpoint, floor sliding friction 1 (XML nominal),
2, and 3, with all other model and policy settings held fixed. Native MuJoCo contact
inspection confirms actual capsule-floor friction coefficients equal 1/2/3. Floor
friction does not impose a no-slip constraint; contact remains compliant.

Six commands (forward .15/.20/.35, reverse .20, left/right spin .5), three seeds,
eight seconds each, discard the first second. MJX at250Hz, policy50Hz. No pushes,
observation noise, latency, or joint-reset noise. Each foot is scored separately:
contact duty, swing peak/duration, height95th percentile, contact sliding speed and
integrated distance, and motion faster than .05m/s below2mm clearance. That last
metric is a geometric scuff proxy, not a measured TPU load.

The previous clean suite's per-foot traces already show shallow front lift at
forward .20: 95th-percentile heights2.49/2.90mm; no front samples above4mm. Aggregate
four-foot swing statistics did not adequately expose this hardware-relevant margin.

Friction is now configurable in both nominal/evaluation models and the DR sampler.
Tests confirm that training's requested friction range survives randomization and
that changing the range preserves all other perturbations for identical seeds.
Default friction settings are still unchanged pending the experiment results.

The friction-only run completed86.5M. The64.9M checkpoint improved front lift but
retained a weak front-left swing and introduced fast-forward underspeed. A second
86.5M run used friction[.8,2.5],8mm swing height, shape weight-3, and linear tracking6.
Its64.9M checkpoint gives nominal forward.20 peaks5.20/5.08mm, error.0231m/s versus
.0210m/s deployed, with all39 nominal rollouts surviving. However fast reverse ring
side overlap rises to4.51% (from.04%). A final43.2M requested-step polish warm-starts
that64.9M checkpoint and changes only ring-side-contact weight from-3 to-6, keeping
all other second-run settings. This is a measured correction of a regression,
not a claim that forward clearance alone certifies the whole policy.
