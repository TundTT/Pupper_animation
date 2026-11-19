# Does the improved video agree with the statistics?

Yes: overall tracking, action smoothness and foot-slip costs improved, and the
fast forward gait has longer swings at roughly half the old cadence. Landings
are not softer, and reverse clearance remains low.

## Completed training evaluation

Training completed at 201,850,880 steps. Best checkpoint: 187,432,960 steps.
Compare the previous policy re-evaluated at step zero of THIS run against its best
checkpoint. Both use the new reward weights, push-free evaluation, 128 evaluation
environments and the same randomized-geometry evaluation distribution. Commands
and resets are not a paired rollout here. The final checkpoint is close to best
(reward 36.279 versus 36.358), so the result is not an isolated last-step collapse.

| Metric | Starting checkpoint | New best |
|---|---:|---:|
| Mean XY velocity error | 0.06778 m/s | 0.04633 m/s |
| Mean per-step action-difference RMS | 0.46909 | 0.30179 |
| Weighted foot-slip cost per control | 0.03285 | 0.01899 |
| Mean fraction of feet with ring-side penetration | 8.70% | 6.39% |
| Mean episode length | 600 steps | 600 steps |
| Episode reward | 32.667 | 36.358 |

Reward and penalties in the original old run are not directly comparable because
weights and evaluation disturbances changed. Metrics above come from the new
run's metrics.jsonl. The normal 12-second evaluation horizon was completed by all
128 environments at both comparison points; this is not a long-duration survival
guarantee. Missing-stance-foot penalty during stops declined about 81%.

## Matched 250 Hz foot measurements

Old step-zero parameters and new best parameters were run through the same nominal
model, showcase commands, seed 2, noise-free observations and push-free dynamics.
Both completed 800 controls (16 seconds) without falling. At each command's first
step, instrumented versus ordinary qpos agreed to 20 micrometres/radians absolute
tolerance. Instrumentation returns substep geometry without altering stepping.

For steady forward 0.35 m/s, use t=4.5--6.0 seconds, excluding the command transition.
All 37 old and 19 new complete airborne events exceed the declared event filter
(duration >40 ms and peak >2 mm), so that filter excludes no complete events in
this fast comparison. Cadence is inverse median same-foot liftoff interval.

| Metric | Old | New |
|---|---:|---:|
| Cadence | 6.25 Hz | 3.33 Hz |
| Median airborne duration | 60 ms | 136 ms |
| Mean swing peak | 4.16 mm | 7.21 mm |
| Action-difference RMS | 0.35461 | 0.24126 |
| XY velocity error | 0.04326 m/s | 0.04157 m/s |
| Foot-slip cost | 0.04670 | 0.02235 |
| Median downward speed just before contact | 0.22254 m/s | 0.28270 m/s |
| Ring-side penetration fraction | 0.67% | 5.07% |

Thus the forward gait is slower in cadence, longer in swing and smoother in its
commands, with substantially less horizontal slip and preserved tracking. However,
pre-contact downward speed increased about 27%; this measures velocity, not impact
force. Ring-side penetration worsened in this particular fast nominal segment
even though it improved across the broader randomized evaluation.

At 0.20 m/s, command roughness fell from 0.264 to 0.197 and velocity error from
0.0394 to 0.0231 m/s. Foot lift is still asymmetric (new steady maxima by foot:
6.83, 0.47, 17.99, 2.12 mm). Reverse tracking improved from 0.0745 to 0.0317 m/s,
but all new reverse foot peaks stay below 2 mm; reverse roughness increased from
0.173 to 0.228. These observations do not support a uniform all-direction gait
improvement or a symmetric swing trajectory.

Measurements and event records: comparison.json, old.npz, new.npz. Reproduction
script: ../compare_gait.py. This is one paired nominal showcase, not a multi-seed
hardware or perturbation robustness test. Multiple training changes were combined,
so the result cannot attribute the improvement to action regularization alone.
