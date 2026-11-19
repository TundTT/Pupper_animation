# Tip support, ring clearance, and reverse walking

Your intended behavior: support the robot on the rounded capsule tip; avoid
leaning onto or dragging the surrounding TPU ring, which is not the intended
load-bearing structure.

![Nominal and tilted foot geometry](ring_intent.png)

## What the existing ring penalties do

The ring itself has no collision forces. The capsule is the physical collision
shape. The trainer transforms 96 outline probes per foot through the actual foot
pose, including randomized leg lengths/mount tilts. These are a convex outer
outline on two rim edges, not a complete deformable TPU simulation.

- **Side clearance, weight -1:** 74 side probes per foot should be at least
  1 mm above the floor. The deepest violation on each foot is squared after
  dividing by 10 mm, capped at four, then averaged across four feet.
- **Bottom compression, weight -1:** 22 fixed lower-arc probes per foot may be
  up to 9 mm below the floor without this penalty. Only excess depth is penalized
  with the same square/cap/foot averaging as sides. This allowance does not apply
  to whichever part happens to become lowest after rotating the foot.
- **Side rubbing, weight -2:** near-floor side probes are penalized for squared
  world-XY velocity. Weight rises linearly from zero at +1 mm height to full
  weight at -1 mm; at the floor it is one half. Take the maximum weighted probe
  speed per foot, then average feet. This does not measure normal force, and on
  a tilted floor world XY is an approximation to the floor tangent plane.

All three costs are averaged over 250 Hz substeps and multiplied by their weights
and the outer 20 ms control timestep. Zero cost is the best possible ring reward.
They are soft penalties, not hard contact prohibitions. The two depth terms each
max out at four before weighting; they can outweigh linear tracking's ceiling of
two. The earlier source comment claiming otherwise has been corrected.

Examples with only one foot violating, held for one control step:

- A side probe 5 mm below the floor has a 6 mm clearance violation and costs
  `(6/10)^2 / 4 * .02 = .0018` reward.
- A bottom probe 12 mm below the floor exceeds its allowance by 3 mm and costs
  `(3/10)^2 / 4 * .02 = .00045` reward.

The 9 mm allowance is an existing training tolerance, not a measured safe TPU
load or a permission to support the robot on ring sides. Normal nominal standing
already overlaps the undeformed bottom outline by about 4.4 mm.
The bottom rim extends roughly 3 mm below the capsule tip in this model. Some
bottom TPU compression is therefore expected before the tip seats; "tip support"
does not mean the bottom ring can never touch first. The simulator does not
calculate real TPU-versus-tip load sharing.

## Added direct support preference

The existing ring settings and geometry are retained. A new contact-only
`tip_support` penalty uses the designated tip site to determine the capsule's
intended downward direction. It permits up to 15 degrees of lean from the floor
normal, then increases smoothly in cosine space: raw cost one at 45 degrees,
capped at four, averaged over feet and substeps, with weight -0.5. Airborne feet
are not penalized by this term. These angles are an engineering starting point,
not measured TPU tolerances. This supplements the ring checks; it does not infer
or simulate a force borne by the TPU ring.

Measured existing contacting-foot orientations are already mostly close to the
tip: reverse median 5.2 degrees, p95 8.4 degrees, max 9.5 degrees in the prior
showcase segment. Thus its reverse shuffle is not evidence of capsule-barrel
support. The new guard mainly discourages future policies from exploiting lean.

## Reverse and general-gait trial

With continuous independent vx/vy/yaw commands, the old sampler assigns only
about 0.86% of all commands to vx below -0.05, |vy| below 0.03 and |yaw| below
0.1. Symmetric vx bounds did not provide frequent straight reverse practice.

Both trial arms warm-start identically from the preceding best checkpoint:

- 20% stand, 25% straight forward, 25% straight reverse, 30% mixed commands.
  Straight speeds span 0.10--0.35 m/s; full mixed ranges are retained.
- Air-time weight 0.3 to 1.5, encouraging swings longer than the existing 80 ms
  threshold. No externally imposed gait phase or enlarged action bounds.
- The tip-support guard described above; ring weights/allowances unchanged.
- Existing action regularization -0.30, foot-slip -0.6, torque -0.0002 retained.

The second arm adds a touchdown penalty, weight -5: on each 250 Hz transition
from airborne to contact, use the last airborne normal velocity and square the
downward speed exceeding 0.15 m/s. Sum events within the control interval and
retain the outer dt factor. Carry contact/velocity history across control
boundaries; standing contact and upward motion incur no cost. This measures
approach velocity, not impact force. Calibration against the prior fast forward
trace gives approximately -0.040 per control before outer dt, versus +0.073
clearance reward. The first arm sets this new penalty to zero for comparison.

Each arm requests 43.2 million training steps on a separate GPU, with identical
seed, geometry DR and optimizer settings. Source snapshots and exact launch
commands are saved. Selection will use direction-specific tracking, survival,
clearance, contact/tip posture, slip, and touchdown speed rather than simply
comparing different total reward scales.

Validation: six existing environment checks and four new command/contact tests
pass, including diagnostic/ordinary stepping agreement and impact-history carry.

## Findings and revised training

The initial 43M-step trials were not selected as replacements. More straight
practice improved reverse tracking/clearance, but lateral tracking regressed.
The -5 touchdown coefficient suppressed lift. The subsequent trials use:

- 20% stand, 20% forward, 20% reverse, 10% pure lateral, 10% pure yaw, 20% mixed.
- Signed duration reward: `clip(air_time - .08, -.08, .25)` at touchdown,
  weight 1.5. Short swings now have a negative score and a graded incentive to
  lengthen; continuous stance remains free. This is not a prescribed trajectory.
- A lighter touchdown weight of -1.5.

The second 86M-step trials exposed a low-speed reward loophole: both stopped
moving for a 0.10 m/s rightward command. The old linear tracker awarded 1.7348 of
its maximum 2 points for doing so, leaving only 0.2652 to pay all stepping costs.

The next trials use tracking variance
`min(.07, .0025 + .55 * ||command_xy||^2)` instead of fixed .07. At 0.10 m/s,
standing now earns only 0.573 of 2 points; at 0.35 m/s the variance is .069875,
essentially the old value. Exact tracking still earns 2. The two arms compare
action-rate weights -0.3 and -0.6, starting from the second stage's softer-landing
checkpoint. Model geometry, motor/action limits and the ring rules are retained.

All comparisons use 13 fixed commands, three seeds each, six seconds per command,
discarding the first second for steady statistics. Both policies must complete
the same cases; mean reward across different objective versions is not used to
claim improvement. Reports include the severity of ring-side penetration as well
as its frequency, since shallow frequent overlap and deep overlap are different.

## Final training correction

Strict low-speed XY tracking restored sideways movement but made turning too
expensive when a small amount of body translation occurred. XY tracking now adds
`.27 * commanded_yaw^2` to its variance, still capped at the original .07. This
restores the old XY tolerance during 0.5 rad/s turns. Yaw tracking uses variance
`min(.25, .05 + .8 * commanded_yaw^2)` so slow turning itself remains worthwhile.

A new clearance-shortfall penalty scores actual 250 Hz touchdown events. Each
foot tracks the maximum capsule clearance during its airborne interval. On
replanting, the cost is `max(1 - peak/.004, 0)^2`, summed across landing feet,
weighted -0.1 and multiplied by control dt. It is gated on a moving command.
This is a soft 4 mm minimum-peak incentive, not a rigid path or a required 28 mm
apex. It complements air time and catches tiny contact flickers between controls.
Peak history is cleared on contact and episode reset, and carried across control
boundaries while airborne. Tip support and all three original ring rules remain.

The stronger-smoothing trial had a late KL spike (0.214) and its final checkpoint
was rejected; its earlier best remains saved. The final trial starts from the
stable precision checkpoint, retains action-rate -0.3, and uses learning rate
1e-4. Warm-start training now defaults to that lower rate unless explicitly
overridden. Eight command, contact, tracking and diagnostic tests pass.

## Completed result

The final run completed 86,507,520 steps. Step 57,671,680 is retained as the
experimental candidate, with improved slow reverse tracking/clearance but remaining
short swings and increased ring-side overlap versus the prior policy. See
[the measured results and video](RESULTS.md) for the selection and all tradeoffs.
