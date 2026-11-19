# Reverse and spin jitter follow-up

The prior policy still has short reverse swings and more ring-side overlap. The new run must be judged on physical motion, not only reward or average velocity error.

Measured from the saved 250 Hz evaluation / 50 Hz action trace:

| Command | Second action-difference MSE | Action power at >=5 Hz |
|---|---:|---:|
| Forward .20 m/s | .0271 | 9.9% |
| Reverse .15 m/s | .1372 | 99.5% |
| Reverse .20 m/s | .1550 | 98.7% |
| Reverse .35 m/s | .1473 | 97.4% |
| Left spin .50 rad/s | .0118 | 8.0% |
| Right spin .50 rad/s | .0249 | 29.0% |

The new objective penalizes mean((a[t]-a[t-1])-(a[t-1]-a[t-2]))^2.
A steady target ramp pays only on acceleration; repetitive rapid reversals pay repeatedly. This is a discrete command-acceleration proxy, not measured mechanical jerk. The existing first-difference penalty stays -0.30. No runtime filter, action-bound, observation-width, model-geometry, or ring-rule change is introduced.

Two identically warm-started trials compare coefficients -0.8 and -2.0, each requesting 43.2M steps, 8192 environments, seed 0, learning rate 5e-5. At reverse .20 these initially contribute -0.124 / -0.310 before outer dt. This is a deliberate frequency penalty; it must not earn an apparent improvement by stopping movement. All prior duration, clearance, touchdown, tip and ring terms remain.

Validation includes action-history clearing across automatic resets and ordinary/diagnostic stepping agreement. Matched evaluation also records joint velocities and torso angular velocities at 250 Hz, to check that smoother targets correspond to smoother physical motion. The showcase now includes both spin directions (right spin 28-30 s, final stop 30-32 s). W&B intermediate, final and best videos stay enabled.

Selection: lower command reversals and physical angular/joint acceleration in reverse and both turns, without freezing, falls, or increased ring-side use. Check airborne duration and cadence rather than reporting a reduction in action deltas alone. Retain forward/lateral and transition checks. If neither trial meets these criteria, continue diagnosis rather than declaring the issue fixed.

Additional baseline check: reverse .20 has no joints above 95% of their command bounds in the steady window, so repeated clipping is not causing its oscillation. In-place +.5/- .5 rad/s spins average +.439/- .362 rad/s with yaw-rate standard deviations .115/.178 rad/s respectively. The right-spin issue combines underspeed with uneven rotation.

## Result of acceleration-only trials and next correction

Both arms completed 43,253,760 steps and selected their 28,835,840 checkpoints. They reduce command acceleration but retain ~5-6 Hz reverse stepping, and increase right-spin ring-side overlap. Neither is selected as the fix. Both runs uploaded six videos, including final and best.

The next two trials replace the height-area reward with a lift-triggered planned swing penalty: a 3 mm sin-squared reference over 160 or 200 ms, weight -0.4, averaged at 250 Hz then multiplied by control dt. Early contact does not clear the clock, and repeated lift during the window does not restart it. The clock is per foot and reward-only; no new observation channel or runtime filter is introduced. Horizontal foot placement is free.

A constrained kinematic check of a 32 mm fore/aft stroke and 3 mm sine-squared lift is feasible with up to 6.34 mm lateral movement: max XZ residual below .0022 mm across all four feet and common scales .864, 1 and 1.0192. A fixed lateral line has up to 1.81 mm residual. Raising/lowering the torso alone did not resolve it, so torso targets and joint bounds are retained. This is a path-feasibility check, not a dynamic or combined-DR guarantee.

Both shape arms retain action-acceleration -0.8 and increase ring-side weight from -1 to -3 because the preceding smoothing trials traded away ring clearance. They warm-start identically from the previous selected policy, with 86.4M requested steps and learning rate 1e-4. Other ring settings and existing tip guard are unchanged.

## Workspace correction after completed shape trials

The 160 ms shape trial completed 86.5M steps. At reverse .20 it reduced joint acceleration from ~138 to ~77 rad/s² and cadence from ~6.1 to ~2.2 Hz. However fast reverse undershot its command and right turning remained poor. The 200 ms arm was worse and retained a frozen right turn. Neither is a final replacement.

The next trials provide the mechanical workspace needed for a centered arc: knee command scales .9 or 1.1 rad (previously .5), while hip .5 and abduction .25 stay fixed; the moving torso target is 3 mm lower. Standing height is unchanged. Both scales are within the existing XML joint limits. A 56 mm fore/aft arc at 4 or 6 mm height is feasible with a fixed lateral foot position: maximum residual .054 mm at the shortest common scale and essentially zero nominally. Physical geometry itself is unchanged.

Tracking-linear weight becomes 4 and yaw weight 3 to avoid rewarding underspeed/freezing when stepping is more expensive. The 4 mm reference uses shape weight -.4, the 6 mm reference -1. Both keep 160 ms duration, action-acceleration -.8, ring-side -3, other ring settings and tip guard. They warm-start from shape160 final/best with approximate rescaling of the tanh action head and exact corresponding action-history normalization adjustment. Step-zero evaluations must verify this adaptation. Smoothness penalties and evaluation action-difference statistics use a fixed reference scale, so wider authority does not hide larger physical motion. Saved action scales are honored separately for every checkpoint in matched evaluation.

## Completing the swing objective and enforcing ring clearance

The workspace trials learned slower reverse motion, but their feet still brushed the ring sides and right turning often used tiny contact flickers. Measurements show that they also stayed about 4 mm above the lowered walking-height target: its old weight was too weak to enforce the intended workspace.

Two follow-up arms add .25 credit per active foot to the planned-swing score. With negative cost weight this becomes `weight * (bonus - normalized_error²)`. A correct full arc earns credit; a full window hovering at floor height has mean error² .375 and therefore loses reward. Early contact is worse. Timers still cannot be reset early. This removes the previous objective's incentive to avoid lifting altogether. Tests check a full arc, floor hover and early-contact trajectories.

Height weight increases from -2 to -12, retaining the geometry-dependent reference and -3 mm moving offset. A new ring-side-contact term penalizes the fraction of feet with any side probe below the floor, averaged over substeps. It supplements the existing depth/rub penalties and explicitly makes shallow frequent brushing costly. The two arms compare weights -1 and -3. Both start identically from workspace6 best, keep its 6 mm/160 ms reference, physical smoothness weights and command scales, and request 172.8M steps with five evaluations.
