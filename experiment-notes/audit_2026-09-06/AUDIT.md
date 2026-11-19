Audit of Pupper V3 walking policy, 2026-09-06

Training source was not modified. This audit uses the current `leg` working tree (including its pre-existing uncommitted edits), the saved run configuration, and `workspace/output/walk_2026-09-06_03-24-30/best_params`. The saved model hash matches the current XML. Checkpoint metrics confirm step 173,015,040, reward 29.189739, mean episode length 586.609375. Findings distinguish direct measurement, mathematical incentives, and proposed experiments; no PPO ablation was trained.

**1. The observed gait is fast, short-duration stepping; the proposed rectangular-height explanation is not supported.**

The reward interpretation is correct: `walk_env.py:121–133,161` integrates the sum of clipped airborne capsule-bottom heights. For an unconstrained swing of duration T and height cap H, a rectangular trajectory approaches area HT; a sinusoid has area 2HT/pi and a sine-squared trajectory HT/2. At H=0.028 m and weight 10, their average reward rates per continuously airborne foot are 0.28, 0.17825, and 0.14. This term alone does not specify an aesthetically smooth trajectory. It also does not select cadence when duty factor and normalized height shape are fixed.

However, the actual policy is not holding the foot at the cap. I reproduced all 800 control steps of the best showcase with seed 2, then repeated it with only pushes disabled. Both log every 4 ms substep. Unmodified `env.step` and instrumented step agree within 9.78e-6 maximum absolute qpos difference over the first 300 steps; numerical compilation differences do not change the finding. Three re-rendered poses match original video frames with about 1.42–1.50 mean absolute RGB levels out of 255, consistent with video compression. Native MuJoCo also confirms the capsule heights at these poses. See `render_check.jpg`.

Original-video evidence: `best_2.96_frames.jpg`, `best_4.80_frames.jpg`, and corresponding final-video sheets contain consecutive 25 fps frames spanning 0.40 s. These span more than a complete stride. At the measured fast cadence there are only four original frames per stride, so extracting 5–10 genuine frames from one fast stride is impossible. The instrumented trace provides 40 samples per stride instead. Final-video inspection is visual only; numeric rollout measurements below are from best_params.

`clean_feet_250hz.csv`, `showcase_feet_250hz.csv`, and `swing_analysis.png` contain the measured height curves. At steady 0.35 m/s the clean trajectory repeats at approximately 0.160 s, or 6.25 Hz. Complete airborne intervals have median 60 ms and maximum 76 ms over 4.5–6.0 s; the representative back-right swing is 72 ms. The 0.20 m/s interval has a median 40 ms airborne interval, maximum 80 ms, and alternating larger and smaller lifts. Rapid action oscillation is present in hip, abduction, and knee.

One complete clean back-right swing, with the next liftoff 160 ms after the current liftoff:

| Time from liftoff, ms | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 | 64 | 72 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Capsule-bottom height, mm | 0.890 | 2.792 | 4.470 | 5.960 | 6.702 | 5.856 | 4.509 | 2.859 | 1.036 | -1.011 |

The last sample is contact. The preceding airborne vertical speed is about -0.28 m/s. Across the larger fast clean swings, mean height/peak is 0.576 and time above 90% of peak is 17.2%; these are rounded pulses, not rectangular dwell near a cap. The original showcase yields similar ratios, 0.577 and 19.7%. Neither trace reaches the 28 mm cap anywhere.

There is a real discrepancy with the supplied earlier 10–35 mm result: the reproduced full best showcase peaks at [4.76, 6.16, 12.95, 7.29] mm for [front right, front left, back right, back left]; with pushes disabled, peaks are [5.70, 6.54, 7.25, 5.73] mm. This does not invalidate another command/seed/model measurement, but that earlier number cannot establish the waveform of this particular showcase. This audit does not substitute peak height for the requested temporal analysis.

`train_walk.py:99,175` disables noise, latency, and reset joint noise for this video, but retains pushes. Nine pushes occur at control indices 71, 278, 296, 359, 387, 395, 474, 528, 741. There are no pushes in the steady 0.20 m/s interval. Removing them preserves the fast stepping. Thus pushes explain some individual disturbances, not the steady cadence. The camera magnifies motion and 25 fps undersamples a 50–75 ms swing, but neither creates the 250 Hz measured pattern.

**2. Try action_rate=-0.30 first; torque regularization is a poor first lever.**

`walk_env.py:170–176` uses mean squared normalized action differences, but sum squared actuator force at the final control-step sample. These are different normalizations and different physical objectives. The following are measured weighted terms before multiplication by ctrl_dt=0.02, from the clean steady windows 2.5–4.0 s and 4.5–6.0 s:

| Term | 0.20 m/s | 0.35 m/s |
|---|---:|---:|
| swing_clearance | +0.013136 | +0.043233 |
| action_rate | -0.003463 | -0.006249 |
| torques | -0.000199 | -0.000174 |
| foot_slip | -0.029405 | -0.046699 |
| air_time | +0.000400 | +0.000720 |
| ring_side | -0.000302 | -0.000257 |
| ring_bottom | 0 | 0 |
| ring_rub | -0.000139 | -0.000188 |
| tracking_linear | +1.938815 | +1.932929 |

At 0.35 m/s, raw action-rate mean is 0.0062488/0.05=0.124976. The coefficient that makes its total penalty equal the current swing reward is 0.0432329/0.124976=0.34593. At 0.20 m/s the equivalent coefficient is 0.18965. Choose -0.30 as an explicit intermediate experiment: the new penalties are -0.020779 and -0.037493, versus the current -0.003463 and -0.006249. At fast speed this changes the objective by only -0.031244, about 1.6% of the linear tracking term, while making roughness comparable to the actual clearance incentive.

This is a scale-calibrated experiment, not proof that the learned policy will smooth out. For a fixed-amplitude sinusoidal action, mean squared differences are proportional to sin²(pi*f*0.02). Halving 6.25 Hz to 3.125 Hz cuts that component to 25.99%, saving approximately 0.02775 at the proposed coefficient if the measured raw cost were sinusoidal. In a walking policy stride amplitude may grow when cadence falls, so this estimate is conditional, not a predicted reward improvement.

The RMS action difference is 0.264 at slow speed and 0.355 at fast speed. Fast back-left hip RMS command difference is 0.496 normalized, or 0.248 rad per 20 ms after action_scale=0.5. The issue is broader than the vertical foot motion. Actual joint-velocity RMS is 2.53 rad/s over all joints/substeps at fast speed; command roughness should not be confused with instantaneous physical joint displacement.

Fast raw torque cost is only 0.0001743/0.0002=0.8716 Nm² summed across motors. Matching the proposed action penalty would require a torque weight about -0.043, over 200 times the present coefficient. This would tax support and propulsion, and still would not directly penalize jerk. Keep torques=-0.0002 in the first experiment. If torque becomes the subject of a later experiment, average squared force over substeps first; the current end-of-control sampling can miss force peaks.

Minimal discriminating experiment: equal-budget warm starts from best_params, identical seeds/commands/DR: baseline versus only action_rate=-0.30. Render 50 fps, evaluate with push_probability=0 as well as the original disturbed showcase, and record cadence, swing duration, action differences, touchdown normal speed, tracking error and survival. If action roughness falls but cadence/duration do not improve, the temporal structure requires its own incentive. This audit has not run those training arms.

A cheap separate cadence arm is air_time=1.5, keeping action_rate=-0.05. For four feet, Ts=0.16 s, stride period 0.32 s, the existing formula would contribute 4*(1/0.32)*0.02*1.5*(0.16-0.08)=0.030 per second of return; coefficient 0.3 provides only 0.006. This deliberately brings the longer-swing incentive into the same 0.03 range as the roughness change. It does not uniquely enforce 160 ms, and contact chatter must be monitored. Do not combine it with the action-rate arm initially if attribution is the purpose.

**3. A smooth 28 mm apex has a kinematic prerequisite; a simple target-height patch is not ready to apply.**

`walk_config.py:10` and `walk_env.py:108–110` restrict commanded offsets to hip +/-0.5, abduction +/-0.25, knee +/-0.5 rad. The nominal standing leg is close to extended. A broad hip/knee grid does reach 32.98 mm nominal and 31.30 mm at scale 0.864 without ring costs, but those maxima occur at large fore/aft excursions. A reachable peak somewhere is not a reachable centered swing arc.

Constrained front-right native-MuJoCo kinematics, fixed nominal home torso and original foot lateral location: maximum centered capsule-bottom height is only 2.909 mm. Allowing lateral movement and holding only fore/aft position centered raises the maximum to 5.920 mm. Holding lateral location fixed, maximal heights at fore/aft offsets -60,-40,-20,0,+20,+40,+60 mm are 15.03,7.55,3.62,2.91,5.36,10.54,19.44 mm. These are static feasibility results, not dynamic impossibility under all body motions.

For a centered nominal front-right foot, inverse kinematics for a 10 mm apex requires offsets (hip,abd,knee)=(0.317,0.056,-0.835) rad; 20 mm requires (0.440,0.095,-1.166); 28 mm requires approximately (0.530,0.165,-1.383), outside current hip/knee action bounds. These poses have zero ring cost. At the short endpoint, widening authority alone still requires checking lateral/path feasibility. Therefore do not replace the current reward with an unconditional 28 mm centered sinusoid and expect a reward weight to solve the constraint. This audit does not recommend silently broadening motor authority as part of the small reward experiment.

The current height objective further favors near-extension: it penalizes deviations from the nominal settled torso height. It currently costs 0.125 for lowering the body 10 mm, larger than the measured +0.043 fast clearance term. A smooth long stride may need a lower torso or more knee authority. This is a separate hypothesis from rectangular reward exploitation.

If a phase-tracking redesign is chosen after checking a feasible path, use a real phase target, not a constant height merely multiplied by horizontal speed:

z_i* = H sin²(pi*phi_i), H=0.028 m, phi_i in [0,1] over a planned 0.16 s swing.

R_shape = -1.25 * moving * sum_i b_i * ((z_i-z_i*)/0.028)²,

where b_i is the PLANNED swing mask, not the observed noncontact mask. Evaluate/average at the simulation substeps and retain the outer ctrl_dt multiplication. Start with a 0.32 s period and paired trot offsets [0,0.5,0.5,0] if an explicit trot is desired; the 0.16 s swing has eight controls, four video samples at 25 fps, and eight at 50 fps. Supply phase to the policy if using an external clock. A liftoff-driven clock needs a minimum swing window that persists through premature contacts; otherwise touching early removes the error penalty. This is an environment/state/observation redesign, not a weight-only change.

Weight derivation: a nominal fixed-pose pure-knee sweep has local ring marginal costs about 13.83 per metre at 0.52 mm clearance, 13.11 at 1.04 mm, 10.55 at 1.96 mm, and 7.18 at 3.02 mm. The old approximately 6.6 figure is not a bound across the range. To produce a restoring slope larger than 13.83 at 5 mm height error, W must exceed 13.83*0.028²/(2*0.005)=1.0843. W=1.25 provides slope 15.94 per metre. This replaces swing_clearance=10. If supplementing the old positive term, use W=2.0: its slope 25.51 exceeds 13.83+10=23.83. These are local calibration checks, not a bound on all poses or every competing reward. Far from feasibility no scalar weight is a solution.

The suggested -k*(z-h_target)²*||v_xy||*~contact does not by itself define a rise/fall profile if h_target is constant. Its multiplier also vanishes when the foot pauses, and contact can remove the penalty. A phase-dependent reference and appropriate gating are necessary to make the claimed trajectory-shaping interpretation true.

**4. Other reward terms and sampling issues.**

- `foot_slip` (`walk_env.py:135–155`) is larger than action_rate and comparable to clearance in the measured intervals. It penalizes horizontal speed of contacting feet, not vertical touchdown speed. It is consistent with spending brief intervals repositioning a foot and planting it quickly; it is not proof of causation. Retain -0.6 initially because reducing it risks skating.
- `air_time` starts paying only above 80 ms, and is sampled at 50 Hz (`walk_env.py:144–155`). Most measured swings barely reach or miss that threshold; its realized reward is tiny. Air-time resets whenever contact is true, not only at a stable touchdown. Very short 4–28 ms noncontact episodes are present, so distinguish true swings from contact flicker before using a contact-based phase clock. Debouncing should not silently redefine physical slip contacts.
- Ring penalties saturate at side excess depth 20 mm or bottom excess depth 20 mm (`walk_geometry.py:68–79`): side points at -19 mm or lower, bottom points at -29 mm or lower with the present allowance. All measured steady-forward saturation fractions are zero. Saturation is not responsible for these particular jabs. Unsaturated ring costs can still discourage pure-knee lift, as quantified above.
- `stance_feet` (`walk_env.py:168`) is zero for moving commands, so it cannot explain steady walking jabs. It can produce sharp planting in the final stop, once commanded rest and speed below 0.05 m/s coincide.
- The ring and swing history inside `mjx.step` uses geometry at the beginning of each 4 ms integration substep; qpos is advanced afterward. The CSV timestamps follow that geometry time. The explicit final `mjx.forward` refreshes the 50 Hz sample. This is a 4 ms sampling convention, not evidence of a new gait oscillation.

**5. Recommended asymmetric length distribution: common U(0.90,0.98), per-leg U(0.96,1.04).**

At `walk_randomize.py:86–88`, change only the common upper endpoint from 1.10 to 0.98 for this range proposal. Its derivation is deliberate: preserve the established lower product 0.90*0.96=0.864, retain +/-4% per-leg build asymmetry, and cap the maximum product at about 1.02. The latter requires common_max <= 1.02/1.04=0.980769; 0.98 is a conservative round value.

Consequences: combined range 0.864–1.0192; mean 0.94; standard deviation 0.03170. Numerical integration of the product distribution gives 2.8437% of individual legs above nominal and 8.9062% of robots with any leg above nominal. Thus 97.1563% of individual samples and 91.0938% of four-leg models are entirely within the requested shorter direction, in the respective senses. The common scale is always below nominal. Per-leg factors remain centered around the robot's common scale; they do not need their own negative bias to achieve this combined asymmetry.

The offset being scaled is (0,-49.4,68.5) mm, norm 84.4548 mm. Proposed endpoint lengths are 72.9689 and 86.0763 mm. This is the parent-to-knee/thigh-offset proxy, not a uniform scaling of the complete leg/foot mesh. A proposed common upper bound of 1.03 combined with current per-leg upper 1.04 would actually allow 1.0712, or +7.12%. A common lower bound 0.85 would introduce the unneeded shorter product 0.816, or -18.4%. Without physical measurements there is no basis to claim that broader range is the manufacturing distribution. The recommendation is an engineering prior constrained by existing support and the stated near-maximum CAD reference; it is not an estimated tolerance fit.

If absolutely no longer-than-CAD individual legs are acceptable, use common_max=1/1.04=0.961538 (or 0.9615), retaining common_min=0.90 and per-leg 0.96–1.04. This further shifts the mean down, so it is not the same distribution as the +1.92% allowance proposal.

**6. Standing and reach checks pass the basic length endpoints, with important limits.**

Direct MJX, zero control/home targets, flat floor, four seconds settling:

| Common product on all four legs | Torso z, mm | Maximum ring penetration, mm | Foot contacts | Ring side/bottom cost |
|---|---:|---:|---:|---:|
| 0.864 | 131.007 | 4.406 | 4 | 0 / 0 |
| 0.940 | 137.409 | 4.408 | 4 | 0 / 0 |
| 1.0192 | 144.081 | 4.410 | 4 | 0 / 0 |
| diagonal [0.864,0.936,0.936,0.864] | 135.915 | 5.620 | 3 | 0 / 0 |

Thus the shorter length alone does not need more than the existing 9 mm allowance. Unequal diagonal legs can stand on three feet with the remaining foot elevated; zero penalty does not imply four-foot stance or policy robustness. Native MuJoCo endpoint checks agree qualitatively, with small solver-dependent penetration differences. They are not substituted for the MJX numbers in this table.

A 128-model native-MuJoCo stress sample using proposed lengths and all other current DR distributions gives settled penetration median 5.293 mm, p95 6.710 mm, p99 7.839 mm, max 9.045 mm after four seconds. One sample slightly exceeds the allowance, and 33/128 have only three feet contacting at fixed home control. At 0.2 s no sample exceeds 9 mm; at 2 s one reaches 9.103 mm. This demonstrates why the current 50-substep tests are not a guarantee over full combined DR. The small worst settled excess is not evidence to widen allowance globally; it corresponds to a very small quadratic penalty. These are sampled checks, not an exhaustive bound or learned-policy evaluation under the new distribution.

The existing tests cover common scales 0.94,1,1.06 (`tests/test_walk_randomize.py:49`) and a few shared tilt pairs (`:105`), with only 0.2 s settling. Their coverage is narrower than either the old combined range or the proposed one.

All present action targets stay inside soft joint limits. For example, right abduction -0.25 is still 0.02 rad above the soft lower limit -0.27; the mirrored left abduction is equally close to its soft upper boundary. DR does not directly change joint bounds. But maintaining the original torso height after shortening is geometrically infeasible at the original planted foot position: constrained IK retains approximately 11.1 mm residual at scale 0.864, and 4.9 mm at 0.94. The robot must settle lower or change posture/foot placement. This is not fixed by asserting that actuator commands remain in range.

**7. Keep ring geometry calibration; adapt the torso-height target to DR and floor normal.**

`walk_geometry.py:15–39` bakes points into each foot body's own frame. `walk_env.py:24–36,89–92` transforms those points through actual runtime body poses and retains the unchanged capsule dimensions. Multiplying an upstream `body_pos`, or modifying that body's `body_quat`, does not alter any foot-local visual mount or capsule size. This argument is sample-by-sample and does not depend on the distribution being symmetric. Keep ring_local and self.sizes as they are for the proposed change. No increase in bottom_allowance is indicated by the basic endpoints.

The asymmetric distribution does expose a reward-target issue. `walk_env.py:33,152` compares every randomized model with the same nominal height 0.142463661 m. At the shortest MJX endpoint, the height penalty is approximately 2*((0.13100654-0.14246366)/0.04)²=0.1641; around the proposed mean it is 0.0319. Nominal neutral-pose thigh shortening changes rest height by approximately 0.08429*(s-1) metres. Since E[s]=0.94, the average reference correction is about -5.057 mm. This is a real persistent incentive to overextend a predominantly shorter robot.

Recommended formula, retaining height weight -2 and normalization 0.04:

height_error = (dot(n, torso_position) - h_ref_DR) / 0.04.

For length-only DR, h_ref_DR = h_nom + 0.08429 * (mean(s_i)-1).

For combined geometry DR, precompute at reset the mean change in foot capsule-bottom projection under identical nominal root pose/home joints, and set h_ref_DR = h_nom - mean_i dot(n, p_i_DR - p_i_nom). This is a neutral-posture reference, not a guarantee all four unequal feet contact. Use actual per-environment sys fields; a value cached once from nominal `self.mj_model` would defeat the adjustment. This does not add a false joint sensor offset or require observing s_i directly.

The floor-normal part matters independently: current height uses world Z even though floor tilt is randomized. At tilt 0.04 rad, moving 1 m upslope changes floor elevation by about 40 mm and produces roughly 2 units of height penalty at otherwise constant terrain-relative body height—comparable to the entire linear tracking ceiling. The correct ring world transform does not fix that body-height objective.

One small geometry caveat: `capsule_bottom()` chooses the endpoint using world-Z axis and subtracts radius along world Z before the result is projected onto the floor normal. The exact tilted-plane capsule clearance is dot(center,n) - half_length*abs(dot(axis,n)) - radius. With radius 12 mm and floor angles up to 0.04 on both axes, same-endpoint projection error is at most about 0.0192 mm. If the capsule axis is nearly tangent to the floor, endpoint selection can also be wrong, giving a bound around 2*18 mm*0.0566 + 0.0192 mm = 2.06 mm. This does not affect the flat-floor waveform diagnosis, and ring-point projection itself is exact. A future cleanup should use the exact capsule formula and tangential velocity (I-nn^T)v for slip/rub on tilted floors.

**8. The current mount tilt already includes real lateral variation; its frame description is wrong.**

`walk_randomize.py:56–60` builds delta=Ry(ry)Rx(rx), then `:111` assigns q_new=delta*q_nominal. That is a perturbation in the PARENT body's coordinates, not the child leg body's own X/Y coordinates. This follows both the installed MJX transform code (`mujoco/mjx/_src/smooth.py:47–81`) and MuJoCo's documented parent-relative body transforms: https://mujoco.readthedocs.io/en/stable/modeling.html.

At encoder reference pose, the four parent frames are essentially identical in torso coordinates, despite the nontrivial/mirrored leg_3 quaternions:

parent X ~= (0,-1,0)
parent Y ~= (-0.841476,0,+0.540294)
parent Z ~= (-0.540294,0,-0.841476).

So rx is effectively rotation around torso -Y, generating fore/aft deflection. ry includes a rotation around torso -X of magnitude 0.8415*ry, which is precisely a lateral tilt component, plus torso yaw of magnitude 0.5403*ry. At the +/-0.07 limit this corresponds to about +/-3.37 degrees of lateral-roll component plus +/-2.17 degrees yaw, to first order. The signs are independently symmetric, so both sideways directions are represented.

Finite-difference capsule-bottom displacement sensitivity at the settled home pose:

| Foot | Lateral displacement / rx, mm/rad | Lateral displacement / ry, mm/rad |
|---|---:|---:|
| front right | -1.089 | -39.988 |
| front left | +1.116 | -40.734 |
| back right | -0.876 | -40.006 |
| back left | +0.893 | -40.738 |

Thus ry=+/-0.07 yields roughly +/-2.80–2.85 mm lateral foot displacement. rx yields roughly +/-3.36 mm fore/aft displacement. ry also introduces approximately +/-1.8 mm vertical displacement, with opposite signs on left/right. Lateral coverage is real; the core mechanism should stay.

Minimal proposed fix: retain axes, multiplication order, and +/-0.07 values; correct comments/names to “parent-frame X/Y mount perturbation,” documenting the torso-frame mapping above. No added encoder bias.

If independently defined torso-lateral and fore/aft tilt is preferred, compute B_i=R_torso_home^T R_parent_i_home once, then set delta_parent_i = B_i^T [Ry(beta_i) Rx(alpha_i)] B_i and R_body_new_i=delta_parent_i R_body_nom_i. Here alpha is torso-X roll (lateral tilt), beta torso-Y pitch (fore/aft tilt), both sampled U(-0.07,0.07). In quaternion form, conjugate the desired torso delta by the nominal parent orientation before the same pre-multiplication. Keep it constant during an episode/environment realization: recomputing B_i from the instantaneous pose would create a moving mount error rather than a manufacturing offset. This alternate axis definition changes the distribution and needs its own standing check; the measured standing stress results above apply to the current mechanism.

Do not simply reverse quaternion multiplication expecting that to produce torso X/Y tilts; that switches to child-frame axes. Also, this mechanism changes the lower-leg mounting orientation at the knee. It does not represent a translational mounting offset or a tilt of the whole upstream hip assembly.

The mount perturbation does not falsify qpos/home encoder readings. Generic training sensor_noise=0.01 does add ordinary observation noise to joint positions (`walk_env.py:57–62`); the accurate statement is “no mount-correlated encoder bias,” not literally “no observation noise.” Evaluation turns that generic noise off.

Finally, the installed Brax DomainRandomizationVmapWrapper samples its model batch once in its constructor (`brax/envs/wrappers/training.py:232`), then reuses it in reset/step. These are per-environment-slot physical parameters, not new per-episode samples. That persistence is compatible with fixed manufacturing differences, but should be described accurately when designing exposure or validation.
