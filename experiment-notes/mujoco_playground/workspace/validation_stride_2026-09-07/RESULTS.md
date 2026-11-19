# Restore long strides while retaining clearance

User prefers the stride in `walk_2026-09-07_15-29-41` (ring6), and finds the
new higher-clearance gait too rapid and tap-like. This preference supersedes
promotion of clearance run front3. The height goal and rigid flush capsules remain.

## Measured difference at forward .20 m/s

| Policy / physics | Major stride cadence | Median major air time | Actual foot stride distance |
|---|---:|---:|---:|
| Preferred ring6 / original soft |1.91Hz|224ms|not recomputed here|
| Preferred ring6 / new rigid, friction2 |1.94Hz|224ms|105.1mm|
| High-clearance front3 final / rigid, friction2 |3.57Hz|144ms|48.2mm|

Stride distance is the world-XY distance between consecutive qualifying liftoffs
of the same foot, pooled over four feet and three reset seeds; qualifying swings
last>40ms and peak>2mm. These existing diagnostic suites discard the first second.
The preferred rigid transfer used6seconds, front3 used8seconds, so this is a clear
style diagnostic rather than a perfectly matched-duration statistical estimate.

## New run

Warm-start **directly from preferred ring6 step43,253,760**, rather than the
new fast-gait lineage. Rigid-flush physics, friction DR.8--2.5, showcase friction2.
Keep16mm height and12mm minimum-peak goal, front error weights3/3/1/1, shape-6,
shortfall-.5. Extend planned swing from200 to**320ms**, double action acceleration
penalty from-.8 to**-1.6**, and use learning rate**5e-5**. This is an experiment
to preserve the preferred longer rhythm while learning clearance, not a guarantee
that a longer reference alone fixes cadence.

[Run zw331stu](https://wandb.ai/QuadMorph/pupper-leg/runs/zw331stu),
`output/stride_trials/preferred_long/walk_2026-09-07_16-48-35`, requests86.4M
steps with5evaluations and50fps videos. [Launch](launch.json), [log](train.log).

Selection checks: actual cadence and stride distance, per-foot clearance,
forward/reverse/turn tracking, survival and the same native10mm obstacle course.

## Selection

Selected **21,626,880** steps, SHA256
`f9b30170c1d927c039204a92b4015fff59ebbdc9ede2137aca4878812290ea5e`.
The run's `selected_params` and `rollout_selected.mp4` point to this checkpoint.
`best_params` remains the higher-reward86.5M checkpoint, which is not preferred
because its stride shortened again.

Matched friction2,13commands ×3seeds ×8seconds, discard1second:

| Forward .20 metric | Preferred ring6, rigid | Selected21.6M | Later43.3M | Final86.5M |
|---|---:|---:|---:|---:|
| Cadence, Hz |1.94|2.17|2.50|2.60|
| Front strides, mm |105.5/107.9|98.2/98.2|87.2/88.7|80/76|
| Median major air time, ms |228|224|196|192|
| Front peaks, mm |~6.9/5.5|9.2/11.3|13.1/13.5|15.1/13.9|
| Action second-difference RMS |.152|.162|.198|.210|

All39 selected-policy flat trials survived, with zero ring-side overlap in the
friction2 suite. Average XY error across all13 commands was.0412m/s versus.0406
for preferred; average yaw error was.0771 versus.0936rad/s. The selected policy
finishes all3 ten-mm obstacle courses, compared with1/3 preferred. Some capsule
side contacts and ring-side overlap remain on the bars. The early checkpoint
is selected to honor the user's long-stride preference rather than maximize
clearance or training reward alone.

[Matched preferred report](preferred/report.json), [selected report](long21/report.json),
[selected10mm report](long21_obstacles/report.json),
[43M report](long43/report.json), [86M report](long86/report.json).

The current defaults match the selected swing settings and action-acceleration
weight; warm-start learning rate is now5e-5. No model geometry, joint bounds,
policy interface or robot-code deployment was changed in this stride follow-up.

## Final verification and media

The selected policy also survived all18 lower-friction (mu1) command/seed
checks, for57/57 flat rollouts across the two frictions. At mu1 forward.20,
cadence2.21Hz and front peaks9.27/10.99mm retain the selected gait.
Independent native checks complete3/3 flat,3/3 five-mm and3/3 ten-mm courses
with no falls. Some obstacle contacts remain.

The W&B run contains **eval/video_selected** for the21.6M checkpoint and
**eval/video_obstacles_10mm** for its obstacle clip.
[Selected walking video](../output/stride_trials/preferred_long/walk_2026-09-07_16-48-35/rollout_selected.mp4),
[obstacle video](selected_obstacles/obstacles_10mm.mp4),
[measured stride comparison](stride_comparison.png),
[friction1 report](selected_mu1/report.json),
[full obstacle report](selected_obstacles/report.json).

13 relevant gait/obstacle tests passed after the front-weight code change.
The final default swing/reward values were asserted equal to the selected run's
saved settings; training and rollout evaluations exercised the320ms reference
and-1.6 action-acceleration weight. Python compilation and git diff checks pass.
