> Superseded by user stride feedback: the higher-clearance candidate steps too rapidly. Restore the preferred ring6 stride while retaining clearance; see [the stride follow-up](../validation_stride_2026-09-07/RESULTS.md).

# Higher clearance for small obstacles

Hardware feedback: the visible gait still does not clear small obstacles reliably.
Obstacle height has not been specified; the initial test uses 5 and 10 mm bars.

## Baseline

Rigid-flush run `wtzmn4vi`, step 86,507,520 (best/final), was trained with an
8 mm,160 ms planned swing target. At forward .20 m/s, three seeds, friction2,
8-second MJX rollouts (first second discarded), front major-swing median peaks
are **4.8/4.6 mm**, rear **5.4/4.0 mm**. XY tracking error is .023 m/s.

Native MuJoCo obstacle stress test: three transverse bars, each 30 mm wide, at
X=.32/.57/.82 m, with a 35 mm placement offset per seed. Policy receives no terrain
input. All cases use .20 m/s, friction2 and eight seconds. Baseline finishes
**3/3** flat controls and **3/3** 5 mm courses, but only **1/3** 10 mm courses.
No falls, but frequent obstacle side strikes and ring-side overlap. Finishing
means the torso passes the final bar by 150 mm; it does not certify clean foot
clearance. Native results are compared only with the same evaluator, not treated
as bit-identical to MJX rollouts.

## New experiment

Native constrained kinematics checked full sin-squared height arcs, free horizontal
placement, all four legs, and length scales .864/1/1.0192. A 16 mm apex is feasible
with the existing (.5,.25,1.1) action bounds and at least1mm ring-side clearance.
A20mm target incurs up to2.69mm height residual at the shortest length. This is
geometry feasibility, not a dynamic or exhaustive combined-DR guarantee.

Run [mz6ipazu](https://wandb.ai/QuadMorph/pupper-leg/runs/mz6ipazu),
`output/clearance_trials/height16/walk_2026-09-07_16-20-16`, warm-starts the rigid
86.5M checkpoint and requests86.4M steps. Target **16mm over200ms**, shape weight
**-6**, touchdown clearance threshold **12mm**, shortfall weight **-.5**.
All other physical settings, command exposure, action bounds, tracking and
smoothness weights are retained. The longer duration avoids doubling required
vertical acceleration when doubling the target height. Videos remain50fps.

[Feasibility sweep](feasibility.json), [baseline foot report](baseline/report.json),
[baseline obstacle report](baseline_obstacles/report.json),
[baseline obstacle video](baseline_obstacles/obstacles_10mm.mp4),
[launch command](launch.json), [training log](train_height16.log).

## Height-only outcome and front correction

At21.6M, front forward.20 peaks rose to8.8/6.1mm, rear12.8/9.6mm;
all39 flat command/seed rollouts survived and the10mm course finished2/3.
At43.3M, fronts remained8.2/6.1mm while rears reached15.3/12.4mm. Front major
swings lasted100/104ms, versus156/152ms behind. No joint command saturated in
the checked forward.20 interval. The target alone is not producing balanced lift.

The86.5M final checkpoint finished1/3 obstacle courses and fell in the other two;
its larger training reward107.84 does not establish improved obstacle handling.
It is not selected for hardware.

A43.2M-step continuation, [5uf5o2f3](https://wandb.ai/QuadMorph/pupper-leg/runs/5uf5o2f3),
`output/clearance_trials/front3/walk_2026-09-07_16-33-12`, multiplies only the
front feet's trajectory error by3: FR/FL/BR/BL weights3/3/1/1. The per-foot bonus
is unchanged, so this does not increase the payoff for in-place stepping. All
other height16 settings are retained. It warm-starts the86.5M height-only policy.
[Launch](launch_front3.json), [log](train_front3.log).

## Final clearance-only measurements (subsequently rejected for stride style)

Front3 step43,253,760 reached13.1/13.9mm front major-swing peaks at forward.20,
with XY error.0374m/s versus.023m/s baseline. It survived all39 friction2 flat
trials and all18 additional friction1 trials. It finished all three5mm and all
three10mm courses, with no falls, but some capsule side contacts and ring-side
overlap remained. Fast forward/reverse tracking worsened (.072/.089m/s error),
and average body angular acceleration rose from34.4 to53.3rad/s².

The user's subsequent video feedback correctly identified much shorter, faster
strides:1.9Hz preferred versus3.57Hz here. Thus the clearance improvement alone
does not satisfy the final goal. The selected_params alias remains a historical
clearance artifact; selection.json marks it superseded. No robot-code update.

13 relevant gait/obstacle tests passed; default clearance config was checked
against this run's saved recipe. [Final flat report](front43/report.json),
[obstacle report](front43_obstacles/report.json),
[obstacle video](front43_obstacles/obstacles_10mm.mp4),
[clearance comparison](selected_comparison.png).
