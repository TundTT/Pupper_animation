# Reverse walking: measured result

**Experimental candidate, not a complete reverse-gait or ring-protection fix.** Training completed 86,507,520 additional environment steps in the final run. The selected checkpoint is step 57,671,680 (also the best training evaluation); the final checkpoint had higher ring costs and slightly worse aggregate tracking. Both are retained.

[Selected parameters](/home/theerawit/Pupper_animation/mujoco_playground/workspace/output/reverse_trials/tip_clearance/walk_2026-09-07_02-04-04/params_000057671680) · [50 fps review video](selected_review.mp4) · [Ring explanation](RING_AND_TRAINING.md) · [Selection metadata](selection.json)

## Matched evaluation

13 commands × three reset seeds, six seconds per rollout, first second discarded; nominal geometry, no pushes, observation noise, action latency or joint reset noise. Small XY/yaw reset variation remains. Every rollout survived for both policies (39/39). This is a short nominal-model comparison, not a combined-DR robustness guarantee. All training arms used geometry DR. Swing statistics include every complete airborne interval, including tiny contact flickers; they are medians rather than maximum lift.

| Aggregate across 13 commands | Previous | Candidate |
|---|---:|---:|
| XY velocity error | 0.0309848 m/s | 0.0206556 m/s |
| Yaw-rate error | 0.0667877 rad/s | 0.0578993 rad/s |
| Action delta RMS | 0.18853 | 0.186702 |
| Foot-slip cost | 0.0113074 | 0.00850288 |
| Ring-side overlap fraction | 1.00462% | 1.95128% |
| Ring-side cost | 0.000466459 | 0.000727362 |
| Ring-bottom cost | 4.80839e-09 | 7.20721e-06 |
| Ring-rub cost | 0.000724951 | 0.000366713 |

Costs are reported before outer dt, as exposed by the environment metrics. Ring-side fraction means the fraction of feet with any modeled side probe below the floor, averaged over time and commands; it is not measured TPU load or a fraction of whole rollouts. Side clearance cost also starts above the floor at +1 mm. Baseline yaw error was recovered per sample from its old yaw reward (variance .25); candidate yaw error is measured directly.

| Reverse command | XY error, previous → candidate (m/s) | Median swing peak (mm) | Median airborne time (ms) |
|---|---:|---:|---:|
| −15 cm/s | 0.0406 → 0.0192 | 1.38 → 2.30 | 40 → 48 |
| −20 cm/s | 0.0349 → 0.0226 | 1.99 → 3.17 | 48 → 58 |
| −35 cm/s | 0.0372 → 0.0414 | 4.14 → 5.23 | 66 → 72 |

## Remaining regressions and limitations

- Slow reverse lifts higher and tracks better, but still has short swings. At −0.15 m/s action delta RMS increases .217 → .281 and median touchdown approach speed .120 → .150 m/s.
- Fast reverse tracking error rises .0372 → .0414 m/s, while lift increases and touchdown approach speed falls .300 → .257 m/s.
- Pure-right-turn yaw error rises .113 → .150 rad/s even though unwanted translation decreases. Rightward walking still contains many tiny, near-ground contact flickers.
- Mean ring-side overlap rises 1.00% → 1.95%; side cost rises .000466 → .000727. Ring rubbing falls .000725 → .000367. These do not demonstrate an improvement in avoiding all ring contact.
- A capsule-orientation penalty expresses tip-down preference, but the simulator has no deformable TPU collision/load-sharing model. It cannot certify that the real ring carries no load.

## Changes used by the candidate

Balanced command practice (20% stand, 20% forward, 20% reverse, 10% lateral, 10% yaw, 20% mixed), stronger signed swing-duration incentive, a soft 4 mm per-swing minimum-peak goal, touchdown approach-speed penalty, contact-only tip-orientation penalty, and tracking tolerances that make slow movement worthwhile while allowing translation during turning. Action-rate weight remains −0.30; existing ring weights, 9 mm bottom allowance, motor limits and physical geometry are retained. Warm starts now default to learning rate 1e-4. The earlier shorter-leg DR and geometry/floor-relative height correction remain active.

These changes were trained in stages; this final comparison measures their combined effect, not the causal effect of each coefficient. Earlier trials and final results are saved under this validation directory and `workspace/output/reverse_trials/`.

## Review video timeline

0–2 s stand; 2–4 forward .20; 4–6 forward .35; 6–8 forward/left turn; 8–10 forward/right turn; 10–12 pure left turn; 12–14 reverse .15; 14–16 stand; 16–18 reverse .20; 18–20 reverse .35; 20–22 reverse/left turn; 22–24 left; 24–26 right; 26–28 stand. Commands use m/s and yaw rad/s.

## Validation

Eight command, contact, tracking and diagnostic-step tests passed (`final_gait_tests.log`). Six existing environment checks passed earlier in this change. The suite captures foot states at 250 Hz; video is 50 fps. See [baseline report](baseline_suite/report.json), [selected report](tip_57m_suite/report.json), and [final-checkpoint report](tip_86m_suite/report.json).
