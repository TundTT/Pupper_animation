# Continuous inverted-triangle robustness follow-up

**Nominal continuous sequence: PASS. Original friction grid: 20/20. Original stress cases: 16/16. Former held-out regression cases: 12/12. Fresh held-out cases: 12/12.**

[W&B review, videos and artifact](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/f715e149f36d49a4) · [Complete nominal rollout](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/29dace359e5a45b4) · [All final case results](validation_results.csv)

This derivative starts at `1139981`. The accepted parent plan and all its failed audits remain unchanged in `../pc_20260912/`. The parent nominal replay reproduced all four complete end-state hashes exactly; see [parent_reproduction.json](parent_reproduction.json). Original tip geometry, the 9 mm spacer gap, model assets, actuator limits and all acceptance gates are unchanged. No hardware or walking-policy activation occurred.

Final plan SHA-256: `204c0a54172874749b6702e2802872b265489bffc18ce9b20bb08b2bde41e369`. Tested source: `fc348333dd70b3cceeaaaf55f30395578f9264a1`. Model SHA-256: `768cbeb0bfbaab0c898e6078d0b778ee2968718d98bdb2857e13538daeb607a8`. The later evidence commit changes no tested Python/model files. The fresh pinned Python 3.12 environment passed all 16 tests.

The final continuous rollout uses 67,278 integrated 520 Hz steps (129.381 s), with the order rear-right −π, rear-left +π, front-right −π, front-left −π. Actual integration state, velocities and delayed-command history propagate at every boundary. [continuation_chain.json](continuation_chain.json) records exact boundary hashes.

The rear-left support pose now keeps the front-left lower motor housing clear under high mass and low gains. Rear-right lift/support targets recover the combined-case rotation margin. A small third-stage lift adjustment improves the front-right shin/rear-right motor gap. The final leg lands with the three support targets held steady; it then keeps its own two proximal targets fixed while the support targets change by less than 0.047 rad. This removes the large post-touchdown active-leg sweep that caused unloading and rebound.

| Original failing case | Measurement | Parent | Final |
| --- | --- | ---: | ---: |
| mass-low | Landing descent (mm/s) | 114.082 | 13.610 |
| gains-high | Landing descent (mm/s) | 102.734 | 13.304 |
| mass-high | Motor/body floor force (N) | 1.566 | 0.000 |
| gains-low | Motor/body floor force (N) | 4.947 | 0.000 |
| combined-seed-22 | Rotation floor gap (mm) | 4.909 | 9.498 |

The parent final landing unloaded the rear-right support for about 1.2 seconds before touchdown in the low-mass and high-gain cases. Recorded tracking/load diagnostics are in [diagnostics/](diagnostics/). Slower landing alone had already failed in the parent work. A first waypoint revision passed all 36 original cases, but passed only 11/12 held-out cases: a mixed low-mass/high-stiffness/delay case reached its waypoint about 1 mm above the floor, then rebounded during the large final transfer. Its peak descent was 133.238 mm/s. [That failed audit and actual W&B video remain available](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/82c4c53f9031458e).

Lowering the waypoint alone also failed in local diagnostic sweeps. Holding the active pose after touchdown and optimizing a small support adjustment resolved the known failure. The original held-out cases are now reported as regression cases because case 08 informed diagnosis. Twelve fresh cases, predeclared with definition seed 271828 before final selection and excluded from tuning, provide the final held-out check. The original 11/12 result is preserved, not relabeled as an independent 12/12 result.

| Flip | Nominal rotation gap (mm) | Refined CAD gap (mm) | Right front/rear wheel gap (mm) | Left front/rear wheel gap (mm) |
| --- | ---: | ---: | ---: | ---: |
| back_r | 12.468 | 3.867 | 22.873 | 50.165 |
| back_l | 14.065 | 2.805 | 79.242 | 65.467 |
| front_r | 7.899 | 2.722 | 19.479 | 101.549 |
| front_l | 10.474 | 3.838 | 40.899 | 20.960 |

The limiting nominal front-right shin/rear-right motor gap improves from 2.491257 to 2.722194 mm (about 9.3%). Detailed CAD is sampled at 520 Hz within ±0.125-second windows around phase minima and same-side wheel minima. Reintegrated review states match saved states exactly. [Three-view key frames](cad/) cover initial stance, maximum lift, half-turn and final stance. The worst original stress-case third flip also receives [a dense review](cad-worst-stress/cad_review.json). Finite sampling is not a continuous-time collision proof.

The original 36-case grid retains friction 0.5/0.65/0.8/1.0 with seeds 1–5, mass/inertia ±10%, COM offsets, 5/10-step delays, 1.5 Nm available torque, gains ×0.8/×1.2, start-height/tilt errors and five combined seeds. Fresh held-out cases mix independent mass, COM, kp/kd, torque, delay, friction and start offsets within the documented ranges. These are sensitivity assumptions, not measured hardware distributions. Exact scenarios and results are in [batches/](batches/) and [fresh-held-out-manifest.json](fresh-held-out-manifest.json).

Across the final 60 replays, worst sampled values are: rotation_floor_m=0.00582705, cad_gap_m=0.00243521, support_N=1.46967, floor_N=0, landing_descent_m_s=0.0210199, tilt_deg=6.98006, requested_torque_Nm=1.11799, hub_error_rad=0.0180125.

All 121 experiment runs have cloud-verified actual video bytes and COMMITTED artifacts before review upload; [wandb_runs.json](wandb_runs.json) indexes them. Full traces and search evaluation streams remain in the isolated checkout and W&B artifacts. This bundle retains every audit, configuration, candidate, state and search summary, including failed and interrupted attempts. Four searches were intentionally stopped after selecting an immutable candidate; their completed-record step totals are explicitly lower bounds because discarded in-flight work was not measured. The final focused finish search completed its configured budget.

Local preload/direct-finish sweeps were explicitly unlogged selection diagnostics; their full results and context states are included in this review artifact. They are not acceptance audits. Search prefixes may include failed prior trajectories for diagnosis; only independent continuous replays establish acceptance. [The parent failure history and videos](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/79b7226026be45d4) remain unchanged.

**Direct walking-policy entry remains unvalidated.** The final posture differs from the walking default, and the rear-left hub retains one extra unwrapped revolution. [policy_compatibility.json](policy_compatibility.json) records this. Encoder/IMU progression, bounded tracking/abort behavior and a reference-preserving entry transition would require separate work. No hardware or policy was activated.

Replay from the repository root using a new output directory and the pinned environment:

```sh
MUJOCO_GL=egl python -m motion.inverted_triangle.sequence --plan motion/inverted_triangle/results/robustness_20260912/plan/plan.json --output runs/inverted_triangle/review-nominal
MUJOCO_GL=egl python -m motion.inverted_triangle.robustness --plan motion/inverted_triangle/results/robustness_20260912/plan/plan.json --output runs/inverted_triangle/review-grid --workers 12
PYTHONPATH=. MUJOCO_GL=egl python motion/inverted_triangle/results/robustness_20260912/drivers/extra_batch.py motion/inverted_triangle/results/robustness_20260912/plan/plan.json motion/inverted_triangle/results/robustness_20260912/fresh-held-out-manifest.json runs/inverted_triangle/review-fresh-held-out
```

These commands log actual rollouts online to the documented QuadMorph project.
