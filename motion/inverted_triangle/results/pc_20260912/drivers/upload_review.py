import hashlib,json,shutil
from pathlib import Path
from training.wandb_logging import ExperimentLogger
root=Path('runs/inverted_triangle');bundle=root/'review-bundle';summary=json.loads((bundle/'summary.json').read_text())
run_dir=root/'review-upload';run_dir.mkdir(exist_ok=False)
logger=ExperimentLogger(run_dir,dict(task='PC continuous inverted-triangle simulation review',**summary),name='inverted-triangle-PC-review-20260912')
summary['review_wandb_url']=logger.state['url'];(bundle/'summary.json').write_text(json.dumps(summary,indent=2))
for filename in ['source_manifest.json','requirements.lock.txt']:
 shutil.copy2(Path('motion/inverted_triangle')/filename,bundle/filename)
shutil.copy2(root/'landing-duration-scout.json',bundle/'diagnostics/landing-duration-scout.json')
failures=[]
for result in json.loads((bundle/'validation_results.json').read_text()):
 if result['status']=='PASS_CONTINUOUS_FOUR_FLIPS':continue
 for path in sorted((root/'final-validation-v5'/result['name']).glob('*/audit.json')):
  a=json.loads(path.read_text())
  if a['status']!='FAILED':continue
  first=a['first_sampled_gate_violation'];derived=False
  if first is None and not a['gates']['gentle_landing']:
   trace_path=path.parent/'trace.json';trace=json.loads(trace_path.read_text());previous=None;leg=['front_r','front_l','back_r','back_l'].index(a['leg'])
   for row in trace:
    if previous and row['phase']=='land' and row['bottom_m'][leg]<.008:
     speed=(previous['bottom_m'][leg]-row['bottom_m'][leg])/(row['time']-previous['time'])
     if speed>=.025:
      first=dict(time_s=row['time'],phase='land',gate='gentle_landing',value=speed,threshold=.025,trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest());derived=True;break
    previous=row
  failures.append(dict(condition=result['name'],leg=a['leg'],first=first,first_derived_from_saved_trace=derived,
      peak_landing_descent_m_s=a['peak_landing_descent_m_s'],minimum_rotation_floor_m=a['minimum_rotation_floor_m'],maximum_unintended_floor_force_N=a['maximum_unintended_floor_force_N'],url=result['url']))
(bundle/'stress_failure_details.json').write_text(json.dumps(failures,indent=2))
audits=[json.loads(p.read_text()) for p in sorted((bundle/'evidence/final-nominal-v5').glob('*/audit.json'))]
stages=['01-back_r','02-back_l','03-front_r','04-front_l'];cad=[json.loads((bundle/'cad'/stage/'cad_review.json').read_text()) for stage in stages]
lines=['# PC simulation result — September 12, 2026','',
 '**The continuous four-flip plan passes nominally and in all 20 required friction/seed replays. Broader stress testing passes 7/16 cases; full uncertainty robustness is not achieved.**','',
 'The robot starts on four rigid point-up triangles and finishes in the audited four-tip stance. The original shin tip and additional 9 mm axial gap are unchanged. This was simulation only; no robot connection, deployment, calibration change or policy activation occurred.','',
 f'[W&B review and videos]({logger.state["url"]}) · [Nominal continuous rollout]({summary["nominal_wandb_url"]}) · [All 114 verified experiment runs](wandb_runs.json)','',
 '## Accepted immutable plan','',
 'Order and hub directions: **back-right −π → back-left +π → front-right −π → front-left −π**. The actual integration state, velocities, actuator state and commanded references carry between stages; the saved state hashes match exactly at every boundary. See [continuation_chain.json](continuation_chain.json).','',
 f'- Plan: [plan/plan.json](plan/plan.json), SHA-256 `{summary["plan_sha256"]}`.',
 f'- Tested source commit: `{summary["source_commit"]}`; 14 model/continuation tests passed.',
 f'- Model SHA-256: `{summary["model_sha256"]}`. CAD/XML hashes are in [source_manifest.json](source_manifest.json).',
 f'- Nominal rollout: **{summary["duration_s"]:.3f} simulated seconds / {summary["environment_steps"]:,} physics steps** at 520 Hz; [complete actual video](continuous_rollout.mp4).',
 '- Dependencies: [requirements.lock.txt](requirements.lock.txt); Python 3.12.3, MuJoCo 3.3.7, NumPy 2.2.6, SciPy 1.16.2. Full environment and source hashes are in [summary.json](summary.json).','',
 '| Flip | Rotation floor gap (mm) | Refined CAD gap (mm) | Minimum support (N) | Peak tilt (deg) | Peak requested torque (Nm) |',
 '| --- | ---: | ---: | ---: | ---: | ---: |']
for a,c in zip(audits,cad):lines.append(f'| {a["leg"]} | {a["minimum_rotation_floor_m"]*1000:.3f} | {c["minimum"]["minimum_m"]*1000:.3f} | {a["minimum_support_force_N"]:.3f} | {a["max_tilt_deg"]:.3f} | {a["peak_requested_torque_Nm"]:.3f} |')
lines += ['',f'Maximum nominal landing descent is {summary["peak_landing_descent_m_s"]*1000:.3f} mm/s against the 25 mm/s gate. Motor/body floor support is zero. All original torque, speed, tilt, support, tracking, landing and CAD gates remain enforced.',
 '', 'Final loads in front-right, front-left, rear-right, rear-left order are '+', '.join(f'{v:.3f} N' for v in summary['final_normal_force_N'])+'. Final long-tip heights are '+', '.join(f'{v*1000:.3f} mm' for v in summary['final_tip_bottom_m'])+' (the screening tolerance is ±3 mm).','',
 '## CAD and visual review','',
 'The limiting detailed pair is the front-right shin against the rear-right motor assembly, with **2.491 mm** sampled clearance. Each phase minimum and both same-side wheel minima were refined at the 1/520-second physics interval within ±0.125-second windows. Reintegrated states matched the saved audit exactly. The first three stages are unchanged from the previously reviewed prefix; [prefix equivalence](cad/prefix_equivalence.json) records exact equality.','',
 '| Flip | Right front/rear wheel minimum (mm) | Left front/rear wheel minimum (mm) | Three-view key frames |',
 '| --- | ---: | ---: | --- |']
for stage,c in zip(stages,cad):lines.append(f'| {stage[3:]} | {c["same_side_front_rear"]["r"]["minimum_m"]*1000:.3f} | {c["same_side_front_rear"]["l"]["minimum_m"]*1000:.3f} | [Initial, lift, half-turn, final](cad/{stage}/views.png) |')
lines += ['', 'Finite sampling is not continuous collision proof. Floor-only convex hulls and independent nonconvex CAD checks remain separate. Polymer compliance, spacer mass, measured friction and calibration uncertainty are not certified by these rigid-model results.','',
 '## Fixed-plan robustness','',
 '| Friction | Perturbation seeds | Passing continuous sequences |','| --- | --- | ---: |',
 '| 0.50 | 1–5 | 5/5 |','| 0.65 | 1–5 | 5/5 |','| 0.80 | 1–5 | 5/5 |','| 1.00 | 1–5 | 5/5 |','',
 'Each initial seed perturbs all eight proximal coordinates by uniform ±0.01 rad. Every replay uses the same plan and propagates the actual preceding end state. No nominal checkpoint is restored between legs.','',
 'Additional sensitivity cases use mass/inertia ±10%, trunk COM offsets ±3 mm horizontally/±2 mm vertically, 5/10-step command delays (9.62/19.23 ms), 1.5 Nm available torque, PD gains ×0.8/×1.2, and initial height ±2 mm with roll/pitch ±1°. Five combined seeds use heavier mass, shifted COM, delay, lower gains/torque and a low tilted start. These are explicit sensitivity assumptions, not measured hardware distributions. The rationale and exact configuration are in [the grid configuration](batches/final-validation-v5/config.json).','',
 '**7/16 additional cases passed.** Both COM cases, both delay cases, half torque, and both start-offset cases passed. Remaining failures are:','',
 '| Condition | First failing leg/phase | Measured gate failure |','| --- | --- | --- |']
for f in failures:
 first=f['first'];phase=first['phase'] if first else 'see audit'
 if f['condition']=='mass-low' or f['condition']=='gains-high':detail=f'Peak near-ground descent {f["peak_landing_descent_m_s"]*1000:.3f} mm/s; limit <25'
 elif f['condition'].startswith('combined'):detail=f'Minimum rotation clearance {f["minimum_rotation_floor_m"]*1000:.3f} mm; minimum 5'
 else:detail=f'Peak motor/body floor force {f["maximum_unintended_floor_force_N"]:.3f} N; limit <0.2'
 lines.append(f'| [{f["condition"]}]({f["url"]}) | {f["leg"]} / {phase} | {detail} |')
lines += ['', 'Exact first sampled failure times and values are in [stress_failure_details.json](stress_failure_details.json). Landing failure times were derived from the retained actual traces where the original audit did not record that timestamp. A separate, unaccepted duration scout found that slowing the final landing to 12 seconds still exceeded the descent gate; that scout has no CAD/video acceptance claim.','',
 '[All 36 results with run links](validation_results.csv) · [Failed lower-gain continuous replay](failed_gains_rollout.mp4)','',
 '## Search history and preserved failures','',
 'The saved rear-right flip was reproduced on clean source commit `5fb4567`. The prescribed nine-parameter Powell sequence failed at the second front-left flip on support and CAD gates. A rear-first alternate order passed two connected flips. Body-shift and independent landing targets enabled the third and fourth flips.','',
 'A lower-cost lift revision then intersected the rear-right motor assembly; its independent audit rejected it. The final optimizer includes those motor/body pairs in its cost, and a saved regression check verifies the previously missed collision is now detected. The accepted plan retains the clear lift path and uses two focused landing searches across friction conditions.','',
 'Earlier plans failed high-friction tracking, including a final-leg error of 0.0350187 rad against the strict 0.035 gate. That failed audit and [its actual video](earlier_tracking_failure.mp4) are preserved. The corrected plan passed the same seed with 0.00950 rad final-leg error. No threshold was relaxed.','',
 'Every experiment audit, candidate, configuration and continuation state is retained under [evidence/](evidence/); earlier plans and exact local drivers are also included. Full traces, evaluation streams and all actual videos remain in the local isolated checkout and the verified W&B artifacts/Media indexed by [wandb_runs.json](wandb_runs.json). Video MD5 values were matched to local bytes and every indexed artifact was COMMITTED in the cloud. [evidence_sha256.json](evidence_sha256.json) hashes this review bundle.','',
 '## Leg-policy compatibility and remaining work','',
 '**Direct leg-policy entry is not validated.** The final proximal posture differs from `policy_walk_v2.json`, and the rear-left commanded hub coordinate is 7.283185 rad versus the policy reference of 1 rad. The physical hub orientation agrees modulo 2π, but silently resetting or clamping that reference would not be an acceptable handoff. See [policy_compatibility.json](policy_compatibility.json).','',
 'A reference-preserving stance transition and measured-state executor remain necessary before a policy/hardware handoff: encoder/IMU tracking, low-speed progression, bounded landing, timeout/abort behavior and valid calibration. Simulation contacts are audit signals, not physical foot-load sensors. No hardware-ready claim is made; the failed uncertainty cases remain unresolved.','',
 '## Replay','',
 'From the repository root in the fresh environment described in [VALIDATION.md](../../VALIDATION.md):','',
 '```sh',
 'python -m motion.inverted_triangle.sequence --plan motion/inverted_triangle/results/pc_20260912/plan/plan.json --output runs/inverted_triangle/review-replay',
 'python -m motion.inverted_triangle.robustness --plan motion/inverted_triangle/results/pc_20260912/plan/plan.json --output runs/inverted_triangle/review-grid --workers 12',
 '```','',
 'Use a new output directory each time. On headless Linux prefix the command with `MUJOCO_GL=egl`. These commands log online to the documented QuadMorph W&B project.']
(bundle/'README.md').write_text('\n'.join(lines)+'\n')
manifest={p.relative_to(bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*') if p.is_file() and p.name!='evidence_sha256.json'}
(bundle/'evidence_sha256.json').write_text(json.dumps(manifest,indent=2))
logger.metrics(summary['environment_steps'],{k:v for k,v in summary.items() if isinstance(v,(int,float,bool,str))})
for name,key,caption in [
 ('continuous_rollout.mp4','trajectory/continuous_nominal',f'SIMULATION fixed plan {summary["plan_sha256"]} | seed 0, friction 0.8 | 4/4 continuous flips accepted | {summary["environment_steps"]} physics steps | early termination none; no trained checkpoint'),
 ('failed_gains_rollout.mp4','stress/lower_gains_failure','SIMULATION | seed 18, friction 0.8, PD gains x0.8 | failed second flip: motor/body floor support | sequence stops after failed audit; no trained checkpoint'),
 ('earlier_tracking_failure.mp4','history/preserved_tracking_failure','SIMULATION earlier fixed plan | seed 3, friction 1.0 | fourth flip rejected: hub error 0.0350187 rad exceeds 0.035 | no simulator early termination; no trained checkpoint')]:
 shutil.copy2(bundle/name,run_dir/name);logger.video(run_dir/name,summary['environment_steps'],key=key,caption=caption)
for stage in stages:logger.run.log({'cad/'+stage:logger.sdk.Image(str(bundle/'cad'/stage/'views.png'),caption='Recorded integrated simulation: initial, maximum lift, half-turn, final; three views. Not hardware validation.')})
rows=json.loads((bundle/'validation_results.json').read_text())
logger.run.log({'validation/cases':logger.sdk.Table(columns=['condition','friction','seed','status','accepted_flips','run_url'],data=[[r[k] for k in ['name','friction','seed','status','completed_flips','url']] for r in rows])})
logger.run.summary.update(dict(simulation_audit_status='PASS continuous nominal and 20/20 friction; 7/16 extended stress',hardware_validated=False,direct_policy_handoff_validated=False,extended_robustness_all_pass=False))
artifact=logger.sdk.Artifact('inverted-triangle-pc-review-'+logger.state['id'],type='simulation-review',metadata=dict(plan_sha256=summary['plan_sha256'],source_commit=summary['source_commit'],hardware_validated=False))
artifact.add_dir(str(bundle));logger.run.log_artifact(artifact);logger.finish()
(run_dir/'logging_status.json').write_text(json.dumps(dict(url=logger.state['url'],status='SDK online finish completed; cloud verification pending'),indent=2));print(logger.state['url'],flush=True)
