"""Publish authorized simulation review results and an actual rollout copy."""
from pathlib import Path
import json,hashlib,shutil,subprocess
from training.wandb_logging import ExperimentLogger
root=Path('runs/roll_to_stand/review-bundle');root.mkdir(exist_ok=False);data=Path('runs/roll_to_stand/review-data');summary=json.loads((data/'summary.json').read_text());rows=json.loads((data/'case-index.json').read_text());selected=Path('motion/inverted_triangle/roll_validation/selected_pipeline.json');setup=dict(review_only=True,new_integration_steps=0,simulation_only=True,hardware_validated=False,base_commit='2f4c5c3fbe50a65cb1b6ad12ff6a748e4c584f59',frozen_candidate_commit='c1d8342',selected_pipeline=json.loads(selected.read_text()),selected_pipeline_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),summary=summary)
(root/'config.json').write_text(json.dumps(setup,indent=2));logger=ExperimentLogger(root,setup,name='roll-to-stand-corrections-review')
try:
 links=dict(review_url=logger.state['url'],review_id=logger.state['id'],nominal_video_run='https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/4f5d491a77534519',front_rear_video_run='https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/686ce5bef5894c4a');(data/'review-links.json').write_text(json.dumps(links,indent=2))
 video=Path('runs/roll_to_stand/selected-development-handoff/nominal_regression/handoff/continuous_rollout.mp4');shutil.copyfile(video,root/'nominal-continuous.mp4');logger.metrics(0,dict(review_conditions=len(rows),roll_robustness_passed=36,formation_development_passed=5,fresh_continuous_passed=5,full_acceptance=False));logger.video(root/'nominal-continuous.mp4',0,key='trajectory/selected_continuous_roll_stand_walk',caption='SIMULATION actual 66-second roll + standing + corrected pinned walking export; byte-identical review copy of run 4f5d491a77534519, no new integration. Nominal gait checks pass; formation and sensor/dynamics robustness NOT accepted. No hardware.')
 columns=['family','case','role','pass','failed_gates','dense_gap_mm','tilt_deg','torque_Nm','forward_vx_m_s','W&B_run']
 values=[[r['family'],r['case'],r.get('role','development_or_diagnostic'),r['passed'],', '.join(r['failed_gates']),None if r['dense_cad_gap_m'] is None else 1000*r['dense_cad_gap_m'],r['tilt_deg'],r['peak_requested_torque_Nm'],r.get('forward_velocity_m_s',[None])[0],r['url']] for r in rows]
 logger.run.log({'results/case_table':logger.sdk.Table(columns=columns,data=values),'train/env_steps':0})
 logger.run.summary.update(dict(simulation_audit_status='PARTIAL_SUCCESS_NOT_FULL_ACCEPTANCE',roll_robustness='36/36',formation_roll='5/7',development_handoff='3/7',fresh_handoff='5/25',fresh_shapes_sensors_dynamics_handoff='0/12',hardware_validated=False,source_frozen_before_held_out=True))
 artifact=logger.sdk.Artifact('roll-corrections-review-'+logger.state['id'],type='simulation-review',metadata=dict(full_acceptance=False,source_commit=setup['source_commit']))
 for f in data.iterdir():
  if f.is_file() and f.suffix in ['.json','.zip']:artifact.add_file(str(f),name=f.name)
 logger.run.log_artifact(artifact)
finally:logger.finish()
print(links)
