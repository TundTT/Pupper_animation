import json
from pathlib import Path
from training.wandb_logging import ExperimentLogger
root=Path('/tmp/handoff-evidence-20260913');index=json.loads((root/'case-index.json').read_text());summary=json.loads((root/'summary.json').read_text());folder=root/'review'
logger=ExperimentLogger(folder,dict(summary=summary,candidate=json.loads((root/'frozen-candidate.json').read_text()),source_commits=['4eb8260790b42829c960a9c47032b9c6b3b88c07','076f46b9c9b3cbbf0e8757ded444e56166b12f05','986d1cb'],simulation_only=True,hardware_validated=False,optimizer='13 retained CPU dynamic grid trials; no neural checkpoint',alignment_source_available=False),name='PC handoff review: entry fixed, cold geometry infeasible')
logger.metrics(summary['environment_steps'],{'handoff/accepted':False,'development/accepted':summary['groups']['development']['passed'],'fresh_conditional/accepted':summary['groups']['fresh-conditional']['passed'],'parent_regression/passed':summary['groups']['parent-regression']['passed'],'experiments':summary['experiments']})
columns=['case','scope','passed','steps','floor_peak_N','failed_gates','run_url'];data=[]
for row in index:data.append([row['case'],row['kind'],row['engineering_pass'],row['steps'],row.get('floor_peak_N'),','.join(k for k,v in row.get('gates',{}).items() if v is not True),row.get('wandb_url')])
logger.run.log({'evidence/acceptance_matrix':logger.sdk.Table(columns=columns,data=data)})
for label,path,caption in [
 ('selected_continuous','development/diagonal_10mm_spread/rollout.mp4','Actual saved alignment endpoint; 10 mm diagonal shape; entry timeout corrected, motor-floor and four-tip support FAIL; continuous simulation after explicit cold boundary.'),
 ('saved_entry_failure','reproduce-diagonal/rollout.mp4','Unchanged saved diagonal timeout reproduced: 6239 steps; FAIL.'),
 ('saved_floor_failure','reproduce-nominal/rollout.mp4','Unchanged actual-endpoint nominal floor failure reproduced: 25548 steps, 68.105525 N; FAIL.'),
 ('published_alignment_failure','alignment-published-0/alignment.mp4','Distinct published deterministic alignment source dde1f96: actual failed upstream replay, completion mask zero; rejected endpoint.')]:
 logger.video(root/path,summary['environment_steps'],key='evidence/'+label,caption=caption)
artifact=logger.sdk.Artifact('pc-handoff-resolution-evidence-'+logger.state['id'],type='motion-audit')
for name in ['case-index.json','summary.json','cloud-verification.json','all-audits-configs-and-provenance.zip','geometry-review.json','contact-convergence-summary.json','alignment-source-availability.json','boundary-parity.json','sensor-baseline-parity.json','frozen-walking-policy-record.json','verification-tests.json','handoff-environment.txt','alignment-environment.txt']:
 artifact.add_file(str(root/name))
logger.run.log_artifact(artifact);logger.run.summary.update({'full_acceptance':False,'entry_timeout_corrected':True,'cold_nominal_infeasible':True,'walking_activated':False,'robot_accessed':False,'missing_alignment_commit':'a7e3bb1e67f73b4fb14ee6658d66d0263fd2bfd8'})
logger.run.finish();(root/'review-links.json').write_text(json.dumps(dict(review_url=logger.state['url'],review_id=logger.state['id'],selected_video_run='https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/6919a38d0564467b'),indent=2));print(logger.state['url'],flush=True)
