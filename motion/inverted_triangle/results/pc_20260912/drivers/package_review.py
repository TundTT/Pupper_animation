"""Collect immutable review evidence; bulky full traces remain in W&B artifacts."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
root=Path('runs/inverted_triangle');out=root/'review-bundle';out.mkdir(exist_ok=False)
shutil.copytree(root/'fixed-plan-v5',out/'plan')
records=json.loads((root/'cloud-verification.json').read_text())
shutil.copy2(root/'cloud-verification.json',out/'wandb_runs.json')
# Preserve every audit, candidate, integration state, configuration and search
# summary. Full JSON traces/evaluation streams and every video remain locally
# and in the verified W&B artifact/Media entries indexed in wandb_runs.json.
for record in records:
 directory=root/record['directory'];target=out/'evidence'/record['directory']
 for path in directory.rglob('*.json'):
  rel=path.relative_to(directory)
  if 'wandb' in rel.parts or path.name=='trace.json':continue
  dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
 for path in directory.glob('error.txt'):
  target.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target/path.name)
for version in [1,2,3,4]:shutil.copytree(root/f'fixed-plan-v{version}',out/'earlier_plans'/f'v{version}')
for name in ['friction-v1','final-validation','final-validation-v5']:
 target=out/'batches'/name;target.mkdir(parents=True)
 for filename in ['config.json','results.json','summary.json']:
  if (root/name/filename).exists():shutil.copy2(root/name/filename,target/filename)
 if (root/name/'scenarios').exists():shutil.copytree(root/name/'scenarios',target/'scenarios')
shutil.copytree(root/'drivers',out/'drivers')
(out/'diagnostics').mkdir()
for name in ['body-shift-scout.json','policy-stance-scout.json','policy-stance-sweep.json','search-cad-regression.json']:
 if (root/name).exists():shutil.copy2(root/name,out/'diagnostics'/name)
(out/'cad').mkdir();prefix=[]
for stage in ['01-back_r','02-back_l','03-front_r','04-front_l']:
 source=root/'cad-review'/stage if stage!='04-front_l' else root/'cad-review-final-stage-v5'
 shutil.copytree(source,out/'cad'/stage)
 if stage!='04-front_l':
  old=json.loads((root/'nominal-v4'/stage/'end_state.json').read_text());new=json.loads((root/'final-nominal-v5'/stage/'end_state.json').read_text())
  error=float(np.max(np.abs(np.asarray(old['state'])-new['state'])));assert error==0
  prefix.append(dict(stage=stage,end_state_max_absolute_difference=error,reason='Unchanged prefix; dense CAD review remains applicable to the identical integrated trajectory.'))
(out/'cad/prefix_equivalence.json').write_text(json.dumps(prefix,indent=2))
shutil.copy2(root/'final-nominal-v5/rollout.mp4',out/'continuous_rollout.mp4')
shutil.copy2(root/'final-validation-v5/gains-low/rollout.mp4',out/'failed_gains_rollout.mp4')
shutil.copy2(root/'final-validation/mu-1.0-seed-3/rollout.mp4',out/'earlier_tracking_failure.mp4')
nominal=json.loads((root/'final-nominal-v5/sequence_audit.json').read_text());results=json.loads((root/'final-validation-v5/results.json').read_text())
audits=[json.loads(p.read_text()) for p in sorted((root/'final-nominal-v5').glob('*/audit.json'))]
assert all(not a['source_dirty'] for a in audits)
chain=[]
for a,b in zip(audits,audits[1:]):
 matched=a['end_state_sha256']==b['start_state_sha256'];assert matched
 chain.append(dict(previous_leg=a['leg'],next_leg=b['leg'],state_sha256=a['end_state_sha256'],identical=True))
(out/'continuation_chain.json').write_text(json.dumps(chain,indent=2))
policy_path=Path('ros2_ws/src/neural_controller/launch/policy_walk_v2.json');policy=json.loads(policy_path.read_text())
end=json.loads((root/'final-nominal-v5/04-front_l/end_state.json').read_text());command=np.asarray(end['command']);default=np.asarray(policy['default_joint_pos']);prox=[0,1,3,4,6,7,9,10];hubs=[2,5,8,11]
compatibility=dict(policy_source=str(policy_path),policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
    final_command=command.tolist(),policy_default_joint_pos=default.tolist(),proximal_difference_rad=(command-default)[prox].tolist(),
    hub_difference_rad=(command-default)[hubs].tolist(),equivalent_hub_difference_rad=((command-default)[hubs]+np.pi)%(2*np.pi)-np.pi,
    direct_policy_handoff_validated=False,geometric_four_tip_stance=nominal['final_four_tips'],
    reason='Final proximal posture differs from policy default and back-left hub remains one full unwrapped turn above the policy coordinate. A reference-preserving entry transition/executor is required; no policy was activated.')
compatibility['equivalent_hub_difference_rad']=compatibility['equivalent_hub_difference_rad'].tolist()
(out/'policy_compatibility.json').write_text(json.dumps(compatibility,indent=2))
rows=[]
for result in sorted(results,key=lambda r:r['name']):
 failures=[]
 for path in sorted((root/'final-validation-v5'/result['name']).glob('*/audit.json')):
  a=json.loads(path.read_text())
  if a['status']=='FAILED':failures.append(dict(leg=a['leg'],gates=[k for k,v in a['gates'].items() if not v],first=a['first_sampled_gate_violation'],minimum_rotation_floor_m=a['minimum_rotation_floor_m'],maximum_unintended_floor_force_N=a['maximum_unintended_floor_force_N']))
 rows.append(dict(name=result['name'],friction=result['friction'],seed=result['seed'],status=result['audit']['status'],completed_flips=result['audit']['completed_flips'],failures=failures,url=result['logging']['url']))
(out/'validation_results.json').write_text(json.dumps(rows,indent=2))
with (out/'validation_results.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['condition','friction','seed','status','accepted_flips','first_failure','wandb_url'])
 for r in rows:writer.writerow([r['name'],r['friction'],r['seed'],r['status'],r['completed_flips'],json.dumps(r['failures']),r['url']])
friction=[r for r in rows if r['name'].startswith('mu-')];stress=[r for r in rows if not r['name'].startswith('mu-')]
summary=dict(status=nominal['status'],plan_sha256=nominal['plan_sha256'],source_commit=nominal['source_commit'],model_sha256=audits[0]['model_sha256'],
 environment_steps=nominal['environment_steps'],duration_s=nominal['environment_steps']/520,
 friction_passed=sum(r['status']=='PASS_CONTINUOUS_FOUR_FLIPS' for r in friction),friction_cases=len(friction),
 stress_passed=sum(r['status']=='PASS_CONTINUOUS_FOUR_FLIPS' for r in stress),stress_cases=len(stress),
 minimum_rotation_floor_m=min(a['minimum_rotation_floor_m'] for a in audits),minimum_support_force_N=min(a['minimum_support_force_N'] for a in audits),
 max_tilt_deg=max(a['max_tilt_deg'] for a in audits),peak_requested_torque_Nm=max(a['peak_requested_torque_Nm'] for a in audits),
 peak_landing_descent_m_s=max(a['peak_landing_descent_m_s'] for a in audits),final_normal_force_N=audits[-1]['final_normal_force_N'],
 final_tip_bottom_m=audits[-1]['final_tip_bottom_m'],source_hashes=audits[0]['source_hashes'],environment=audits[0]['environment'],
 simulation_only=True,hardware_validated=False,direct_policy_handoff_validated=False,tests_passed=14,
 cloud_verified_runs=sum(r['verified'] for r in records),cloud_checked_runs=len(records),
 nominal_wandb_url=json.loads((root/'final-nominal-v5/logging_status.json').read_text())['url'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))
manifest={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'evidence_sha256.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary,indent=2))
