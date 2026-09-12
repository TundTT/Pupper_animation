"""Build the derivative review bundle without replacing the accepted parent."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
root=Path('runs/inverted_triangle');out=root/'review-bundle';out.mkdir(exist_ok=False)
records=json.loads((root/'cloud-verification.json').read_text())
assert len(records)==121 and all(r['verified'] for r in records)
shutil.copytree(root/'combined-plan-v2',out/'plan')
shutil.copy2(root/'cloud-verification.json',out/'wandb_runs.json')
for r in records:
 src=root/r['directory'];dst=out/'evidence'/r['directory']
 for p in src.rglob('*'):
  rel=p.relative_to(src)
  if not p.is_file() or 'wandb' in rel.parts or p.name=='trace.json':continue
  if p.suffix not in ['.json','.txt','.log']:continue
  d=dst/rel;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
for name in ['validation-v1','held-out-v1','validation-v2','former-held-out-regression-v2','fresh-held-out-v2']:
 dst=out/'batches'/name;dst.mkdir(parents=True)
 for f in ['config.json','results.json','summary.json']:shutil.copy2(root/name/f,dst/f)
 shutil.copytree(root/name/'scenarios',dst/'scenarios')
shutil.copytree(root/'landing-only-plan-v1',out/'earlier_plans/landing-only-v1')
shutil.copytree(root/'cad-v2',out/'cad')
shutil.copytree(root/'combined-plan-v1',out/'earlier_plans/combined-v1')
if (root/'cad-worst-stress').exists():shutil.copytree(root/'cad-worst-stress',out/'cad-worst-stress')
shutil.copytree(root/'diagnostics',out/'diagnostics');
for name in ['preload-scout','direct-finish-scout','direct-finish-scout-v2']:
 shutil.copytree(root/name,out/'diagnostics'/name)
shutil.copy2(root/'fresh-held-out-manifest.json',out/'fresh-held-out-manifest.json');shutil.copytree(root/'drivers',out/'drivers')
for f in ['requirements.lock.txt','source_manifest.json']:shutil.copy2(Path('motion/inverted_triangle')/f,out/f)
nom=json.loads((root/'nominal-v2/sequence_audit.json').read_text())
assert nom['plan_sha256']==hashlib.sha256((out/'plan/plan.json').read_bytes()).hexdigest()
for batch in ['validation-v2','former-held-out-regression-v2','fresh-held-out-v2']:
 assert json.loads((root/batch/'summary.json').read_text())['plan_sha256']==nom['plan_sha256']
stages=['01-back_r','02-back_l','03-front_r','04-front_l']
audits=[json.loads((root/'nominal-v2'/s/'audit.json').read_text()) for s in stages]
assert all(not a['source_dirty'] for a in audits)
chain=[]
for a,b in zip(audits,audits[1:]):
 assert a['end_state_sha256']==b['start_state_sha256']
 chain.append(dict(previous_leg=a['leg'],next_leg=b['leg'],identical=True,state_sha256=a['end_state_sha256']))
(out/'continuation_chain.json').write_text(json.dumps(chain,indent=2))
parent=Path('motion/inverted_triangle/results/pc_20260912')
reproduction=[]
for s in stages:
 a=json.loads((root/'baseline-nominal'/s/'audit.json').read_text());b=json.loads((parent/'evidence/final-nominal-v5'/s/'audit.json').read_text())
 assert a['end_state_sha256']==b['end_state_sha256']
 reproduction.append(dict(stage=s,status=a['status'],identical_complete_end_state=True,end_state_sha256=a['end_state_sha256']))
(out/'parent_reproduction.json').write_text(json.dumps(reproduction,indent=2))
rows=[];all_audits=[]
for batch in ['validation-v2','former-held-out-regression-v2','fresh-held-out-v2']:
 for r in json.loads((root/batch/'results.json').read_text()):
  stage_audits=[json.loads(p.read_text()) for p in sorted((root/batch/r['name']).glob('0*/audit.json'))];all_audits+=stage_audits
  failures=[dict(leg=a['leg'],gates=[k for k,v in a['gates'].items() if not v],first=a['first_sampled_gate_violation']) for a in stage_audits if a['status']=='FAILED']
  rows.append(dict(batch=batch,name=r['name'],friction=r['friction'],seed=r['seed'],scenario=r['scenario'],status=r['audit']['status'],completed_flips=r['audit']['completed_flips'],failures=failures,url=r['logging']['url']))
(out/'validation_results.json').write_text(json.dumps(rows,indent=2))
with (out/'validation_results.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['batch','case','friction','seed','status','completed_flips','failures','wandb_url'])
 for r in rows:w.writerow([r[k] if k!='failures' else json.dumps(r[k]) for k in ['batch','name','friction','seed','status','completed_flips','failures','url']])
groups={name:[r for r in rows if test(r)] for name,test in [('friction',lambda r:r['name'].startswith('mu-')),('stress',lambda r:r['batch']=='validation-v2' and not r['name'].startswith('mu-')),('held_out',lambda r:r['batch']=='fresh-held-out-v2'),('former_held_out_regression',lambda r:r['batch']=='former-held-out-regression-v2')]}
summary=dict(status=nom['status'],parent_commit='1139981cac240986e1deb8d53b2a63441b79373f',source_commit=nom['source_commit'],plan_sha256=nom['plan_sha256'],parent_plan_sha256=hashlib.sha256((parent/'plan/plan.json').read_bytes()).hexdigest(),model_sha256=audits[0]['model_sha256'],environment_steps=nom['environment_steps'],duration_s=nom['environment_steps']/520,simulation_only=True,hardware_validated=False,direct_policy_handoff_validated=False,tests_passed=16,source_hashes=audits[0]['source_hashes'],environment=audits[0]['environment'],cloud_verified_runs=len(records),nominal_wandb_url=json.loads((root/'nominal-v2/logging_status.json').read_text())['url'])
for name,rs in groups.items():summary[name+'_passed']=sum(r['status']=='PASS_CONTINUOUS_FOUR_FLIPS' for r in rs);summary[name+'_cases']=len(rs)
summary['all_60_worst_metrics']=dict(rotation_floor_m=min(a['minimum_rotation_floor_m'] for a in all_audits),cad_gap_m=min(a['minimum_cad_gap_m'] for a in all_audits),support_N=min(a['minimum_support_force_N'] for a in all_audits),floor_N=max(a['maximum_unintended_floor_force_N'] for a in all_audits),landing_descent_m_s=max(a['peak_landing_descent_m_s'] for a in all_audits),tilt_deg=max(a['max_tilt_deg'] for a in all_audits),requested_torque_Nm=max(a['peak_requested_torque_Nm'] for a in all_audits),hub_error_rad=max(a['final_hub_error_rad'] for a in all_audits))
comparison=[]
for case,stage,metric in [('mass-low','04-front_l','peak_landing_descent_m_s'),('gains-high','04-front_l','peak_landing_descent_m_s'),('mass-high','02-back_l','maximum_unintended_floor_force_N'),('gains-low','02-back_l','maximum_unintended_floor_force_N'),('combined-seed-22','01-back_r','minimum_rotation_floor_m')]:
 old=json.loads((parent/'evidence/final-validation-v5'/case/stage/'audit.json').read_text());new=json.loads((root/'validation-v2'/case/stage/'audit.json').read_text())
 comparison.append(dict(case=case,stage=stage,metric=metric,before=old[metric],after=new[metric],old_status=old['status'],new_status=new['status']))
summary['targeted_comparison']=comparison
(out/'summary.json').write_text(json.dumps(summary,indent=2))
policy_path=Path('ros2_ws/src/neural_controller/launch/policy_walk_v2.json');policy=json.loads(policy_path.read_text());end=json.loads((root/'nominal-v2/04-front_l/end_state.json').read_text());command=np.asarray(end['command']);default=np.asarray(policy['default_joint_pos'])
compat=dict(policy_source=str(policy_path),policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),final_command=command.tolist(),policy_default_joint_pos=default.tolist(),difference_rad=(command-default).tolist(),direct_policy_handoff_validated=False,geometric_four_tip_stance=nom['final_four_tips'],reason='Posture differs from walking default; rear-left hub preserves its extra unwrapped revolution. A measured-state reference-preserving entry transition remains unvalidated. No hardware or policy activation.')
(out/'policy_compatibility.json').write_text(json.dumps(compat,indent=2))
print(json.dumps(summary,indent=2))
