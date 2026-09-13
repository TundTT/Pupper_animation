"""Assemble explicit scope/pass tables; preserve all failed and incomplete audits."""
import json,hashlib,zipfile,collections
from pathlib import Path
old=Path('/tmp/pupper-roll-to-stand-20260912/runs/roll_to_stand');new=Path('runs/roll_to_stand');out=new/'review-data';out.mkdir(exist_ok=True)
def passed(a):return all(v is True for v in a['gates'].values())
def row(f,family):
 a=json.loads(f.read_text());p=f.parent
 while not (p/'wandb_run.json').exists() and p!=p.parent:p=p.parent
 w=json.loads((p/'wandb_run.json').read_text()) if (p/'wandb_run.json').exists() else {}
 d=dict(family=family,case=a['config']['name'],audit=str(f),run_id=w.get('id'),url=w.get('url'),passed=passed(a),gates=a['gates'],failed_gates=[k for k,v in a['gates'].items() if v is not True],configuration=a['config'],model_sha256=a.get('model_sha256',a['model_manifest']['model_sha256']),tilt_deg=a['max_tilt_deg'],peak_requested_torque_Nm=a['peak_requested_torque_Nm'],max_measured_joint_speed_rad_s=a['max_measured_joint_speed'],floor_force_N=a.get('max_unintended_floor_force_N',a.get('max_motor_body_floor_N')),dense_cad_gap_m=a.get('dense_dynamics_audit',{}).get('cad_minimum',{}).get('minimum_m'),steps=a['environment_steps'])
 if 'forward_mean_body_velocity_m_s' in a:d['forward_velocity_m_s']=a['forward_mean_body_velocity_m_s'];d['transition_pass']=a['transition_pass']
 return d
families={
 'baseline_formation':list((old/'formation-baseline').glob('*/audit.json')),
 'direct_original_families':list((old/'direct-original-families').glob('*/*/audit.json')),
 'support_only_original_families':list((new/'corrected-original-families').glob('*/*/audit.json')),
 'final_original_families':list((new/'damped-original-families').glob('*/*/audit.json')),
 'final_development_roll':list((new/'damped-development').glob('*/*/audit.json')),
 'unmodified_development_handoff':list((new/'corrected-development-walking').glob('*/handoff/audit.json')),
 'final_development_handoff':list((new/'selected-development-handoff').glob('*/handoff/audit.json')),
 'length_permutation_and_common_offset_validation':list((new/'held-out-shape-permutations').glob('*/handoff/audit.json')),
 'angle_permutation_validation':list((new/'held-out-angle-permutations').glob('*/handoff/audit.json')),
 'fresh_shapes_sensors_dynamics':list((new/'held-out-shapes-sensors').glob('*/handoff/audit.json')),
 'expanded_reach_rejected':list((new/'expanded-reach-diagnostic').glob('*/*/audit.json')),
}
rows=[row(f,family) for family,files in families.items() for f in sorted(files)]
forms=json.load(open('motion/inverted_triangle/formation_cases.json'));seen={json.dumps(c.get('formation'),sort_keys=True) for c in forms}
coverage=[r for r in rows if r['family'] in ['length_permutation_and_common_offset_validation','angle_permutation_validation','fresh_shapes_sensors_dynamics']]
for r in coverage:
 r['role']='fresh_held_out' if r['family']=='fresh_shapes_sensors_dynamics' or json.dumps(r['configuration'].get('formation'),sort_keys=True) not in seen else 'development_repeat'
 roll=Path(r['audit']).parents[1]/'roll/audit.json';r['roll_pass']=passed(json.load(open(roll)));r['roll_failed_gates']=[k for k,v in json.load(open(roll))['gates'].items() if v is not True]
summary={}
for family in families:
 batch=[r for r in rows if r['family']==family];summary[family]=dict(cases=len(batch),passed=sum(r['passed'] for r in batch),failed_gates=dict(collections.Counter(k for r in batch for k in r['failed_gates'])))
for role in ['fresh_held_out','development_repeat']:
 batch=[r for r in coverage if r['role']==role];summary[role]=dict(cases=len(batch),roll_passed=sum(r['roll_pass'] for r in batch),continuous_passed=sum(r['passed'] for r in batch))
summary['coverage_all']=dict(cases=len(coverage),roll_passed=sum(r['roll_pass'] for r in coverage),continuous_passed=sum(r['passed'] for r in coverage))
(out/'case-index.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
# Compact exact audit/config archive in Git. Full states, traces and all MP4s are
# retained locally and in their W&B artifacts/Media, not repeated in this zip.
count=0
with zipfile.ZipFile(out/'all-audits-and-configs.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for prefix,root in [('initial',old),('corrections',new)]:
  for f in root.rglob('*.json'):
   rel=f.relative_to(root)
   if any(p in rel.parts for p in ['wandb','review-data','review-bundle']):continue
   if f.name not in ['audit.json','provenance.json','summary.json','interrupted_status.json','interrupted_status_history.json','configs.json'] and 'configs' not in rel.parts:continue
   z.write(f,prefix+'/'+str(rel));count+=1
manifest=dict(files=count,sha256=hashlib.sha256((out/'all-audits-and-configs.zip').read_bytes()).hexdigest(),bytes=(out/'all-audits-and-configs.zip').stat().st_size,full_dynamics_location='W&B artifacts and original local run directories; indexed by case and run ID',directory_label_note='Some directories use held-out prefix but contain five exact development repeats. Index role explicitly separates 25 fresh conditions from those five regressions.')
(out/'archive-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary,indent=2));print(manifest)
