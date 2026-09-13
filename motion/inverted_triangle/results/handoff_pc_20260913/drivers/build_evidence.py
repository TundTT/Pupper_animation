"""Build lightweight, reviewable evidence; full arrays stay in per-run W&B artifacts."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
root=Path('/tmp/handoff-evidence-20260913');rows=[]
for group in ['reproduce-nominal','reproduce-diagonal','entry-search','splay-search','development','contact-convergence','fresh-conditional']:
 for p in sorted((root/group).rglob('report.json')):
  if 'wandb' in p.relative_to(root).parts:continue
  r=json.loads(p.read_text());trace=json.loads((p.parent/'trace.json').read_text());last=trace[-1] if trace else {};gates=r['engineering_gates']
  row=dict(case=str(p.parent.relative_to(root)),kind='actual historical endpoint + explicitly configured synthetic trials',config=r['config'],source_commit=r['source_commit'],source_dirty=r['source_dirty'],engineering_pass=r['engineering_pass'],gates=gates,early_termination=r['early_termination'],steps=r['environment_steps'],floor_peak_N=r['dense_audit']['max_floor_force_N'],peak_requested_torque_Nm=r['peak_requested_torque_Nm'],final_tip_bottom_m=last.get('tip_bottom_m'),final_normal_force_N=last.get('normal_force_N'),final_pose_error_rad=last.get('pose_error_rad'),wandb_url=r.get('wandb_url'),dense_replay_error=r['dense_audit'].get('replay_max_state_error'),cad_minimum=r['dense_audit'].get('cad_minimum'))
  rows.append(row)
for p in sorted((root/'parent-regression').rglob('audit.json')):
 r=json.loads(p.read_text());state=json.loads((p.parent.parent/'wandb_run.json').read_text());rows.append(dict(case=str(p.parent.relative_to(root)),kind='parent fixed-keyframe regression, NOT actual alignment handoff',config=r['config'],engineering_pass=all(v is True for v in r['gates'].values()),gates=r['gates'],steps=r['environment_steps'],floor_peak_N=r['max_unintended_floor_force_N'],peak_requested_torque_Nm=r['peak_requested_torque_Nm'],wandb_url=state.get('url'),dense_replay_error=r['dense_dynamics_audit'].get('replay_max_state_error'),cad_minimum=r['dense_dynamics_audit'].get('cad_minimum')))
for p in sorted(root.glob('alignment-published-*/terminal.json')):
 r=json.loads(p.read_text());state=json.loads((p.parent/'wandb_run.json').read_text());rows.append(dict(case=str(p.parent.relative_to(root)),kind='actual fresh published-source alignment replay; rejected endpoint',source_commit=r['source_commit'],engineering_pass=False,alignment_passed=r['alignment_passed'],completed_mask=r['completed_mask'],failed_mask=r['failed_mask'],steps=r.get('environment_steps',round(r['simulation_time_s']*520)),wandb_url=state.get('url')))
(root/'case-index.json').write_text(json.dumps(rows,indent=2)+'\n')
parity=[]
for original,repeat in [('nominal_endpoint','baseline-nominal'),('diagonal_10mm_spread','baseline-diagonal')]:
 a=np.load(root/'development'/original/'integration.npz');b=np.load(root/'fresh-conditional'/repeat/'integration.npz');parity.append(dict(original=original,repeat=repeat,arrays_exactly_equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files}))
(root/'sensor-baseline-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
archive=root/'all-audits-configs-and-provenance.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(root.rglob('*.json')):
  parts=p.relative_to(root).parts
  if 'wandb' in parts or 'selected-download' in parts:continue
  if p.name in ('trace.json','dense_cad_samples.json','initial_integration_state.json','final_state.json','results.json','summary.json'):continue
  z.write(p,str(p.relative_to(root)))
 for p in sorted((root/'drivers').glob('*.py')):z.write(p,str(p.relative_to(root)))
summary=dict(experiments=len(rows),environment_steps=sum(r['steps'] for r in rows),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),archive_bytes=archive.stat().st_size,groups={})
for group in ['development','fresh-conditional','parent-regression']:
 subset=[r for r in rows if r['case'].startswith(group+'/') and 'baseline-' not in r['case']]
 summary['groups'][group]=dict(count=len(subset),passed=sum(r['engineering_pass'] for r in subset))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
