import json,pathlib,numpy as np
root=pathlib.Path('/tmp/handoff-evidence-20260913');out=[]
for shape in ['zero','diagonal']:
 trials=[root/'contact-convergence'/f'{shape}-parts-{n}' for n in [2,4,8]]
 if shape=='zero':trials.insert(0,root/'development'/'nominal_endpoint')
 base=np.load(trials[0]/'integration.npz')
 for p in trials:
  r=json.load(open(p/'report.json'));a=np.load(p/'integration.npz');same=a['qpos'].shape==base['qpos'].shape
  out.append(dict(case=p.name,steps=r['environment_steps'],floor_peak_N=r['dense_audit']['max_floor_force_N'],gates=r['engineering_gates'],reference_case=trials[0].name,same_steps=same,max_qpos_difference=float(abs(a['qpos']-base['qpos']).max()) if same else None,max_qvel_difference=float(abs(a['qvel']-base['qvel']).max()) if same else None,max_joint_difference_rad=float(abs(a['qpos'][:,7:]-base['qpos'][:,7:]).max()) if same else None,max_root_translation_difference_m=float(np.linalg.norm(a['qpos'][:,:3]-base['qpos'][:,:3],axis=1).max()) if same else None,final_qpos_difference=float(abs(a['qpos'][-1]-base['qpos'][-1]).max())))
(root/'contact-convergence-summary.json').write_text(json.dumps(out,indent=2));print([(r['case'],r['max_qpos_difference'],r['final_qpos_difference'],r['floor_peak_N']) for r in out])
