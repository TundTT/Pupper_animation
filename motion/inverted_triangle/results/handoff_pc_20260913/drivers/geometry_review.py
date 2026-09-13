import json,hashlib,subprocess
from pathlib import Path
import numpy as np
import mujoco as mj
from motion.inverted_triangle.handoff_probe import initialize
from motion.inverted_triangle.clearance import CADClearance
from motion.inverted_triangle.core import provenance,versions
terminal=json.load(open('motion/inverted_triangle/results/handoff_local_20260913/alignment-terminal-0.json'))
cases=json.load(open('motion/inverted_triangle/handoff_development_cases.json'))['cases']
records=[]
for case in cases:
 r,b=initialize(case['config'],terminal);checker=CADClearance(r);q=r.d.qpos.copy()
 def measure():
  floor={n:float((v@r.d.xmat[checker.body_ids[n]].reshape(3,3).T+r.d.xpos[checker.body_ids[n]])[:,2].min()) for n,v in checker.local_vertices.items()}
  foot=min(z for n,z in floor.items() if n.endswith('_3'))
  other={n:z-foot for n,z in floor.items() if not n.endswith('_3')}
  return dict(minimum_motor_body_minus_foot_m=min(other.values()),part=min(other,key=other.get),part_margins_m=other,foot_heights_relative_min_m=(r.bottoms()-min(r.bottoms())).tolist())
 row=dict(name=case['name'],config=case['config'],actual_endpoint_geometry=measure(),static_preformation_sweep=[])
 for splay in np.linspace(0,.32,33):
  r.d.qpos[:]=q;r.d.qpos[7+np.array([1,4,7,10])]=splay*np.array([-1,1,-1,1]);mj.mj_forward(r.m,r.d)
  row['static_preformation_sweep'].append(dict(splay_rad=float(splay),**measure()))
 row['first_sample_with_1mm_margin']=next((x for x in row['static_preformation_sweep'] if x['minimum_motor_body_minus_foot_m']>=.001),None)
 records.append(row);print(case['name'],row['actual_endpoint_geometry']['minimum_motor_body_minus_foot_m'], None if row['first_sample_with_1mm_margin'] is None else row['first_sample_with_1mm_margin']['splay_rad'],flush=True)
out=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes=provenance(),versions=versions(),terminal_sha256=hashlib.sha256(Path('motion/inverted_triangle/results/handoff_local_20260913/alignment-terminal-0.json').read_bytes()).hexdigest(),scope='Static CAD only. Swept poses are hypothetical pre-formation prerequisites, NOT measured endpoints, dynamic rollouts, or a deployable motion.',physical_discrepancy='User estimated about 5 mm clearance. Fixed CAD nominal actual endpoint predicts penetration. No dimension is changed.',records=records)
Path('/tmp/handoff-evidence-20260913/geometry-review.json').write_text(json.dumps(out,indent=2))
