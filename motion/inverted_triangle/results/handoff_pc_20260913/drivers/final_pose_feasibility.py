"""Static numerical witnesses only; no qpos assignment is dynamic evidence."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.optimize import least_squares
from motion.inverted_triangle.handoff_probe import initialize
from motion.inverted_triangle.handoff_entry import EntryRoll
root=Path('/tmp/handoff-evidence-20260913');terminal=json.load(open('motion/inverted_triangle/results/handoff_local_20260913/alignment-terminal-0.json'));cases=json.load(open('motion/inverted_triangle/handoff_development_cases.json'))['cases'];records=[]
for i in [3,7]:
 case=cases[i];r,b=initialize(case['config'],terminal);goal=EntryRoll(r.initial[7:],previous_command=b['entry_initial_command']).goal
 def apply(x):
  roll,pitch=x[12:];r.d.qpos[7:]=goal+x[:12];r.d.qpos[:3]=[0,0,1]
  r.d.qpos[3:7]=[np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)];r.d.qvel[:]=0;mj.mj_forward(r.m,r.d)
  h=r.tip_bottoms();return h-h.mean()
 trials=[];rng=np.random.default_rng(20260913+i)
 for j in range(12):
  x=np.zeros(14) if j==0 else rng.uniform(-.08,.08,14)
  result=least_squares(lambda x:np.r_[apply(x)*1000,.001*x],x,bounds=(-.099999,.099999),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-11)
  heights=apply(result.x);trials.append(dict(trial=j,tip_height_spread_m=float(np.ptp(heights)),max_pose_error_rad=float(abs(result.x[:12]).max()),tilt_deg=float(np.rad2deg(r.tilt())),variables=result.x.tolist(),optimizer_success=bool(result.success),evaluations=result.nfev))
 records.append(dict(name=case['name'],config=case['config'],trials=trials,best=min(trials,key=lambda t:t['tip_height_spread_m'])))
 print(case['name'],records[-1]['best'],flush=True)
(root/'final-pose-feasibility.json').write_text(json.dumps(dict(scope='Numerical static witnesses only. Coplanarity is not loaded support, CAD clearance, or controller/dynamic acceptance. Lack of a witness is not proof of global infeasibility.',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records),indent=2))
