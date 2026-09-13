"""Offline geometry diagnostic: coplanar witnesses, not dynamic acceptance."""
import json,concurrent.futures
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.optimize import minimize
from motion.inverted_triangle.core import Robot,HERE,provenance
from motion.inverted_triangle.walking_policy import WalkingPolicy

def solve(case):
 r=Robot(formation=case.get('formation'));goal=WalkingPolicy(np.zeros(12)).home
 def heights(x):
  roll,pitch=x[12:14];r.d.qpos[:3]=[0,0,.15];r.d.qpos[3:7]=[np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)];r.d.qpos[7:]=goal+x[:12];mj.mj_forward(r.m,r.d)
  return r.tip_bottoms()
 def equal(x):
  z=heights(x);return (z[1:]-z[0])*1000
 def limits(x):return np.r_[x[14]-x[:12],x[14]+x[:12],np.cos(x[12])*np.cos(x[13])-np.cos(np.deg2rad(8))]
 rng=np.random.default_rng(712);trials=[]
 for k in range(16):
  x=np.r_[rng.uniform(-.15,.15,12),rng.uniform(-.05,.05,2),.3] if k else np.r_[np.zeros(14),.2]
  fit=minimize(lambda x:x[14]+1e-8*np.sum(x[:12]**2),x,method='SLSQP',bounds=[(-.5,.5)]*12+[(-.14,.14)]*2+[(0,.5)],constraints=[dict(type='eq',fun=equal),dict(type='ineq',fun=limits)],options=dict(maxiter=400,ftol=1e-11))
  h=heights(fit.x);ok=np.ptp(h)<1e-7 and limits(fit.x).min()>-1e-7
  trials.append(dict(success=bool(ok),solver_success=bool(fit.success),cap_rad=float(abs(fit.x[:12]).max()),joint_offsets_rad=fit.x[:12].tolist(),body_roll_pitch_rad=fit.x[12:14].tolist(),tip_height_spread_m=float(np.ptp(h)),message=fit.message))
 best=min([x for x in trials if x['success']],key=lambda x:x['cap_rad'],default=None)
 return dict(case=case,best_coplanar_witness=best,trials=trials,not_global_infeasibility_certificate=True,not_dynamic_or_load_acceptance=True)
if __name__=='__main__':
 cases=json.loads((HERE/'formation_cases.json').read_text())
 with concurrent.futures.ProcessPoolExecutor(7) as pool:results=list(pool.map(solve,cases))
 Path('runs/roll_to_stand/stance-adjustment.json').write_text(json.dumps(dict(source_hashes=provenance(),results=results),indent=2))
 for r in results:print(r['case']['name'],r['best_coplanar_witness'],flush=True)
