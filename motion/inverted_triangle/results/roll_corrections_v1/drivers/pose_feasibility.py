"""Kinematic reach diagnostic only; never dynamic acceptance."""
import json,multiprocessing as mp
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import differential_evolution,minimize
from motion.inverted_triangle.roll_to_stand import initialize

def evaluate_case(case):
 r,h,goal=initialize(formation=case.get('formation'))
 def heights(x):
  roll,pitch=x[-2:];r.d.qpos[2]=.15;r.d.qpos[3:7]=[np.cos(roll/2)*np.cos(pitch/2),np.sin(roll/2)*np.cos(pitch/2),np.cos(roll/2)*np.sin(pitch/2),-np.sin(roll/2)*np.sin(pitch/2)];r.d.qpos[7:]=goal+x[:12];mj.mj_forward(r.m,r.d);return r.tip_bottoms()
 def objective(x):
  b=heights(x);return float(np.ptp(b)**2*1e6+np.sum(x*x)*1e-4)
 bounds=[(-.1,.1)]*12+[(-np.deg2rad(5.65),np.deg2rad(5.65))]*2
 result=differential_evolution(objective,bounds,popsize=7,maxiter=80,seed=61,polish=True,tol=1e-7)
 b=heights(result.x);row=dict(name=case['name'],formation=case.get('formation'),kinematic_only=True,not_a_global_infeasibility_certificate=True,within_point_one_rad=dict(tip_height_spread_m=float(np.ptp(b)),x=result.x.tolist(),tip_heights_m=b.tolist(),objective=float(result.fun),evaluations=result.nfev))
 # Find a larger-range coplanar witness and report its required joint departure.
 def minimum_departure(x):return float(np.max(np.abs(x[:12]))+np.ptp(heights(x))*1e4)
 bigger=minimize(minimum_departure,result.x,method='Powell',bounds=[(-.5,.5)]*12+bounds[-2:],options=dict(maxiter=30,maxfev=10000,xtol=1e-6,ftol=1e-7))
 b=heights(bigger.x);row['expanded_pose_witness']=dict(max_joint_departure_rad=float(np.abs(bigger.x[:12]).max()),tip_height_spread_m=float(np.ptp(b)),x=bigger.x.tolist(),tip_heights_m=b.tolist(),evaluations=bigger.nfev)
 return row
if __name__=='__main__':
 cases=json.loads(Path('motion/inverted_triangle/formation_cases.json').read_text())
 with mp.get_context('spawn').Pool(7) as pool:rows=pool.map(evaluate_case,cases)
 Path('runs/roll_to_stand/pose-feasibility.json').write_text(json.dumps(rows,indent=2))
 for r in rows:print(r['name'],r['within_point_one_rad']['tip_height_spread_m'],r['expanded_pose_witness']['max_joint_departure_rad'],r['expanded_pose_witness']['tip_height_spread_m'],flush=True)
