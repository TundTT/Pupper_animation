"""Contact/floor diagnostics on saved physics states; no gate changes."""
import json,concurrent.futures
from pathlib import Path
import numpy as np,mujoco as mj
from motion.inverted_triangle.core import Robot
from motion.inverted_triangle.clearance import CADClearance

def run(directory):
 audit=json.loads((directory/'audit.json').read_text());cfg=audit['config'];r=Robot(cfg.get('friction',.8),cfg.get('scenario',{}).get('dynamics'),cfg.get('formation'));checker=CADClearance(r);z=np.load(directory/'integration.npz');lowest=(1.,None,None);tip_events=[None]*4
 for k in range(0,len(z['qpos']),13):
  r.d.qpos[:]=z['qpos'][k];r.d.qvel[:]=z['qvel'][k];mj.mj_forward(r.m,r.d)
  for name,points in checker.local_vertices.items():
   if name.endswith('_3'):continue
   bid=checker.body_ids[name];height=float((points@r.d.xmat[bid].reshape(3,3).T+r.d.xpos[bid])[:,2].min())
   if height<lowest[0]:lowest=(height,name,(k+1)/520)
  if (k+1)/520>2+cfg['seconds']/2:
   tip=r.tip_bottoms()
   for i in range(4):
    if tip_events[i] is None and abs(tip[i])<=.003 and z['loads'][k,i]>=1:
     tip_events[i]=dict(time_s=(k+1)/520,record_step=k,tip_height_m=float(tip[i]),load_N=float(z['loads'][k,i]))
 for i,event in enumerate(tip_events):
  if event is None:continue
  start=max(0,event['record_step']-260);end=min(len(z['qpos']),event['record_step']+261);max_slip=0.;max_normal=0.
  for k in range(start,end):
   r.d.qpos[:]=z['qpos'][k];r.d.qvel[:]=z['qvel'][k];mj.mj_forward(r.m,r.d)
   for c in r.d.contact:
    if r.floor not in (c.geom1,c.geom2) or r.geoms[i] not in (c.geom1,c.geom2):continue
    jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac);mj.mj_jac(r.m,r.d,jac,rot,c.pos,r.bodies[i]);v=jac@r.d.qvel;max_slip=max(max_slip,float(np.linalg.norm(v[:2])));max_normal=max(max_normal,float(abs(v[2])))
  event.update(window_s=[start/520,end/520],normal_impulse_Ns=float(z['loads'][start:end,i].sum()/520),peak_step_impulse_Ns=float(z['loads'][start:end,i].max()/520),max_contact_slip_m_s=max_slip,max_contact_normal_speed_m_s=max_normal)
 return dict(case=cfg['name'],source_directory=str(directory),sampled_motor_body_backpack_cad_floor_clearance_m=lowest[0],minimum_part=lowest[1],minimum_time_s=lowest[2],floor_cad_sample_period_s=.025,tip_arrival_events=tip_events,scope='Geometric diagnostics on saved integrated states. Loads/impulses use original physics records; point velocities use FK/Jacobians of those states. No new dynamic acceptance claim.')
if __name__=='__main__':
 cases=list(Path('runs/roll_to_stand/damped-development').glob('*/*/audit.json'))+[Path('runs/roll_to_stand/damped-original-families/combined-seed-22/combined-seed-22/audit.json')]
 with concurrent.futures.ProcessPoolExecutor(4) as pool:result=list(pool.map(run,[p.parent for p in cases]))
 Path('runs/roll_to_stand/review-data/contact-review.json').write_text(json.dumps(result,indent=2))
 for x in result:print(x['case'],x['sampled_motor_body_backpack_cad_floor_clearance_m'],flush=True)
