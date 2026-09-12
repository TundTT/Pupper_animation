"""Simulation-only robust landing search; the accepted lift path is immutable."""
import hashlib,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution,minimize
from motion.inverted_triangle.core import Robot,LEGS,PROX,HUB,duration,smooth,provenance,versions
from motion.inverted_triangle.simulate import trajectory,run
from training.wandb_logging import ExperimentLogger
W=None

def init(states,base):
 global W
 W=[]
 for mu,state,command in states:
  r=Robot(mu);r.restore(state);W.append((r,state,np.array(command),np.array(base)))

def evaluate(x):
 rows=[];total=0;cost=0
 for r,state,previous,base in W:
  r.restore(state);target=base.copy();target[PROX]=x;n=int(np.ceil(duration(previous,target,4)/r.m.opt.timestep));floor=0;tilt=0;peak=0;descent=0;last=None
  for k in range(n+1040):
   tau=r.tick(previous+smooth(min((k+1)/n,1))*(target-previous));total+=1;peak=max(peak,float(np.abs(tau).max()))
   if k%13:continue
   floor=max(floor,r.unintended_floor_force());tilt=max(tilt,float(np.rad2deg(r.tilt())));bottom=float(r.bottoms()[1])
   if last is not None and bottom<.008:descent=max(descent,(last-bottom)/.025)
   last=bottom
  f=r.contacts_precise();err=abs(r.d.qpos[7+HUB[1]]-base[HUB[1]]);tip=abs(float(r.tip_bottoms()[1]));speed=float(np.abs(r.d.qvel[6:]).max())
  value=float(np.maximum(2.8-f,0).dot(np.maximum(2.8-f,0)))*20+max(err-.018,0)**2*40000+max(floor-.02,0)**2*20+max(tilt-5,0)**2+max(descent-.015,0)**2*80000+max(tip-.001,0)**2*80000+max(speed-.05,0)**2*20+max(peak-2.5,0)**2*8
  rows.append(dict(friction=r.friction,force_N=f.tolist(),hub_error_rad=float(err),tip_m=tip,descent_m_s=descent,tilt_deg=tilt,floor_N=floor,final_speed_rad_s=speed,cost=value));cost+=value
 return dict(cost=cost+float(np.sum((x-W[0][3][PROX])**2))*.01,x=np.asarray(x).tolist(),conditions=rows,environment_steps=total)

def main():
 out=Path('runs/inverted_triangle/robust-last-landing');out.mkdir(exist_ok=False)
 source=Path('runs/inverted_triangle/fixed-plan-v4/04-front_l.json');candidate=json.loads(source.read_text());base=np.array(candidate['landing_pose']);states=[];prefix_steps=0
 for mu in [.5,.65,.8,1.]:
  r=Robot(mu);original=json.loads(Path(f'runs/inverted_triangle/final-validation/mu-{mu}-seed-1/04-front_l/start_state.json').read_text());home=r.restore(original);previous=home.copy()
  for phase,target,seconds in trajectory(home,np.array(candidate['full_pose']),1,direction=-1,pre_shift_pose=candidate['pre_shift_pose'],landing_pose=base):
   if phase=='land':break
   n=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
   for k in range(n):r.tick(previous+smooth((k+1)/n)*(target-previous));prefix_steps+=1
   previous=target.copy()
  states.append((mu,r.snapshot(previous),previous.tolist()))
 config=dict(simulation_only=True,source_hashes=provenance(),environment=versions(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),frictions=[.5,.65,.8,1.],seed=8,prefix_steps=prefix_steps)
 (out/'config.json').write_text(json.dumps(config,indent=2));(out/'landing_start_states.json').write_text(json.dumps(states,indent=2));logger=ExperimentLogger(out,config)
 limits=Robot().m.jnt_range[1:][PROX];x=base[PROX].copy();radius=np.array([.12,.12,.2,.12,.12,.12,.12,.12]);bounds=np.c_[np.maximum(limits[:,0],x-radius),np.minimum(limits[:,1],x+radius)]
 rng=np.random.default_rng(8);pop=np.clip(x+rng.normal(0,.03,(64,8)),bounds[:,0],bounds[:,1]);pop[0]=x;count=0;steps=prefix_steps;best=None;begin=time.monotonic()
 def record(items):
  nonlocal count,steps,best
  values=[]
  with (out/'evaluations.jsonl').open('a') as f:
   for item in items:
    count+=1;steps+=item['environment_steps'];item['evaluation']=count;f.write(json.dumps(item)+'\n');values.append(item['cost'])
    if best is None or item['cost']<best['cost']:
     best=item;c=dict(candidate);land=base.copy();land[PROX]=item['x'];c['landing_pose']=land.tolist();c['status']='UNACCEPTED_ROBUST_LANDING_CANDIDATE';c.pop('measurements',None);c.pop('search_vector',None);c['landing_search']=item;c['parent_sha256']=config['parent_sha256'];(out/'candidate.json').write_text(json.dumps(c,indent=2));print(json.dumps(item),flush=True)
  return values
 with mp.get_context('spawn').Pool(16,initializer=init,initargs=(states,base.tolist())) as pool:
  def mapper(function,values):return record(pool.map(evaluate,list(values)))
  differential_evolution(lambda x:0,bounds,init=pop,seed=8,maxiter=20,workers=mapper,updating='deferred',polish=False,tol=1e-5)
 init(states,base.tolist())
 minimize(lambda x:record([evaluate(x)])[0],np.asarray(best['x']),method='Powell',bounds=bounds,options=dict(maxiter=2,maxfev=160,xtol=.001,ftol=.001))
 (out/'search.json').write_text(json.dumps(dict(evaluations=count,environment_steps=steps,seconds=time.monotonic()-begin,best=best),indent=2))
 audit=run(out/'candidate.json',out/'replay',video=True,direction=-1,start_override=json.loads(Path('runs/inverted_triangle/final-nominal/04-front_l/start_state.json').read_text()));steps+=audit['environment_steps']
 logger.metrics(steps,{k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))});logger.video(out/'replay/rollout.mp4',steps,key='trajectory/front_l',caption=f'SIMULATION robust landing candidate {audit["candidate_sha256"]} | search seed 8, replay seed 0 | {audit["status"]} | early termination {audit["early_termination"]}; no trained checkpoint')
 logger.run.summary.update(dict(simulation_audit_status=audit['status'],gates=audit['gates'],hardware_validated=False));artifact=logger.sdk.Artifact('robust-landing-'+logger.state['id'],type='trajectory-run')
 for p in out.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl'] and 'wandb' not in p.relative_to(out).parts:artifact.add_file(str(p),name=p.relative_to(out).as_posix())
 artifact.add_file(__file__,name='driver.py');logger.run.log_artifact(artifact);logger.finish()
 (out/'logging_status.json').write_text(json.dumps(dict(status='SDK finish completed; verify cloud',url=logger.state['url']),indent=2))

if __name__=='__main__':main()
