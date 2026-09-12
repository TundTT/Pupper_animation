"""CPU search for small support transfer with fixed active pose after touchdown."""
import json,hashlib,time,multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from motion.inverted_triangle.robust_search import initialize,evaluate,decode
from motion.inverted_triangle.core import PROX,provenance,versions
from motion.inverted_triangle.simulate import run
from training.wandb_logging import ExperimentLogger

def objective(v):
 r=evaluate(np.r_[v[2:4],v]);costs=[]
 for c in r['conditions']:
  f=np.array(c['final_force_N']);lack=np.maximum(1.8-f,0);lack[1]=max(3.-f[1],0)
  value=float(lack@lack)*150+max(c['tilt_deg']-7.2,0)**2*150+max(c['descent_m_s']-.018,0)**2*5e5+max(c['hub_error_rad']-.025,0)**2*5e4+max(c['floor_N']-.01,0)**2*100+max(.003-c['cad_gap_m'],0)**2*3e6+max(c['tip_m']-.001,0)**2*3e5+c['pre_touchdown_support_deficit']*10+max(c['speed_rad_s']-.04,0)**2*100
  costs.append(value)
 r['cost']=float(max(costs)+np.mean(costs));r['direct_vector']=np.asarray(v).tolist();return r

def main():
 out=Path('runs/inverted_triangle/finish-search');out.mkdir(exist_ok=False)
 c=json.loads(Path('runs/inverted_triangle/combined-plan-v1/04-front_l.json').read_text());states=json.loads(Path('runs/inverted_triangle/preload-scout/contexts.json').read_text())
 config=dict(simulation_only=True,source_hashes=provenance(),environment=versions(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=52,training_contexts=[s['job'] for s in states],scope='Former held-out-08 included in diagnosis; fresh seed-271828 set excluded.',generations=6,population=32,workers=16)
 (out/'config.json').write_text(json.dumps(config,indent=2));(out/'contexts.json').write_text(json.dumps(states,indent=2));logger=ExperimentLogger(out,config)
 v=np.array(c['full_pose'])[PROX];v[2:4]=[-.98,-.4];rad=np.repeat(.06,8);rad[2:4]=[.12,.06];bounds=np.c_[v-rad,v+rad];rng=np.random.default_rng(52);pop=np.clip(v+rng.normal(size=(32,8))*rad*.45,bounds[:,0],bounds[:,1]);pop[0]=v
 best=None;steps=0;count=0;begin=time.monotonic()
 def record(items):
  nonlocal best,steps,count
  values=[]
  with (out/'evaluations.jsonl').open('a') as f:
   for r in items:
    count+=1;steps+=r['environment_steps'];r['evaluation']=count;f.write(json.dumps(r)+'\n');values.append(r['cost'])
    if best is None or r['cost']<best['cost']:
     best=r;candidate=decode(r['x'],c,'landing');candidate['search_measurements']=r;(out/'candidate.json').write_text(json.dumps(candidate,indent=2));print(json.dumps(r),flush=True)
  return values
 with mp.get_context('spawn').Pool(16,initializer=initialize,initargs=(states,c,'landing')) as pool:
  def mapper(function,values):return record(pool.map(objective,list(values)))
  differential_evolution(lambda x:0,bounds,init=pop,seed=52,maxiter=6,workers=mapper,updating='deferred',polish=False,tol=1e-5)
 (out/'search.json').write_text(json.dumps(dict(best=best,evaluations=count,environment_steps=steps,prefix_steps_reused_from_preload_scout=True,seconds=time.monotonic()-begin),indent=2))
 context=next(s for s in states if s['job']['name']=='mu-0.8-seed-1')
 audit=run(out/'candidate.json',out/'replay',video=True,start_override=context['start'],direction=-1,landing_delta=0.)
 steps+=audit['environment_steps'];logger.metrics(steps,{k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))});logger.video(out/'replay/rollout.mp4',steps,key='trajectory/front_l',caption=f'SIMULATION fixed-active-pose landing candidate | {audit["status"]} | actual single-stage audit from integrated prefix; not a continuous-plan claim')
 logger.run.summary.update(dict(simulation_audit_status=audit['status'],gates=audit['gates'],hardware_validated=False));artifact=logger.sdk.Artifact('finish-search-'+logger.state['id'],type='trajectory-run')
 for p in out.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.jsonl'] and 'wandb' not in p.relative_to(out).parts:artifact.add_file(str(p),name=p.relative_to(out).as_posix())
 artifact.add_file(__file__,name='finish_search.py');logger.run.log_artifact(artifact);logger.finish();(out/'logging_status.json').write_text(json.dumps(dict(url=logger.state['url'],status='SDK finish completed; verify cloud'),indent=2))
if __name__=='__main__':main()
