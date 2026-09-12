"""Selection-only preload sweep after the held-out-08 failure; no acceptance claim."""
import json,multiprocessing as mp
from pathlib import Path
import numpy as np
from motion.inverted_triangle.robust_search import contexts,phases,integrate,initialize,evaluate
from motion.inverted_triangle.core import Robot,PROX

def main():
 out=Path('runs/inverted_triangle/preload-scout');out.mkdir(exist_ok=False)
 c,states,steps=contexts(Path('runs/inverted_triangle/combined-plan-v1/plan.json'),3,True)
 directory=Path('runs/inverted_triangle/held-out-v1/held-out-08/04-front_l')
 a=json.loads((directory/'audit.json').read_text());start=json.loads((directory/'start_state.json').read_text());r=Robot(a['friction'],a['effective_dynamics']);home=r.restore(start);prefix=[]
 for item in phases(home,c):
  if item[0]=='land':break
  prefix.append(item)
 home,n=integrate(r,home,prefix);steps+=n
 states.append(dict(job=dict(name='former-held-out-08-now-diagnostic',friction=a['friction'],scenario=a['scenario']),start=start,state=r.snapshot(home)))
 (out/'contexts.json').write_text(json.dumps(states,indent=2))
 x=np.r_[np.asarray(c['touchdown_pose'])[3:5],np.asarray(c['landing_pose'])[PROX]]
 vectors=[]
 for delta in [.0,.05,.10,.15,.20]:
  v=x.copy();v[1]+=delta;vectors.append(v)
 with mp.get_context('spawn').Pool(5,initializer=initialize,initargs=(states,c,'landing')) as pool:results=pool.map(evaluate,vectors)
 (out/'results.json').write_text(json.dumps(dict(selection_only=True,wandb_mode='disabled for local diagnostic sweep; subsequent full audits/videos online',original_held_out_case_now_used_for_diagnosis=True,prefix_steps=steps,offsets_rad=[0,.05,.10,.15,.20],results=results),indent=2))
 for delta,result in zip([0,.05,.10,.15,.20],results):print(delta,result['cost'],[(v['name'],v['descent_m_s'],v['hub_error_rad'],v['floor_N']) for v in result['conditions']],flush=True)
if __name__=='__main__':main()
