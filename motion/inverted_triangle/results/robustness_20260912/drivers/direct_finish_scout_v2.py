"""Local physics selection: retain support targets through the four-tip finish."""
import json,multiprocessing as mp
from pathlib import Path
import numpy as np
from motion.inverted_triangle.robust_search import initialize,evaluate,decode
from motion.inverted_triangle.core import PROX

def main():
 out=Path('runs/inverted_triangle/direct-finish-scout-v2');out.mkdir(exist_ok=False)
 c=json.loads(Path('runs/inverted_triangle/combined-plan-v1/04-front_l.json').read_text());states=json.loads(Path('runs/inverted_triangle/preload-scout/contexts.json').read_text())
 vectors=[]
 for hip in [-1.08,-.98,-.88]:
  for knee in [-.4,-.3,-.2]:
   touch=np.array(c['touchdown_pose']);touch[3:5]=[hip,knee];vectors.append(np.r_[touch[3:5],touch[PROX]])
 with mp.get_context('spawn').Pool(12,initializer=initialize,initargs=(states,c,'landing')) as pool:results=pool.map(evaluate,vectors)
 report=dict(selection_only=True,wandb_mode='disabled local diagnostic; full acceptance audits online',results=results)
 (out/'results.json').write_text(json.dumps(report,indent=2))
 best=min(results,key=lambda r:r['cost']);(out/'candidate.json').write_text(json.dumps(decode(best['x'],c,'landing'),indent=2))
 for r in results:print(r['x'][:2],round(r['cost'],3),[(v['name'],round(v['descent_m_s']*1000,1),round(min(v['final_force_N']),2),round(v['hub_error_rad'],3)) for v in r['conditions']],flush=True)
if __name__=='__main__':main()
