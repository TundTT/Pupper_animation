"""Explicit local physics search for rolling support-transfer tracking."""
import json,concurrent.futures,hashlib
from pathlib import Path
from motion.inverted_triangle.core import HERE,provenance,versions,KP,KD
import motion.inverted_triangle.roll_to_stand as module
original_initialize=module.initialize
root=Path('runs/roll_to_stand/damping-screen-v1')
def run(c):
 def initialize(*args,**kwargs):
  r,home,goal=original_initialize(*args,**kwargs);tick=r.tick
  def controlled(target,kp=KP,kd=KD,feedforward=None):
   scale=c['roll_damping_scale'] if r.d.time<2+c['seconds'] else 1.
   return tick(target,kp=kp,kd=kd*scale,feedforward=feedforward)
  r.tick=controlled;return r,home,goal
 module.initialize=initialize
 return module.probe(c,root/c['name'],cad=False,video=False)
if __name__=='__main__':
 root.mkdir(exist_ok=False);allcases=json.loads((HERE/'roll_validation/original_families.json').read_text());names=['mu-1.0-seed-3']+[f'combined-seed-{i}' for i in range(22,27)];cases=[c for c in allcases if c['name'] in names];formations=json.loads((HERE/'formation_cases.json').read_text());cases+=[formations[i] for i in [0,1,4]];controller=json.loads((HERE/'roll_validation/selected_controller.json').read_text());configs=[]
 variants=[dict(seconds=t,roll_damping_scale=d) for t in [12.,16.] for d in [1.5,2.,3.]]
 for k,v in enumerate(variants):configs.extend([dict(c,**controller,**v,name=f'v{k}-'+c['name']) for c in cases])
 (root/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),control_change='Multiply KD only during initial hold and rolling; normal KD restored for feedback settling',environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(8) as pool:reports=list(pool.map(run,configs))
 (root/'summary.json').write_text(json.dumps(reports,indent=2))
 for k in range(len(variants)):
  batch=reports[k*len(cases):(k+1)*len(cases)];print('VARIANT',k,'PASS',sum(all(v for key,v in r['gates'].items() if 'cad' not in key) for r in batch),[r['config']['name']+':'+','.join(key for key,v in r['gates'].items() if v is False) for r in batch],flush=True)
