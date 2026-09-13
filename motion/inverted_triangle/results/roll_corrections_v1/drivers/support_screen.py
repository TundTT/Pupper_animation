"""Explicit local physics optimization screen; all failures retained for upload."""
import json,concurrent.futures
from pathlib import Path
from motion.inverted_triangle.core import HERE,provenance,versions
from motion.inverted_triangle.roll_to_stand import probe
root=Path('runs/roll_to_stand/support-screen-v1')
def run(c):return probe(c,root/c['name'],cad=False,video=False)
if __name__=='__main__':
 root.mkdir(exist_ok=False);cases=json.loads((HERE/'formation_cases.json').read_text());configs=[]
 variants=[dict(force_gain=f,force_target_N=t,integral_gain=i) for f,t,i in [(0,4,.35),(.008,4,.35),(.016,4,.35),(.008,8,.35),(.016,8,.35),(.008,4,.7),(0,4,.7),(.004,4,.15)]]
 for k,v in enumerate(variants):
  configs.extend([dict(c,name=f'v{k}-'+c['name'],hold_seconds=12.,adaptive_support=v) for c in cases])
 (root/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(12) as pool:reports=list(pool.map(run,configs))
 (root/'summary.json').write_text(json.dumps(reports,indent=2))
 for k in range(len(variants)):
  batch=reports[k*7:k*7+7];print('VARIANT',k,'PASS',sum(all(v for key,v in r['gates'].items() if 'cad' not in key) for r in batch),[r['config']['name']+':'+','.join(key for key,v in r['gates'].items() if v is False) for r in batch],flush=True)
