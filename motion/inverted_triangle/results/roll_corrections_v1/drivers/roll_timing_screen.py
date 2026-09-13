import json,concurrent.futures,hashlib
from pathlib import Path
from motion.inverted_triangle.core import HERE,provenance,versions
from motion.inverted_triangle.roll_to_stand import probe
root=Path('runs/roll_to_stand/timing-screen-v1')
def run(c):return probe(c,root/c['name'],cad=False,video=False)
if __name__=='__main__':
 root.mkdir(exist_ok=False);allcases=json.loads((HERE/'roll_validation/original_families.json').read_text());names=['mu-1.0-seed-2','mu-1.0-seed-3','mass-high','gains-low','combined-seed-22'];cases=[c for c in allcases if c['name'] in names]+json.loads((HERE/'formation_cases.json').read_text())[:1];controller=json.loads((HERE/'roll_validation/selected_controller.json').read_text());configs=[]
 variants=[dict(seconds=t,splay_power=p,shoulder_bump=b) for t,p,b in [(16,1,0),(20,1,0),(12,.7,0),(16,.7,0),(12,1.3,0),(16,1.3,0),(16,1,-.1),(16,1,.1)]]
 for k,v in enumerate(variants):configs.extend([dict(c,**controller,**v,name=f'v{k}-'+c['name']) for c in cases])
 (root/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(8) as pool:reports=list(pool.map(run,configs))
 (root/'summary.json').write_text(json.dumps(reports,indent=2))
 for k in range(len(variants)):
  batch=reports[k*len(cases):(k+1)*len(cases)];print('VARIANT',k,'PASS',sum(all(v for key,v in r['gates'].items() if 'cad' not in key) for r in batch),[r['config']['name']+':'+','.join(key for key,v in r['gates'].items() if v is False) for r in batch],flush=True)
