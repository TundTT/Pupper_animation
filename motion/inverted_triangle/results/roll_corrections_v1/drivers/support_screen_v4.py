import json,concurrent.futures,hashlib
from pathlib import Path
from motion.inverted_triangle.core import HERE,provenance,versions
from motion.inverted_triangle.roll_to_stand import probe
root=Path('runs/roll_to_stand/support-screen-v4')
def run(c):return probe(c,root/c['name'],cad=False,video=False)
if __name__=='__main__':
 root.mkdir(exist_ok=False);cases=json.loads((HERE/'formation_cases.json').read_text());configs=[]
 variants=[dict(estimate_mode='effort_norm',force_gain=f,reference_cap=cap,integral_gain=.5) for f in [.003,.008,.016] for cap in [.075,.09]]
 for k,v in enumerate(variants):configs.extend([dict(c,name=f'v{k}-'+c['name'],hold_seconds=25.,balanced_support=v) for c in cases])
 (root/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(12) as pool:reports=list(pool.map(run,configs))
 (root/'summary.json').write_text(json.dumps(reports,indent=2))
 for k in range(len(variants)):
  batch=reports[k*7:k*7+7];print('VARIANT',k,'PASS',sum(all(v for key,v in r['gates'].items() if 'cad' not in key) for r in batch),[r['config']['name']+':'+','.join(key for key,v in r['gates'].items() if v is False) for r in batch],flush=True)
