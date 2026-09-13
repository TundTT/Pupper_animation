import json,concurrent.futures,hashlib,argparse
from pathlib import Path
from motion.inverted_triangle.core import provenance,versions
from motion.inverted_triangle.roll_to_stand import probe
p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
def run(c):return probe(c,a.output/c['name'],cad=False,video=False)
if __name__=='__main__':
 a.output.mkdir(exist_ok=False);configs=json.loads(a.configs.read_text())
 (a.output/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:reports=list(pool.map(run,configs))
 (a.output/'summary.json').write_text(json.dumps(reports,indent=2))
 for r in reports:print('RESULT',r['config']['name'],[k for k,v in r['gates'].items() if v is False],flush=True)
