import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
from motion.inverted_triangle.core import provenance,versions
p=argparse.ArgumentParser();p.add_argument('--configs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--controller',type=Path);p.add_argument('--walking',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);(a.output/'configs').mkdir()
configs=json.loads(a.configs.read_text());controller=json.loads(a.controller.read_text()) if a.controller else {}
configs=[dict(c,**controller) for c in configs]
(a.output/'provenance.json').write_text(json.dumps(dict(source_hashes=provenance(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),environment=versions(),configs=configs,definition_sha256=hashlib.sha256(a.configs.read_bytes()).hexdigest(),walking=a.walking,driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
def run(c):
 cfg=a.output/'configs'/(c['name']+'.json');cfg.write_text(json.dumps(c if a.walking else [c]))
 cmd=[sys.executable,'-m','motion.inverted_triangle.'+('policy_handoff' if a.walking else 'roll_to_stand'),'--config',str(cfg),'--output',str(a.output/c['name'])]
 if not a.walking:cmd+=['--cad','--video','--wandb','online']
 with (a.output/(c['name']+'.log')).open('w') as f:ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl')).returncode
 path=a.output/c['name']/('handoff' if a.walking else c['name'])/'audit.json'
 return dict(name=c['name'],returncode=ret,audit=json.loads(path.read_text()) if path.exists() else None)
results=[]
with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
 for result in pool.map(run,configs):
  results.append(result);(a.output/'results.json').write_text(json.dumps(results,indent=2));print(result['name'],result['returncode'],None if result['audit'] is None else [k for k,v in result['audit']['gates'].items() if v is not True],flush=True)
if any(r['returncode'] for r in results):raise SystemExit(1)
