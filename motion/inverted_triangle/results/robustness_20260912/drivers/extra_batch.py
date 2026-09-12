"""Replay an explicit immutable scenario manifest with actual online videos."""
import concurrent.futures,hashlib,json,subprocess,sys,os
from pathlib import Path
from motion.inverted_triangle.core import provenance,versions
plan=Path(sys.argv[1]).resolve();manifest=Path(sys.argv[2]);root=Path(sys.argv[3]);root.mkdir(exist_ok=False);(root/'scenarios').mkdir()
jobs=json.loads(manifest.read_text())['conditions'];config=dict(plan=str(plan),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),conditions=jobs,source_hashes=provenance(),environment=versions(),simulation_only=True)
(root/'config.json').write_text(json.dumps(config,indent=2))
def go(j):
 path=root/'scenarios'/(j['name']+'.json');path.write_text(json.dumps(j['scenario'],indent=2))
 with (root/(j['name']+'.log')).open('w') as log:
  p=subprocess.run([sys.executable,'-m','motion.inverted_triangle.sequence','--plan',str(plan),'--output',str(root/j['name']),'--friction',str(j['friction']),'--seed',str(j['seed']),'--scenario',str(path)],stdout=log,stderr=subprocess.STDOUT)
 a=root/j['name']/'sequence_audit.json';l=root/j['name']/'logging_status.json'
 return dict(**j,returncode=p.returncode,audit=json.loads(a.read_text()) if a.exists() else None,logging=json.loads(l.read_text()) if l.exists() else None)
rows=[]
with concurrent.futures.ThreadPoolExecutor(12) as pool:
 for result in pool.map(go,jobs):
  rows.append(result);(root/'results.json').write_text(json.dumps(rows,indent=2));print(result['name'],result['audit']['status'] if result['audit'] else 'ERROR',flush=True)
summary=dict(plan_sha256=config['plan_sha256'],conditions=len(rows),passed=sum(r['audit'] and r['audit']['status']=='PASS_CONTINUOUS_FOUR_FLIPS' for r in rows),process_errors=sum(r['returncode']!=0 for r in rows),simulation_only=True)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
