import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--terminal',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--cad',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
plan=json.loads(a.plan.read_text());(a.output/'plan.json').write_text(json.dumps(plan,indent=2))
(a.output/'driver.json').write_text(json.dumps(dict(driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),terminal_sha256=hashlib.sha256(a.terminal.read_bytes()).hexdigest(),cad=a.cad),indent=2))
def run(c):
 name=c['name'];cfg=a.output/(name+'.json');cfg.write_text(json.dumps(c['config'],indent=2))
 cmd=[sys.executable,'-m','motion.inverted_triangle.handoff_probe','--config',str(cfg),'--terminal',str(a.terminal),'--output',str(a.output/name)]
 if not a.cad:cmd+=['--no-cad']
 with (a.output/(name+'.log')).open('w') as f:ret=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 path=a.output/name/'report.json';report=json.loads(path.read_text()) if path.exists() else None
 row=dict(name=name,role=c.get('role','diagnostic'),returncode=ret,report=str(path))
 if report:row.update({k:report[k] for k in ['engineering_gates','engineering_pass','early_termination','environment_steps','wandb_url'] if k in report})
 return row
index=[]
with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
 for row in pool.map(run,plan):
  index.append(row);(a.output/'index.json').write_text(json.dumps(index,indent=2));print(json.dumps(row),flush=True)
raise SystemExit(0 if all(r.get('engineering_pass',False) for r in index) else 1)
