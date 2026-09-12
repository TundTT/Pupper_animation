import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path('runs/inverted_triangle/friction-v1');root.mkdir(exist_ok=False)
config=dict(plan='runs/inverted_triangle/fixed-plan-v1/plan.json',frictions=[.5,.65,.8,1.],seeds=[1,2,3,4,5],initial_proximal_perturbation_rad=[-.01,.01],simulation_only=True,workers=12)
(root/'config.json').write_text(json.dumps(config,indent=2))
def evaluate(condition):
 mu,seed=condition;name=f'mu-{mu}-seed-{seed}'
 env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 with (root/(name+'.log')).open('w') as log:
  result=subprocess.run([sys.executable,'-m','motion.inverted_triangle.sequence','--plan',config['plan'],'--output',str(root/name),'--friction',str(mu),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,env=env)
 audit=root/name/'sequence_audit.json'
 return dict(name=name,returncode=result.returncode,audit=json.loads(audit.read_text()) if audit.exists() else None)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=config['workers']) as pool:
 for result in pool.map(evaluate,[(mu,seed) for mu in config['frictions'] for seed in config['seeds']]):
  results.append(result);(root/'results.json').write_text(json.dumps(results,indent=2));print(result['name'],result['returncode'],result['audit']['status'] if result['audit'] else 'ERROR',flush=True)
