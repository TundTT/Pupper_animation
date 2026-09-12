"""CPU simulation acceptance matrix; each scenario preserves its report and trace."""
import argparse,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from .simulate import run
from .native import CONFIG
from .provenance import provenance

def evaluate(job):
 label,options,config,folder,video,initial,digest=job
 folder=Path(folder);folder.mkdir();(folder/'config.json').write_text(json.dumps(config,indent=2))
 try:
  report,trace=run(config,video=str(folder/'rollout.mp4') if video else None,**options)
  report.update(initial);report['config_sha256']=digest;report['provenance_unchanged']=initial==provenance()
  if not report['provenance_unchanged']:report['passed']=False
  np.savetxt(folder/'trace.csv',trace,delimiter=',')
 except Exception as exc:
  report=dict(passed=False,error=repr(exc),options=options,**initial)
 (folder/'audit.json').write_text(json.dumps(report,indent=2))
 return label,report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=CONFIG)
 p.add_argument('--quick',action='store_true');p.add_argument('--video',action='store_true');p.add_argument('--workers',type=int,default=1)
 args=p.parse_args();assert 1<=args.workers<=4;args.output.mkdir(parents=True,exist_ok=False)
 config=json.loads(args.config.read_text());initial=provenance();digest=hashlib.sha256(args.config.read_bytes()).hexdigest()
 cases=[('nominal',{})]
 cases += [(f'bias-{a:+.2f}-{b:+.2f}',dict(tilt=(a,b))) for a in [-.04,0,.04] for b in [-.04,0,.04] if a or b]
 cases += [(f'friction-seed-{seed}',dict(seed=seed,friction=.06)) for seed in range(1,5)]
 cases += [(f'combined-{seed}',dict(seed=seed,friction=.06,tilt=tilt)) for seed,tilt in [(5,(-.04,-.04)),(6,(-.04,.04)),(7,(.04,-.04)),(8,(.04,.04))]]
 cases += [(f'cancel-phase-{phase}',dict(interrupt_phase=phase)) for phase in range(1,6)]
 cases += [('payload-heavy',dict(payload_scale=1.2)),('payload-light',dict(payload_scale=.8)),('wheel-mass-20g',dict(added_wheel_mass=.02)),('heavy-combined',dict(payload_scale=1.2,added_wheel_mass=.02,seed=9,friction=.06))]
 if args.quick:cases=[cases[0],cases[8],cases[9],cases[21]]
 jobs=[(label,options,config,str(args.output/label),args.video and label=='nominal',initial,digest) for label,options in cases]
 results={}
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for future in as_completed([pool.submit(evaluate,job) for job in jobs]):
   label,report=future.result();results[label]=report['passed']
   print(label,report['passed'],report.get('error',[(x['leg'],round(x['final_error_deg'],2),x['timed_out']) for x in report.get('legs',[])]),flush=True)
 summary=dict(passed=all(results.values()),scenarios=results,hardware_validated=False,optimizer_updates=0,**initial)
 (args.output/'suite.json').write_text(json.dumps(summary,indent=2));raise SystemExit(0 if summary['passed'] else 1)
if __name__=='__main__':main()
