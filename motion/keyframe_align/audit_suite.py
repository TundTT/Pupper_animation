"""CPU-only acceptance matrix. No training or hardware connection."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from .simulate import run
from .native import CONFIG

def provenance():
    root=Path(__file__).resolve().parents[2]
    paths=list((root/'motion/keyframe_align').glob('*'))
    paths += list((root/'ros2_ws/src/neural_controller/include/neural_controller').glob('wheel_align*.hpp'))
    paths += [root/'training/wheel_align'/f for f in ('model.xml','geometry.json','geometry.py','configs.py')]
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    git='git.exe' if os.name!='nt' and str(root).startswith('/mnt/c/') else 'git'
    return dict(source_commit=subprocess.check_output([git,'rev-parse','HEAD'],text=True).strip(),
      source_dirty=bool(subprocess.check_output([git,'status','--porcelain'],text=True).strip()),source_sha256=files,
      controller_binary_sha256=hashlib.sha256(Path(os.environ['KEYFRAME_ALIGN_LIBRARY']).read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--quick',action='store_true');p.add_argument('--video',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());initial=provenance()
    cases=[('nominal',{})]
    cases += [(f'bias-{a:+.2f}-{b:+.2f}',dict(tilt=(a,b))) for a in [-.04,0,.04] for b in [-.04,0,.04] if a or b]
    cases += [(f'friction-seed-{seed}',dict(seed=seed,friction=.06)) for seed in range(1,5)]
    cases += [(f'combined-{seed}',dict(seed=seed,friction=.06,tilt=tilt)) for seed,tilt in
              [(5,(-.04,-.04)),(6,(-.04,.04)),(7,(.04,-.04)),(8,(.04,.04))]]
    cases += [(f'cancel-phase-{phase}',dict(interrupt_phase=phase)) for phase in range(1,6)]
    if args.quick:cases=[cases[0],cases[8],cases[9],cases[-1]]
    results={}
    for label,options in cases:
        folder=args.output/label;folder.mkdir()
        video=str(folder/'rollout.mp4') if args.video and label=='nominal' else None
        report,trace=run(config,video=video,**options);report.update(initial)
        report['config_sha256']=hashlib.sha256(args.config.read_bytes()).hexdigest()
        report['provenance_unchanged']=initial==provenance()
        if not report['provenance_unchanged']:report['passed']=False
        (folder/'audit.json').write_text(json.dumps(report,indent=2));(folder/'config.json').write_text(json.dumps(config,indent=2))
        np.savetxt(folder/'trace.csv',trace,delimiter=',')
        results[label]=report['passed']
        print(label,report['passed'],[(x['leg'],round(x['final_error_deg'],2),x['timed_out']) for x in report['legs']],flush=True)
    summary=dict(passed=all(results.values()),scenarios=results,hardware_validated=False,optimizer_updates=0,**initial)
    (args.output/'suite.json').write_text(json.dumps(summary,indent=2));raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__':main()
