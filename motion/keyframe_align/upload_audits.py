"""Upload saved audits/video, including failed development cases. Never runs motion."""
import argparse,json
from pathlib import Path
from training.wandb_logging import ExperimentLogger

def upload(root,mode='online'):
    root=Path(root);suite=json.loads((root/'suite.json').read_text())
    logger=ExperimentLogger(root,dict(suite,controller='deterministic-keyframes',checkpoint_step=None),
        mode=mode,name='keyframes-'+suite['source_commit'][:7])
    logger.run.summary.update({'simulation_audit_status':'passed' if suite['passed'] else 'failed',
        'hardware_validated':False,'checkpoint_applicable':False,'optimizer_updates':0,
        'handoff_status':'Simulation checks complete; see integration handoff for target-specific checks.'})
    artifact=logger.sdk.Artifact('keyframe-audits-'+logger.state['id'],type='motion-audit')
    artifact.add_file(str(root/'suite.json'),name='suite.json')
    for path in sorted(root.rglob('audit.json')):
        # Audit discovery is intentionally narrow; no credentials/calibration data.
        if 'wandb' in path.relative_to(root).parts:continue
        audit=json.loads(path.read_text());label=path.parent.relative_to(root).as_posix()
        prefix='audit' if label in suite['scenarios'] else 'development_audit'
        logger.run.summary[f'{prefix}/{label}/passed']=audit['passed']
        for leg in audit['legs']:
            for key,value in leg.items():
                if isinstance(value,(int,float,bool)):
                    logger.run.summary[f'{prefix}/{label}/{leg["leg"]}/{key}']=value
        for name in ('audit.json','config.json','trace.csv'):
            file=path.parent/name
            if file.is_file():artifact.add_file(str(file),name=file.relative_to(root).as_posix())
    video=root/'nominal/rollout.mp4'
    if video.exists():logger.video(video,0,key='motion/keyframes',caption=f'SIMULATION | deterministic keyframes | seed 0 | source {suite["source_commit"][:7]} | no neural checkpoint | suite passed={suite["passed"]}')
    logger.run.log_artifact(artifact);logger.finish()
    if mode=='online':
        remote=logger.sdk.Api().run(f'{logger.state["entity"]}/{logger.state["project"]}/{logger.state["id"]}')
        media=[f.name for f in remote.files() if f.name.startswith('media/videos/')]
        if video.exists() and not media:raise RuntimeError('Remote rollout video not verified')
        result={'url':remote.url,'video_files':media,'simulation_audit_status':remote.summary.get('simulation_audit_status')}
        (root/'upload_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--wandb',choices=['online','offline'],default='online');args=p.parse_args();upload(args.directory,args.wandb)
