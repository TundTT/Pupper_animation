"""Audit a fixed four-flip plan by propagating the actual full physics state."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from .core import LEGS,provenance,HERE
from .simulate import run
from training.wandb_logging import ExperimentLogger


def replay(plan_path,output,friction=.8,seed=0,video=True,scenario=None):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    stages=plan['stages']
    if len(stages)!=4 or sorted(s['leg'] for s in stages)!=sorted(LEGS):raise ValueError('A full plan must flip each of four legs exactly once')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state=None;audits=[]
    for i,stage in enumerate(stages):
        candidate=plan_path.parent/stage['candidate']
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=stage['candidate_sha256']:raise ValueError('Plan candidate hash mismatch')
        if i==0 and 'start_state' in json.loads(candidate.read_text()):raise ValueError('A full plan must start from the initial stance, not an embedded continuation state')
        if json.loads(candidate.read_text())['leg']!=stage['leg']:raise ValueError('Plan leg mismatch')
        directory=output/f'{i+1:02d}-{stage["leg"]}'
        audit=run(candidate,directory,video=video,friction=friction,landing_delta=stage['landing_delta'],direction=stage['direction'],seed=seed if i==0 else 0,start_override=state,scenario=scenario)
        audits.append(audit)
        if audit['status']!='PASS_NOMINAL_SINGLE_FLIP':break
        state=json.loads((directory/'end_state.json').read_text())
    final_tips=bool(len(audits)==4 and all(abs(z)<.003 for z in audits[-1]['final_tip_bottom_m']))
    passed=len(audits)==4 and all(a['status']=='PASS_NOMINAL_SINGLE_FLIP' for a in audits) and final_tips
    result=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1]).decode().strip(),scenario=scenario or {},status='PASS_CONTINUOUS_FOUR_FLIPS' if passed else 'FAILED',simulation_only=True,hardware_validated=False,seed=seed,friction=friction,completed_flips=sum(a['status']=='PASS_NOMINAL_SINGLE_FLIP' for a in audits),final_four_tips=final_tips,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),source_hashes=provenance(),stage_status=[a['status'] for a in audits],environment_steps=sum(a['environment_steps'] for a in audits))
    if video:
        import imageio_ffmpeg
        clips=sorted(output.glob('*/rollout.mp4'))
        listing=output/'video_concat.txt'
        listing.write_text(''.join("file '"+p.resolve().as_posix().replace("'", "'\\''")+"'\n" for p in clips))
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(output/'rollout.mp4')],check=True)
        result['rollout_sha256']=hashlib.sha256((output/'rollout.mp4').read_bytes()).hexdigest()
    (output/'sequence_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--friction',type=float,default=.8);p.add_argument('--seed',type=int,default=0);p.add_argument('--no-video',action='store_true')
    p.add_argument('--scenario',type=Path)
    p.add_argument('--wandb',choices=['online','offline','disabled'],default='online')
    a=p.parse_args()
    if a.no_video and a.wandb!='disabled':p.error('Logged audits require actual rollout video; no-video is for explicitly disabled local checks')
    scenario=json.loads(a.scenario.read_text()) if a.scenario else None
    result=replay(a.plan,a.output,a.friction,a.seed,not a.no_video,scenario)
    logging=dict(mode=a.wandb,status='not attempted');logger=None
    try:
        if a.wandb!='disabled':
            logger=ExperimentLogger(a.output.resolve(),dict(task='continuous fixed-plan robustness replay',**result),mode=a.wandb)
            logger.metrics(result['environment_steps'],{k:v for k,v in result.items() if isinstance(v,(int,float,bool,str))})
            steps=0
            for audit_path in sorted(a.output.glob('*/audit.json')):
                audit=json.loads(audit_path.read_text());steps+=audit['environment_steps']
                logger.video(audit_path.parent/'rollout.mp4',steps,key='trajectory/'+audit['leg'],caption=f'SIMULATION fixed plan {result["plan_sha256"][:12]} | seed {a.seed}, friction {a.friction} | {audit["leg"]}: {audit["status"]} | early termination {audit["early_termination"]}; no trained checkpoint')
            logger.video(a.output/'rollout.mp4',result['environment_steps'],key='trajectory/continuous_sequence',caption=f'SIMULATION fixed plan {result["plan_sha256"]} | seed {a.seed}, friction {a.friction} | {result["status"]}, {result["completed_flips"]}/4 accepted flips | actual preceding states and velocities propagated; no trained checkpoint')
            logger.run.summary.update(dict(simulation_audit_status=result['status'],hardware_validated=False))
            artifact=logger.sdk.Artifact('inverted-sequence-'+logger.state['id'],type='trajectory-run')
            for path in sorted(a.output.rglob('*.json')):
                if 'wandb' not in path.relative_to(a.output).parts:artifact.add_file(str(path),name=path.relative_to(a.output).as_posix())
            logger.run.log_artifact(artifact);logger.finish()
            logging.update(status='SDK finish completed; verify cloud Media/artifact' if a.wandb=='online' else 'offline only; not uploaded',url=logger.state.get('url'))
    except Exception:
        logging['status']='failed or upload unverified'
        raise
    finally:
        (a.output/'logging_status.json').write_text(json.dumps(logging,indent=2))

if __name__=='__main__':main()
