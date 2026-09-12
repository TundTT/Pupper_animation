"""PC entry point: whole-flip search, CAD audit, state continuation and W&B."""
import argparse,hashlib,json,subprocess,traceback
from pathlib import Path
from .core import HERE,LEGS,provenance,versions
from .search_flip import optimize
from .simulate import run
from training.wandb_logging import ExperimentLogger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--order',nargs='+',choices=LEGS,default=['back_r','front_l','back_l','front_r'])
    p.add_argument('--maxiter',type=int,default=20);p.add_argument('--method',choices=['powell','de'],default='powell');p.add_argument('--radius',type=float,default=.4);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--direction',type=int,choices=[-1,1],default=-1);p.add_argument('--candidate',type=Path);p.add_argument('--audit-only',action='store_true');p.add_argument('--wandb',choices=['online','offline','disabled'],default='online')
    a=p.parse_args()
    if len(set(a.order))!=len(a.order):raise ValueError('Each leg can appear only once')
    if a.audit_only and (not a.candidate or len(a.order)!=1):raise ValueError('audit-only requires one leg and a candidate')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    config=dict(task='post-cooled inverted triangle trajectory search',**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[1]).decode().strip(),source_hashes=provenance(),environment=versions(),simulation_only=True,hardware_validated=False)
    (output/'config.json').write_text(json.dumps(config,indent=2))
    logger=None;state=None;stages=[];total_steps=0;exit_code=0;status='INCOMPLETE';error=None
    try:
        if a.wandb!='disabled':logger=ExperimentLogger(output,config,mode=a.wandb)
        for i,leg in enumerate(a.order):
            directory=output/f'{i+1:02d}-{leg}';directory.mkdir()
            if a.audit_only:
                candidate=directory/'candidate.json';candidate.write_bytes(a.candidate.read_bytes())
            else:
                report=optimize(directory/'search',leg,a.candidate if i==0 else None,state,a.maxiter,a.method,a.radius,a.seed,a.direction)
                total_steps+=report['environment_steps'];candidate=directory/'search/best_flip.json'
            selected=json.loads(candidate.read_text())
            if selected['leg']!=leg:raise ValueError('Candidate leg does not match experiment order')
            delta=selected.get('landing_delta',.3);direction=selected.get('direction',a.direction)
            audit=run(candidate,directory/'replay',video=True,landing_delta=delta,direction=direction,start_override=state)
            total_steps+=audit['environment_steps']
            if logger:
                logger.metrics(total_steps,{f'flip/{leg}/'+k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))})
                logger.video(directory/'replay/rollout.mp4',total_steps,key=f'trajectory/{leg}/rollout',caption=f'SIMULATION trajectory candidate {audit["candidate_sha256"][:12]} | search seed {a.seed}, replay seed {audit["seed"]}, env steps {total_steps} | {audit["status"]} | early termination {audit["early_termination"]}; no trained checkpoint')
                logger.run.summary[f'flip/{leg}/gates']=audit['gates']
            stages.append(dict(leg=leg,candidate=candidate.relative_to(output).as_posix(),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),landing_delta=delta,direction=direction,status=audit['status']))
            (output/'plan.json').write_text(json.dumps(dict(stages=stages,simulation_only=True),indent=2))
            if audit['status']!='PASS_NOMINAL_SINGLE_FLIP':status='FAILED';break
            state=json.loads((directory/'replay/end_state.json').read_text())
        else:
            status='PASS_CONTINUOUS_FOUR_FLIPS' if len(stages)==4 and all(abs(z)<.003 for z in audit['final_tip_bottom_m']) else 'PASS_PARTIAL_SEQUENCE'
    except Exception:
        error=traceback.format_exc();(output/'error.txt').write_text(error);status='ERROR';exit_code=1
    finally:
        summary=dict(status=status,environment_steps=total_steps,simulation_only=True,hardware_validated=False,robustness_validated=False,wandb_mode=a.wandb,wandb_upload_status='not attempted' if a.wandb=='disabled' else 'pending',stages=stages,error=error)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        if logger:
            try:
                logger.run.summary.update(dict(simulation_audit_status=status,hardware_validated=False,environment_steps=total_steps))
                artifact=logger.sdk.Artifact('inverted-triangle-'+logger.state['id'],type='trajectory-run')
                for path in sorted(output.rglob('*.json')):
                    if 'wandb' not in path.relative_to(output).parts:artifact.add_file(str(path),name=path.relative_to(output).as_posix())
                for path in sorted(output.rglob('*.jsonl')):
                    if 'wandb' not in path.relative_to(output).parts:artifact.add_file(str(path),name=path.relative_to(output).as_posix())
                logger.run.log_artifact(artifact);logger.finish(exit_code)
                summary['wandb_upload_status']='SDK online finish completed; verify run Media/artifact in cloud' if a.wandb=='online' else 'offline only; not uploaded'
                summary['wandb_url']=logger.state.get('url')
            except Exception:
                summary['wandb_upload_status']='logging failed or upload unverified';summary['wandb_error']=traceback.format_exc();exit_code=1
        elif a.wandb!='disabled':summary['wandb_upload_status']='initialization failed; not confirmed uploaded'
        (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if exit_code:raise SystemExit(exit_code)

if __name__=='__main__':main()
