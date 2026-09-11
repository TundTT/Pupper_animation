"""Select the newest checkpoint that passes balanced task audits, never reward alone."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from . import configs as c


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenarios(stage):
    if stage in ('foundation','single'):
        return [(stage,20260910,stage=='foundation',False),
                (stage+'-holdout',20260912,stage=='foundation',False)]
    if stage=='sequence':
        return [('nominal',20260910,True,False),('randomized',20260910,False,False),
                ('interrupted',20260911,False,True)]
    raise ValueError('Unknown stage')


def select(run_dir,auditor,hashes,envs=64):
    if envs<64 or envs%4:raise ValueError('Selection requires at least 64 environments, balanced across four wheels')
    run_dir=Path(run_dir).resolve();config=json.loads((run_dir/'config.json').read_text())
    if config['motion_contract_version']!=c.MOTION_VERSION or config['source_hashes']!=hashes:
        raise ValueError('Use the exact current-contract training checkout for selection')
    stage=config['curriculum_stage'];candidates=[]
    for path in run_dir.glob('params_*'):
        if path.is_file() and path.name[7:].isdigit() and int(path.name[7:])>0:
            candidates.append((int(path.name[7:]),path))
    if not candidates:raise ValueError('No positive-step checkpoints; params_0 is not a trained candidate')
    report=dict(status='failed',stage=stage,envs=envs,source_hashes=hashes,candidates=[])
    review=run_dir/'selection';review.mkdir(exist_ok=True)
    for step,path in sorted(candidates,reverse=True):
        sha=digest(path);directory=review/f'step-{step}';directory.mkdir(exist_ok=True)
        entry=dict(step=step,checkpoint_sha256=sha,audits=[],passed=True)
        for label,seed,nominal,interrupt in scenarios(stage):
            output=directory/f'audit-{label}.json'
            if output.exists():
                result=json.loads(output.read_text())
                expected=(sha,envs,seed,nominal,interrupt,stage)
                actual=tuple(result.get(k) for k in ('checkpoint_sha256','envs','seed','nominal','interrupt','stage'))
                if actual!=expected:raise ValueError('Cached audit differs; preserve it and use a separate selection directory')
            else:
                result=auditor(path,stage=stage,envs=envs,seed=seed,nominal=nominal,interrupt=interrupt)
                if result['checkpoint_sha256']!=sha:raise ValueError('Checkpoint changed during audit')
                output.write_text(json.dumps(result,indent=2)+'\n')
            entry['audits'].append(str(output.relative_to(run_dir)))
            if not result['passes_simulation_gate']:
                entry['passed']=False;break
        report['candidates'].append(entry)
        (review/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Checkpoint {step}: '+('PASS' if entry['passed'] else 'FAIL'),flush=True)
        if entry['passed']:
            selected=run_dir/'selected';selected.mkdir(exist_ok=True)
            target=selected/'mjx_params'
            if target.exists() and digest(target)!=sha:raise ValueError('Refusing to overwrite a different selected checkpoint')
            if not target.exists():shutil.copyfile(path,target)
            selected_config=dict(config,selected_checkpoint_step=step,selected_checkpoint_sha256=sha)
            (selected/'config.json').write_text(json.dumps(selected_config,indent=2)+'\n')
            report.update(status='passed',params=str(target),step=step,checkpoint_sha256=sha,audits=entry['audits'])
            (run_dir/'selected_checkpoint.json').write_text(json.dumps(report,indent=2)+'\n')
            (review/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
    if (run_dir/'selected_checkpoint.json').exists():
        raise ValueError('Selection failed but an older selection manifest exists; inspect it before proceeding')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--envs',type=int,default=64);args=p.parse_args()
    from .evaluate import audit,load_policy
    from .train import source_hashes
    from .env import AlignEnv
    from .policy_video import record_policy
    from training.wandb_logging import ExperimentLogger
    directory=args.run_dir.resolve();config=json.loads((directory/'config.json').read_text())
    logger=None;identity=directory/'wandb_run.json'
    if identity.exists():
        saved=json.loads(identity.read_text())
        if saved['mode']=='online':
            logger=ExperimentLogger(directory,config,entity=saved['entity'],project=saved['project'],mode='online')
        else:print('Offline run: selection remains local until explicitly uploaded',flush=True)
    success=False
    try:
        report=select(directory,audit,source_hashes(),args.envs)
        if logger:
            logger.run.summary['selected_checkpoint_simulation_status']=report['status']
            logger.run.summary['selected_checkpoint_stage']=report['stage']
            logger.run.summary['hardware_validated']=False
        if report['status']!='passed':raise RuntimeError('No checkpoint passed; stop the curriculum and report selection/report.json')
        stage=report['stage'];policy=load_policy(report['params'],AlignEnv(noise=False,stage=stage))
        scopes=[stage] if stage=='sequence' else [stage,'sequence']
        for scope in scopes:
            path=record_policy(policy,directory/'selected'/f'policy-{scope}.mp4',training_step=report['step'],stage=scope)
            if logger:logger.video(path,report['step'],key='policy/selected_'+scope,caption=f'Selected checkpoint {report["step"]}; {scope} nominal diagnostic')
        if logger:
            logger.run.summary['selected_checkpoint_step']=report['step']
            logger.run.summary['selected_checkpoint_sha256']=report['checkpoint_sha256']
            logger.run.summary['handoff_status']='selected checkpoint passed '+stage+' audits; hardware not validated'
        success=True
        print(json.dumps({k:report[k] for k in ('status','stage','step','params','checkpoint_sha256')},indent=2))
    finally:
        if logger:
            logger.run.summary['selection_handoff_complete']=success
            try:logger.artifacts()
            finally:logger.finish(exit_code=0 if success else 1)


if __name__=='__main__':main()
