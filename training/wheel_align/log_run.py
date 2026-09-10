"""Upload a completed local run and render its checkpoint without retraining."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from training.wandb_logging import ExperimentLogger, ENTITY, PROJECT, replay_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True, help='Unchanged original training checkout')
    p.add_argument('--entity', default=ENTITY); p.add_argument('--project', default=PROJECT)
    p.add_argument('--mode', choices=['online','offline'], default='online')
    args = p.parse_args(); directory = args.run_dir.resolve()
    config = json.loads((directory/'config.json').read_text())
    if not (directory/'mjx_params').is_file():
        raise ValueError('This command requires the completed run and its final mjx_params checkpoint')
    logger = ExperimentLogger(directory, config, entity=args.entity, project=args.project, mode=args.mode)
    try:
        step = replay_metrics(logger, directory/'metrics.jsonl')
        status = logger.audits()
        logger.run.summary['training_status'] = 'completed; imported from saved local run'
        logger.run.summary['policy_video_status'] = 'rendering'
        video = directory/'videos'/'policy-final.mp4'
        environment = os.environ.copy()
        # A rendering-only process does not need to reserve most of the GPU memory.
        environment.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        subprocess.run([sys.executable, str(Path(__file__).with_name('policy_video.py').resolve()),
            '--source-root', str(args.source_root.resolve()), '--params', str(directory/'mjx_params'),
            '--out', str(video), '--step', str(step)], env=environment, check=True)
        logger.video(video, step, caption=f'Final checkpoint, nominal simulation; full audit status: {status}')
        logger.artifacts()
    except BaseException:
        logger.run.summary['upload_status'] = 'incomplete; inspect local error and retry same command'
        logger.finish(exit_code=1)
        raise
    logger.run.summary['upload_status'] = 'complete'
    url = logger.state['url']; logger.finish()
    if args.mode=='online':
        verified=False
        for attempt in range(4):
            remote=logger.sdk.Api().run(f'{args.entity}/{args.project}/{logger.state["id"]}')
            has_video=any(f.name.startswith('media/videos/') and f.name.endswith('.mp4') for f in remote.files())
            verified=(has_video and int(remote.summary.get('last_env_step',-1))>=step
                and remote.summary.get('simulation_audit_status')==status)
            if verified:break
            if attempt<3:time.sleep(3)
        if not verified:
            raise RuntimeError(f'Upload finished but cloud metrics/media are not yet verified. Inspect {url}; retry the same command, preserving wandb_run.json.')
    print(json.dumps({'url': url, 'run_id': logger.state['id'], 'mode': args.mode,
        'status': 'Uploaded metrics, audits, policy video and artifacts; cloud metrics/media verified' if args.mode=='online' else 'Saved offline; not uploaded',
        'simulation_audit_status': status}, indent=2))


if __name__ == '__main__':
    main()
