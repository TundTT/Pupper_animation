"""Attach an existing checkpoint video to its original W&B training run."""
import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--run', required=True, help='entity/project/run_id of the existing run')
    parser.add_argument('--key', default='eval/video_best')
    args = parser.parse_args()
    params = Path(args.params).resolve()
    video = Path(args.video).resolve()
    metadata = json.loads((params.parent / 'run.json').read_text())
    if not video.is_file():
        raise FileNotFoundError(video)
    checkpoint_sha = hashlib.sha256(params.read_bytes()).hexdigest()
    entity, project, run_id = args.run.split('/')
    remote = wandb.Api().run(args.run)
    if (remote.name != params.parent.name or remote.config.get('model_sha256') != metadata['model_sha256']
            or remote.config.get('config') != metadata['config']):
        raise ValueError('W&B run name/model does not match the checkpoint directory')
    best_path = params.parent / 'best_params'
    best_sha = hashlib.sha256(best_path.read_bytes()).hexdigest() if best_path.exists() else None
    best = json.loads((params.parent / 'best_checkpoint.json').read_text()) if checkpoint_sha == best_sha else {}
    step = best.get('step')
    if step is None and params.name.startswith('params_') and params.name[7:].isdigit():
        step = int(params.name[7:])
    caption = f'{params.parent.name}; checkpoint step {step}; {params.name}'
    # Append media at the current history position. Logging at the old checkpoint
    # step would be discarded by W&B, whose history only advances forward.
    with wandb.init(entity=entity, project=project, id=run_id, resume='must') as run:
        payload = {
            args.key: wandb.Video(str(video), format='mp4', caption=caption),
            args.key + '_checkpoint_sha256': checkpoint_sha,
        }
        if step is not None:
            payload[args.key + '_checkpoint_step'] = step
        run.log(payload)
    uploaded = wandb.Api().run(args.run).summary.get(args.key)
    if not uploaded or uploaded.get('_type') != 'video-file':
        raise RuntimeError('W&B did not confirm the uploaded video in the run summary')
    receipt = dict(run=args.run, key=args.key, video=str(video), params=str(params),
                   checkpoint_sha256=checkpoint_sha, checkpoint_step=step)
    (video.parent / (video.stem + '_wandb.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
