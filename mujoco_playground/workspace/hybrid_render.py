"""Render audit qpos and log eval/video to the run identified by wandb.json.

Use --upload_only --out existing.mp4 to backfill media without rendering or
training. --no_wandb keeps a render local even when run metadata is available.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlparse
import mujoco
import numpy as np
import mediapy
from workspace import configs


def log_video(path, run_dir, fps=25):
    """Append media to an existing run, then verify history and uploaded bytes."""
    import wandb

    path=Path(path);run_dir=Path(run_dir)
    metadata=json.loads((run_dir/'wandb.json').read_text())
    parts=[unquote(part) for part in urlparse(metadata['url']).path.strip('/').split('/')]
    if len(parts)!=4 or parts[2]!='runs' or parts[3]!=metadata['id']:
        raise ValueError('wandb.json must identify one existing entity/project/runs/id URL')
    entity,project,_,run_id=parts
    size=path.stat().st_size
    if size==0:
        raise ValueError(f'Video is empty: {path}')
    sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    # resume="must" prevents an accidental new run. Read the server's history:
    # the SDK's run.step can still be 0 immediately after resuming a finished
    # run, which would make W&B discard an otherwise uploaded video.
    previous=wandb.Api(timeout=30).run(f'{entity}/{project}/{run_id}')
    logged_step=previous.lastHistoryStep+1
    run=wandb.init(entity=entity,project=project,id=run_id,resume='must',mode='online')
    run.log({'eval/video':wandb.Video(str(path),fps=fps,format='mp4')},step=logged_step)
    run.finish()

    for attempt in range(6):
        remote=wandb.Api(timeout=30).run(f'{entity}/{project}/{run_id}')
        media=dict(remote.summary.get('eval/video',{}))
        if media.get('_type')=='video-file' and media.get('sha256')==sha256:
            uploaded=remote.file(media['path'])
            history=list(remote.scan_history(keys=['_step','eval/video'],min_step=logged_step,
                                             max_step=logged_step+1,use_cache=False))
            if uploaded.size==size and any(row['eval/video'].get('path')==media['path'] for row in history):
                receipt=dict(run_id=run_id,run_url=metadata['url'],key='eval/video',step=logged_step,
                             local_video=str(path),media=media,uploaded_bytes=uploaded.size,
                             history_verified=True)
                (run_dir/'video_media_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
                print(f"Verified W&B eval/video: {metadata['url']} ({uploaded.size} bytes)",flush=True)
                return receipt
        if attempt<5:
            time.sleep(2)
    raise RuntimeError(f'W&B media verification failed for {metadata["url"]}; no success assumed')


def render(audit, robot, out):
    data=np.load(audit);q=data['qpos'][:,robot]
    m=mujoco.MjModel.from_xml_path(str(configs.resolve_model_path()));d=mujoco.MjData(m)
    renderer=mujoco.Renderer(m,height=480,width=640)
    cam=mujoco.MjvCamera();cam.distance=.75;cam.azimuth=135;cam.elevation=-20
    with mediapy.VideoWriter(str(out),shape=(480,640),fps=25) as video:
        for i in range(0,len(q),2):
            d.qpos[:]=q[i];mujoco.mj_forward(m,d)
            cam.lookat[:]=[q[i,0],q[i,1],.08]
            renderer.update_scene(d,camera=cam)
            frame=renderer.render()
            # Label replay because visual model is nominal geometry; trajectories
            # come from randomized physics and are never resimulated for rendering.
            from PIL import Image, ImageDraw
            im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,640,48),fill='black')
            draw.text((8,5),f"Hybrid audit robot {robot} | t={i*.02:.1f}s | phase={data['phase'][i,robot]} | completed={data['completed'][i,robot].sum()}/4",fill='white')
            draw.text((8,26),'Recorded randomized-physics motion; nominal visual geometry',fill='white')
            video.add_image(np.asarray(im))
    renderer.close()
    data.close()
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',help='recorded audit .npz; required unless --upload_only')
    p.add_argument('--robot',type=int,default=0)
    p.add_argument('--out',required=True,help='output MP4, or existing MP4 for --upload_only')
    p.add_argument('--run_dir',help='directory containing wandb.json; defaults to audit/output directory')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--upload_only',action='store_true')
    mode.add_argument('--no_wandb',action='store_true')
    args=p.parse_args()
    out=Path(args.out)
    run_dir=Path(args.run_dir) if args.run_dir else (Path(args.audit).parent if args.audit else out.parent)
    if args.upload_only:
        if not out.is_file():
            p.error(f'existing video not found: {out}')
        if not (run_dir/'wandb.json').is_file():
            p.error(f'run metadata not found: {run_dir / "wandb.json"}')
    else:
        if not args.audit:
            p.error('--audit is required to render a video')
        out.parent.mkdir(parents=True,exist_ok=True)
        render(args.audit,args.robot,out)
    if not args.no_wandb and (run_dir/'wandb.json').is_file():
        log_video(out,run_dir)
    else:
        print('Video saved locally; W&B logging disabled or no wandb.json present.',flush=True)


if __name__=='__main__':main()
