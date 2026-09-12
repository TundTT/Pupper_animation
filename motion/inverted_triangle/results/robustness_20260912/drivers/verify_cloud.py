"""Verify W&B media bytes and committed artifacts without displaying credentials."""
import base64,hashlib,json
from pathlib import Path
import wandb
root=Path('runs/inverted_triangle');path=root/'cloud-verification.json'
old=json.loads(path.read_text()) if path.exists() else []
cache={r['id']:r for r in old if r.get('verified')};api=wandb.Api();rows=[];seen=set()
for state_path in sorted(root.rglob('wandb_run.json')):
 if any(part in state_path.relative_to(root).parts for part in ['wandb','review-bundle']):continue
 state=json.loads(state_path.read_text())
 if state.get('mode')!='online' or state['id'] in seen:continue
 seen.add(state['id'])
 if state['id'] in cache:rows.append(cache[state['id']]);continue
 directory=state_path.parent
 try:
  remote=api.run(f"{state['entity']}/{state['project']}/{state['id']}")
  local={}
  for video in directory.rglob('*.mp4'):
   if 'wandb' in video.relative_to(directory).parts:continue
   digest=hashlib.md5(video.read_bytes()).digest()
   for key in [digest.hex(),base64.b64encode(digest).decode()]:local.setdefault(key,[]).append(video.relative_to(directory).as_posix())
  media=[dict(name=f.name,bytes=f.size,md5=f.md5,matching_local_videos=local.get(f.md5,[])) for f in remote.files() if f.name.endswith('.mp4')]
  artifacts=[dict(name=a.name,digest=a.digest,state=a.state) for a in remote.logged_artifacts()]
  row=dict(directory=directory.relative_to(root).as_posix(),id=state['id'],url=remote.url,cloud_state=remote.state,
           simulation_audit_status=remote.summary.get('simulation_audit_status'),media=media,artifacts=artifacts)
  row['verified']=bool(remote.state=='finished' and media and all(m['matching_local_videos'] for m in media) and artifacts and all(a['state']=='COMMITTED' for a in artifacts))
 except Exception as e:row=dict(directory=directory.relative_to(root).as_posix(),id=state['id'],verified=False,error=type(e).__name__+': '+str(e))
 rows.append(row);path.write_text(json.dumps(rows,indent=2));print(row['directory'],row['verified'],flush=True)
path.write_text(json.dumps(rows,indent=2));print('verified',sum(r['verified'] for r in rows),'of',len(rows),flush=True)
