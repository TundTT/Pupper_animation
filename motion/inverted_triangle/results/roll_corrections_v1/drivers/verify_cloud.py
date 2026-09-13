"""Verify authorized W&B result/video uploads by cloud bytes and artifact state."""
import argparse,base64,hashlib,json,concurrent.futures
from pathlib import Path
import wandb
p=argparse.ArgumentParser();p.add_argument('--roots',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cache={x['id']:x for x in json.loads(a.output.read_text()) if x.get('verified')} if a.output.exists() else {};jobs={}
for root in a.roots:
 for f in root.rglob('wandb_run.json'):
  if 'wandb' in f.relative_to(root).parts or 'review-bundle' in f.relative_to(root).parts:continue
  s=json.loads(f.read_text())
  if s.get('mode')=='online':jobs[s['id']]=(f,s)
def verify(job):
 f,s=job;directory=f.parent
 if s['id'] in cache:return cache[s['id']]
 try:
  remote=wandb.Api().run(f"{s['entity']}/{s['project']}/{s['id']}");local={};digests={}
  for video in directory.rglob('*.mp4'):
   if 'wandb' in video.relative_to(directory).parts:continue
   data=video.read_bytes();digest=hashlib.md5(data).digest();name=video.relative_to(directory).as_posix();digests[name]=dict(md5=digest.hex(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   for key in [digest.hex(),base64.b64encode(digest).decode()]:local.setdefault(key,[]).append(name)
  media=[]
  for v in remote.files():
   if not v.name.endswith('.mp4'):continue
   matches=[name for name in local.get(v.md5,[]) if digests[name]['bytes']==v.size];media.append(dict(name=v.name,bytes=v.size,md5=v.md5,matching_local_videos=matches))
  artifacts=[dict(name=x.name,digest=x.digest,state=x.state) for x in remote.logged_artifacts()]
  expected=['handoff/continuous_rollout.mp4'] if (directory/'handoff/continuous_rollout.mp4').exists() else sorted(digests)
  covered={name for m in media for name in m['matching_local_videos']};missing=sorted(set(expected)-covered)
  row=dict(directory=str(directory),id=s['id'],url=remote.url,cloud_state=remote.state,simulation_audit_status=remote.summary.get('simulation_audit_status',remote.summary.get('status')),media=media,artifacts=artifacts,local_video_hashes=digests,missing_expected_media=missing)
  row['verified']=bool(remote.state in ['finished','failed'] and media and not missing and all(m['matching_local_videos'] for m in media) and artifacts and all(x['state']=='COMMITTED' for x in artifacts))
 except Exception as e:row=dict(directory=str(directory),id=s['id'],verified=False,error=type(e).__name__+': '+str(e))
 return row
rows=[]
with concurrent.futures.ThreadPoolExecutor(6) as pool:
 for row in pool.map(verify,jobs.values()):
  rows.append(row);a.output.write_text(json.dumps(rows,indent=2));print(row['id'],row['verified'],row.get('cloud_state'),flush=True)
print('verified',sum(x['verified'] for x in rows),'of',len(rows),flush=True)
