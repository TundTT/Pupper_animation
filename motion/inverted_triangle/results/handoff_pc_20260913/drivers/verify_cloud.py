import argparse,hashlib,json
from pathlib import Path
import wandb
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--download-selected',action='store_true');a=p.parse_args();api=wandb.Api(timeout=90);rows=[]
cache_path=a.root/'cloud-verification.json'
cache={r['id']:r for r in json.loads(cache_path.read_text()) if r.get('verified')} if cache_path.exists() else {}
for path in sorted(a.root.rglob('wandb_run.json')):
 if 'wandb' in path.relative_to(a.root).parts:continue
 state=json.loads(path.read_text())
 if state['id'] in cache:
  rows.append(cache[state['id']]);continue
 identity=f"{state['entity']}/{state['project']}/{state['id']}"
 try:
  run=api.run(identity);videos=[dict(name=f.name,size=f.size,md5=f.md5) for f in run.files() if f.name.startswith('media/videos/') and f.name.endswith('.mp4')]
  artifacts=[dict(name=x.name,digest=x.digest,state=x.state,files=list(x.manifest.entries)) for x in run.logged_artifacts()]
  row=dict(local=str(path.parent.relative_to(a.root)),url=run.url,id=run.id,state=run.state,videos=videos,artifacts=artifacts,verified=run.state=='finished' and bool(videos) and all(x['size']>0 for x in videos) and bool(artifacts))
  if a.download_selected and path.parent==a.root/'development'/'diagonal_10mm_spread':
   remote=run.file(videos[0]['name']).download(root=str(a.root/'selected-download'),replace=True)
   local=path.parent/'rollout.mp4';download=Path(remote.name)
   row['selected_download_sha256']=hashlib.sha256(download.read_bytes()).hexdigest();row['selected_local_sha256']=hashlib.sha256(local.read_bytes()).hexdigest();row['selected_bytes_identical']=row['selected_download_sha256']==row['selected_local_sha256']
  rows.append(row)
 except Exception as e:rows.append(dict(local=str(path.parent.relative_to(a.root)),id=state['id'],verified=False,error=str(e)))
 (a.root/'cloud-verification.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(runs=len(rows),verified=sum(r['verified'] for r in rows),failures=[r for r in rows if not r['verified']]),indent=2))
