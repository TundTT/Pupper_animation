import json
from pathlib import Path
from training.wandb_logging import ExperimentLogger
root=Path('/tmp/handoff-evidence-20260913')
for seed in (0,1,5):
 folder=root/f'alignment-published-{seed}';terminal=json.loads((folder/'terminal.json').read_text());audit=json.loads((folder/'alignment_report.json').read_text())
 steps=terminal.get('environment_steps',round(terminal['simulation_time_s']*520))
 log=ExperimentLogger(folder,dict(source_commit=terminal['source_commit'],terminal=terminal,checkpoint_applicable=False,optimizer_updates=0),name=f'alignment-source-unavailable-fallback-{seed}')
 log.metrics(steps,{'alignment/passed':audit['passed'],'alignment/completed_mask':audit['completed_mask'],'alignment/failed_mask':audit['failed_mask']})
 log.video(folder/'alignment.mp4',steps,key='alignment/actual_rollout',caption=f'SIMULATION published deterministic controller dde1f96; seed {seed}; actual failed replay; rejected handoff endpoint; no checkpoint')
 artifact=log.sdk.Artifact('published-alignment-failure-'+log.state['id'],type='motion-audit')
 for name in ('terminal.json','alignment_report.json','alignment_trace.csv'):artifact.add_file(str(folder/name))
 log.run.log_artifact(artifact);log.run.finish()
 print(log.state['url'],flush=True)
