"""Correct interrupted-run counts from actual retained files; retain prior metadata."""
from pathlib import Path
import json
from training.wandb_logging import ExperimentLogger
root=Path('/tmp/pupper-roll-to-stand-20260912/runs/roll_to_stand/adaptive-default');path=root/'interrupted_status.json';old=json.loads(path.read_text());directories=[d for d in root.iterdir() if d.is_dir() and (d/'integration.npz').exists()]
new=dict(status='INTERRUPTED_REJECTED_FEEDBACK',reason='Superseded feedback audit stopped; all completed and partial physics records/videos retained.',completed_audits=sum((d/'audit.json').exists() for d in directories),partial_audits=sum(not (d/'audit.json').exists() for d in directories),counts_source='Actual retained audit.json and integration.npz files',metadata_correction='Previous manually supplied counts lagged the process completing its second case before interruption.')
(root/'interrupted_status_history.json').write_text(json.dumps(dict(previous=old,corrected=new),indent=2));path.write_text(json.dumps(new,indent=2));logger=ExperimentLogger(root,json.loads((root/'provenance.json').read_text()))
try:
 logger.run.summary.update(new);artifact=logger.sdk.Artifact('interrupted-support-'+logger.state['id'],type='interrupted-simulation')
 for f in root.rglob('*'):
  if f.is_file() and f.suffix in ['.json','.npz'] and 'wandb' not in f.relative_to(root).parts:artifact.add_file(str(f),name=f.relative_to(root).as_posix())
 logger.run.log_artifact(artifact)
finally:logger.finish(exit_code=1)
print(new)
