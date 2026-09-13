"""Upload retained simulation results/videos, with source hashes only."""
import argparse,concurrent.futures,json,subprocess
from pathlib import Path
import numpy as np
from motion.inverted_triangle.core import Robot
from motion.inverted_triangle.roll_to_stand import render_states
from training.wandb_logging import ExperimentLogger

def render(args):
 path,config=args;r=Robot(config.get('friction',.8),config.get('scenario',{}).get('dynamics'),config.get('formation'));z=np.load(path/'states.npz');render_states(r,list(zip(z['times'],z['qpos'])),path/'rollout.mp4',config['name']+' | LOCAL OPTIMIZATION FAILED/INCOMPLETE AUDIT | no checkpoint')
 return path
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=12);a=p.parse_args();setup=json.loads((a.root/'provenance.json').read_text());reports=json.loads((a.root/'summary.json').read_text());logger=ExperimentLogger(a.root,dict(archival_upload_of_local_screen=True,original_setup=setup,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 try:
  with concurrent.futures.ProcessPoolExecutor(a.workers) as pool:paths=list(pool.map(render,[(a.root/r['config']['name'],r['config']) for r in reports]))
  step=0
  for path,report in zip(paths,reports):
   step+=report['environment_steps'];logger.metrics(step,dict(trial=report['config']['name'],audit_status=report['status']));logger.video(path/'rollout.mp4',step,key='trajectory/'+report['config']['name'],caption='SIMULATION actual recorded physics; local optimization; CAD incomplete; no checkpoint or hardware; '+json.dumps(report['gates']))
  logger.run.summary.update(dict(simulation_audit_status='REJECTED_OPTIMIZATION_SCREEN',trials=len(reports),hardware_validated=False,dense_cad=False))
  artifact=logger.sdk.Artifact('support-screen-'+logger.state['id'],type='failed-physics-optimization')
  for f in a.root.rglob('*'):
   if f.is_file() and f.suffix in ['.json','.npz'] and 'wandb' not in f.relative_to(a.root).parts:artifact.add_file(str(f),name=f.relative_to(a.root).as_posix())
  logger.run.log_artifact(artifact)
 finally:logger.finish()
