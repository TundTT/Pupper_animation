"""Upload saved simulation evidence and its real rollout to W&B; never rerun motion."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.wandb_logging import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=ROOT/'hardware_testing/backpack_align_2026-09-12')
    parser.add_argument('--mode', choices=['online', 'offline'], default='online')
    args = parser.parse_args()
    directory = args.directory.resolve()
    suite = json.loads((directory/'suite.json').read_text())
    config = dict(suite, controller='deterministic-position-pid-keyframes',
                  checkpoint_applicable=False, optimizer_updates=0,
                  evidence_sha256=hashlib.sha256((directory/'evidence.tar.gz').read_bytes()).hexdigest())
    logger = ExperimentLogger(directory/'wandb-upload', config, mode=args.mode,
                              name='backpack-9mm-position-pid-26-case-audit')
    try:
        logger.run.summary.update({'simulation_audit_status': 'passed' if suite['passed'] else 'failed',
                                   'hardware_validated': False, 'scenario_count': len(suite['scenarios']),
                                   'checkpoint_applicable': False})
        for label, passed in suite['scenarios'].items():
            logger.run.summary['audit/'+label+'/passed'] = passed
        logger.video(directory/'rollout.mp4', 0, key='motion/keyframes',
                     caption='SIMULATION; seed 0; full FL/FR/BR/BL rollout; no early termination; '
                             'deterministic position PID, no neural checkpoint. Hardware pending.')
        artifact = logger.sdk.Artifact('backpack-alignment-'+logger.state['id'], type='motion-audit')
        for name in ('suite.json', 'evidence.tar.gz', 'rollout.mp4'):
            artifact.add_file(str(directory/name), name=name)
        logger.run.log_artifact(artifact)
    except Exception:
        logger.finish(exit_code=1)
        raise
    logger.finish()
    print(json.dumps({'mode': args.mode, 'url': logger.state['url']}))


if __name__ == '__main__':
    main()
