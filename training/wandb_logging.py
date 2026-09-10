"""Shared experiment logging for current and future QuadMorph policies."""
import hashlib
import json
from pathlib import Path
import uuid

ENTITY = 'QuadMorph'
PROJECT = 'wheel-leg lift and align triangle base'


class ExperimentLogger:
    def __init__(self, directory, config, *, entity=ENTITY, project=PROJECT,
                 mode='online', name=None, sdk=None):
        if sdk is None:
            import wandb as sdk
        self.sdk = sdk
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / 'wandb_run.json'
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            'id': uuid.uuid4().hex[:16], 'entity': entity, 'project': project,
            'last_env_step': -1, 'mode': mode,
        }
        if (self.state['entity'], self.state['project'], self.state['mode']) != (entity, project, mode):
            raise ValueError('Existing W&B identity/mode differs. Sync offline logs with wandb sync; do not create a duplicate run.')
        if mode=='offline' and self.state['last_env_step']>=0:
            raise ValueError('W&B offline sessions cannot resume. Sync the existing saved session rather than reimporting it.')
        self._save()
        self.run = sdk.init(entity=entity, project=project, id=self.state['id'],
            resume='allow' if mode == 'online' else None, mode=mode,
            name=name or self.directory.name, config=config, dir=str(self.directory),
            tags=['quadmorph', 'simulation'], settings=sdk.Settings(save_code=False))
        self.run.define_metric('train/env_steps')
        self.run.define_metric('*', step_metric='train/env_steps')
        if mode == 'online':
            # Local queueing is not a cloud acknowledgement. Replay anything
            # beyond the server's saved summary after an interrupted upload.
            remote_step = int(self.run.summary.get('last_env_step', -1))
            self.state['last_env_step'] = min(self.state['last_env_step'], remote_step)
        self.state['url'] = self.run.url if mode == 'online' else None
        self._save()

    def _save(self):
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2) + '\n')
        temporary.replace(self.state_path)

    def metrics(self, step, data):
        step = int(step)
        if step <= self.state['last_env_step']:
            return
        payload = {k: v for k, v in data.items() if k not in ('step', 'seconds')}
        payload.update({'train/env_steps': step})
        if 'seconds' in data:
            payload['train/wall_seconds'] = data['seconds']
        # Use a custom axis rather than SDK _step, allowing later videos/audits.
        self.run.log(payload)
        self.run.summary['last_env_step'] = step
        self.state['last_env_step'] = step
        self._save()

    def video(self, path, step, *, key='policy/rollout', caption='Nominal simulated policy rollout'):
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing policy video: {path}')
        self.run.log({'train/env_steps': int(step), key: self.sdk.Video(str(path), format='mp4', caption=caption)})
        self.run.summary['policy_video_status'] = 'logged'

    def audits(self):
        files = sorted(self.directory.glob('audit-*.json'))
        gates = {}
        for path in files:
            audit = json.loads(path.read_text())
            label = path.stem.removeprefix('audit-')
            gates[label] = audit.get('passes_simulation_gate')
            def record(prefix, values):
                for key, value in values.items():
                    if isinstance(value, dict):
                        record(f'{prefix}/{key}', value)
                    elif isinstance(value, (int, float, bool, str)):
                        self.run.summary[f'{prefix}/{key}'] = value
            record(f'audit/{label}', audit)
        expected = ('nominal', 'randomized', 'interrupted')
        status = ('failed' if any(value is False for value in gates.values()) else
                  'passed' if all(gates.get(key) is True for key in expected) else 'pending')
        self.run.summary['simulation_audit_status'] = status
        self.run.summary['hardware_validated'] = False
        return status

    def artifacts(self):
        artifact = self.sdk.Artifact(f'quadmorph-{self.state["id"]}', type='policy-run',
            metadata={'source_commit': self.run.config.get('source_commit'),
                      'simulation_audit_status': self.run.summary.get('simulation_audit_status', 'pending'),
                      'hardware_validated': False})
        files = [self.directory / name for name in ('config.json', 'metrics.jsonl', 'latest.json', 'mjx_params')]
        for pattern in ('audit-*.json', 'policy*.json', 'policy*.reference.csv', 'videos/*.trace.csv', 'videos/*.json'):
            files.extend(self.directory.glob(pattern))
        for path in sorted(set(files)):
            if path.is_file() and not path.is_symlink():
                artifact.add_file(str(path), name=path.relative_to(self.directory).as_posix())
                if path.name == 'mjx_params':
                    self.run.summary['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.run.log_artifact(artifact)

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)


def replay_metrics(logger, path):
    """Replay actual saved measurements; no estimates from a prose report."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('No saved training metrics to upload')
    steps = [int(row['step']) for row in rows]
    if steps != sorted(steps):
        raise ValueError('Training steps are out of order')
    for row in rows:
        logger.metrics(row['step'], row)
    return max(steps)
