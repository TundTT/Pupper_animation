import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from training.wandb_logging import ExperimentLogger, replay_metrics, ENTITY, PROJECT


class FakeSDK:
    def __init__(self):self.runs=[]
    def Settings(self,**kwargs):return kwargs
    def init(self,**kwargs):
        summary={}
        for identity,previous in self.runs:
            if identity['id']==kwargs['id']:summary=dict(previous.summary)
        run=SimpleNamespace(config=kwargs['config'],summary=summary,url='https://wandb.ai/test/run',logs=[],artifacts=[])
        run.define_metric=lambda *a,**kw:None
        run.log=lambda data:run.logs.append(data)
        run.log_artifact=lambda data:run.artifacts.append(data)
        run.finish=lambda **kw:None
        self.runs.append((kwargs,run));return run
    def Video(self,path,**kwargs):return {'video':path,**kwargs}
    def Artifact(self,*a,**kw):
        files=[];return SimpleNamespace(files=files,add_file=lambda path,**kw:files.append((path,kw)))


def test_backfill_identity_metrics_audits_and_media(tmp_path):
    sdk=FakeSDK();logger=ExperimentLogger(tmp_path,{'source_commit':'test-only'},sdk=sdk)
    metrics=tmp_path/'metrics.jsonl'
    metrics.write_text('\n'.join(json.dumps(dict(step=i,seconds=i/10,reward=i)) for i in (0,100,200)))
    assert replay_metrics(logger,metrics)==200
    assert len(logger.run.logs)==3
    for label,passed in [('nominal',False),('randomized',False),('interrupted',False)]:
        (tmp_path/f'audit-{label}.json').write_text(json.dumps(dict(passes_simulation_gate=passed,
            all_four_completed=0,phase_seconds_mean=dict(lift=25.))))
    assert logger.audits()=='failed'
    assert logger.run.summary['audit/nominal/phase_seconds_mean/lift']==25.
    video=tmp_path/'policy.mp4';video.write_bytes(b'test-only-placeholder')
    logger.video(video,200)
    assert logger.run.logs[-1]['policy/rollout']['video']==str(video)
    assert logger.run.summary['hardware_validated'] is False
    logger.finish();again=ExperimentLogger(tmp_path,{'source_commit':'test-only'},sdk=sdk)
    replay_metrics(again,metrics)
    assert not again.run.logs
    assert sdk.runs[0][0]['id']==sdk.runs[1][0]['id']
    assert sdk.runs[0][0]['entity']==ENTITY and sdk.runs[0][0]['project']==PROJECT
    with pytest.raises(ValueError):ExperimentLogger(tmp_path,{},entity='wrong-destination',sdk=sdk)


def test_actual_render_and_offline_wandb(tmp_path):
    import mujoco
    import imageio.v2 as imageio
    from training.wheel_align.configs import MODEL_PATH
    from training.wheel_align.policy_video import write_video
    m=mujoco.MjModel.from_xml_path(str(MODEL_PATH));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
    row=dict(seconds=0.,phase=0,command=0,active_command=0,progress=0.,completed=0,rotating=False,gate_steps=0,done=False)
    row.update({leg+'_error':0. for leg in ('FR','FL','BR','BL')})
    video=tmp_path/'unit-test.mp4'
    write_video([d.qpos.copy()]*2,[row]*2,MODEL_PATH,video,caption='UNTRAINED RENDER TEST')
    frames=imageio.mimread(str(video))
    assert len(frames)==2 and frames[0].shape==(480,640,3)
    assert np.std(frames[0][90:])>5
    (tmp_path/'config.json').write_text(json.dumps({'source_commit':'offline-unit-test'}))
    (tmp_path/'mjx_params').write_bytes(b'UNTRAINED OFFLINE ARTIFACT TEST')
    logger=ExperimentLogger(tmp_path,{'source_commit':'offline-unit-test'},mode='offline')
    logger.metrics(100,{'reward':1.});logger.video(video,100);logger.audits();logger.artifacts();logger.finish()
    assert logger.state['url'] is None
    assert list(tmp_path.glob('wandb/offline-run-*/files/media/videos/**/*.mp4'))


def test_checkpoint_video_worker(tmp_path):
    # Four control steps of an untrained actor: verifies exact-checkout
    # import/provenance, real rollout, MP4 and trace. No optimizer is invoked.
    import os
    import subprocess
    import sys
    import jax
    from jax import numpy as jp
    from brax.io import model
    from brax.training.acme import running_statistics,specs
    from training.wheel_align.train import network_factory,source_hashes,ROOT
    net=network_factory()(83,8,preprocess_observations_fn=running_statistics.normalize)
    stats=running_statistics.init_state(specs.Array((83,),jp.float32))
    params=(stats,net.policy_network.init(jax.random.PRNGKey(42)))
    model.save_params(str(tmp_path/'mjx_params'),params)
    (tmp_path/'config.json').write_text(json.dumps(dict(source_hashes=source_hashes(),curriculum_stage="foundation")))
    output=tmp_path/'untrained-rollout-test.mp4'
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    subprocess.run([sys.executable,str(ROOT/'training/wheel_align/policy_video.py'),
        '--source-root',str(ROOT),'--params',str(tmp_path/'mjx_params'),'--out',str(output),
        '--step','0','--max-steps','4'],env=env,check=True)
    metadata=json.loads(output.with_suffix('.json').read_text())
    assert metadata['frames']==1 and metadata['training_step']==0 and metadata['stage']=='foundation'
    assert output.with_suffix('.trace.csv').is_file()


def test_training_callbacks_log_final_video_without_optimizer(tmp_path,monkeypatch):
    import sys
    from training.wheel_align import train as entry,policy_video
    sdk=FakeSDK()
    monkeypatch.setattr(entry,'ExperimentLogger',lambda directory,config,**kw:ExperimentLogger(directory,config,sdk=sdk,**kw))
    monkeypatch.setattr(entry.jax,'devices',lambda:[SimpleNamespace(platform='gpu')])
    monkeypatch.setattr(entry,'AlignEnv',lambda **kw:object())
    monkeypatch.setattr(entry.model,'save_params',lambda path,params:Path(path).write_bytes(b'UNTRAINED TEST CHECKPOINT'))
    def render(policy,path,**kwargs):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'UNIT TEST VIDEO PLACEHOLDER');return path
    monkeypatch.setattr(policy_video,'record_policy',render)
    def fake_train(environment,**kwargs):
        assert kwargs['restore_value_fn'] is False
        assert kwargs['restore_params'] is None
        make_policy=lambda params,deterministic:object()
        for step in (0,100):
            kwargs['progress_fn'](step,{'eval/reward':float(step)})
            kwargs['policy_params_fn'](step,make_policy,object())
        return make_policy,object(),{}
    monkeypatch.setattr(entry.train,'train',fake_train)
    directory=tmp_path/'test-run'
    monkeypatch.setattr(sys,'argv',['train','--stage','foundation','--steps','100','--envs','256','--out',str(directory),'--video-every-steps','50'])
    entry.main()
    run=sdk.runs[0][1]
    assert run.summary['last_env_step']==100
    assert run.summary['training_status']=='completed'
    assert any('policy/full_sequence_diagnostic' in row for row in run.logs)
    assert any('policy/final' in row for row in run.logs)
    assert any('policy/rollout' in row for row in run.logs)
    assert any(Path(path).name=='mjx_params' for artifact in run.artifacts for path,kw in artifact.files)
