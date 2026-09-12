"""Retain an intentionally interrupted search and audit its last recorded best.

The immutable plan being validated was selected earlier. Do not substitute this
search's latest candidate into it. In-flight discarded evaluations are explicitly
not counted as measured steps.
"""
import json,sys,hashlib
from pathlib import Path
from motion.inverted_triangle.simulate import run
from training.wandb_logging import ExperimentLogger
out=Path(sys.argv[1]);config=json.loads((out/'config.json').read_text())
records=[json.loads(s) for s in (out/'evaluations.jsonl').read_text().splitlines()]
contexts=json.loads((out/'contexts.json').read_text());context=next(c for c in contexts if c['job']['name']=='mu-0.8-seed-1')
prefix_steps=sum(round(c['state']['state'][0]*520) for c in contexts)
summary=dict(status='INTENTIONALLY_STOPPED_FOR_INDEPENDENT_VALIDATION',completed_evaluations=len(records),
 environment_steps_lower_bound=prefix_steps+sum(c['environment_steps'] for c in records),
 unrecorded_inflight_steps='Unknown; excluded from the lower bound, not reported as zero.',
 best=min(records,key=lambda c:c['cost']),reason='An immutable combined candidate passed nominal and combined-case audits. Stop further candidate selection and retain all recorded evaluations and the interrupted process log.')
(out/'search.json').write_text(json.dumps(summary,indent=2))
logger=ExperimentLogger(out,config)
c=json.loads((out/'candidate.json').read_text())
a=run(out/'candidate.json',out/'replay',video=True,start_override=context['start'],direction=c['direction'],landing_delta=c['landing_delta'])
steps=summary['environment_steps_lower_bound']+a['environment_steps']
logger.metrics(steps,dict(simulation_audit_status=a['status'],environment_steps_lower_bound=steps,search_intentionally_stopped=True))
logger.video(out/'replay/rollout.mp4',steps,key='trajectory/'+c['leg'],caption=f'SIMULATION | last recorded search candidate {a["candidate_sha256"]} | {a["status"]} single-stage audit from an actually integrated prefix | intentionally stopped search; not the selected full-plan claim')
logger.run.summary.update(dict(simulation_audit_status=a['status'],search_intentionally_stopped=True,environment_steps_are_lower_bound=True,gates=a['gates'],hardware_validated=False))
artifact=logger.sdk.Artifact('sensitivity-search-'+logger.state['id'],type='trajectory-run')
for p in out.rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.txt'] and 'wandb' not in p.relative_to(out).parts:artifact.add_file(str(p),name=p.relative_to(out).as_posix())
artifact.add_file(str(out.with_suffix('.log')),name='interrupted-process.log')
artifact.add_file(__file__,name='finalize_search.py')
logger.run.log_artifact(artifact);logger.finish()
(out/'logging_status.json').write_text(json.dumps(dict(url=logger.state['url'],status='SDK finish completed; verify cloud; intentional search interruption retained'),indent=2))
