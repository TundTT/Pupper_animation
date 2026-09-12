"""Audit one immutable plan across the required grid and explicit stress cases."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .core import provenance, versions


def conditions():
    jobs=[dict(name=f'mu-{mu}-seed-{seed}',friction=mu,seed=seed,scenario={})
          for mu in [.5,.65,.8,1.] for seed in range(1,6)]
    corners=[
        ('mass-low',dict(dynamics=dict(mass_scale=.9))),
        ('mass-high',dict(dynamics=dict(mass_scale=1.1))),
        ('com-front-right-high',dict(dynamics=dict(base_com_offset_m=[.003,-.003,.002]))),
        ('com-rear-left-low',dict(dynamics=dict(base_com_offset_m=[-.003,.003,-.002]))),
        ('delay-5-steps',dict(dynamics=dict(delay_steps=5))),
        ('delay-10-steps',dict(dynamics=dict(delay_steps=10))),
        ('half-torque',dict(dynamics=dict(torque_limit_Nm=1.5))),
        ('gains-low',dict(dynamics=dict(kp_scale=.8,kd_scale=.8))),
        ('gains-high',dict(dynamics=dict(kp_scale=1.2,kd_scale=1.2))),
        ('start-low-tilted',dict(initial_offset=dict(height_m=-.002,roll_rad=math.radians(1),pitch_rad=-math.radians(1)))),
        ('start-high-tilted',dict(initial_offset=dict(height_m=.002,roll_rad=-math.radians(1),pitch_rad=math.radians(1)))),
    ]
    jobs += [dict(name=name,friction=.8,seed=11+i,scenario=scenario)
             for i,(name,scenario) in enumerate(corners)]
    combined=dict(dynamics=dict(mass_scale=1.1,base_com_offset_m=[.003,.003,.002],
                               delay_steps=5,torque_limit_Nm=1.5,kp_scale=.8,kd_scale=.8),
                  initial_offset=dict(height_m=-.002,roll_rad=math.radians(1),pitch_rad=-math.radians(1)))
    jobs += [dict(name=f'combined-seed-{seed}',friction=.8,seed=seed,scenario=combined)
             for seed in range(22,27)]
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'scenarios').mkdir();plan=a.plan.resolve();jobs=conditions()
    config=dict(plan=str(plan),plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                source_hashes=provenance(),environment=versions(),simulation_only=True,
                workers=a.workers,conditions=jobs,initial_joint_perturbation_rad=[-.01,.01],
                rationale={
                    'scope':'Sensitivity cases selected for simulation; these are not measured hardware uncertainty distributions.',
                    'mass':'Uniform +/-10% scales mass and inertia together to probe source-model and omitted payload/spacer mass uncertainty.',
                    'com':'Trunk COM offsets +/-3 mm horizontally and +/-2 mm vertically probe small assembly/payload placement errors.',
                    'delay':'5 and 10 physics steps are 9.62 and 19.23 ms command delay at 520 Hz; command history propagates between flips.',
                    'torque':'1.5 Nm is half the 3 Nm software ceiling, probing lower available torque without increasing any limit.',
                    'gains':'PD gain factors 0.8 and 1.2 probe moderate gain mismatch; these are explicitly varied dynamics.',
                    'start':'Height +/-2 mm and roll/pitch +/-1 degree are small setup errors relative to the approximate 5 mm housing clearance.',
                    'combined':'Five seeds combine the heavier, displaced-COM, delayed, lower-gain/torque and tilted low-start settings.'})
    (root/'config.json').write_text(json.dumps(config,indent=2))
    def evaluate(job):
        scenario=root/'scenarios'/(job['name']+'.json');scenario.write_text(json.dumps(job['scenario'],indent=2))
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        if sys.platform.startswith('linux'):env.setdefault('MUJOCO_GL','egl')
        cmd=[sys.executable,'-m','motion.inverted_triangle.sequence','--plan',str(plan),
             '--output',str(root/job['name']),'--friction',str(job['friction']),
             '--seed',str(job['seed']),'--scenario',str(scenario)]
        with (root/(job['name']+'.log')).open('w') as log:
            process=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        audit=root/job['name']/'sequence_audit.json';logging=root/job['name']/'logging_status.json'
        return dict(**job,returncode=process.returncode,
                    audit=json.loads(audit.read_text()) if audit.exists() else None,
                    logging=json.loads(logging.read_text()) if logging.exists() else None)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(evaluate,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            (root/'results.json').write_text(json.dumps(results,indent=2))
            print(result['name'],result['audit']['status'] if result['audit'] else 'ERROR',flush=True)
    summary=dict(plan_sha256=config['plan_sha256'],conditions=len(results),
                 passed=sum(bool(r['audit'] and r['audit']['status']=='PASS_CONTINUOUS_FOUR_FLIPS') for r in results),
                 process_errors=sum(r['returncode']!=0 for r in results),simulation_only=True,hardware_validated=False)
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if summary['process_errors']:raise SystemExit(1)


if __name__=='__main__':main()
