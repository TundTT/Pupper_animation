"""Summarize recorded all-leg audits without changing controller or training."""
import argparse
import json
from pathlib import Path
import numpy as np


def diagnose(path):
    config=json.loads(Path(path).with_suffix('.json').read_text())
    abduction_guard=config.get('abduction_guard',.25)
    with np.load(path) as archive:
        d={key:archive[key] for key in ['qpos','done','phase','clearance']}
    n=d['qpos'].shape[1]
    orders=np.array([[1,0,2,3],[0,1,3,2]])[np.arange(n)%2]
    rows=[]
    names=['front_r','front_l','back_r','back_l']
    for leg,name in enumerate(names):
        samples=[]
        for robot in range(n):
            stage=int(np.where(orders[robot]==leg)[0][0])
            start=100+1100*stage
            end=start+1000
            valid=(d['done'][:end,robot]==0)
            alive=np.logical_and.accumulate(valid)[start:end]
            if not alive.any():continue
            q=d['qpos'][start:end,robot][alive]
            phase=d['phase'][start:end,robot][alive]
            clear=d['clearance'][start:end,robot][alive]
            hip=q[:,8+3*leg]*(1 if leg%2==0 else -1)
            abd_error=np.abs(q[:,7+3*leg]-(1 if leg%2==0 else -1))
            samples.append(dict(ever_rotate=bool(np.any((phase==2)|(phase==3))),
                                ever_verify=bool(np.any(phase==3)),max_clearance=float(clear.max()),
                                max_hip=float(hip.max()),hip_guard_fraction=float((hip>1.1).mean()),
                                abduction_guard_fraction=float((abd_error<abduction_guard).mean()),
                                clearance_fraction=float((clear>.005).mean()),
                                lift_phase_fraction=float((phase==1).mean())))
        rows.append(dict(leg=name,abduction_guard=abduction_guard,robots_with_alive_samples=len(samples),
                         reached_rotation=sum(s['ever_rotate'] for s in samples),
                         reached_verify=sum(s['ever_verify'] for s in samples),
                         **{k+'_median':float(np.median([s[k] for s in samples])) for k in ['max_clearance','max_hip','hip_guard_fraction','abduction_guard_fraction','clearance_fraction','lift_phase_fraction']}))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('audit');p.add_argument('--out',required=True);a=p.parse_args()
    result=diagnose(a.audit);Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
