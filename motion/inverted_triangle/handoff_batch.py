"""Run an explicit, retained handoff plan; failures never disappear from the index."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--terminal',type=Path,required=True)
    p.add_argument('--cad',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    plan=json.loads(args.plan.read_text())
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    index=[]
    for case in plan:
        name=case['name'];folder=args.output/name
        cfg=args.output/(name+'.json');cfg.write_text(json.dumps(case['config'],indent=2)+'\n')
        command=[sys.executable,'-m','motion.inverted_triangle.handoff_probe','--config',str(cfg),
                 '--terminal',str(args.terminal),'--output',str(folder)]
        if not args.cad:command.append('--no-cad')
        with (args.output/(name+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,config=case['config'],returncode=result.returncode,report=str(folder/'report.json'))
        if (folder/'report.json').exists():
            r=json.loads((folder/'report.json').read_text())
            row.update({k:r[k] for k in ('engineering_gates','engineering_pass','early_termination','environment_steps','wandb_url') if k in r})
            row['max_floor_force_N']=r['dense_audit']['max_floor_force_N']
        index.append(row)
        (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
        print(json.dumps(row),flush=True)
    raise SystemExit(0 if all(r.get('engineering_pass',False) for r in index) else 1)


if __name__=='__main__':main()
