"""Plot matched action and physical yaw traces from evaluate_gait_suite outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from workspace.evaluate_gait_suite import COMMANDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True,help='Baseline NPZ')
    parser.add_argument('--candidate',required=True,help='Candidate NPZ')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    before=np.load(args.baseline);after=np.load(args.candidate)
    np.testing.assert_array_equal(before['commands'],after['commands'])
    assert before['action'].shape==after['action'].shape
    seeds=len(before['commands'])//len(COMMANDS)
    names=['reverse20','turn_left','turn_right']
    fig,axes=plt.subplots(3,2,figsize=(13,9))
    summary={}
    for row,name in enumerate(names):
        index=list(COMMANDS).index(name)*seeds
        # Compare the same joint and reset seed for both policies. Select the
        # baseline joint with most command variation, not a favorable candidate.
        joint=int(np.argmax(np.var(before['action'][50:,index],axis=0)))
        for label,data,color in [('Previous',before,'#bf4b45'),('New',after,'#247d9b')]:
            scale=data['action_scale'][joint] if 'action_scale' in data else (.5,.25,.5)[joint%3]
            action=data['action'][50:150,index,joint]*scale
            yaw=data['angular_velocity'][50:150,index].reshape(-1,3)[:,2]
            axes[row,0].plot(np.arange(len(action))*.02,action,label=label,color=color,lw=1.4)
            axes[row,1].plot(np.arange(len(yaw))*.004,yaw,label=label,color=color,lw=1,alpha=.85)
        axes[row,0].set_title(f'{name}: joint {joint} target offset')
        axes[row,0].set_ylabel('Radians')
        axes[row,1].set_title(f'{name}: measured torso yaw rate')
        axes[row,1].axhline(COMMANDS[name][2],color='black',ls='--',lw=1,label='Command')
        axes[row,1].set_ylabel('Radians / second')
        summary[name]={}
        for label,data in [('baseline',before),('candidate',after)]:
            yaw=data['angular_velocity'][50:,index:index+seeds,...,2]
            summary[name][label]=dict(yaw_rate_mean=float(yaw.mean()),yaw_rate_std=float(yaw.std()))
    for ax in axes.flat:
        ax.set_xlabel('Seconds after steady window begins');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Matched reverse and spin traces: identical commands and reset seed')
    fig.tight_layout()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=150)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
