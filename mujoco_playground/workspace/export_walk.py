"""Export walking weights with the current locomotion observation convention."""
import argparse
import json
from pathlib import Path
from brax.io import model
import numpy as np


def convert_walk_params(params, history, activation):
    """Fold normalization precisely and substitute the fixed upright command.

    Constant channels can have near-zero standard deviation. Substituting their
    known values avoids huge weights and float32 cancellation on the robot.
    Only upright orientation commands are supported by this walking policy.
    """
    mean = np.asarray(params[0].mean, dtype=np.float64)
    std = np.asarray(params[0].std, dtype=np.float64)
    fixed = np.concatenate([np.arange(9, 12) + 36*i for i in range(history)])
    fixed_values = np.tile([0., 0., 1.], history)
    layers = []
    entries = list(params[1]['params'].values())
    for i, entry in enumerate(entries):
        kernel = np.asarray(entry['kernel'], dtype=np.float64)
        bias = np.asarray(entry['bias'], dtype=np.float64)
        if i == 0:
            if kernel.shape[0] != 36*history:
                raise ValueError('Observation width mismatch')
            bias = bias + ((fixed_values-mean[fixed])/std[fixed]) @ kernel[fixed]
            variable = np.ones(len(mean), dtype=bool)
            variable[fixed] = False
            bias = bias - (mean[variable]/std[variable]) @ kernel[variable]
            kernel = kernel / std[:, None]
            kernel[fixed] = 0.
        if i == len(entries)-1:
            if kernel.shape[1] != 24:
                raise ValueError('Expected a 12-action tanh-normal policy head')
            kernel = kernel[:, :12]
            bias = bias[:12]
        layers.append(dict(type='dense', activation='tanh' if i == len(entries)-1 else activation,
                           shape=[None, len(bias)], weights=[kernel.tolist(), bias.tolist()]))
    return dict(in_shape=[None, 36*history], layers=layers)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',required=True);p.add_argument('--out')
    args=p.parse_args();run=json.loads((Path(args.params).parent/'run.json').read_text());c=run['config']
    net=convert_walk_params(model.load_params(args.params),c['observation_history'],c['activation'])
    expected=36*c['observation_history']
    if net['in_shape'][1]!=expected:raise ValueError(f'Expected {expected} observation inputs')
    data={**net,'behavior':'locomotion','action_types':['position']*12,'action_scale':c['action_scale'],
          'default_joint_pos':run['home_joint_pos'],'kp':run['kp'][0],'kd':run['kd'][0],
          'joint_lower_limits':run['joint_lower_limits'],'joint_upper_limits':run['joint_upper_limits'],'observation_history':c['observation_history'],
          'observation_layout':['ang_vel[3]','gravity[3]','command_xyyaw[3]','desired_world_z[3]','joint_pos_minus_default[12]','last_action[12]'],
          'joint_names':[f'leg_{leg}_{j}' for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)],
          'model_sha256':run['model_sha256'],'ring_signal':'reward_only','orientation_command':[0.,0.,1.],
          'command_low':c['command_low'],'command_high':c['command_high']}
    output=Path(args.out) if args.out else Path(args.params).parent/'policy_walk.json'
    output.write_text(json.dumps(data,indent=2))
    print(f'Wrote {output}. Export only: validate in simulation before hardware deployment.')

if __name__=='__main__':main()
