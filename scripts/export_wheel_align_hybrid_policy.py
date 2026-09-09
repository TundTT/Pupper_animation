#!/usr/bin/env python3
"""Export the audited guard actor directly to RTNeural; run with the training venv.

The reference worktree is only read. Normalization is folded into layer zero;
NormalTanh's scale head is discarded and tanh(location) gives eight actions.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import jax
import mujoco
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks

BLOCKS = [
    ("body_angular_velocity", 3, "rad/s, body-aligned IMU frame"),
    ("projected_gravity", 3, "body-aligned IMU frame; world [0,0,-1]"),
    ("effective_command_one_hot", 5, "active command in LIFT/ROTATE/VERIFY, otherwise stand"),
    ("joint_position", 12, "position rows: q-default; wheel rows: atan2(sin(q),cos(q))"),
    ("joint_velocity", 12, "all encoder velocities times 0.1; no wheel sign correction"),
    ("last_action", 8, "previous direct network output, initially zero"),
    ("target_error_sin", 4, "sin(wrap(target-wheel_angle)), all wheels"),
    ("target_error_cos", 4, "cos(wrap(target-wheel_angle)), all wheels"),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference-out', type=Path, required=True)
    args = p.parse_args()
    root = args.reference_root.resolve()
    run = root / 'mujoco_playground/workspace/hybrid_results/attempt2_guard'
    audit = json.loads((run / 'audit64.json').read_text())
    checkpoint = root / audit['params']  # selection comes from the audit, not latest.json
    cfg = json.loads((run / 'config.json').read_text())
    latest = json.loads((run / 'latest.json').read_text())
    assert checkpoint.parent == run and checkpoint.name == 'mjx_params'
    assert audit['all_four_success_count'] == 50 and audit['fall_count'] == 0
    assert sha(checkpoint) == sha(root / latest['path'])
    hashes = json.loads((run / 'source_hashes.json').read_text())
    for name in ('hybrid_env.py', 'hybrid_train.py'):
        relative = 'mujoco_playground/workspace/' + name
        assert sha(root / relative) == sha(run / 'source' / name) == hashes[relative]
    xml_relative = 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
    assert sha(root / xml_relative) == hashes[xml_relative]
    assert cfg['policy_layers'] == [128, 128, 128] and cfg['activation'] == 'elu'
    assert cfg['action_size'] == 8 and cfg['observation_size'] == 51
    assert cfg['ppo']['normalize_observations'] and cfg['slew_rate'] == .25
    assert cfg['abduction_guard'] == .35
    spec = importlib.util.spec_from_file_location('hybrid_configs', root / 'mujoco_playground/workspace/configs.py')
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    mj = mujoco.MjModel.from_xml_path(str(root / xml_relative))
    pos, wheels = cfg['position_rows'], cfg['wheel_rows']
    assert pos == constants.POSITION_ACTUATOR_ROWS == [0, 1, 3, 4, 6, 7, 9, 10]
    assert wheels == constants.WHEEL_ACTUATOR_ROWS == [2, 5, 8, 11]
    for row, name in enumerate(constants.JOINT_NAMES):
        assert mj.joint(name).id == row + 1 == mj.actuator_trnid[row, 0]
    params = model.load_params(str(checkpoint))
    mean, std = (np.asarray(x, dtype=np.float64) for x in (params[0].mean, params[0].std))
    actor = params[1]['params']
    assert mean.shape == std.shape == (51,) and np.all(std > 0)
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert set(actor) == {f'hidden_{i}' for i in range(4)}
    layers = []
    for i, shape in enumerate(((51, 128), (128, 128), (128, 128), (128, 16))):
        layer = actor[f'hidden_{i}']
        kernel, bias = np.asarray(layer['kernel'], dtype=np.float64), np.asarray(layer['bias'], dtype=np.float64)
        assert kernel.shape == shape and bias.shape == (shape[1],)
        if i == 0:
            bias = bias - (mean / std) @ kernel
            kernel = kernel / std[:, None]
        if i == 3:
            kernel, bias = kernel[:, :8], bias[:8]
        layers.append(dict(type='dense', activation='tanh' if i == 3 else 'elu',
                           shape=[None, len(bias)],
                           weights=[kernel.astype(np.float32).tolist(), bias.astype(np.float32).tolist()]))
    layout, offset = [], 0
    for name, size, transform in BLOCKS:
        layout.append(dict(name=name, offset=offset, size=size, transform=transform))
        offset += size
    assert offset == mean.size == actor['hidden_0']['kernel'].shape[0]  # one frame, no history
    lo, hi = mj.jnt_range[1:, 0].copy(), mj.jnt_range[1:, 1].copy()
    lo[wheels], hi[wheels] = -2., 2.  # wheel velocity bounds, not position limits
    payload = dict(in_shape=[None, 51], layers=layers, behavior='wheel_align_hybrid',
                   observation_history=1, single_observation_size=51, observation_layout=layout,
                   observation_normalization='checkpoint mean/std folded into layer 0', observation_clip=None,
                   policy_action_size=8, position_joint_rows=pos, wheel_joint_rows=wheels,
                   command_states=constants.COMMAND_STATES, command_leg=[-1, 1, 0, 2, 3],
                   joint_names=constants.JOINT_NAMES,
                   action_types=['velocity' if i in wheels else 'position' for i in range(12)],
                   default_joint_pos=constants.DEFAULT_POSE.tolist(), action_scale=[.5, 1.6, 0.] * 4,
                   joint_lower_limits=lo.tolist(), joint_upper_limits=hi.tolist(), ctrl_dt=.02,
                   training_config=cfg, provenance=dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                   final_step=latest['step'], audit=audit, source_sha256=hashes,
                   config_sha256=sha(run / 'config.json'), constants_sha256=sha(root / 'mujoco_playground/workspace/configs.py')))
    serialized = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    payload = json.loads(serialized)
    net = networks.make_ppo_networks(51, 8, policy_hidden_layer_sizes=(128, 128, 128),
                                   activation=jax.nn.elu, preprocess_observations_fn=running_statistics.normalize)
    policy = networks.make_inference_fn(net)(params, deterministic=True)
    rng = np.random.default_rng(1901)
    obs = (mean + rng.normal(size=(128, 51)) * std * 2).astype(np.float32)
    for i in range(5):
        obs[i] = 0
        obs[i, 5] = -1
        obs[i, 6:11] = np.eye(5)[i]
        obs[i, 47:51] = -1
    expected = np.asarray(policy(jax.numpy.asarray(obs), jax.random.PRNGKey(0))[0])
    actual = obs
    for layer in payload['layers']:
        kernel, bias = [np.asarray(x, dtype=np.float32) for x in layer['weights']]
        actual = actual @ kernel + bias
        if layer['activation'] == 'elu':
            negative = actual < 0
            actual[negative] = np.expm1(actual[negative])
        else:
            actual = np.tanh(actual)
    error = float(np.max(np.abs(expected - actual)))
    assert np.isfinite(actual).all() and error < 3e-4, error
    args.out.write_text(serialized)
    np.savetxt(args.reference_out, np.concatenate([obs, expected], axis=1), delimiter=',', fmt='%.9g')
    print(f'PASS: {len(obs)} Brax/JSON cases, max action error {error:.9g}; 51 -> 128 -> 128 -> 128 -> 8')
    print(f'Checkpoint: {checkpoint}, SHA256 {sha(checkpoint)}, final step {latest["step"]}')


if __name__ == '__main__':
    sys.dont_write_bytecode = True  # reference worktree remains read-only
    main()
