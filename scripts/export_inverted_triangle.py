#!/usr/bin/env python3
"""Export the reviewed simulation's exact phase targets; offline preparation only."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

COMMIT = '8904c2a836951250317300b2722e36376a810a50'
PLAN_HASH = '204c0a54172874749b6702e2802872b265489bffc18ce9b20bb08b2bde41e369'
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != COMMIT:
        raise ValueError('Use the immutable reviewed checkout at ' + COMMIT)
    sys.path.insert(0, str(source))
    core = importlib.import_module('motion.inverted_triangle.core')
    sim = importlib.import_module('motion.inverted_triangle.simulate')
    import numpy as np
    plan_dir = source / 'motion/inverted_triangle/results/robustness_20260912/plan'
    plan_bytes = (plan_dir / 'plan.json').read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != PLAN_HASH:
        raise ValueError('Plan hash mismatch')
    # Check code and candidates against Git, not just the mutable working tree.
    paths = list((source / 'motion/inverted_triangle').glob('*.py')) + list(plan_dir.glob('*.json'))
    for path in paths:
        committed = subprocess.check_output(['git', 'show', COMMIT + ':' + path.relative_to(source).as_posix()], cwd=source)
        if path.read_bytes().replace(b'\r\n', b'\n') != committed.replace(b'\r\n', b'\n'):
            raise ValueError('Edited export source: ' + str(path))
    robot = core.Robot()
    initial = robot.initial[7:].copy()
    previous = initial.copy()
    segments = []
    for stage, item in enumerate(json.loads(plan_bytes)['stages']):
        raw = (plan_dir / item['candidate']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['candidate_sha256']:
            raise ValueError('Candidate hash mismatch')
        candidate = json.loads(raw)
        phases = sim.trajectory(previous, candidate['full_pose'], core.LEGS.index(item['leg']),
                                item['landing_delta'], item['direction'],
                                pre_shift_pose=candidate.get('pre_shift_pose'),
                                landing_pose=candidate.get('landing_pose'),
                                touchdown_pose=candidate.get('touchdown_pose'))
        for name, target, minimum in phases:
            steps = int(np.ceil(core.duration(previous, target, minimum) / robot.m.opt.timestep))
            segments.append(dict(stage=stage, leg=item['leg'], phase=name, steps=steps, target=target.tolist()))
            previous = target.copy()
    result = dict(schema_version=1, source_commit=COMMIT, plan_sha256=PLAN_HASH,
                  model_sha256=robot.manifest['model_sha256'], axial_gap_m=.009,
                  joint_names=['leg_' + leg + '_' + str(i) for leg in core.LEGS for i in (1, 2, 3)],
                  hz=520, initial=initial.tolist(), kp=core.KP.tolist(), kd=core.KD.tolist(),
                  speed=core.SPEED.tolist(), acceleration=core.ACCEL.tolist(), segments=segments,
                  total_steps=sum(s['steps'] for s in segments))
    assert result['total_steps'] == 67278
    dest = ROOT / 'ros2_ws/src/neural_controller/launch/inverted_triangle_plan.json'
    dest.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    evidence = ROOT / 'hardware_testing/inverted_triangle/source'
    evidence.mkdir(parents=True, exist_ok=True)
    for path in plan_dir.glob('*.json'):
        shutil.copyfile(path, evidence / path.name)
    (evidence / 'export_provenance.json').write_text(json.dumps(dict(
        source_commit=COMMIT, source_hashes=core.provenance(),
        export_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        plan_sha256=PLAN_HASH, model_sha256=robot.manifest['model_sha256']), indent=2) + '\n')
    print(f'Exported {len(segments)} segments, {result["total_steps"]} steps to {dest}')


if __name__ == '__main__':
    main()
