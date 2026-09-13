"""Convert a preserved W&B integration artifact to a motor-free C++ parity test.

Usage: python prepare_measured_fixture.py ARTIFACT_DIRECTORY OUTPUT.csv
No simulations, network access or robot access are performed here.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def prepare(source, destination):
    source = Path(source)
    initial = json.loads((source / 'initial_integration_state.json').read_text())
    report = json.loads((source / 'report.json').read_text())
    data = np.load(source / 'integration.npz')
    assert report['engineering_pass'] and report['environment_steps'] == 24051
    assert report['source_hashes']['handoff_entry.py'] == 'a2dd16334d08bd3cbfd166da98dac660b33e2b423b78787e8691c9583b07b9fe'
    assert report['source_hashes']['balanced_support.py'] == 'a1f166f2bbb044ae50350996b5b0bee8043ad5c70386b5cc1eaff79523438962'
    # mjSTATE_INTEGRATION: time, qpos(19), qvel(18), then remaining integrator state.
    state = np.asarray(initial['state'])
    assert initial['state_spec'] == 8191
    np.testing.assert_allclose(state[1:20], report['initial_q'], atol=0, rtol=0)
    qpos = np.vstack([state[1:20], data['qpos'][:-1]])
    qvel = np.vstack([state[20:38], data['qvel'][:-1]])
    quat = qpos[:, 3:7]
    w, x, y, z = quat.T
    gravity = -np.column_stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    rows = np.column_stack([qpos[:, 7:], qvel[:, 6:], gravity, data['commands'], data['kd']])
    with Path(destination).open('w') as stream:
        np.savetxt(stream, [np.r_[qpos[0, 7:], initial['command']]], delimiter=',', fmt='%.17g')
        np.savetxt(stream, rows, delimiter=',', fmt='%.17g')
    result = dict(source_run='360ffcd09344487d', rows=len(rows),
                  integration_sha256=hashlib.sha256((source/'integration.npz').read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    prepare(*sys.argv[1:])
