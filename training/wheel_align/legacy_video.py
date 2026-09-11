"""Historical v2/v3 video schedule. Imports physics from the original checkout."""
from pathlib import Path
import csv
import json

def record_policy(policy, output, *, seed=20260910, max_steps=6656, training_step=0, renderer=None):
    if max_steps < 4 or max_steps > 6656 or max_steps % 4:
        raise ValueError('Video steps must be a multiple of four between 4 and 6656')
    import jax
    from jax import numpy as jp
    import numpy as np
    from training.wheel_align.env import AlignEnv
    from training.wheel_align import configs as c, contract as ct
    env = AlignEnv(noise=False)
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    @jax.jit
    def advance(state, command):
        state = env.select_command(state, command)
        def tick(s, _):
            def move(s):
                action, _ = policy(s.obs, jax.random.PRNGKey(0))
                return env.step(s, action)
            return jax.lax.cond(s.done > 0, lambda s: s, move, s), None
        return jax.lax.scan(tick, state, None, length=4)[0]
    qposes, rows = [], []
    for step in range(0, max_steps, 4):
        slot = min(step // 1664, 3)
        command = 0 if step % 1664 < 104 else slot + 1
        state = advance(state, jp.asarray(command))
        q = np.asarray(state.pipeline_state.q)
        motion = jax.tree.map(np.asarray, state.info['motion'])
        row = dict(seconds=(step+4)*c.CONTROL_DT, phase=int(motion['phase']), command=command,
            active_command=int(motion['active_command']), progress=float(motion['progress']),
            completed=int(motion['completed'].sum()), rotating=bool(motion['was_rotating']),
            gate_steps=int(motion['gate_steps']), done=bool(state.done))
        for k, leg in enumerate(('FR','FL','BR','BL')):
            target = float(motion['target'][k]); actual = float(q[9+3*k])
            row.update({leg+'_target': target, leg+'_actual': actual,
                        leg+'_error': float(ct.wrap(target-actual))})
        for key in ('wheel_gap','body_gap','impact_speed','unsafe_rotation'):
            row[key] = float(state.metrics[key])
        qposes.append(q); rows.append(row)
        if row['done']:
            break
    output = Path(output)
    renderer(qposes, rows, c.MODEL_PATH, output,
                caption=f'Policy step {training_step} | nominal rollout | seed {seed}')
    with output.with_suffix('.trace.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(seed=seed, training_step=int(training_step), fps=13, frames=len(rows),
        simulated_seconds=rows[-1]['seconds'], terminated=rows[-1]['done'],
        completed_wheels=rows[-1]['completed'], scenario='single nominal four-wheel sequence',
        status='Diagnostic rollout; consult the full audit results')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return output
