"""Render the actual policy in MJX; also usable with an unchanged historical checkout."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys


def write_video(qposes, diagnostics, model_path, output, *, caption, fps=13):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = .65
    camera.azimuth = 135
    camera.elevation = -25
    phases = ['IDLE', 'LIFT', 'ROTATE', 'VERIFY', 'LOWER', 'HOLD']
    commands = ['stand', 'FL', 'FR', 'BR', 'BL']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.mp4')
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer, imageio.get_writer(
                str(temporary), fps=fps, codec='libx264', quality=7, macro_block_size=16) as writer:
            for q, row in zip(qposes, diagnostics):
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                camera.lookat[:] = q[:3]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 83), fill='black')
                draw.text((8, 5), caption, fill='white')
                phase = phases[int(row['phase'])]
                draw.text((8, 21), f"t={row['seconds']:.2f}s  {phase}  request={commands[row['command']]}  active={commands[row['active_command']]}  progress={row['progress']:.2f}", fill='white')
                errors = '  '.join(f'{leg} {row[leg+"_error"]:+.2f}' for leg in ('FR','FL','BR','BL'))
                draw.text((8, 37), 'Target error (rad): ' + errors, fill='white')
                draw.text((8, 53), f"Completed={row['completed']}  rotating={row['rotating']}  gate_steps={row['gate_steps']}", fill='white')
                draw.text((8, 69), 'TERMINATED' if row['done'] else 'SIMULATION - not a hardware validation', fill='orange')
                writer.append_data(np.asarray(frame))
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def record_policy(policy, output, *, seed=20260910, max_steps=None, training_step=0, stage="sequence"):
    from training.wheel_align import configs as c
    budget=c.SEQUENCE_STEPS if stage=="sequence" else c.SINGLE_STEPS
    max_steps=budget if max_steps is None else max_steps
    if max_steps < 4 or max_steps > budget or max_steps % 4:
        raise ValueError('Video steps must be a multiple of four within the configured sequence budget')
    import jax
    from jax import numpy as jp
    import numpy as np
    from training.wheel_align.env import AlignEnv
    from training.wheel_align import configs as c, contract as ct, schedule
    env = AlignEnv(noise=False,automatic=True,stage=stage)
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    state.info['order']=jp.arange(1,5)
    state.info['early_interrupt']=jp.asarray(False)
    @jax.jit
    def advance(state):
        def tick(s, _):
            def move(s):
                action, _ = policy(s.obs, jax.random.PRNGKey(0))
                return env.step(s, action)
            return jax.lax.cond(s.done > 0, lambda s: s, move, s), None
        return jax.lax.scan(tick, state, None, length=4)[0]
    qposes, rows = [], []
    for step in range(0, max_steps, 4):
        state = advance(state)
        command = int(state.info['motion']['command'])
        q = np.asarray(state.pipeline_state.q)
        motion = jax.tree.map(np.asarray, state.info['motion'])
        row = dict(seconds=(step+4)*c.CONTROL_DT, phase=int(motion['phase']), command=command,
            active_command=int(motion['active_command']), progress=float(motion['progress']),
            completed=int(motion['completed'].sum()), rotating=bool(motion['was_rotating']),
            gate_steps=int(motion['gate_steps']), residual_gain=float(motion['residual_gain']), done=bool(state.done))
        for k, leg in enumerate(('FR','FL','BR','BL')):
            target = float(motion['target'][k]); actual = float(q[9+3*k])
            row.update({leg+'_target': target, leg+'_actual': actual,
                        leg+'_error': float(ct.wrap(target-actual))})
        for key in ('wheel_gap','body_gap','impact_speed','unsafe_rotation','clearance','floor_estimate','tilt','support_fraction','active_load','contact_peak_cost','floor_gate_blocked','stability_gate_blocked'):
            row[key] = float(state.metrics[key])
        for i in range(8):row['action_'+str(i)]=float(motion['last_action'][i]);row['proximal_q_'+str(i)]=float(q[7+ct.POS[i]])
        qposes.append(q); rows.append(row)
        if row['done'] or bool(schedule.finished(state.info['sequence'],state.info['motion'],4 if stage=='sequence' else 1)):
            break
    output = Path(output)
    write_video(qposes, rows, c.MODEL_PATH, output,
                caption=f'Policy step {training_step} | {stage} nominal | seed {seed}')
    with output.with_suffix('.trace.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = dict(seed=seed, training_step=int(training_step), fps=13, frames=len(rows),
        simulated_seconds=rows[-1]['seconds'], terminated=rows[-1]['done'],
        completed_wheels=rows[-1]['completed'], stage=stage, scenario='nominal four-wheel sequence' if stage=='sequence' else 'nominal front-left '+stage,
        lowering_and_settling_finished=bool(schedule.finished(state.info['sequence'],state.info['motion'],4 if stage=='sequence' else 1)),
        status='Diagnostic rollout; consult the full audit results')
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--max-steps', type=int, default=None, help='Short diagnostic/test video limit; default is the contract sequence budget')
    args = p.parse_args()
    # This worker imports the ORIGINAL implementation without changing it or its hashes.
    sys.path.insert(0, str(args.source_root.resolve()))
    if sys.platform.startswith('linux'):
        os.environ.setdefault('MUJOCO_GL', 'egl')
    import jax
    from brax.io import model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from training.wheel_align.train import network_factory, source_hashes
    config = json.loads(args.params.parent.joinpath('config.json').read_text())
    if config['source_hashes'] != source_hashes():
        raise ValueError('Video source checkout differs from the recorded training sources. Use the original checkout.')
    from training.wheel_align import configs as original_config
    net = network_factory()(original_config.OBSERVATION_SIZE, 8, preprocess_observations_fn=running_statistics.normalize)
    policy = networks.make_inference_fn(net)(model.load_params(str(args.params)), deterministic=True)
    version=config.get('motion_contract_version',original_config.MOTION_VERSION)
    if version<4:
        import runpy
        legacy=runpy.run_path(str(Path(__file__).with_name('legacy_video.py')))['record_policy']
        legacy(policy,args.out,training_step=args.step,max_steps=args.max_steps or 6656,renderer=write_video)
    elif version==4:
        from training.wheel_align.policy_video import record_policy as historical_record
        historical_record(policy,args.out,training_step=args.step,max_steps=args.max_steps)
    else:
        record_policy(policy, args.out, training_step=args.step, max_steps=args.max_steps,stage=config.get("curriculum_stage","sequence"))


if __name__ == '__main__':
    main()
