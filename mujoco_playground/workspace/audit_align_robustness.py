"""Checkpoint comparison and per-stage traces under full wheel randomization."""
import argparse
import functools
import json
from pathlib import Path

import jax
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from jax import numpy as jp
from ml_collections import ConfigDict

from workspace import configs
from workspace.wheel_align_env import PupperWheelAlignEnv, domain_randomize_wheel_align
from workspace.wheel_align_evaluate import make_evaluator
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv


class ProbeEnv(PupperWheelAlignEnv):
    def reset(self, rng):
        state = super().reset(rng)
        state.info['probe'] = jp.zeros((4, 4))
        return state

    def step(self, state, action):
        state = super().step(state, action)
        state.info['probe'] = jp.stack((
            self._wheel_clearance(state.pipeline_state),
            self._wheel_loads(state.pipeline_state),
            jp.abs(self._errors(state.pipeline_state, state.info)),
            jp.abs(state.pipeline_state.qd[self._wheel_dof]),
        ), axis=-1)
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--params', default='mjx_params_best')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--current-config', action='store_true')
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log-wandb', action='store_true')
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    cfg = (configs.get_wheel_align_config() if args.current_config else
           ConfigDict(json.loads((args.run / 'config.json').read_text())))
    cfg.align.initial_completed_max = 0
    direct = cfg.align.wheel_control_mode == 'direct_joint_position'
    if direct and args.trace:
        parser.error('--trace uses teacher-specific reference fields; omit it for direct policies')
    env_type = PupperWheelAlignDirectEnv if direct else ProbeEnv
    if direct and cfg.direct.get('encoder_hold', False):
        from workspace.wheel_align_refine_env import PupperWheelAlignRefineEnv
        env_type = PupperWheelAlignRefineEnv
        cfg.direct.reference_reset_probability = 0.0
        if cfg.direct.get('encoder_gate', False):
            from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
            env_type = PupperWheelAlignGateEnv
    env = env_type(cfg)
    net = networks.make_ppo_networks(
        env.observation_size, env.action_size,
        policy_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
        activation=jax.nn.elu, preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = networks.make_inference_fn(net)
    params = model.load_params(args.run / args.params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluate = make_evaluator(cfg, make_policy, domain_randomize_wheel_align,
                              count=args.count, seed=args.seed, env_type=env_type)
    report = evaluate(params)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.log_wandb:
        import wandb
        metadata=json.loads((args.run/'wandb_run.json').read_text())
        run=wandb.init(entity='QuadMorph',project='wheel-leg lift and align triangle base',
                       id=metadata['id'],resume='must')
        run.log(report)
        if args.video:
            import mediapy as media
            from workspace import wheel_align_visualize
            video_cfg=cfg.copy_and_resolve_references()
            video_cfg.automatic_commands=False
            video_env=env_type(video_cfg)
            frames,fps=wheel_align_visualize.render(video_env,make_policy(params,deterministic=True),jax.random.PRNGKey(0))
            video_path=args.output.with_suffix('.mp4')
            media.write_video(video_path,frames,fps=fps)
            run.log({'eval/video':wandb.Video(str(video_path),fps=fps,format='mp4')})
        run.finish()
    if not args.trace:
        return
    policy = make_policy(params, deterministic=True)
    wrapped = DomainRandomizationVmapWrapper(env, functools.partial(
        domain_randomize_wheel_align,
        rng=jax.random.split(jax.random.PRNGKey(args.seed), args.count)))

    @jax.jit
    def rollout():
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(args.seed + 1), args.count))

        def step(carry, _):
            state, alive, rng = carry
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            nxt = wrapped.step(state, action)
            nxt = jax.tree.map(lambda old, new: jp.where(
                alive.reshape(alive.shape + (1,) * (new.ndim - alive.ndim)), new, old), state, nxt)
            sample = (state.info['command'], state.info['phase'], nxt.info['probe'],
                      nxt.info['lift_reference'], nxt.info['stance_shift'], action,
                      nxt.metrics['body_drift_dist'], alive, nxt.info['completed'])
            return (nxt, alive & ~nxt.done.astype(bool), rng), sample

        return jax.lax.scan(step, (state, jp.ones(args.count, dtype=bool),
                                  jax.random.PRNGKey(args.seed + 2)), None,
                            length=cfg.episode_length)[1]

    arrays = jax.device_get(rollout())
    keys = ('command', 'phase', 'probe', 'lift_reference', 'shift', 'action', 'drift', 'alive', 'completed')
    np.savez_compressed(args.output.with_suffix('.npz'), **dict(zip(keys, arrays)))
    command, phase, probe, refs, shift, action, drift, alive, completed = arrays
    failures = []
    for leg, row in configs.FOOT_ROW_BY_LEG.items():
        for episode in np.where(~completed[-1, :, row])[0]:
            selected = (command[:, episode] == configs.COMMAND_STATES.index(leg)) & alive[:, episode]
            idx = np.where(selected)[0]
            tail = idx[-50:]
            failures.append(dict(leg=leg, episode=int(episode),
                phases={str(p): int(np.sum(phase[idx, episode, row] == p)) for p in range(5)},
                end_probe=probe[tail, episode, row].mean(axis=0).tolist(),
                max_ref=float(refs[idx, episode, row].max()),
                end_action=action[tail, episode].mean(axis=0).tolist(),
                drift_max=float(drift[idx, episode].max())))
    report['failures'] = failures
    report['drift_by_quarter_p95'] = [float(np.percentile(x, 95)) for x in np.array_split(drift, 4)]
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
