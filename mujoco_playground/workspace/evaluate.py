"""Evaluate a trained leg-lift policy across the DOMAIN-RANDOMIZED model distribution.

Why this exists
---------------
The rollout videos (`visualize.render`, and the ones train.py logs each eval) are
rendered from the raw env object, which bypasses brax's randomization wrapper. They
therefore show the policy on the NOMINAL model: step-time DR is still active (sensor
noise, kicks, action latency all live inside leg_lift_env.step/_get_obs), but the
PHYSICS randomization in randomize.py -- friction, kp/kd, mass, inertia, torso CoM --
is not. A video can look clean while the policy is still brittle to the parameter
spread the real robot actually sits somewhere inside.

This script closes that gap: it runs the O-button showcase across `--num_envs`
independently randomized copies of the robot and reports the DISTRIBUTION of outcomes,
which is the number that matters for sim-to-real. Use `--nominal` to reproduce the
video's conditions for a side-by-side.

Usage:
    python -m workspace.evaluate --params workspace/output/<run>/mjx_params
    python -m workspace.evaluate --params ... --nominal   # no physics DR, for comparison
"""

import argparse
import functools

import jax
import numpy as np
from brax.envs import training as brax_training
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from workspace import configs, visualize
from workspace.leg_lift_env import PupperLegLiftEnv
from workspace.randomize import domain_randomize

_ACTIVATIONS = {"swish": jax.nn.swish, "relu": jax.nn.relu, "tanh": jax.nn.tanh, "elu": jax.nn.elu}


def _build_inference(config, env, params):
    networks = ppo_networks.make_ppo_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(config.policy.hidden_layer_sizes),
        activation=_ACTIVATIONS[config.policy.activation],
    )
    return ppo_networks.make_inference_fn(networks)(params, deterministic=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a leg-lift policy under domain randomization.")
    p.add_argument("--params", required=True, help="path to brax mjx_params from train.py")
    p.add_argument("--num_envs", type=int, default=512, help="independently randomized robots")
    p.add_argument("--steps_per_command", type=int, default=100, help="hold each command this many steps")
    p.add_argument("--model_path", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--nominal", action="store_true",
        help="disable PHYSICS randomization (reproduces the rollout video's conditions)",
    )
    args = p.parse_args()

    config = configs.get_config()
    env = PupperLegLiftEnv(config, model_path=args.model_path)
    inference_fn = _build_inference(config, env, model.load_params(args.params))

    schedule = visualize.showcase_schedule(args.steps_per_command)
    # episode_length must cover the whole showcase or EpisodeWrapper truncates it and
    # the fall statistics below would silently measure the wrapper, not the policy.
    randomization_fn = (
        None if args.nominal
        else functools.partial(domain_randomize, rng=jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs))
    )
    wrapped = brax_training.wrap(
        env, episode_length=len(schedule) + 1, action_repeat=1, randomization_fn=randomization_fn
    )

    reset = jax.jit(wrapped.reset)
    step = jax.jit(wrapped.step)
    infer = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(args.seed)
    state = reset(jax.random.split(rng, args.num_envs))

    mode = "NOMINAL physics (video conditions)" if args.nominal else "RANDOMIZED physics"
    print(f"Evaluating {args.params}")
    print(f"  {mode}, {args.num_envs} envs, {len(schedule)} steps "
          f"({len(schedule) * config.ctrl_dt:.0f}s showcase)\n")

    ever_done = jp.zeros(args.num_envs, dtype=bool)
    per_cmd: dict[int, dict[str, list]] = {}
    for i, cmd in enumerate(schedule):
        state.info["command"] = jp.full((args.num_envs,), cmd, dtype=jp.int32)
        state.info["command_switch_step"] = jp.full((args.num_envs,), 2_000_000_000, dtype=jp.int32)
        rng, act_rng = jax.random.split(rng)
        action, _ = infer(state.obs, act_rng)
        state = step(state, action)
        ever_done = ever_done | (state.done > 0)
        # Only score the second half of each command window: the first half is the
        # raise/lower transient, not the hold we actually care about.
        if (i % args.steps_per_command) >= args.steps_per_command // 2:
            d = per_cmd.setdefault(cmd, {"foot": [], "drift": [], "tilt": [], "z": [], "alive": [], "stance": [], "yaw": []})
            d["foot"].append(np.asarray(state.metrics["lifted_foot_height"]))
            d["drift"].append(np.asarray(state.metrics["body_drift_dist"]))
            d["tilt"].append(np.asarray(state.metrics["tilt_deg"]))
            d["z"].append(np.asarray(state.metrics["torso_z"]))
            d["yaw"].append(np.asarray(state.metrics["yaw_deg"]))
            d["alive"].append(np.asarray(~ever_done))
            # How far the PLANTED legs sit from the home pose, in degrees -- the
            # directly readable version of the stance_pose reward, and the "does it
            # still look like it is standing normally" number.
            mask = np.asarray(env._stance_joint_mask[cmd])
            dev = np.asarray(state.pipeline_state.q[:, 7:]) - np.asarray(env._default_pose)
            d["stance"].append(np.rad2deg(np.sqrt((mask * dev**2).sum(-1) / mask.sum())))

    names = configs.COMMAND_STATES
    print(f"{'command':9} {'foot clearance (m)':>34} {'drift (m)':>16} {'tilt (deg)':>15} "
          f"{'torso_z':>9} {'stance dev':>13} {'yaw deg':>13}")
    print(f"{'':9} {'mean':>9}{'p10':>9}{'p50':>8}{'p90':>8} {'mean':>8}{'p90':>8} "
          f"{'mean':>7}{'p90':>8} {'mean':>9} {'deg rms':>8}{'p90':>5} {'mean':>7}{'p90':>6}")
    for cmd in [c for c in [0, 1, 2, 3, 4] if c in per_cmd]:
        d = per_cmd[cmd]
        alive = np.concatenate(d["alive"])
        sel = lambda k: np.concatenate(d[k])[alive]  # noqa: E731  score only upright robots
        f, dr, ti, z, st, yw = (sel("foot"), sel("drift"), sel("tilt"), sel("z"),
                                sel("stance"), sel("yaw"))
        foot = (f"{f.mean():>9.4f}{np.percentile(f,10):>9.4f}{np.percentile(f,50):>8.4f}{np.percentile(f,90):>8.4f}"
                if cmd != 0 else f"{'-- (standing) --':>34}")
        print(f"{names[cmd]:9} {foot} {dr.mean():>8.4f}{np.percentile(dr,90):>8.4f} "
              f"{ti.mean():>7.2f}{np.percentile(ti,90):>8.2f} {z.mean():>9.4f} "
              f"{st.mean():>8.2f}{np.percentile(st,90):>5.1f} "
              f"{yw.mean():>7.2f}{np.percentile(yw,90):>6.1f}")

    fell = float(jp.mean(ever_done.astype(jp.float32)))
    print(f"\nFELL / terminated at any point in the showcase: {fell * 100:.1f}% of {args.num_envs} robots")
    print("  (termination = tilt > "
          f"{np.rad2deg(config.terminal_body_angle):.0f} deg, torso < {config.terminal_body_z} m, "
          f"knee on floor, drift > {config.terminal_body_drift} m, "
          f"or yaw > {np.rad2deg(config.terminal_body_yaw):.0f} deg)")


if __name__ == "__main__":
    main()
