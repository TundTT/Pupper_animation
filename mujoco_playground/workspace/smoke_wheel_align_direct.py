"""Direct-control, sensor-only, reward-gating, and checkpoint rollout checks."""

import argparse
import functools
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import HOLD, LOWER, ROTATE, domain_randomize_wheel_align
from workspace.wheel_align_evaluate import make_evaluator
from workspace.wheel_align_training import wrap_for_alignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--position-velocities", action="store_true")
    parser.add_argument("--history-taps", type=int, nargs=4)
    parser.add_argument("--clearance-band", action="store_true")
    args = parser.parse_args()
    cfg = configs.get_wheel_align_direct_config()
    if args.params:
        from ml_collections import ConfigDict

        cfg = ConfigDict(json.loads((args.params.parent / "config.json").read_text()))
    if args.position_velocities:
        cfg.direct.observation_frame_size = 64
    if args.history_taps is not None:
        cfg.direct.history_taps = tuple(args.history_taps)
    if args.clearance_band:
        cfg.direct.lift_clearance_upper = 0.010
        cfg.reward_config.scales.hold = -1.0
    frame_size = cfg.direct.observation_frame_size
    obs_size = frame_size * cfg.observation_history
    cfg.align.initial_completed_max = 0
    cfg.automatic_commands = False
    env = PupperWheelAlignDirectEnv(cfg)
    if args.clearance_band:
        cost = np.asarray(
            env._lift_cost(jp.array([0.0, 0.003, 0.008, 0.010, 0.012, 0.025]))
        )
        assert np.all(cost[1:4] == 0.0)
        assert cost[4] < cost[0] < cost[5]
        assert cfg.align.unload_dwell_steps == 3 and cfg.align.settle_dwell_steps == 5
        print(
            "PASS: reward favors small unloaded clearance over ground contact and excessive lift; success gates unchanged",
            flush=True,
        )
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(1))
    assert state.obs.shape == (obs_size,) and env.action_size == 12
    np.testing.assert_array_equal(
        state.obs.reshape(4, frame_size), jp.tile(state.obs[:frame_size], (4, 1))
    )
    state = env.set_command(state, 1)
    for i in range(12):
        a = jp.zeros(12).at[i].set(0.1)
        delta = env.position_targets(state.info, a) - env.position_targets(
            state.info, jp.zeros(12)
        )
        expected = np.zeros(12)
        expected[i] = cfg.action_scale[i] * 0.1
        np.testing.assert_allclose(delta, expected, atol=1e-6)
    print(
        f"PASS: {obs_size} sensor observations, repeated startup frames, all 12 independent position outputs",
        flush=True,
    )

    # Neither privileged measurements nor training-only stage labels may affect
    # the actor input or its affine action mapping.
    info = dict(state.info)
    info.update(
        phase=jp.full(4, HOLD),
        lift_reference=jp.full(4, 100.0),
        stance_shift=jp.full(2, 100.0),
        init_xy=jp.full(2, 100.0),
    )
    ps = state.pipeline_state.replace(
        x=state.pipeline_state.x.replace(
            pos=state.pipeline_state.x.pos.at[:, :2].add(50.0)
        )
    )
    original = env._get_obs(
        state.pipeline_state, dict(state.info), state.obs, jax.random.PRNGKey(4)
    )
    changed = env._get_obs(ps, info, state.obs, jax.random.PRNGKey(4))
    np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(
        env.position_targets(info, jp.zeros(12)),
        env.position_targets(state.info, jp.zeros(12)),
    )
    shifted = dict(state.info)
    turns = 2 * jp.pi * jp.array([50.0, -40.0, 30.0, -20.0])
    shifted["wheel_zero"] += turns
    shifted["wheel_idle_target"] += turns
    ps = state.pipeline_state.replace(
        q=state.pipeline_state.q.at[env._wheel_qpos].add(turns)
    )
    shifted_obs = env._get_obs(ps, shifted, state.obs, jax.random.PRNGKey(4))
    np.testing.assert_allclose(original, shifted_obs, atol=3e-5)
    np.testing.assert_allclose(
        env.position_targets(shifted, jp.zeros(12))[env._wheel_rows]
        - env.position_targets(state.info, jp.zeros(12))[env._wheel_rows],
        turns,
        atol=3e-5,
    )
    print(
        "PASS: no privileged actor/control inputs; whole-turn encoder-frame invariance",
        flush=True,
    )

    for i in range(16):
        action = jp.clip(
            state.info["last_policy_action"]
            + 0.005 * jax.random.normal(jax.random.PRNGKey(i + 20), (12,)),
            -1.0,
            1.0,
        )
        state = step(state, action)
        assert all(
            np.isfinite(v).all()
            for v in jax.tree.leaves((state.obs, state.reward, state.metrics))
        )
    print(
        "PASS: 16 randomized direct-action physics steps, finite rewards/observations",
        flush=True,
    )
    if frame_size >= 56:
        assert np.isclose(state.obs[55], 16 * env.dt)
        repeated = env.set_command(state, 1)
        assert np.isclose(repeated.obs[55], state.obs[55])
        changed_command = env.set_command(state, 2)
        assert changed_command.obs[55] == 0.0
        print(
            "PASS: button-age observation increments on ticks, resets only on command change"
        )
        if env._sparse_history:
            np.testing.assert_array_equal(
                state.obs.reshape((4, frame_size)),
                state.info["sensor_history"][jp.array(env._history_taps)],
            )
            np.testing.assert_array_equal(
                repeated.info["sensor_history"][1:], state.info["sensor_history"][1:]
            )
            print(
                "PASS: sparse sensor history uses specified real-time taps; command refresh does not advance history"
            )
    if frame_size == 64:
        ps = state.pipeline_state.replace(
            qd=state.pipeline_state.qd.at[env._joint_dof[env._pos_rows]].add(1.0)
        )
        before = env._get_obs(
            state.pipeline_state, dict(state.info), state.obs, jax.random.PRNGKey(4)
        )
        after = env._get_obs(ps, dict(state.info), state.obs, jax.random.PRNGKey(4))
        np.testing.assert_allclose(
            after[56:64] - before[56:64],
            1.0 / cfg.direct.position_velocity_scale,
            atol=1e-6,
        )
        print("PASS: native encoder velocity features follow named joint rows")

    # Reaching the angle at speed earns nothing; settled/unloaded verified
    # rotation earns exactly one +2 event; actual touchdown earns +3 separately.
    info = dict(state.info)
    info.update(
        phase=jp.array([ROTATE, 0, 0, 0]),
        dwell=jp.zeros(4, dtype=jp.int32),
        rotation_verified=jp.zeros(4, dtype=bool),
        completed=jp.zeros(4, dtype=bool),
        initial_completed=jp.zeros(4, dtype=bool),
        settle_paid=jp.zeros(4, dtype=bool),
    )
    z = jp.array([0.005, 0.0, 0.0, 0.0])
    load = jp.array([0.0, 10.0, 10.0, 10.0])
    err = jp.zeros(4)
    for _ in range(10):
        env._advance_phase(info, z, load, err, jp.ones(4))
    assert not info["rotation_verified"][0]
    for _ in range(cfg.align.settle_dwell_steps):
        env._advance_phase(info, z, load, err, jp.zeros(4))
    assert info["phase"][0] == LOWER
    reward, _ = env._alignment_reward(
        state.pipeline_state,
        info,
        jp.zeros(12),
        z,
        load,
        err,
        jp.zeros(4),
        info["phase"],
    )
    assert np.isclose(reward["settle"] * cfg.reward_config.scales.settle * env.dt, 2.0)
    again, _ = env._alignment_reward(
        state.pipeline_state,
        info,
        jp.zeros(12),
        z,
        load,
        err,
        jp.zeros(4),
        info["phase"],
    )
    assert again["settle"] == 0.0
    completed = 0.0
    for _ in range(cfg.align.touchdown_dwell_steps):
        completed += env._advance_phase(
            info, jp.zeros(4), jp.ones(4) * 10, err, jp.zeros(4)
        )
    assert completed == 1.0 and info["completed"][0]
    info["initial_completed"] = jp.ones(4, dtype=bool)
    info["settle_paid"] = jp.zeros(4, dtype=bool)
    pre, _ = env._alignment_reward(
        state.pipeline_state,
        info,
        jp.zeros(12),
        z,
        load,
        err,
        jp.zeros(4),
        info["phase"],
    )
    assert pre["settle"] == 0.0
    print(
        "PASS: gated one-time +2 settle / +3 touchdown; no pre-completed or repeated credit",
        flush=True,
    )

    wrapped = wrap_for_alignment(
        env,
        episode_length=2,
        action_repeat=1,
        randomization_fn=functools.partial(
            domain_randomize_wheel_align,
            rng=jax.random.split(jax.random.PRNGKey(22), 2),
        ),
    )
    batch = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(23), 2))
    advance = jax.jit(wrapped.step)
    for _ in range(2):
        batch = advance(batch, batch.info["last_policy_action"])
    assert np.all(batch.done) and np.all(batch.info["step"] == 0)
    assert not np.any(batch.info["settle_paid"])
    print(
        "PASS: DR/PPO auto-reset refreshes latched encoder defaults and reward latches",
        flush=True,
    )

    if args.params:
        cfg.automatic_commands = True
        net = networks.make_ppo_networks(
            obs_size,
            12,
            policy_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.policy.hidden_layer_sizes),
            activation=jax.nn.elu,
            preprocess_observations_fn=running_statistics.normalize,
        )
        audit = make_evaluator(
            cfg,
            networks.make_inference_fn(net),
            domain_randomize_wheel_align,
            env_type=PupperWheelAlignDirectEnv,
        )
        metrics = audit(model.load_params(args.params))
        assert all(np.isfinite(v) for v in metrics.values())
        output = args.output or args.params.parent / "cloned_audit.json"
        output.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics, indent=2), flush=True)
    print("ALL DIRECT SMOKE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
