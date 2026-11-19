"""Executable regression smoke: .venv/bin/python -m workspace.smoke_wheel_align."""

import functools

import jax
import numpy as np
from jax import numpy as jp

from workspace import configs
from workspace.randomize import domain_randomize_wheeled
from workspace.wheel_align_env import (
    HOLD,
    LIFT,
    LOWER,
    ROTATE,
    PupperWheelAlignEnv,
    domain_randomize_wheel_align,
    wrap,
)
from workspace.wheel_align_training import wrap_for_alignment
from workspace.wheel_env import PupperWheelEnv


def main():
    cfg = configs.get_wheel_align_config()
    cfg.automatic_commands = False
    cfg.align.initial_completed_max = 0
    env = PupperWheelAlignEnv(cfg)
    reset, step, select = (
        jax.jit(env.reset),
        jax.jit(env.step),
        jax.jit(env.set_command),
    )
    phases = []
    for seed in range(12):
        s = reset(jax.random.PRNGKey(seed))
        phases.append(
            np.asarray(
                wrap(s.pipeline_state.q[env._wheel_qpos] - s.info["calibration_home"])
            )
        )
        assert s.obs.shape == (376,) and env.action_size == 3
    assert np.ptp(phases, axis=0).min() > 4.0
    print(
        "PASS: independent full-circle reset phases; 376 observations / 3 actions",
        flush=True,
    )
    s = select(s, 1)
    saved_target = np.asarray(s.info["target"]).copy()
    assert int(s.info["phase"][1]) == LIFT
    for i in range(16):
        action = jax.random.uniform(
            jax.random.PRNGKey(100 + i), (3,), minval=-0.5, maxval=0.5
        )
        s = step(s, action)
        assert all(
            np.all(np.isfinite(x))
            for x in jax.tree.leaves((s.obs, s.reward, s.metrics, s.pipeline_state.q))
        )
    np.testing.assert_array_equal(s.info["target"], saved_target)
    print(
        "PASS: 16 random physics steps; finite reward components, persistent targets",
        flush=True,
    )
    scoped = select(reset(jax.random.PRNGKey(5)), 1)
    scoped = step(scoped, jp.zeros(3))
    assert np.max(np.abs(scoped.info["stance_shift"])) <= cfg.align.shift_step + 1e-7
    assert np.all(scoped.info["lift_reference"] == 0.0)
    for _ in range(30):
        scoped = step(scoped, jp.zeros(3))
    assert np.all(np.asarray(scoped.info["lift_reference"])[[0, 2, 3]] == 0.0)
    scoped = select(scoped, 0)
    for _ in range(100):
        scoped = step(scoped, jp.zeros(3))
    np.testing.assert_allclose(scoped.info["lift_reference"], 0.0, atol=1e-6)
    np.testing.assert_allclose(scoped.info["stance_shift"], 0.0, atol=1e-6)
    print(
        "PASS: coherent bounded stance shift, selected hip only, gradual return to home",
        flush=True,
    )
    # Necessary support motion is free inside 35 mm; idle retains a tighter
    # 20 mm band. The fixed episode origin is never re-anchored on a button.
    shifted = select(reset(jax.random.PRNGKey(12)), 1)
    ps = shifted.pipeline_state
    ps = ps.replace(x=ps.x.replace(pos=ps.x.pos.at[env._torso_idx - 1, 0].add(0.03)))
    arguments = (
        ps,
        shifted.info,
        jp.zeros(12),
        env._wheel_clearance(ps),
        env._wheel_loads(ps),
        env._errors(ps, shifted.info),
        ps.qd[env._wheel_dof],
    )
    active_rewards, _ = env._alignment_reward(*arguments, shifted.info["phase"])
    idle_rewards, _ = env._alignment_reward(*arguments, jp.zeros(4, dtype=jp.int32))
    assert active_rewards["body_drift"] == 0.0 and idle_rewards["body_drift"] > 0.0
    print("PASS: 35 mm active support allowance; 20 mm idle drift band", flush=True)

    # Weight transfer must earn progress even before clearance becomes positive,
    # and the signal should scale with supported weight, not randomized mass.
    unload_costs = []
    for loads in (jp.array([10., 10., 10., 10.]),
                  jp.array([12., 4., 12., 12.]),
                  jp.array([14., 0., 13., 13.])):
        args = list(arguments)
        args[4] = loads
        rewards, _ = env._alignment_reward(*args, shifted.info["phase"])
        unload_costs.append(float(rewards["unload"]))
        args[4] = loads * 1.3
        scaled, _ = env._alignment_reward(*args, shifted.info["phase"])
        np.testing.assert_allclose(scaled["unload"], rewards["unload"], atol=1e-6)
    assert unload_costs[0] > unload_costs[1] > unload_costs[2] == 0.0
    assert idle_rewards["unload"] == 0.0
    print("PASS: dense unloading progress before liftoff; mass-normalized; no idle bonus", flush=True)

    # Deterministic phase guards: touching cannot rotate/complete, settling needs
    # low speed AND angle, touchdown needs actual force, then hold never expires.
    s = reset(jax.random.PRNGKey(9))
    for command in (1, 2, 3, 4):
        s = select(s, command)
        leg = int(env._command_leg[command])
        info = s.info
        z, load, error, speed = jp.zeros(4), jp.ones(4) * 5, jp.ones(4), jp.zeros(4)
        for _ in range(6):
            env._advance_phase(info, z, load, error, speed)
        assert int(info["phase"][leg]) == LIFT
        z, load = z.at[leg].set(0.004), load.at[leg].set(0.0)
        for _ in range(cfg.align.unload_dwell_steps):
            env._advance_phase(info, z, load, error, speed)
        assert int(info["phase"][leg]) == ROTATE
        error = error.at[leg].set(0.0)
        for _ in range(cfg.align.settle_dwell_steps):
            env._advance_phase(info, z, load, error, speed)
        assert int(info["phase"][leg]) == LOWER
        z = z.at[leg].set(0.0)
        for _ in range(6):
            env._advance_phase(info, z, load, error, speed)
        assert not bool(info["completed"][leg])
        load = load.at[leg].set(5.0)
        for _ in range(cfg.align.touchdown_dwell_steps):
            env._advance_phase(info, z, load, error, speed)
        assert int(info["phase"][leg]) == HOLD
    assert np.all(s.info["completed"])
    saved_target = np.asarray(s.info["target"]).copy()
    s = select(select(s, 0), 1)
    np.testing.assert_array_equal(s.info["target"], saved_target)
    assert np.all(s.info["phase"] == HOLD)
    disturbed = s.pipeline_state.replace(
        q=s.pipeline_state.q.at[env._wheel_qpos].set(s.info["target"] + 0.3)
    )
    np.testing.assert_allclose(env._errors(disturbed, s.info), -0.3, atol=1e-6)
    print(
        "PASS: guarded lift/rotate/lower/reweight, all-leg hold, reselection and disturbed angle feedback",
        flush=True,
    )

    # Geometry must reflect each randomized sys, with mixed gains intact.
    randomized, _ = domain_randomize_wheel_align(
        env.sys, jax.random.split(jax.random.PRNGKey(88), 3)
    )
    radii = np.asarray(randomized.geom_size[:, env._wheel_geom_id, 0])
    assert radii.min() >= 0.0455 and radii.max() <= 0.0505 and np.ptp(radii) > 0.001
    np.testing.assert_array_equal(
        randomized.actuator_biasprm[:, configs.WHEEL_ACTUATOR_ROWS, 1],
        -randomized.actuator_gainprm[:, configs.WHEEL_ACTUATOR_ROWS, 0],
    )
    ps = s.pipeline_state
    original = env.sys
    before = env._wheel_clearance(ps)
    env.sys = original.replace(
        geom_size=original.geom_size.at[env._wheel_geom_id, 0].add(0.002)
    )
    after = env._wheel_clearance(ps)
    assert np.all(np.asarray(before - after) > 0.0019)
    env.sys = original
    print(
        "PASS: per-wheel randomized geometry affects clearance, position PD survives DR",
        flush=True,
    )

    # Schedule traverses ALL four independent of performance, with idle gaps.
    info = dict(reset(jax.random.PRNGKey(10)).info)
    seen = []
    for _ in range(10):
        info["step"] = info["command_switch_step"]
        env._schedule(info)
        seen.append(int(info["command"]))
    assert set(seen) == {0, 1, 2, 3, 4}
    assert bool(jp.all(info["tracked"])) and int(info["command"]) == 0
    assert not bool(jp.any(info["completed"]))
    print(
        "PASS: clock-driven full sequence and idle, no false completion on timeout",
        flush=True,
    )

    # Exercise the actual PPO wrappers, vectorized DR and a timeout reset.
    randomizer = functools.partial(
        domain_randomize_wheel_align, rng=jax.random.split(jax.random.PRNGKey(77), 2)
    )
    wrapped = wrap_for_alignment(
        env, episode_length=2, action_repeat=1, randomization_fn=randomizer
    )
    ws = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(6), 2))
    initial_home = np.asarray(ws.info["home"]).copy()
    wstep = jax.jit(wrapped.step)
    ws = wstep(ws, jp.zeros((2, 3)))
    ws = wstep(ws, jp.zeros((2, 3)))
    assert np.all(ws.done) and np.all(ws.info["step"] == 0)
    assert not np.array_equal(ws.info["home"], initial_home)
    ws = wstep(ws, jp.zeros((2, 3)))
    assert np.all(ws.info["step"] == 1) and np.all(np.isfinite(ws.obs))
    print(
        "PASS: batched DR + PPO timeout restores fresh references/phase/physics together",
        flush=True,
    )

    # Shared wheel-drive path still works after factoring out its plumbing.
    wheel = PupperWheelEnv(configs.get_wheel_config())
    np.testing.assert_array_equal(
        wheel.sys.actuator_biasprm[configs.WHEEL_ACTUATOR_ROWS, 1], 0.0
    )
    assert np.all(wheel.sys.actuator_ctrllimited[configs.WHEEL_ACTUATOR_ROWS])
    drive_dr, _ = domain_randomize_wheeled(
        wheel.sys, jax.random.split(jax.random.PRNGKey(3), 2)
    )
    np.testing.assert_array_equal(
        drive_dr.actuator_biasprm[:, configs.WHEEL_ACTUATOR_ROWS, 1], 0.0
    )
    from workspace.smoke_wheel_position import check_position_control

    check_position_control(env, wheel)
    ds = jax.jit(wheel.reset)(jax.random.PRNGKey(0))
    ds = jax.jit(wheel.step)(ds, jp.zeros(12))
    assert np.isfinite(ds.reward) and ds.obs.shape == (132,)
    print("PASS: wheel-drive regression reset/step", flush=True)
    print("ALL SMOKE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
