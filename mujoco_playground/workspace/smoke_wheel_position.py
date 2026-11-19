"""Physics checks for alignment-only position PD, called by smoke_wheel_align."""

import jax
import mujoco
import numpy as np
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_env import (
    HOLD,
    LIFT,
    LOWER,
    ROTATE,
    domain_randomize_wheel_align,
    wrap,
)


def check_position_control(env, drive):
    wheel = np.array(configs.WHEEL_ACTUATOR_ROWS)
    kp = env._config.align.wheel_position_kp
    kd = env._config.align.wheel_position_kd
    assert not np.any(env.sys.actuator_ctrllimited[wheel])
    assert not np.any(env.sys.mj_model.actuator_ctrllimited[wheel])
    assert not np.any(
        env.sys.mj_model.jnt_limited[env.sys.mj_model.actuator_trnid[wheel, 0]]
    )
    np.testing.assert_array_equal(env.sys.actuator_gainprm[wheel, 0], kp)
    np.testing.assert_array_equal(env.sys.actuator_biasprm[wheel, 1], -kp)
    np.testing.assert_allclose(env.sys.actuator_biasprm[wheel, 2], -kd)
    np.testing.assert_array_equal(
        env.sys.actuator_forcerange[wheel], np.tile([-3.0, 3.0], (4, 1))
    )
    # Gains on the ordinary position joints are identical to the drive env.
    pos = np.array(configs.POSITION_ACTUATOR_ROWS)
    np.testing.assert_array_equal(
        env.sys.actuator_gainprm[pos], drive.sys.actuator_gainprm[pos]
    )
    np.testing.assert_array_equal(
        env.sys.actuator_biasprm[pos], drive.sys.actuator_biasprm[pos]
    )

    state = jax.jit(env.reset)(jax.random.PRNGKey(45))
    # Latency buffers start at valid angles, never raw zero. The policy has
    # eight leg outputs and cannot offset held wheel goals.
    scale_sign = env._action_scale[wheel] * env._ctrl_sign[wheel]
    buffered = state.info["action_buffer"][wheel] * scale_sign[:, None]
    np.testing.assert_allclose(
        wrap(buffered - state.info["wheel_reference"][:, None]), 0.0, atol=1e-6
    )
    idle = state.pipeline_state.q[env._wheel_qpos]

    @jax.jit
    def advance_reference(s):
        s = s.replace(info=dict(s.info))
        env._advance_wheel_reference(s)
        return s

    high = state.pipeline_state.q.at[2].add(0.1)
    airborne = state.replace(
        pipeline_state=jax.jit(env.pipeline_init)(high, jp.zeros(env._nv))
    )
    airborne = airborne.replace(
        info=dict(state.info, phase=jp.full(4, ROTATE), target=idle + 1.0)
    )
    moved = advance_reference(airborne)
    np.testing.assert_allclose(
        wrap(moved.info["wheel_reference"] - idle),
        env._config.align.wheel_reference_speed * env.dt,
        atol=1e-6,
    )
    airborne = airborne.replace(info=dict(airborne.info, phase=jp.full(4, LIFT)))
    waiting = advance_reference(airborne)
    np.testing.assert_allclose(
        wrap(waiting.info["wheel_reference"] - idle), 0.0, atol=1e-6
    )
    ground_q = state.pipeline_state.q.at[2].add(-0.02)
    grounded = state.replace(
        pipeline_state=jax.jit(env.pipeline_init)(ground_q, jp.zeros(env._nv))
    )
    grounded = grounded.replace(info=dict(moved.info, phase=jp.full(4, ROTATE)))
    stopped = advance_reference(grounded)
    np.testing.assert_allclose(
        wrap(stopped.info["wheel_reference"] - idle), 0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        wrap(
            stopped.info["action_buffer"][wheel] * scale_sign[:, None] - idle[:, None]
        ),
        0.0,
        atol=1e-6,
    )
    grounded = grounded.replace(
        info=dict(
            state.info,
            phase=jp.full(4, HOLD),
            wheel_reference=idle + 0.3,
            target=idle + 0.3,
        )
    )
    held = advance_reference(grounded)
    np.testing.assert_allclose(
        wrap(held.info["wheel_reference"] - idle), 0.3, atol=1e-6
    )
    aborted = dict(
        state.info, phase=jp.full(4, LOWER), rotation_verified=jp.zeros(4, dtype=bool)
    )
    for _ in range(10):
        env._advance_phase(
            aborted, jp.zeros(4), jp.ones(4) * 5, jp.zeros(4), jp.zeros(4)
        )
    assert not np.any(aborted["completed"])
    print(
        "PASS: phase-gated bounded rotation, grounded latency flush, persistent holds, no aborted false completion",
        flush=True,
    )

    # Reachability uses the actual cylinder orientation, including half-width.
    d = mujoco.MjData(env.sys.mj_model)
    d.qpos[:] = np.asarray(env._init_q)
    d.qpos[env._hip_qpos] += np.asarray(env._hip_sign) * 0.6
    mujoco.mj_forward(env.sys.mj_model, d)
    ids = env._wheel_geom_id
    axis_z = np.abs(d.geom_xmat[ids].reshape(4, 3, 3)[:, 2, 2])
    extent = (
        0.0505 * np.sqrt(1.0 - axis_z**2) + env.sys.mj_model.geom_size[ids, 1] * axis_z
    )
    clearance = d.geom_xpos[ids, 2] - extent
    assert np.all(clearance > env._config.align.clearance_min), clearance
    print(
        f"PASS: largest randomized wheels reachable; hip +0.6 rad clearances={clearance * 1000} mm",
        flush=True,
    )

    # Probe the real MuJoCo actuator force, not just our ctrl formula. Raw ctrl
    # exceeds the old +/-21 range; it must produce exactly kp*wrapped_error.
    for turns in (-50, -1, 0, 1, 50):
        q = state.pipeline_state.q.at[env._wheel_qpos].set(idle + turns * 2 * jp.pi)
        ps = state.pipeline_state.replace(q=q)
        goals = env._default_pose.at[wheel].set(wrap(idle + 0.3))
        ctrl = env._position_ctrl(ps, goals)
        np.testing.assert_allclose(ctrl[wheel] - q[env._wheel_qpos], 0.3, atol=5e-5)
        d = mujoco.MjData(env.sys.mj_model)
        d.qpos[:] = np.asarray(q)
        d.qvel[:] = 0.0
        d.ctrl[:] = np.asarray(ctrl)
        mujoco.mj_forward(env.sys.mj_model, d)
        np.testing.assert_allclose(d.actuator_force[wheel], kp * 0.3, atol=1e-4)
    # Both directions across the +/-pi seam take the short path.
    for current, goal, delta in [
        (np.pi - 0.02, -np.pi + 0.02, 0.04),
        (-np.pi + 0.02, np.pi - 0.02, -0.04),
    ]:
        ps = state.pipeline_state.replace(
            q=state.pipeline_state.q.at[env._wheel_qpos].set(current)
        )
        ctrl = env._position_ctrl(ps, env._default_pose.at[wheel].set(goal))
        np.testing.assert_allclose(ctrl[wheel] - current, delta, atol=1e-6)
    print(
        "PASS: real PD restoring torque at rest; +/-50 revolutions, seam crossings, no velocity ctrl clipping",
        flush=True,
    )

    # Deterministically force DR endpoint multipliers, and verify no accidental
    # restoration of velocity bias or changes to other randomized physics.
    keys = jax.random.split(jax.random.PRNGKey(8), 2)
    randomized, _ = domain_randomize_wheel_align(
        env.sys, keys, kp_multiplier_range=(0.5, 0.5), kd_multiplier_range=(2.0, 2.0)
    )
    np.testing.assert_allclose(randomized.actuator_gainprm[:, wheel, 0], 0.5 * kp)
    np.testing.assert_allclose(randomized.actuator_biasprm[:, wheel, 1], -0.5 * kp)
    np.testing.assert_allclose(randomized.actuator_biasprm[:, wheel, 2], -2.0 * kd)

    # Compare zero-action angular restoration in actual physics. Put the robot
    # in free fall high enough to isolate wheels from ground forces for 0.6 s.
    # Recompute ctrl each 4 ms exactly as the align env does, with a fixed goal.
    def restore(model, position):
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(env._init_q)
        data.qpos[2] = 4.0
        target = data.qpos[env._wheel_qpos].copy()
        data.qpos[env._wheel_qpos] += 0.3
        data.ctrl[:] = np.asarray(env._default_pose)
        for _ in range(150):
            if position:
                current = data.qpos[env._wheel_qpos]
                data.ctrl[wheel] = current + np.arctan2(
                    np.sin(target - current), np.cos(target - current)
                )
            else:
                data.ctrl[wheel] = 0.0
            mujoco.mj_step(model, data)
        error = np.arctan2(
            np.sin(target - data.qpos[env._wheel_qpos]),
            np.cos(target - data.qpos[env._wheel_qpos]),
        )
        return np.mean(np.abs(error))

    pd_error = restore(env.sys.mj_model, True)
    velocity_error = restore(drive.sys.mj_model, False)
    assert pd_error < 0.025 and velocity_error > 0.20, (pd_error, velocity_error)
    print(
        f"PASS: 0.3 rad perturbation, 0.6 s recovery: PD error={pd_error:.5f} rad, velocity-zero error={velocity_error:.5f} rad",
        flush=True,
    )
