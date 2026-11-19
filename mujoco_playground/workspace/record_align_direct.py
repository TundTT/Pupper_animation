"""Record phase-diverse, randomized successful teacher demonstrations."""

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
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import (
    PupperWheelAlignEnv,
    domain_randomize_wheel_align,
    wrap,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("workspace/output/wheel-align_2026-09-07_20-41-57"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/output/align_direct_reference")
    )
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--save-reset-states", action="store_true",
                        help="save successful phase-matched physical states for training-only resets")
    parser.add_argument(
        "--teacher-settle-steps",
        type=int,
        help="teacher-only airborne settled dwell; student success gates remain unchanged",
    )
    parser.add_argument(
        "--clearance-target",
        type=float,
        help="training-reference lift margin; leaves unload/settle gates unchanged",
    )
    parser.add_argument(
        "--sensor-noise",
        type=float,
        help="student IMU/position-encoder observation noise; physical DR and teacher unchanged",
    )
    parser.add_argument(
        "--position-velocities",
        action="store_true",
        help="include the eight native abduction/hip encoder velocities",
    )
    args = parser.parse_args()
    cfg = ConfigDict(json.loads((args.teacher / "config.json").read_text()))
    cfg.align.initial_completed_max = 0
    if args.clearance_target is not None:
        cfg.align.clearance_target = args.clearance_target
    if args.teacher_settle_steps is not None:
        if args.teacher_settle_steps < cfg.align.settle_dwell_steps:
            parser.error("Teacher dwell must not weaken the original settle gate")
        cfg.align.settle_dwell_steps = args.teacher_settle_steps
    teacher = PupperWheelAlignEnv(cfg)
    direct_cfg = configs.get_wheel_align_direct_config()
    if args.position_velocities:
        direct_cfg.direct.observation_frame_size = 64
    if args.sensor_noise is not None:
        for key in ("motor_angle_noise", "gravity_noise", "angular_velocity_noise"):
            direct_cfg.dr[key] = args.sensor_noise
    direct_cfg.direct.reference_path = ""
    direct_cfg.reward_config.scales.imitation = 0.0
    student = PupperWheelAlignDirectEnv(direct_cfg)
    net = networks.make_ppo_networks(
        376,
        3,
        policy_hidden_layer_sizes=(128, 128),
        value_hidden_layer_sizes=(128, 128),
        activation=jax.nn.elu,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = networks.make_inference_fn(net)(
        model.load_params(args.teacher / "mjx_params_best"), deterministic=True
    )
    wrapped = DomainRandomizationVmapWrapper(
        teacher,
        functools.partial(
            domain_randomize_wheel_align,
            rng=jax.random.split(jax.random.PRNGKey(args.seed), args.count),
        ),
    )

    def initial_sensor(state, key):
        info = dict(state.info)
        student._initialize_direct_info(state.pipeline_state, info)
        obs = student.sensor_observation(
            state.pipeline_state,
            info,
            jp.zeros(student.observation_size),
            key,
            reset_history=True,
        )
        return obs, info["wheel_zero"], info["last_policy_action"], info["imu_buffer"]

    def next_sensor(
        state, history, zero, previous_action, imu_buffer, command_age, key
    ):
        info = dict(state.info)
        info.update(
            wheel_zero=zero,
            last_policy_action=previous_action,
            last_act=previous_action,
            imu_buffer=imu_buffer,
            command_age=command_age,
        )
        obs = student.sensor_observation(state.pipeline_state, info, history, key)
        return obs, info["imu_buffer"]

    @jax.jit
    def record():
        state = wrapped.reset(
            jax.random.split(jax.random.PRNGKey(args.seed + 1), args.count)
        )
        obs, zero, last, imu = jax.vmap(initial_sensor)(
            state, jax.random.split(jax.random.PRNGKey(args.seed + 2), args.count)
        )
        initial_delta = wrap(
            state.info["target"] - state.pipeline_state.q[:, teacher._wheel_qpos]
        )

        def advance(carry, _):
            state, obs, last, imu, age, alive, rng = carry
            rng, key, sensor_key = jax.random.split(rng, 3)
            action, _ = policy(state.obs, key)
            next_state = wrapped.step(state, action)
            # Desired teacher positions, before sampled actuator latency. Wheel
            # labels use a local calibrated frame, never an absolute-angle clip.
            desired = (
                teacher._default_pose
                + next_state.info["last_applied_action"] * teacher._action_scale
            )
            current = state.pipeline_state.q[:, teacher._wheel_qpos]
            wheel = current + wrap(next_state.info["wheel_reference"] - current)
            desired = desired.at[:, teacher._wheel_rows].set(wheel)
            default = (
                jp.broadcast_to(student._default_pose, desired.shape)
                .at[:, student._wheel_rows]
                .set(zero)
            )
            label = (desired - default) / student._action_scale
            command = state.info["command"]
            row = jp.maximum(teacher._command_leg[command], 0)
            phase = jp.where(
                command != 0,
                jp.take_along_axis(state.info["phase"], row[:, None], axis=1)[:, 0],
                0,
            )
            pose = (
                state.pipeline_state.q[:, teacher._joint_qpos[teacher._pos_rows]]
                - teacher._default_pose[teacher._pos_rows]
            )
            sample = (obs, label, command, phase, pose, alive)
            if args.save_reset_states:
                snapshot = dict(state.info)
                # Only small reset fields, never MJX contact/solver arrays.
                snapshot = {k: snapshot[k] for k in (
                    'step', 'command', 'phase_age', 'calibration_home', 'home',
                    'target', 'wheel_idle_target', 'rotation_verified', 'tracked',
                    'completed', 'phase', 'dwell', 'sequence_order',
                    'sequence_cursor', 'command_switch_step', 'init_xy', 'init_forward')}
                snapshot.update(q=state.pipeline_state.q, qd=state.pipeline_state.qd,
                                wheel_zero=zero, command_age=age,
                                last_policy_action=last, imu_buffer=imu)
                sample += (snapshot,)
            age = jp.where(
                next_state.info["command"] != state.info["command"], 0, age + 1
            )
            new_obs, imu = jax.vmap(next_sensor)(
                next_state,
                obs,
                zero,
                label,
                imu,
                age,
                jax.random.split(sensor_key, args.count),
            )
            return (
                next_state,
                new_obs,
                label,
                imu,
                age,
                alive & ~next_state.done.astype(bool),
                rng,
            ), sample

        (state, _, _, _, _, alive, _), samples = jax.lax.scan(
            advance,
            (
                state,
                obs,
                last,
                imu,
                jp.zeros(args.count, dtype=jp.int32),
                jp.ones(args.count, dtype=bool),
                jax.random.PRNGKey(args.seed + 3),
            ),
            None,
            length=cfg.episode_length,
        )
        success = jp.all(state.info["completed"], axis=1) & alive
        return samples, success, initial_delta

    samples, success, delta = jax.device_get(record())
    obs, actions, command, phase, pose, alive = samples[:6]
    success_ids = np.where(success)[0]
    if len(success_ids) < 16:
        raise RuntimeError(
            f"Only {len(success_ids)} full teacher successes; need >=16 diverse sequences"
        )
    valid = alive & success[None, :] & (np.max(np.abs(actions), axis=-1) < 1.0)
    episode = np.broadcast_to(np.arange(args.count), command.shape)
    group = command * 5 + phase
    poses = np.zeros((5, 5, 32, 8), dtype=np.float32)
    bank_valid = np.zeros((5, 5, 32), dtype=bool)
    counts = {}
    # Farthest-point pose coverage within each measured phase, not time bins.
    for cmd in range(5):
        for ph in range(5):
            mask = valid & (command == cmd) & (phase == ph)
            points = pose[mask]
            counts[f"{cmd}/{ph}"] = len(points)
            if not len(points):
                continue
            points = points[:: max(len(points) // 10000, 1)]
            distance = np.full(len(points), np.inf)
            idx = int(np.argmin(np.sum(points**2, axis=1)))
            for k in range(min(32, len(points))):
                poses[cmd, ph, k] = points[idx]
                bank_valid[cmd, ph, k] = True
                distance = np.minimum(
                    distance, np.mean(((points - points[idx]) / 0.15) ** 2, axis=1)
                )
                idx = int(np.argmax(distance))
    for cmd in range(1, 5):
        for ph in (1, 2, 3, 4):
            if not bank_valid[cmd, ph].any():
                raise RuntimeError(
                    f"Missing successful command/phase reference: {cmd}/{ph}"
                )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_reset_states:
        snapshots = samples[6]
        random = np.random.default_rng(args.seed + 13)
        selected = []
        for cmd in range(1, 5):
            for ph in (1, 2, 3):
                candidates = np.argwhere(valid & (command == cmd) & (phase == ph))
                selected.extend(candidates[random.choice(len(candidates), 128,
                                                        replace=len(candidates) < 128)])
        ticks, episodes = np.asarray(selected).T
        saved = {k: v[ticks, episodes] for k, v in snapshots.items()}
        # Real teacher sensor history supports any existing <=50-tick tap layout.
        past = np.maximum(ticks[:, None] - np.arange(51)[None, :], 0)
        saved['sensor_history'] = obs[past, episodes[:, None], :student._single_obs_dim]
        np.savez_compressed(args.output / 'reset_states.npz', **saved)
    np.savez_compressed(
        args.output / "demonstrations.npz",
        obs=obs[valid],
        actions=actions[valid],
        group=group[valid],
        episode=episode[valid],
    )
    np.savez_compressed(args.output / "poses.npz", poses=poses, valid=bank_valid)
    manifest = dict(
        teacher=str(args.teacher / "mjx_params_best"),
        seed=args.seed,
        teacher_clearance_target=float(cfg.align.clearance_target),
        teacher_settle_steps=int(cfg.align.settle_dwell_steps),
        student_sensor_noise={
            key: float(direct_cfg.dr[key])
            for key in ("motor_angle_noise", "gravity_noise", "angular_velocity_noise")
        },
        episodes=args.count,
        successful_episodes=success_ids.tolist(),
        samples=int(valid.sum()),
        phase_counts=counts,
        initial_rotation_delta_bins={
            leg: np.histogram(delta[success, row], bins=8, range=(-np.pi, np.pi))[
                0
            ].tolist()
            for leg, row in configs.FOOT_ROW_BY_LEG.items()
        },
        observation_size=student.observation_size,
        action_size=12,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
