"""Query the teacher on states visited by a student/expert action mixture.

Dataset aggregation is training-only. The deployed student remains one ordinary
MLP and receives neither the teacher's state nor its privileged observations.
"""

import argparse
import functools
import json
import shutil
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
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("workspace/output/align_direct_reference"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace/output/align_direct_aggregated")
    )
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--expert-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=505)
    parser.add_argument(
        "--clearance-target",
        type=float,
        help="teacher-only clearance curriculum; student physics and gates are unchanged",
    )
    parser.add_argument(
        "--wheel-goal-labels",
        action="store_true",
        help="after verified unloading, supervise the calibrated wheel goal directly instead of teacher slew increments",
    )
    parser.add_argument(
        "--position-step-multiplier",
        type=float,
        default=1.0,
        help="training-only amplification of teacher stance/lift correction relative to the previous commanded target",
    )
    parser.add_argument(
        "--stance-lookahead",
        type=int,
        default=1,
        help="teacher steps for stance/lift labels; wheel labels always use the current one-step decision",
    )
    args = parser.parse_args()
    teacher_path = Path("workspace/output/wheel-align_2026-09-07_20-41-57")
    teacher_cfg = ConfigDict(json.loads((teacher_path / "config.json").read_text()))
    teacher_cfg.align.initial_completed_max = 0
    if args.clearance_target is not None:
        teacher_cfg.align.clearance_target = args.clearance_target
    cfg = configs.get_wheel_align_direct_config()
    student_cfg = ConfigDict(
        json.loads((args.student.parent / "config.json").read_text())
    )
    gated = student_cfg.direct.get('encoder_gate', False)
    env_type = PupperWheelAlignDirectEnv
    if gated:
        from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
        env_type = PupperWheelAlignGateEnv
        cfg.direct.encoder_gate = True
        cfg.direct.encoder_hold = True
        cfg.direct.reference_reset_probability = 0.
        cfg.direct.reference_reset_path = ''
        if args.wheel_goal_labels:
            parser.error('Wheel-goal regression labels do not apply to enable outputs')
    cfg.policy.hidden_layer_sizes = student_cfg.policy.hidden_layer_sizes
    cfg.direct.observation_frame_size = student_cfg.direct.observation_frame_size
    if "history_taps" in student_cfg.direct:
        cfg.direct.history_taps = student_cfg.direct.history_taps
    for name in ("motor_angle_noise", "gravity_noise", "angular_velocity_noise"):
        cfg.dr[name] = student_cfg.dr[name]
    cfg.align.initial_completed_max = 0
    source_metadata = json.loads((args.reference / "manifest.json").read_text())
    # Older datasets predate sparse history and contain four adjacent frames.
    source_taps = tuple(source_metadata.get("history_taps", (0, 1, 2, 3)))
    if source_taps != tuple(cfg.direct.history_taps):
        raise ValueError("Reference and student sensor-history taps must match")

    def policy(size, actions, layers, path):
        net = networks.make_ppo_networks(
            size,
            actions,
            policy_hidden_layer_sizes=layers,
            value_hidden_layer_sizes=layers,
            activation=jax.nn.elu,
            preprocess_observations_fn=running_statistics.normalize,
        )
        return networks.make_inference_fn(net)(
            model.load_params(path), deterministic=True
        )

    expert_policy = policy(376, 3, (128, 128), teacher_path / "mjx_params_best")
    student_policy = policy(
        cfg.direct.observation_frame_size * cfg.observation_history,
        12,
        tuple(cfg.policy.hidden_layer_sizes),
        args.student,
    )

    class QueryTeacher(PupperWheelAlignEnv):
        def step(self, state, actual):
            info = dict(state.info)
            # Use actual phase labels and reconstruct the teacher's controller
            # state from the commands that really drove the student. Keeping a
            # counterfactual lift integrator produces non-causal cloning labels.
            for name in (
                "command",
                "phase",
                "phase_age",
                "dwell",
                "tracked",
                "completed",
                "rotation_verified",
                "calibration_home",
                "home",
                "target",
                "wheel_idle_target",
                "init_xy",
                "init_forward",
            ):
                info[name] = actual.info[name]
            previous = actual.info['issued_action'] if gated else actual.info['last_policy_action']
            position_offset = previous * jp.array(
                cfg.action_scale
            )
            abdomen = np.array(
                [
                    i
                    for i, row in enumerate(self._pos_rows)
                    if configs.JOINT_NAMES[row].endswith("_1")
                ]
            )
            hips = np.array(
                [
                    i
                    for i, row in enumerate(self._pos_rows)
                    if configs.JOINT_NAMES[row].endswith("_2")
                ]
            )
            shift = (
                jp.linalg.pinv(self._shift_map[abdomen])
                @ position_offset[self._pos_rows[abdomen]]
            )
            info["stance_shift"] = jp.clip(shift, -0.025, 0.025)
            lifted = self._hip_sign * (
                position_offset[self._hip_rows] - (self._shift_map @ shift)[hips]
            )
            lifting = (info["phase"] == 1) | (info["phase"] == 2) | (info["phase"] == 3)
            info["lift_reference"] = jp.where(
                lifting, jp.clip(lifted, 0.0, self._config.align.max_lift_angle), 0.0
            )
            info["last_applied_action"] = position_offset
            previous_wheel = (
                actual.info["wheel_zero"] + position_offset[self._wheel_rows]
            )
            info["wheel_reference"] = wrap(
                jp.where(info["tracked"], previous_wheel, info["wheel_idle_target"])
            )
            owner = jp.maximum(self._command_leg[info["command"]], 0)
            base_shift = jp.where(
                info["command"] != 0, self._support_shifts[owner], jp.zeros(2)
            )
            info["last_policy_action"] = (
                info["last_policy_action"]
                .at[1:]
                .set(
                    jp.clip(
                        (shift - base_shift) / self._config.align.balance_adjust,
                        -1.0,
                        1.0,
                    )
                )
            )
            info["rng"], key = jax.random.split(info["rng"])
            obs = self._get_obs(
                actual.pipeline_state, info, jp.roll(state.obs, -94), key
            )
            state = state.replace(
                pipeline_state=actual.pipeline_state, info=info, obs=obs
            )
            action, _ = expert_policy(obs, key)
            first = super().step(state, action)
            if args.stance_lookahead == 1:
                return first

            def preview(state, _):
                key = state.info["rng"]
                action, _ = expert_policy(state.obs, key)
                return super(QueryTeacher, self).step(state, action), None

            future, _ = jax.lax.scan(
                preview, first, None, length=args.stance_lookahead - 1
            )
            # Only the supervision label uses future leg targets. Keep the
            # real-time wheel decision, teacher history and phase unchanged.
            first.info["last_applied_action"] = (
                first.info["last_applied_action"]
                .at[self._pos_rows]
                .set(future.info["last_applied_action"][self._pos_rows])
            )
            return first

    expert = QueryTeacher(teacher_cfg)
    student = env_type(cfg)
    randomizer = functools.partial(
        domain_randomize_wheel_align,
        rng=jax.random.split(jax.random.PRNGKey(args.seed), args.count),
    )
    ew = DomainRandomizationVmapWrapper(expert, randomizer)
    sw = DomainRandomizationVmapWrapper(student, randomizer)

    @jax.jit
    def record():
        keys = jax.random.split(jax.random.PRNGKey(args.seed + 1), args.count)
        es, ss = ew.reset(keys), sw.reset(keys)

        def advance(carry, _):
            es, ss, alive, key = carry
            key, action_key, mix_key = jax.random.split(key, 3)
            es = ew.step(es, ss)
            current = ss.pipeline_state.q[:, student._wheel_qpos]
            desired = (
                expert._default_pose
                + es.info["last_applied_action"] * expert._action_scale
            )
            desired = desired.at[:, student._wheel_rows].set(
                current + wrap(es.info["wheel_reference"] - current)
            )
            default = (
                jp.broadcast_to(student._default_pose, desired.shape)
                .at[:, student._wheel_rows]
                .set(ss.info["wheel_zero"])
            )
            label = (desired - default) / student._action_scale
            if args.position_step_multiplier != 1.0:
                previous = ss.info["last_policy_action"][:, student._pos_rows]
                label = label.at[:, student._pos_rows].set(
                    previous
                    + args.position_step_multiplier
                    * (label[:, student._pos_rows] - previous)
                )
            if args.wheel_goal_labels:
                label = label.at[:, student._wheel_rows].set(
                    jp.where(ss.info["phase"] == 2, 0.0, label[:, student._wheel_rows])
                )
            predicted, _ = student_policy(ss.obs, action_key)
            use_expert = jax.random.bernoulli(
                mix_key, args.expert_probability, (args.count,)
            )
            command = ss.info["command"]
            row = jp.maximum(student._command_leg[command], 0)
            phase = jp.where(
                command != 0,
                jp.take_along_axis(ss.info["phase"], row[:, None], axis=1)[:, 0],
                0,
            )
            expert_action = label
            if gated:
                enables = (ss.info['phase'] == 2) & (
                    jp.arange(4)[None, :] == student._command_leg[command][:, None])
                expert_action = label.at[:, student._wheel_rows].set(jp.where(enables, .8, -.8))
            action = jp.where(use_expert[:, None], expert_action, predicted)
            nxt = sw.step(ss, action)
            if gated:
                # Wheel labels in the first12 columns are feedback features,
                # not regression targets: retain the command actually issued
                # by the small controller on this trajectory.
                label = label.at[:, student._wheel_rows].set(nxt.info['issued_action'][:, student._wheel_rows])
                label = jp.concatenate((label, enables.astype(jp.float32)), axis=-1)
            sample = (ss.obs, label, command * 5 + phase, alive)
            ss = nxt
            return (es, ss, alive & ~ss.done.astype(bool), key), sample

        (_, ss, alive, _), samples = jax.lax.scan(
            advance,
            (
                es,
                ss,
                jp.ones(args.count, dtype=bool),
                jax.random.PRNGKey(args.seed + 2),
            ),
            None,
            length=cfg.episode_length,
        )
        return samples, ss.info["completed"], alive

    samples, completed, survived = jax.device_get(record())
    obs, actions, group, alive = samples
    if not np.isfinite(obs[alive]).all() or not np.isfinite(actions[alive]).all():
        raise RuntimeError('Nonfinite live corrective examples')
    valid = alive & (np.max(np.abs(actions[..., :12]), axis=-1) < 1.0)
    with np.load(args.reference / "demonstrations.npz") as original:
        if original['actions'].shape[-1] != actions.shape[-1]:
            raise ValueError('Corrective and original supervision contracts differ')
        episode = np.broadcast_to(
            np.arange(args.count) + int(original["episode"].max()) + 1, group.shape
        )
        combined = {
            k: np.concatenate((original[k], v[valid]), axis=0)
            for k, v in dict(
                obs=obs, actions=actions, group=group, episode=episode
            ).items()
        }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "demonstrations.npz", **combined)
    # The pose prior stays the original successful reference family; recovery
    # states expand cloning data, not the definition of a good stance.
    shutil.copy2(args.reference / "poses.npz", args.output / "poses.npz")
    manifest = dict(
        student=str(args.student),
        teacher=str(teacher_path / "mjx_params_best"),
        successful_episodes=json.loads((args.reference / "manifest.json").read_text())[
            "successful_episodes"
        ],
        expert_probability=args.expert_probability,
        episodes=args.count,
        seed=args.seed,
        teacher_clearance_target=float(teacher_cfg.align.clearance_target),
        student_sensor_noise={
            name: float(cfg.dr[name])
            for name in ("motor_angle_noise", "gravity_noise", "angular_velocity_noise")
        },
        stance_lookahead=args.stance_lookahead,
        wheel_goal_labels=args.wheel_goal_labels,
        position_step_multiplier=args.position_step_multiplier,
        history_taps=list(cfg.direct.history_taps),
        additional_samples=int(valid.sum()),
        total_samples=len(combined["obs"]),
        survived=int(survived.sum()),
        full_successes=int((np.all(completed, axis=1) & survived).sum()),
        completed_per_leg=np.sum(completed, axis=0).tolist(),
        observation_size=student.observation_size,
        action_size=12,
        encoder_gate=gated,
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
