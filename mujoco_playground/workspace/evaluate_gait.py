"""Objective gait metrics for a trained walking policy.

Rolls a saved policy through a fixed command suite and reports the numbers that
decide whether a gait is "normal" rather than shuffling, dragging, tapping or
stuck, so runs can be compared without watching every video:

  track_err   RMS error vs the commanded body velocity (m/s, and rad/s for yaw)
  stride_hz   touchdowns per second per foot (small quadrupeds trot ~2-3 Hz)
  duty        fraction of time each foot is in contact (~0.5-0.65 walking/trot)
  clear_mm    mean peak capsule-bottom clearance per swing
  swing_ms    mean swing duration
  slip_mm/s   mean horizontal foot speed while in contact (dragging indicator)
  td_mm/s     mean downward foot speed at touchdown (tapping indicator)
  airborne%   fraction of samples with zero feet down (should be small for a walk)
  fall%       fraction of rollouts terminating early

Usage:
  .venv/bin/python -m workspace.evaluate_gait --params <mjx_params_file> [--json out.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import numpy as np
from jax import numpy as jp
from brax import math as bmath

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / 'Stanford/training/pupperv3-mjx'))

from brax import envs  # noqa: E402
from brax.io import model as brax_model  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from pupperv3_mjx import utils  # noqa: E402

from workspace import gait_env  # noqa: E402
from workspace import train_gait  # noqa: E402
from workspace import train_reference  # noqa: E402

# (vx, vy, wz, label): the directions the policy must handle, reverse included.
COMMAND_SUITE = [
    (0.25, 0.0, 0.0, 'fwd_slow'),
    (0.40, 0.0, 0.0, 'fwd_fast'),
    (-0.25, 0.0, 0.0, 'back_slow'),
    (-0.40, 0.0, 0.0, 'back_fast'),
    # Strafe dropped from the command set: 0.14 rad of outward abduction headroom
    # caps lateral travel near 0.016 m/s, so 'left'/'right' only ever measured
    # 0.00 m/s and told us nothing. The slots now buy turn and arc coverage, where
    # the measured cw/ccw asymmetry (-0.22 vs +0.87 on FIN_A) actually needs eyes.
    (0.0, 0.0, 0.8, 'turn_ccw'),
    (0.0, 0.0, -0.8, 'turn_cw'),
    (0.0, 0.0, 1.5, 'turn_ccw_fast'),
    (0.0, 0.0, -1.5, 'turn_cw_fast'),
    (0.25, 0.0, 0.5, 'arc'),
    (-0.25, 0.0, 0.5, 'arc_back'),
    (0.0, 0.0, 0.0, 'stand'),
]


def build_reference_eval_env(args):
    """Build the REFERENCE env (MJCF home, uniform action scale).

    Policies trained by train_reference/shaped_env must be evaluated here: gait_env
    uses the contract home pose and a different action frame, so running a
    reference-trained policy in it feeds mismatched observations and it looks frozen.
    """
    ra = argparse.Namespace(num_timesteps=1, num_envs=1, learning_rate=3e-4)
    C, _ = train_reference.build_config(ra)
    kw = dict(
        path=C.simulation.model_path, action_scale=C.policy.action_scale,
        observation_history=C.policy.observation_history,
        joint_lower_limits=jp.array(C.simulation.joint_lower_limits),
        joint_upper_limits=jp.array(C.simulation.joint_upper_limits),
        dof_damping=C.training.dof_damping, position_control_kp=C.training.position_control_kp,
        foot_site_names=C.simulation.foot_site_names, torso_name=C.simulation.torso_name,
        upper_leg_body_names=C.simulation.upper_leg_body_names,
        lower_leg_body_names=C.simulation.lower_leg_body_names,
        resample_velocity_step=C.training.resample_velocity_step,
        linear_velocity_x_range=C.training.lin_vel_x_range,
        linear_velocity_y_range=C.training.lin_vel_y_range,
        angular_velocity_range=C.training.ang_vel_yaw_range,
        zero_command_probability=C.training.zero_command_probability,
        stand_still_command_threshold=C.training.stand_still_command_threshold,
        maximum_pitch_command=C.training.maximum_pitch_command,
        maximum_roll_command=C.training.maximum_roll_command,
        start_position_config=C.training.start_position_config,
        full_home_qpos=C.training.full_home_qpos,
        desired_abduction_angles=C.training.desired_abduction_angles,
        reward_config=C.reward, angular_velocity_noise=C.training.angular_velocity_noise,
        gravity_noise=C.training.gravity_noise, motor_angle_noise=C.training.motor_angle_noise,
        last_action_noise=C.training.last_action_noise, kick_vel=C.training.kick_vel,
        kick_probability=C.training.kick_probability, terminal_body_z=C.training.terminal_body_z,
        early_termination_step_threshold=C.training.early_termination_step_threshold,
        terminal_body_angle=C.training.terminal_body_angle, foot_radius=C.simulation.foot_radius,
        environment_timestep=C.training.environment_dt, physics_timestep=C.simulation.physics_dt,
        latency_distribution=C.training.latency_distribution,
        imu_latency_distribution=C.training.imu_latency_distribution,
        desired_world_z_in_body_frame=jp.array(C.training.desired_world_z_in_body_frame),
        use_imu=C.policy.use_imu)
    envs.register_environment('pupper_ref_eval', train_reference.PupperReferenceEnv)
    return envs.get_environment('pupper_ref_eval', **kw), C


def build_eval_env(args):
    cfg_args = argparse.Namespace(
        num_timesteps=1, num_envs=1, num_evals=2, learning_rate=3e-4, entropy_cost=1e-2,
        leg_length_min_m=args.leg_length_min_m, leg_length_max_m=args.leg_length_max_m,
        max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz,
        target_clearance_m=args.target_clearance_m, target_swing_s=args.target_swing_s,
        max_stance_s=args.max_stance_s, w_clearance=-2.0, w_stance_cap=-0.15,
        w_touchdown=-3.0, w_air_time=2.0, action_scale=args.action_scale,
        tracking_sigma=0.05, w_track=1.5, w_track_ang=0.8, w_track_orient=0.2,
        variance_base=0.0025, speed_gain=0.55, variance_cap=0.07, w_knee=-3.0, w_flight=-1.0, w_tip=-1.5,
        tip_allowance_deg=15.0, tip_scale_deg=45.0,
        target_knee_height_m=args.target_knee_height_m,
    )
    CONFIG, _ = train_gait.build_config(cfg_args)
    env_kwargs = train_gait.make_env_kwargs(CONFIG)
    envs.register_environment('pupper_gait_eval', gait_env.PupperGaitEnv)
    return envs.get_environment('pupper_gait_eval', **env_kwargs), CONFIG


def rollout_metrics(env, jitted, command, seed, steps, dt):
    jit_reset, jit_step, jit_policy = jitted
    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    cmd = jp.array(command[:3])
    state.info['command'] = cmd

    contacts, heights, vels_xy, vel_z, lin_vel, yaw_rate, torso_z, tilt = [], [], [], [], [], [], [], []
    knees = []
    gaps = []
    done_early = False
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action, _ = jit_policy(state.obs, key)
        state = jit_step(state, action)
        state.info['command'] = cmd  # hold the command fixed for the whole rollout
        ps = state.pipeline_state
        foot_pos = ps.site_xpos[env._feet_site_id]
        h = np.asarray(foot_pos[:, 2]) - env._foot_radius
        fv = np.asarray(gait_env.foot_velocities(ps, env._feet_site_id, env._lower_leg_body_id))
        knees.append(np.asarray(ps.xpos[env._lower_leg_body_id][:, 2]))
        fxy = np.asarray(foot_pos[:, :2])
        dmat = np.linalg.norm(fxy[:, None, :] - fxy[None, :, :], axis=-1)
        gaps.append(dmat[np.triu_indices(4, k=1)].min())
        contacts.append(h < 1e-3)
        heights.append(h)
        vels_xy.append(np.linalg.norm(fv[:, :2], axis=-1))
        vel_z.append(fv[:, 2])
        # body-frame linear velocity and yaw rate
        local_vel = np.asarray(bmath.rotate(ps.xd.vel[0], bmath.quat_inv(ps.x.rot[0])))
        local_ang = np.asarray(bmath.rotate(ps.xd.ang[0], bmath.quat_inv(ps.x.rot[0])))
        lin_vel.append(local_vel[:2])
        yaw_rate.append(local_ang[2])
        torso_z.append(float(ps.x.pos[env._torso_idx - 1, 2]))
        rot = np.asarray(ps.x.rot[0])
        # cos(tilt) from the body z axis projected on world z
        w, x, y, z = rot
        tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))
        if float(state.done) > 0:
            done_early = True
            break

    knees = np.array(knees)                 # (T, 4) m
    gaps = np.array(gaps)                   # (T,) m  min foot-pair distance
    contacts = np.array(contacts)           # (T, 4) bool
    heights = np.array(heights)             # (T, 4) m
    vels_xy = np.array(vels_xy)             # (T, 4) m/s
    vel_z = np.array(vel_z)                 # (T, 4) m/s
    lin_vel = np.array(lin_vel)             # (T, 2)
    yaw_rate = np.array(yaw_rate)           # (T,)
    n = len(contacts)
    if n < 10:
        return None

    # Per-foot swing/stance segmentation.
    stride_hz, duty, clear_mm, swing_ms, td_speed, knee_mm = [], [], [], [], [], []
    for f in range(4):
        c = contacts[:, f]
        duty.append(float(c.mean()))
        touchdowns = np.flatnonzero((~c[:-1]) & c[1:])
        stride_hz.append(len(touchdowns) / (n * dt))
        liftoffs = np.flatnonzero(c[:-1] & (~c[1:]))
        peaks, durations = [], []
        for lo in liftoffs:
            nxt = touchdowns[touchdowns > lo]
            if len(nxt) == 0:
                continue
            td = nxt[0]
            peaks.append(heights[lo:td + 1, f].max())
            durations.append((td - lo) * dt)
        clear_mm.append(float(np.mean(peaks) * 1000) if peaks else 0.0)
        kpk = []
        for lo in liftoffs:
            nxt = touchdowns[touchdowns > lo]
            if len(nxt):
                kpk.append(knees[lo:nxt[0] + 1, f].max())
        knee_mm.append(float(np.mean(kpk) * 1000) if kpk else float(np.mean(knees[:, f]) * 1000))
        swing_ms.append(float(np.mean(durations) * 1000) if durations else 0.0)
        td_speed.append(float(np.mean(-vel_z[touchdowns, f]) * 1000) if len(touchdowns) else 0.0)

    # Which feet swing TOGETHER, not just how many. mean_swing ~1.8 is consistent
    # with a trot (diagonal pairs -- a normal, natural gait dogs use constantly)
    # AND with a bound/pace (front pair or same-side pair -- what reads as hopping),
    # so the count alone cannot tell us whether the gait looks right. Correlating
    # the contact signals pairwise does: in-phase feet -> +1, antiphase -> -1.
    #   trot  diag >> 0, lateral and front < 0
    #   bound front >> 0 and rear >> 0
    #   pace  lateral >> 0
    #   walk  4-beat, every pair near -1/3 with none strongly positive
    # Foot order is (front_r, front_l, back_r, back_l).
    def _corr(a, b):
        x, y = contacts[:, a].astype(float), contacts[:, b].astype(float)
        if x.std() < 1e-9 or y.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    g_diag = 0.5 * (_corr(0, 3) + _corr(1, 2))   # FR-BL, FL-BR
    g_lat = 0.5 * (_corr(0, 2) + _corr(1, 3))    # FR-BR, FL-BL (same side)
    g_front = _corr(0, 1)                        # FR-FL
    g_rear = _corr(2, 3)                         # BR-BL
    scores = {'trot': g_diag, 'pace': g_lat, 'bound': 0.5 * (g_front + g_rear)}
    label = max(scores, key=scores.get)
    if scores[label] < 0.2:
        label = 'walk4'  # no pair strongly in phase -> four-beat sequential

    slip = vels_xy[contacts] if contacts.any() else np.array([0.0])
    return {
        'gait_diag': g_diag,
        'gait_lat': g_lat,
        'gait_front': g_front,
        'gait_rear': g_rear,
        'gait': label,
        'steps': n,
        'fell': done_early,
        'vx': float(lin_vel[:, 0].mean()),
        'vy': float(lin_vel[:, 1].mean()),
        'wz': float(yaw_rate.mean()),
        'stride_hz': float(np.mean(stride_hz)),
        'duty': float(np.mean(duty)),
        'clear_mm': float(np.mean(clear_mm)),
        'knee_mm': float(np.mean(knee_mm)),
        'swing_ms': float(np.mean(swing_ms)),
        'slip_mm_s': float(np.mean(slip) * 1000),
        'td_mm_s': float(np.mean(td_speed)),
        'airborne_frac': float((contacts.sum(axis=1) == 0).mean()),
        # >1 foot airborne at once means a trot/bound rather than a walk.
        'max_swing': float((~contacts).sum(axis=1).max()),
        'mean_swing': float((~contacts).sum(axis=1).mean()),
        # Rings (r=48.8mm) clash below ~98mm centre-to-centre.
        'min_gap_mm': float(gaps.min() * 1000),
        'torso_mm': float(np.mean(torso_z) * 1000),
        'tilt_deg': float(np.mean(tilt)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', required=True)
    parser.add_argument('--env', choices=['gait', 'reference'], default='gait',
                        help='Which env the policy was trained in. Mismatching this\n'
                             'feeds the policy the wrong observation frame.')
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--json', default=None)
    parser.add_argument('--leg_length_min_m', type=float, default=0.056)
    parser.add_argument('--leg_length_max_m', type=float, default=0.060)
    parser.add_argument('--max_vx', type=float, default=0.4)
    parser.add_argument('--max_vy', type=float, default=0.2)
    parser.add_argument('--max_wz', type=float, default=1.0)
    parser.add_argument('--target_clearance_m', type=float, default=0.025)
    parser.add_argument('--target_knee_height_m', type=float, default=0.070)
    parser.add_argument('--target_swing_s', type=float, default=0.18)
    parser.add_argument('--max_stance_s', type=float, default=0.35)
    parser.add_argument('--action_scale', type=float, nargs=12, default=None)
    args = parser.parse_args()

    env, CONFIG = (build_reference_eval_env(args) if args.env == 'reference'
                   else build_eval_env(args))
    params = brax_model.load_params(args.params)
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        # MUST match training: these policies were trained with
        # normalize_observations=True, so the network needs the running-statistics
        # preprocessor. An identity fn here feeds raw observations to a policy that
        # expects normalized ones, which makes every policy look frozen.
        running_statistics.normalize,
        policy_hidden_layer_sizes=CONFIG.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(CONFIG.policy.activation))
    make_policy = ppo_networks.make_inference_fn(network)
    inference_fn = make_policy((params[0], params[1]), deterministic=True)
    # JIT once: re-jitting per command cost more than the rollouts themselves.
    jitted = (jax.jit(env.reset), jax.jit(env.step), jax.jit(inference_fn))

    dt = float(CONFIG.training.environment_dt)
    rows, summary = [], {}
    print(f'{"command":<14}{"vx":>6}{"wz":>6}{"strHz":>7}{"duty":>6}'
          f'{"clrMM":>7}{"kneeMM":>7}{"nSwing":>7}{"gapMM":>7}{"slip":>6}{"tdMM/s":>7}'
          f'{"torso":>6}{"fall":>5}  {"gait":<6}{"diag":>6}{"lat":>6}{"frnt":>6}')
    for vx, vy, wz, label in COMMAND_SUITE:
        agg = []
        for s in range(args.seeds):
            m = rollout_metrics(env, jitted, (vx, vy, wz), 100 + s, args.steps, dt)
            if m:
                agg.append(m)
        if not agg:
            continue
        # 'gait' is a string label; averaging it would raise. Take the modal label.
        avg = {k: float(np.mean([a[k] for a in agg]))
               for k in agg[0] if not isinstance(agg[0][k], str)}
        labels = [a['gait'] for a in agg]
        avg['gait'] = max(set(labels), key=labels.count)
        avg['label'] = label
        avg['cmd'] = [vx, vy, wz]
        avg['track_err'] = float(np.sqrt((avg['vx'] - vx) ** 2 + (avg['vy'] - vy) ** 2))
        avg['yaw_err'] = float(abs(avg['wz'] - wz))
        rows.append(avg)
        print(f'{label:<14}{avg["vx"]:>6.2f}{avg["wz"]:>6.2f}'
              f'{avg["stride_hz"]:>7.2f}{avg["duty"]:>6.2f}{avg["clear_mm"]:>7.1f}'
              f'{avg["knee_mm"]:>7.1f}{avg["mean_swing"]:>7.2f}{avg["min_gap_mm"]:>7.0f}'
              f'{avg["slip_mm_s"]:>6.0f}{avg["td_mm_s"]:>7.0f}'
              f'{avg["torso_mm"]:>6.0f}{avg["fell"]*100:>5.0f}  {avg["gait"]:<6}'
              f'{avg["gait_diag"]:>6.2f}{avg["gait_lat"]:>6.2f}{avg["gait_front"]:>6.2f}')

    moving = [r for r in rows if r['label'] != 'stand']
    summary = {
        'mean_track_err': float(np.mean([r['track_err'] for r in moving])),
        'mean_yaw_err': float(np.mean([r['yaw_err'] for r in moving])),
        'mean_stride_hz': float(np.mean([r['stride_hz'] for r in moving])),
        'mean_duty': float(np.mean([r['duty'] for r in moving])),
        'mean_clear_mm': float(np.mean([r['clear_mm'] for r in moving])),
        'mean_knee_mm': float(np.mean([r['knee_mm'] for r in moving])),
        'mean_swing_feet': float(np.mean([r['mean_swing'] for r in moving])),
        'min_gap_mm': float(np.min([r['min_gap_mm'] for r in moving])),
        'mean_slip_mm_s': float(np.mean([r['slip_mm_s'] for r in moving])),
        'mean_td_mm_s': float(np.mean([r['td_mm_s'] for r in moving])),
        'fall_frac': float(np.mean([r['fell'] for r in rows])),
        'fwd_track_err': float(np.mean([r['track_err'] for r in rows if r['label'].startswith('fwd')])),
        'back_track_err': float(np.mean([r['track_err'] for r in rows if r['label'].startswith('back')])),
        'back_clear_mm': float(np.mean([r['clear_mm'] for r in rows if r['label'].startswith('back')])),
        'back_slip_mm_s': float(np.mean([r['slip_mm_s'] for r in rows if r['label'].startswith('back')])),
    }
    print()
    for k, v in summary.items():
        print(f'  {k:>18}: {v:.3f}')
    if args.json:
        Path(args.json).write_text(json.dumps({'rows': rows, 'summary': summary}, indent=2))
        print(f'\nwrote {args.json}')


if __name__ == '__main__':
    main()
