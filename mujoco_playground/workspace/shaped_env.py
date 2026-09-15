"""Reference walking env plus the gait constraints, added on top of a policy that
already walks.

`gait_env.PupperGaitEnv` rebuilt the environment from scratch and, across 12 training
runs, never learned to translate: with clearance/knee/stance/tip penalties active,
walking costs more than it earns while standing is free and safe, so PPO stands. The
reference env (`train_reference.PupperReferenceEnv`) does produce forward walking, so
this keeps that env exactly as-is and only *adds* reward terms, to be applied by
fine-tuning from an already-walking checkpoint rather than from scratch.

Added terms (reward-only; observation layout and action contract are untouched):
  - knee_lift (POSITIVE): pays for raising the knee ("elbow") while a foot swings.
  - foot_lift (POSITIVE): pays for foot height while swinging, up to a target.
  - drag (penalty): horizontal foot speed while below the clearance target.
  - tip_vertical (penalty): weight-bearing capsule far from vertical, so load goes
    through the rounded tip rather than the side of the capsule. Zero inside an
    allowance, rising to full at a scale angle.

Lift is rewarded rather than penalized, deliberately. An earlier version penalized
*insufficient* knee/foot height gated on the foot being airborne; because a planted
foot scores zero on such a term, the cheapest way to avoid the penalty is to never
lift a foot at all, and a warm-started walking policy froze within one fine-tune
(duty 0.99, clearance 0.0 mm). A bounded positive reward has the opposite gradient:
standing earns nothing extra, lifting well earns the bonus.

Joint 3 rotation moves the foot tip 6.8 -> 16.1 mm while moving the knee exactly
0.0 mm, so foot height alone is satisfiable by pawing; knee height is what
distinguishes a real stride.
"""
import numpy as np
from brax import math
from pupperv3_mjx import rewards as ref_rewards
from jax import numpy as jp
import jax

from workspace import gait_env
from workspace.train_reference import PupperReferenceEnv


class PupperShapedEnv(PupperReferenceEnv):
    """PupperReferenceEnv with gait-shaping penalties layered on its reward."""

    def __init__(self, *args,
                 w_foot_clearance: float = 0.0,
                 w_knee_clearance: float = 0.0,
                 w_tip_vertical: float = 0.0,
                 w_drag: float = 0.0,
                 w_touchdown: float = 0.0,
                 w_multi_swing: float = 0.0,
                 w_single_support: float = 0.0,
                 w_ring_clash: float = 0.0,
                 w_over_lift: float = 0.0,
                 over_lift_m: float = 0.022,
                 ring_clearance_m: float = 0.105,
                 touchdown_allowance: float = 0.25,
                 knee_rest_m: float = 0.0584,
                 target_clearance_m: float = 0.022,
                 target_knee_height_m: float = 0.068,
                 tip_allowance_deg: float = 15.0,
                 tip_scale_deg: float = 45.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._w_foot_clearance = w_foot_clearance
        self._w_knee_clearance = w_knee_clearance
        self._w_tip_vertical = w_tip_vertical
        self._w_drag = w_drag
        self._w_touchdown = w_touchdown
        self._touchdown_allowance = touchdown_allowance
        self._w_multi_swing = w_multi_swing
        self._w_single_support = w_single_support
        self._w_ring_clash = w_ring_clash
        self._w_over_lift = w_over_lift
        self._over_lift_m = over_lift_m
        self._ring_clearance_m = ring_clearance_m
        self._target_clearance_m = target_clearance_m
        self._target_knee_height_m = target_knee_height_m
        self._knee_rest_m = knee_rest_m
        self._tip_allowance_cos = float(np.cos(np.radians(tip_allowance_deg)))
        self._tip_scale_cos = float(np.cos(np.radians(tip_scale_deg)))

        import mujoco as _mj
        mj_model = self.sys.mj_model
        axes = []
        for leg, sid in zip(('front_r', 'front_l', 'back_r', 'back_l'), self._feet_site_id):
            gid = mj_model.geom(f'leg_{leg}_3_foot_collision').id
            rot = np.zeros(9)
            _mj.mju_quat2Mat(rot, mj_model.geom_quat[gid])
            local_z = rot.reshape(3, 3)[:, 2]
            to_site = mj_model.site_pos[sid] - mj_model.geom_pos[gid]
            axes.append(local_z * (1.0 if np.dot(to_site, local_z) >= 0 else -1.0))
        self._capsule_axis_local = jp.array(np.array(axes))

    _EXTRA_METRICS = ('knee_lift', 'foot_lift', 'drag', 'tip_vertical', 'touchdown',
                      'multi_swing', 'ring_clash', 'over_lift',
                      'single_support')

    def sample_command(self, rng):
        """Explicit mode mix so reverse gets as much training signal as forward.

        The reference samples vx/vy/yaw jointly from uniform ranges, so pure-axis
        commands are rare. Measured on the reference policy, forward tracks at
        +0.182 m/s under a 0.30 command while reverse manages -0.001 m/s, i.e.
        reverse is effectively untrained.
        """
        keys = jax.random.split(rng, 7)
        vx_lo, vx_hi = self._linear_velocity_x_range
        wz_lo, wz_hi = self._angular_velocity_range
        speed = jax.random.uniform(keys[0], (), minval=0.06, maxval=vx_hi)
        yaw = jax.random.uniform(keys[2], (), minval=0.3, maxval=wz_hi)
        sign = jp.where(jax.random.bernoulli(keys[3]), 1.0, -1.0)
        sign2 = jp.where(jax.random.bernoulli(keys[4]), 1.0, -1.0)
        mode = jax.random.randint(keys[5], (), 0, 3)
        zero = jp.zeros(())
        # Strafe modes removed: lateral travel is capped at ~0.016 m/s by 0.14 rad
        # of outward abduction headroom (see lin_vel_y_range in train_reference),
        # so those samples trained an impossible task. vy is always commanded 0.
        candidates = jp.stack([
            jp.array([sign * speed, zero, zero]),            # forward / reverse
            jp.array([zero, zero, sign * yaw]),              # turn in place
            jp.array([sign * speed, zero, sign2 * yaw]),     # arc
        ])
        command = candidates[mode]
        stand = jax.random.uniform(keys[6], ()) < 0.05
        return jp.where(stand, jp.zeros(3), command)

    def reset(self, rng):
        state = super().reset(rng)
        # Metrics keys must match between reset and step or brax's scan over the
        # episode fails on a changed pytree structure.
        for key in self._EXTRA_METRICS:
            state.metrics[key] = jp.zeros(())
        # The reference env overwrites last_contact with the CURRENT contact at the
        # end of step(), so landing detection needs its own previous-contact memory.
        state.info['prev_contact'] = jp.zeros(4, dtype=bool)
        # Filter state for move_gate; must exist in reset or brax's scan over the
        # episode fails on a changed pytree structure.
        state.info['vel_filt'] = jp.zeros(3)
        state.info['ang_filt'] = jp.zeros(3)
        return state

    def step(self, state, action):
        state = super().step(state, action)
        ps = state.pipeline_state

        foot_height = ps.site_xpos[self._feet_site_id][:, 2] - self._foot_radius
        foot_vel = gait_env.foot_velocities(ps, self._feet_site_id, self._lower_leg_body_id)
        knee_height = ps.xpos[self._lower_leg_body_id][:, 2]
        contact = state.info['last_contact']
        airborne = ~contact

        # Gate lift rewards on the velocity ACTUALLY achieved along the commanded
        # direction, not on the command being nonzero. Gating on the command alone
        # lets the policy collect the lift bonus while standing still: measured
        # clearance 30.6 mm and knee 85.8 mm at 0.02 m/s, i.e. it marched in place
        # to farm the bonus. This pays in proportion to how much of the commanded
        # velocity is realized, so standing earns nothing.
        command = state.info['command']
        local_vel = math.rotate(ps.xd.vel[0], math.quat_inv(ps.x.rot[0]))
        local_ang = math.rotate(ps.xd.ang[0], math.quat_inv(ps.x.rot[0]))

        # The gate runs on a LOW-PASS FILTERED velocity, not the instantaneous one.
        # Using the instantaneous value with clip(progress, 0, 1) rectifies it: a
        # robot rocking back and forth has zero net displacement but scores positive
        # on every forward half-cycle, so the lift and single-support rewards pay out
        # for oscillating in place. seq_f did exactly that -- vx = 0.00 on every
        # command while still reporting duty 0.74 and one foot airborne. The filter
        # (tau ~ 0.4s at dt=0.02) averages the rocking to ~0, so only sustained
        # travel opens the gate.
        alpha = jp.minimum(self.dt / 0.4, 1.0)
        vel_filt = (1.0 - alpha) * state.info['vel_filt'] + alpha * local_vel
        ang_filt = (1.0 - alpha) * state.info['ang_filt'] + alpha * local_ang
        state.info['vel_filt'] = vel_filt
        state.info['ang_filt'] = ang_filt

        cmd_sq = jp.sum(jp.square(command[:2]))
        progress = jp.sum(vel_filt[:2] * command[:2]) / (cmd_sq + 1e-6)
        lin_gate = jp.clip(progress, 0.0, 1.0)
        yaw_gate = jp.clip(ang_filt[2] * command[2] / (jp.square(command[2]) + 1e-6), 0.0, 1.0)
        # Turning in place is legitimate locomotion, so either one can open the gate.
        move_gate = jp.maximum(lin_gate, yaw_gate)
        moving = jp.linalg.norm(command[:3]) > 0.05

        capsule_axis_world = jax.vmap(math.rotate)(
            self._capsule_axis_local, ps.x.rot[self._lower_leg_body_id - 1])
        tip_alignment = -capsule_axis_world[:, 2]

        # Positive, bounded lift rewards: standing earns nothing, lifting well pays.
        knee_span = max(self._target_knee_height_m - self._knee_rest_m, 1e-6)
        knee_lift = jp.mean(
            jp.clip((knee_height - self._knee_rest_m) / knee_span, 0.0, 1.0) * airborne) * moving * move_gate
        foot_lift = jp.mean(
            jp.clip(foot_height / self._target_clearance_m, 0.0, 1.0) * airborne) * moving * move_gate
        # Excess height ABOVE the target. The lift reward is clipped at the target,
        # so once a foot clears it the extra height earns nothing -- measured 29mm
        # against an 18mm target, i.e. 11mm the policy was not being paid for. That
        # height is a byproduct of swinging fast, so lowering the target cannot
        # remove it; only an explicit penalty on the overshoot can. This is zero for
        # any foot at or below the target, so unlike the earlier clearance penalties
        # it never makes not-lifting the cheapest option.
        over_lift = jp.mean(jp.clip(foot_height - self._over_lift_m, 0.0, None)) * moving
        drag = gait_env.reward_foot_clearance(
            foot_height, foot_vel, self._target_clearance_m) * moving
        tip_vertical = gait_env.reward_tip_vertical(
            tip_alignment, contact, self._tip_allowance_cos, self._tip_scale_cos)
        # Excess downward foot speed on the step a foot lands, so the lift rewards
        # cannot be cashed in as hard floor tapping.
        landing = contact & (~state.info['prev_contact'])
        touchdown = jp.sum(landing * jp.square(
            jp.clip(-foot_vel[:, 2] - self._touchdown_allowance, 0.0, None)))

        # Sequential stepping: a walk lifts one foot at a time (duty ~0.75). Without
        # this the policy converged on a bound/pronk, moving the front pair and rear
        # pair together and effectively hopping, which is what it looked like on video.
        # Gated on move_gate as well as `moving`: without it the term is a pure
        # penalty on flight, and a policy that simply stands collects zero. Paying
        # it only in proportion to realized velocity means the comparison is
        # "walk sequentially vs walk with a bound", not "walk vs stand".
        multi_swing = jp.clip(jp.sum(airborne) - 1.0, 0.0, 3.0) * moving * move_gate

        # POSITIVE reward for true single support: exactly one foot off the ground.
        # The penalty above proved too weak on its own -- seq_d reached four-beat
        # PHASING (offsets 0/100/180/280) but at duty 0.55 the stance arcs only span
        # 198 deg, so the feet actually DOWN at any instant are the diagonal pair:
        # a trot's support pattern wearing a walk's timing, which is what reads as
        # two-legs-together hopping on video. Only duty >= 0.75 keeps three feet
        # planted. Gated by move_gate so standing (where this is trivially satisfied
        # by all four feet down, scoring 0 here) earns nothing and cannot freeze.
        n_air = jp.sum(airborne)
        single_support = jp.clip(1.0 - jp.abs(n_air - 1.0), 0.0, 1.0) * moving * move_gate

        # Ring clash: the TPU ring around each foot is ~48.8 mm in radius, so two
        # rings overlap once foot centres come within ~98 mm. Same-side front/rear
        # feet sit only 150 mm apart at home while a 0.6 rad hip sweep moves a foot
        # ~80 mm, so long strides visibly interpenetrate. The ring is visual-only
        # (no collision geom), so nothing in physics prevents it -- hence this term.
        foot_xy = ps.site_xpos[self._feet_site_id][:, :2]
        diff = foot_xy[:, None, :] - foot_xy[None, :, :]
        dist = jp.sqrt(jp.sum(jp.square(diff), axis=-1) + 1e-9)
        pair_mask = jp.triu(jp.ones((4, 4)), k=1)
        shortfall = jp.clip((self._ring_clearance_m - dist) / self._ring_clearance_m, 0.0, 1.0)
        ring_clash = jp.sum(jp.square(shortfall) * pair_mask)

        extra = (self._w_knee_clearance * knee_lift
                 + self._w_foot_clearance * foot_lift
                 + self._w_drag * drag
                 + self._w_tip_vertical * tip_vertical
                 + self._w_touchdown * touchdown
                 + self._w_multi_swing * multi_swing
                 + self._w_ring_clash * ring_clash
                 + self._w_over_lift * over_lift
                 + self._w_single_support * single_support) * self.dt

        state.metrics['knee_lift'] = knee_lift
        state.metrics['foot_lift'] = foot_lift
        state.metrics['drag'] = drag
        state.metrics['tip_vertical'] = tip_vertical
        state.metrics['touchdown'] = touchdown
        state.metrics['multi_swing'] = multi_swing
        state.metrics['ring_clash'] = ring_clash
        state.metrics['over_lift'] = over_lift
        state.metrics['single_support'] = single_support
        state.info['prev_contact'] = contact
        return state.replace(reward=state.reward + extra)
