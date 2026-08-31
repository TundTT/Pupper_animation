"""Physics domain randomization for the Pupper V3 training envs.

Mirrors pupperv3-mjx/pupperv3_mjx/domain_randomization.py but with parameter
ranges tuned to the Pupper_RL_PUBLIC notebook (cell 21): tighter kp/kd range,
higher floor on mass/inertia scale. Passed as `randomization_fn` to ppo.train;
brax vmaps it over the parallel envs before each eval.

Two variants:
  * `domain_randomize`         -- the original, all-actuators-are-position-PD
                                  version. Correct for the leg-lift quadruped.
  * `domain_randomize_wheeled` -- for the wheeled robot, whose `_3` actuators are
                                  VELOCITY actuators. Randomizing those with the
                                  position-PD formula would write a nonzero
                                  biasprm[1] onto them, which a velocity actuator
                                  does not have -- the same silent-corruption trap
                                  wheel_env.py avoids when it overrides gains.
"""

from typing import Tuple

import jax
import numpy as np
from jax import numpy as jp

from workspace import configs


def domain_randomize(
    sys,
    rng,
    torso_body_idx: int = 1,
    # Widened for sim-to-real. Training uses an idealized direct-position actuator
    # (pupper_v3_complete.mjx.position.xml) while the robot and pupperv3_mujoco_sim
    # drive a real torque motor with backlash and voltage/current limits. We cannot
    # model that actuator here, so the next best thing is to make the policy
    # indifferent to a WIDE spread of effective gains, which is what an unmodeled
    # actuator looks like from the policy's point of view.
    friction_range: Tuple = (0.4, 1.5),
    kp_multiplier_range: Tuple = (0.5, 1.6),
    kd_multiplier_range: Tuple = (0.5, 2.0),
    # Biased backward-and-right of center, widened from the old symmetric
    # (-0.03, 0.03) / (-0.01, 0.01). Axis convention verified directly against
    # the model (not assumed): leg_front_r sits at y=-0.0835, leg_front_l at
    # y=+0.0835, both at x=+0.075 vs. -0.075 for the back legs -- so with the
    # robot facing +x, +y=left/-y=right (standard robotics convention).
    #
    # Three rounds of correction so far, all from on-hardware testing (see
    # workspace/README.md Status):
    #  - 2026-08-17/18: back legs (back_r especially) visibly struggled vs. sim,
    #    consistent with the real CoM sitting further back than this MJCF
    #    assumes -- x center shifted from 0 to -0.025.
    #  - 2026-08-18 (2nd test, of the run trained with that first correction):
    #    user asked for a further ~2cm back, ~2cm right correction on top of
    #    that -- x center shifted an additional -0.02 (to -0.045), y center
    #    shifted -0.02 (right) off its previous 0.
    #  - 2026-08-19 (hardware test of that 2nd-correction run): SUCCESS overall
    #    (all four legs lifted and stabilized cleanly, first fully clean session)
    #    but with a regression -- front_l didn't lift as high as before,
    #    suspected to be this correction overshooting. Backed off to a less
    #    extreme 1cm-back/1cm-right correction (x center -0.045 -> -0.035, y
    #    center -0.02 -> -0.01) and also trained a 1.5cm/1.5cm variant as an
    #    extra data point.
    #  - 2026-08-20 (hardware test of both 1cm and 1.5cm): NEITHER beat the
    #    original 2cm/2cm -- 1cm was "too central" (undercorrected), 1.5cm
    #    didn't improve on 2cm either. Verdict: revert to 2cm/2cm as the base.
    #    Also found in this round: front_l's shortfall persists identically
    #    across all three CoM variants (2cm, 1cm, 1.5cm), so it's CoM-invariant
    #    -- not something further CoM tuning will fix. See
    #    Stanford/pupperv3-monorepo/LEG_LIFT_TESTING.md's 2026-08-20 entry.
    #    Half-widths kept the same across all rounds, just re-centered, so
    #    there's still DR coverage around the best-guess offset rather than a
    #    fixed value.
    body_com_x_shift_range: Tuple = (-0.09, 0.0),
    body_com_y_shift_range: Tuple = (-0.035, -0.005),
    body_com_z_shift_range: Tuple = (-0.025, 0.025),
    body_inertia_scale_range: Tuple = (0.9, 1.3),
    body_mass_scale_range: Tuple = (0.9, 1.3),
):
    """Randomize friction, kp/kd, torso CoM, body inertia and mass per env."""

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)

        rng, key_kp, key_kd = jax.random.split(rng, 3)
        kp = (
            jax.random.uniform(key_kp, (1,), minval=kp_multiplier_range[0], maxval=kp_multiplier_range[1])
            * sys.actuator_gainprm[:, 0]
        )
        kd = (
            jax.random.uniform(key_kd, (1,), minval=kd_multiplier_range[0], maxval=kd_multiplier_range[1])
            * (-sys.actuator_biasprm[:, 2])
        )
        gain = sys.actuator_gainprm.at[:, 0].set(kp)
        bias = sys.actuator_biasprm.at[:, 1].set(-kp).at[:, 2].set(-kd)

        rng, key_com = jax.random.split(rng)
        body_com_shift = jax.random.uniform(
            key_com,
            (3,),
            minval=jp.array([body_com_x_shift_range[0], body_com_y_shift_range[0], body_com_z_shift_range[0]]),
            maxval=jp.array([body_com_x_shift_range[1], body_com_y_shift_range[1], body_com_z_shift_range[1]]),
        )
        body_com = sys.body_ipos.at[torso_body_idx].set(sys.body_ipos[torso_body_idx] + body_com_shift)

        rng, key_inertia, key_mass = jax.random.split(rng, 3)
        body_inertia = sys.body_inertia * jax.random.uniform(
            key_inertia, sys.body_inertia.shape, minval=body_inertia_scale_range[0], maxval=body_inertia_scale_range[1]
        )
        body_mass = sys.body_mass * jax.random.uniform(
            key_mass, sys.body_mass.shape, minval=body_mass_scale_range[0], maxval=body_mass_scale_range[1]
        )

        return friction, gain, bias, body_com, body_inertia, body_mass

    friction, gain, bias, body_com, body_inertia, body_mass = rand(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "body_ipos": 0,
        "body_inertia": 0,
        "body_mass": 0,
    })

    sys = sys.tree_replace({
        "geom_friction": friction,
        "actuator_gainprm": gain,
        "actuator_biasprm": bias,
        "body_ipos": body_com,
        "body_inertia": body_inertia,
        "body_mass": body_mass,
    })

    return sys, in_axes


def domain_randomize_wheeled(
    sys,
    rng,
    torso_body_idx: int = 1,
    friction_range: Tuple = (0.4, 1.5),
    kp_multiplier_range: Tuple = (0.5, 1.6),
    kd_multiplier_range: Tuple = (0.5, 2.0),
    # Wheel velocity-actuator gain multiplier. Same intent as kp/kd above: the
    # real wheel motor's effective velocity loop is not modeled here, so make the
    # policy indifferent to a wide spread of it.
    kv_multiplier_range: Tuple = (0.5, 1.6),
    # The leg-lift CoM offsets are deliberately NOT carried over. They encode
    # hardware findings about a quadruped standing on feet (see the long comment
    # on domain_randomize above) and there is no reason to think the wheeled
    # robot's CoM error has the same sign or magnitude -- it has different end
    # effectors and a different stance. Symmetric ranges until wheeled hardware
    # says otherwise.
    body_com_x_shift_range: Tuple = (-0.02, 0.02),
    body_com_y_shift_range: Tuple = (-0.02, 0.02),
    body_com_z_shift_range: Tuple = (-0.02, 0.02),
    body_inertia_scale_range: Tuple = (0.9, 1.3),
    body_mass_scale_range: Tuple = (0.9, 1.3),
):
    """Randomize friction, actuator gains, torso CoM, body inertia and mass per env.

    Actuator gains are randomized PER GROUP, because the two groups have
    different affine-bias semantics:
        position actuator: gainprm[0]=kp, biasprm=(0, -kp, -kd)
        velocity actuator: gainprm[0]=kv, biasprm=(0,   0, -kv)
    Writing the position form onto a velocity row (what the un-suffixed
    domain_randomize does) leaves it with a spurious -kp position term.
    """
    pos_mask = np.zeros(12)
    pos_mask[configs.POSITION_ACTUATOR_ROWS] = 1.0
    wheel_mask = np.zeros(12)
    wheel_mask[configs.WHEEL_ACTUATOR_ROWS] = 1.0
    pos_mask = jp.array(pos_mask)
    wheel_mask = jp.array(wheel_mask)

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)

        rng, key_kp, key_kd, key_kv = jax.random.split(rng, 4)
        kp_mult = jax.random.uniform(key_kp, (), minval=kp_multiplier_range[0], maxval=kp_multiplier_range[1])
        kd_mult = jax.random.uniform(key_kd, (), minval=kd_multiplier_range[0], maxval=kd_multiplier_range[1])
        kv_mult = jax.random.uniform(key_kv, (), minval=kv_multiplier_range[0], maxval=kv_multiplier_range[1])

        base_gain = sys.actuator_gainprm[:, 0]          # kp on position rows, kv on wheel rows
        base_damp = -sys.actuator_biasprm[:, 2]         # kd on position rows, kv on wheel rows

        gain_new = base_gain * (pos_mask * kp_mult + wheel_mask * kv_mult)
        damp_new = base_damp * (pos_mask * kd_mult + wheel_mask * kv_mult)
        # biasprm[1] is -kp for a position actuator and must stay 0 for a velocity
        # actuator, hence the mask rather than a blanket -gain_new.
        bias1_new = -gain_new * pos_mask

        gain = sys.actuator_gainprm.at[:, 0].set(gain_new)
        bias = sys.actuator_biasprm.at[:, 1].set(bias1_new).at[:, 2].set(-damp_new)

        rng, key_com = jax.random.split(rng)
        body_com_shift = jax.random.uniform(
            key_com,
            (3,),
            minval=jp.array([body_com_x_shift_range[0], body_com_y_shift_range[0], body_com_z_shift_range[0]]),
            maxval=jp.array([body_com_x_shift_range[1], body_com_y_shift_range[1], body_com_z_shift_range[1]]),
        )
        body_com = sys.body_ipos.at[torso_body_idx].set(sys.body_ipos[torso_body_idx] + body_com_shift)

        rng, key_inertia, key_mass = jax.random.split(rng, 3)
        body_inertia = sys.body_inertia * jax.random.uniform(
            key_inertia, sys.body_inertia.shape,
            minval=body_inertia_scale_range[0], maxval=body_inertia_scale_range[1],
        )
        body_mass = sys.body_mass * jax.random.uniform(
            key_mass, sys.body_mass.shape,
            minval=body_mass_scale_range[0], maxval=body_mass_scale_range[1],
        )

        return friction, gain, bias, body_com, body_inertia, body_mass

    friction, gain, bias, body_com, body_inertia, body_mass = rand(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "body_ipos": 0,
        "body_inertia": 0,
        "body_mass": 0,
    })

    sys = sys.tree_replace({
        "geom_friction": friction,
        "actuator_gainprm": gain,
        "actuator_biasprm": bias,
        "body_ipos": body_com,
        "body_inertia": body_inertia,
        "body_mass": body_mass,
    })

    return sys, in_axes
