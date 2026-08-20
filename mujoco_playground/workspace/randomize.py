"""Physics domain randomization for the Pupper V3 leg-lift training env.

Mirrors pupperv3-mjx/pupperv3_mjx/domain_randomization.py but with parameter
ranges tuned to the Pupper_RL_PUBLIC notebook (cell 21): tighter kp/kd range,
higher floor on mass/inertia scale. Passed as `randomization_fn` to ppo.train;
brax vmaps it over the parallel envs before each eval.
"""

from typing import Tuple

import jax
from jax import numpy as jp


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
    # Two rounds of correction so far, both from on-hardware testing (see
    # workspace/README.md Status):
    #  - 2026-08-17/18: back legs (back_r especially) visibly struggled vs. sim,
    #    consistent with the real CoM sitting further back than this MJCF
    #    assumes -- x center shifted from 0 to -0.025.
    #  - 2026-08-18 (2nd test, of the run trained with that first correction):
    #    user asked for a further ~2cm back, ~2cm right correction on top of
    #    that -- x center shifted an additional -0.02 (to -0.045), y center
    #    shifted -0.02 (right) off its previous 0. Half-widths kept the same
    #    as before this second shift, just re-centered, so there's still DR
    #    coverage around the new best-guess offset rather than a fixed value.
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
