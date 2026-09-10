"""Mixed actuator randomization adapted from align-hybrid bfd74cc."""
from typing import Tuple
import jax
from jax import numpy as jp
import numpy as np
import mujoco
from . import configs

def _wheel_collision_geom_ids(sys) -> np.ndarray:
    """Resolve the four wheel collision geoms BY NAME, erroring if any is missing.

    By name rather than index so that editing the MJCF cannot silently repoint the
    size randomization at some other geom (the torso box or a knee sphere).
    """
    import mujoco

    mj_model = getattr(sys, "mj_model", None)
    if mj_model is None:
        raise RuntimeError(
            "domain_randomize_wheeled needs sys.mj_model to resolve the wheel collision "
            "geoms by name; got a System without one."
        )
    ids = []
    for name in configs.WHEEL_COLLISION_GEOM_NAMES:
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM.value, name)
        if gid == -1:
            raise ValueError(
                f"wheel collision geom '{name}' not found in the model. The wheel geoms "
                "must be named for wheel-diameter randomization; see configs."
            )
        if mj_model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError(f"geom '{name}' is not a cylinder; refusing to randomize its size.")
        ids.append(gid)
    return np.array(ids)


def domain_randomize_wheeled(
    sys,
    rng,
    torso_body_idx: int = 1,
    # Widened at the bottom from 0.4 after the first hardware session: the real floor
    # is slippery, and a policy that has only ever driven on grippy ground leans on
    # traction it will not have.
    friction_range: Tuple = (0.25, 1.5),
    # Manufacturing spread in wheel DIAMETER (m); the radius moves by half of this.
    # Each wheel is drawn independently -- see configs.WHEEL_DIAMETER_JITTER.
    wheel_diameter_jitter: float = configs.WHEEL_DIAMETER_JITTER,
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
    wheel_geom_ids = jp.array(_wheel_collision_geom_ids(sys))
    proxy_ids = jp.array([sys.mj_model.geom(n.replace('_wheel_collision','_self_collision')).id
                          for n in configs.WHEEL_COLLISION_GEOM_NAMES])
    radius_jitter = wheel_diameter_jitter / 2.0

    @jax.vmap
    def rand(rng):
        rng, key = jax.random.split(rng)
        friction = jax.random.uniform(key, (1,), minval=friction_range[0], maxval=friction_range[1])
        friction = sys.geom_friction.at[:, 0].set(friction)

        # Wheel diameter: one INDEPENDENT draw per wheel (shape (4,)), applied to the
        # cylinder radius (geom_size[:, 0]) only. geom_size[:, 1] (half-length) and
        # geom_pos are untouched, so each wheel keeps its width and stays mounted at
        # the same point on the motor; only its rolling radius, and hence its contact
        # height, changes.
        rng, key_r = jax.random.split(rng)
        dr_radius = jax.random.uniform(
            key_r, (wheel_geom_ids.shape[0],), minval=-radius_jitter, maxval=radius_jitter
        )
        geom_size = sys.geom_size.at[wheel_geom_ids, 0].add(dr_radius)
        geom_size = geom_size.at[proxy_ids,0].set(jp.sqrt(geom_size[wheel_geom_ids,0]**2 + geom_size[wheel_geom_ids,1]**2))

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

        return friction, geom_size, gain, bias, body_com, body_inertia, body_mass

    friction, geom_size, gain, bias, body_com, body_inertia, body_mass = rand(rng)

    in_axes = jax.tree.map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "geom_size": 0,
        "actuator_gainprm": 0,
        "actuator_biasprm": 0,
        "body_ipos": 0,
        "body_inertia": 0,
        "body_mass": 0,
    })

    sys = sys.tree_replace({
        "geom_friction": friction,
        "geom_size": geom_size,
        "actuator_gainprm": gain,
        "actuator_biasprm": bias,
        "body_ipos": body_com,
        "body_inertia": body_inertia,
        "body_mass": body_mass,
    })

    return sys, in_axes
