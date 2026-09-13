"""Walking randomization: manufacturing geometry, friction, and actuator variation.

`LEG_BODY_IDS`/`FOOT_GEOM_IDS` are resolved once from the nominal MJCF at import
time (ids are structural, not affected by per-slot randomization) so this stays
a plain (sys, rng) -> (sys, axes) function, matching brax's expected signature.

Leg-length randomization scales `body_pos` for the leg_*_3 bodies rather than any
mesh/geom field local to the foot: `ring_local` (walk_geometry.py) and `self.sizes`
(walk_env.py) are baked once from the foot body's own local mesh/geom mounts, which
this randomization never touches. `world_points()` re-projects those fixed local
points through the ACTUAL per-step `d.xpos`/`d.xmat` of the foot body, which already
reflects the randomized body_pos upstream in the kinematic chain -- so ring-clearance
rewards stay correctly calibrated across leg lengths with no other change required
(verified for the basic length endpoints; combined DR can change penetration).
Do NOT add mesh_pos/mesh_quat/geom_pos/geom_quat/geom_size for the FOOT to `fields`
below without re-baking ring_local per-sample -- that desyncs the ring from the true
mesh position (see .notes and the walk-policy review this followed from). `body_quat`
for the leg_*_3 bodies is different and safe by the same argument as `body_pos`: it
changes how that body's frame sits relative to its PARENT (modeling a leg bolted on a
few degrees off), not the foot's own local mesh mount, so world_points() still
re-projects ring_local correctly through the resulting (tilted) runtime body pose.

Floor `geom_quat` is randomized too (a small terrain-tilt proxy): the floor stays a
flat MuJoCo plane (real, MJX-native contact physics, no heightfield needed), just
reoriented per env. Reward code must measure height-above-ground as a projection onto
this floor's actual normal (walk_geometry.quat_axis_z) rather than assuming world +Z --
walk_env.py uses this for ring clearance and torso height as well as the swing
height projection. Brax samples geometry once per environment slot and reuses it
across episode resets, representing persistent manufacturing differences.
"""
import mujoco
import numpy as np
import jax
from jax import numpy as jp
from workspace import walk_geometry as geom

_nominal = mujoco.MjModel.from_xml_path(str(geom.MODEL_PATH))
LEG_BODY_IDS = np.array([_nominal.body(f'leg_{leg}_3').id for leg in geom.LEGS])
FOOT_GEOM_IDS = np.array([_nominal.geom(f'leg_{leg}_3_foot_collision').id for leg in geom.LEGS])
FLOOR_GEOM_ID = _nominal.geom('floor').id
_JOINT_DOFS = slice(_nominal.nv - 12, _nominal.nv)
if _nominal.nv < 12:
    raise ValueError('Expected at least 12 actuated DOFs')

def _quat_mul(q1, q2):
    """Hamilton product, batched over any leading shape ending in the 4 quat comps."""
    w1,x1,y1,z1 = q1[...,0],q1[...,1],q1[...,2],q1[...,3]
    w2,x2,y2,z2 = q2[...,0],q2[...,1],q2[...,2],q2[...,3]
    return jp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

def _small_tilt_quat(rx, ry):
    """Compose Ry(ry) Rx(rx); pre-multiplication applies parent-frame X/Y tilt."""
    qx = jp.stack([jp.cos(rx/2), jp.sin(rx/2), jp.zeros_like(rx), jp.zeros_like(rx)], axis=-1)
    qy = jp.stack([jp.cos(ry/2), jp.zeros_like(ry), jp.sin(ry/2), jp.zeros_like(ry)], axis=-1)
    return _quat_mul(qy, qx)

def domain_randomize(sys,rng,friction_range=(.8,2.5),foot_model='rigid_flush',
                     leg_length_common_range=(.90,.98),leg_length_per_leg_range=(.96,1.04)):
    if not np.all(np.isfinite(friction_range)) or not 0. < friction_range[0] <= friction_range[1]:
        raise ValueError('friction_range must be positive and ordered')
    if foot_model not in ('rigid_flush','legacy_soft'):
        raise ValueError(f'Unknown foot_model: {foot_model}')
    for bounds in (leg_length_common_range,leg_length_per_leg_range):
        if not np.all(np.isfinite(bounds)) or not 0. < bounds[0] <= bounds[1]:
            raise ValueError('Leg length ranges must be positive and ordered')
    @jax.vmap
    def sample(key):
        keys=jax.random.split(key,17)
        friction=sys.geom_friction.at[:,0].set(jax.random.uniform(keys[0],(),minval=friction_range[0],maxval=friction_range[1]))
        kp=sys.actuator_gainprm[:,0]*jax.random.uniform(keys[1],(12,),minval=.8,maxval=1.2)
        kd=-sys.actuator_biasprm[:,2]*jax.random.uniform(keys[2],(12,),minval=.8,maxval=1.2)
        gain=sys.actuator_gainprm.at[:,0].set(kp)
        # Preserve offset control when changing kp: zero ctrl must still target home.
        bias=sys.actuator_biasprm.at[:,0].set(kp*sys.qpos0[7:]).at[:,1].set(-kp).at[:,2].set(-kd)
        factor=jax.random.uniform(keys[3],(),minval=.9,maxval=1.1)
        per_body=jax.random.uniform(keys[4],sys.body_mass.shape,minval=.95,maxval=1.05)
        mass=sys.body_mass*factor*per_body
        inertia=sys.body_inertia*factor*per_body[:,None]
        # Wider than the original +-5mm: that range never crossed the robot's
        # natural 55/45 front/rear load split, so training never saw a rear-biased
        # robot -- which is exactly the load state backward walking needs.
        com=sys.body_ipos.at[1].add(jax.random.uniform(keys[5],(3,),minval=-.02,maxval=.02))
        # Leg-length variation: a common whole-robot scale plus a smaller
        # per-leg term for build asymmetry. See module docstring for why this field
        # (and not a foot-local mesh/geom field) keeps the ring reward consistent.
        # Shorter-biased engineering prior: combined range .864--1.0192, mean .94,
        # preserving +/-4% per-leg asymmetry. These are upstream-offset scales,
        # not capsule-size scales. The new 62.65mm capsule and ring move together;
        # its distal extension is present at every length, with no random tip gap.
        thigh_common=jax.random.uniform(keys[6],(),minval=leg_length_common_range[0],maxval=leg_length_common_range[1])
        thigh_per_leg=jax.random.uniform(keys[7],(4,),minval=leg_length_per_leg_range[0],maxval=leg_length_per_leg_range[1])
        body_pos=sys.body_pos.at[LEG_BODY_IDS].multiply((thigh_common*thigh_per_leg)[:,None])
        # Rigid mode keeps compiled contact stiffness fixed. Preserve key indices
        # so changing contact mode does not change unrelated random draws.
        solref,solimp=sys.geom_solref,sys.geom_solimp
        if foot_model == 'legacy_soft':
            solref=solref.at[FOOT_GEOM_IDS,0].set(jax.random.uniform(keys[8],(4,),minval=.02,maxval=.045)) \
                          .at[FOOT_GEOM_IDS,1].set(jax.random.uniform(keys[9],(4,),minval=.8,maxval=1.2))
            solimp=solimp.at[FOOT_GEOM_IDS,0].set(jax.random.uniform(keys[10],(4,),minval=.010,maxval=.025))
        # Servo torque varies with battery voltage/temperature by well over +-15%
        # in practice; forcerange is symmetric so a positive scale keeps it so.
        forcerange=sys.actuator_forcerange*jax.random.uniform(keys[11],(12,1),minval=.85,maxval=1.15)
        armature=sys.dof_armature.at[_JOINT_DOFS].multiply(jax.random.uniform(keys[12],(12,),minval=.7,maxval=1.3))
        # Leg-mount misalignment: manufacturing/assembly tolerance means the lower
        # leg isn't always bolted on perfectly square to the knee axis, so at the
        # commanded home pose the foot isn't always exactly plumb. Tilts body_quat
        # (how leg_*_3's frame sits relative to its parent), not the foot's own mesh
        # mount -- see module docstring for why that keeps the ring reward calibrated.
        # Raised to 0.07 rad (~4deg) per axis, independent per leg: this and the
        # thigh-length range above are now the priority (foot lift has been dialed
        # back to make room for exactly this).
        # Pre-multiplication is in the PARENT frame: X mainly deflects fore/aft;
        # Y supplies real lateral tilt plus yaw. No mount-correlated encoder bias.
        leg_tilt = _small_tilt_quat(jax.random.uniform(keys[13],(4,),minval=-.07,maxval=.07),
                                     jax.random.uniform(keys[14],(4,),minval=-.07,maxval=.07))
        body_quat=sys.body_quat.at[LEG_BODY_IDS].set(_quat_mul(leg_tilt,sys.body_quat[LEG_BODY_IDS]))
        # Terrain proxy: small floor tilt (roll+pitch, ~0.04 rad / ~2.3deg, eased down
        # from 0.07 for the same reason as leg_tilt above). The floor stays a real,
        # flat MuJoCo plane -- MJX handles a tilted plane's contact physics natively,
        # no heightfield needed -- just reoriented per env. Reward code must read
        # walk_env.py's floor normal, not raw world Z, for anything calling this
        # "ground height" (see walk_geometry.quat_axis_z/ring_costs).
        floor_tilt = _small_tilt_quat(jax.random.uniform(keys[15],(),minval=-.04,maxval=.04),
                                       jax.random.uniform(keys[16],(),minval=-.04,maxval=.04))
        geom_quat=sys.geom_quat.at[FLOOR_GEOM_ID].set(_quat_mul(floor_tilt,sys.geom_quat[FLOOR_GEOM_ID]))
        return friction,gain,bias,mass,inertia,com,body_pos,solref,solimp,forcerange,armature,body_quat,geom_quat
    values=sample(rng)
    fields=('geom_friction','actuator_gainprm','actuator_biasprm','body_mass','body_inertia','body_ipos',
            'body_pos','geom_solref','geom_solimp','actuator_forcerange','dof_armature','body_quat','geom_quat')
    axes=jax.tree.map(lambda _:None,sys).tree_replace({k:0 for k in fields})
    return sys.tree_replace(dict(zip(fields,values))),axes
