"""Walking randomization: nominal symmetric terms plus leg-compliance and contact DR.

`LEG_BODY_IDS`/`FOOT_GEOM_IDS` are resolved once from the nominal MJCF at import
time (ids are structural, not affected by per-episode randomization) so this stays
a plain (sys, rng) -> (sys, axes) function, matching brax's expected signature.

Leg-length randomization scales `body_pos` for the leg_*_3 bodies rather than any
mesh/geom field local to the foot: `ring_local` (walk_geometry.py) and `self.sizes`
(walk_env.py) are baked once from the foot body's own local mesh/geom mounts, which
this randomization never touches. `world_points()` re-projects those fixed local
points through the ACTUAL per-step `d.xpos`/`d.xmat` of the foot body, which already
reflects the randomized body_pos upstream in the kinematic chain -- so ring-clearance
rewards stay correctly calibrated across leg lengths with no other change required
(verified: standing penetration is unchanged across a wide body_pos scale sweep).
Do NOT add mesh_pos/mesh_quat/geom_pos/geom_quat/geom_size for the foot to `fields`
below without re-baking ring_local per-sample -- that desyncs the ring from the true
mesh position (see .notes and the walk-policy review this followed from).
"""
import mujoco
import numpy as np
import jax
from jax import numpy as jp
from workspace import walk_geometry as geom

_nominal = mujoco.MjModel.from_xml_path(str(geom.MODEL_PATH))
LEG_BODY_IDS = np.array([_nominal.body(f'leg_{leg}_3').id for leg in geom.LEGS])
FOOT_GEOM_IDS = np.array([_nominal.geom(f'leg_{leg}_3_foot_collision').id for leg in geom.LEGS])
_JOINT_DOFS = slice(_nominal.nv - 12, _nominal.nv)
if _nominal.nv < 12:
    raise ValueError('Expected at least 12 actuated DOFs')

def domain_randomize(sys,rng):
    @jax.vmap
    def sample(key):
        keys=jax.random.split(key,13)
        friction=sys.geom_friction.at[:,0].set(jax.random.uniform(keys[0],(),minval=.4,maxval=1.2))
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
        # Leg-length/compliance proxy: a common whole-robot scale plus a smaller
        # per-leg term for build asymmetry. See module docstring for why this field
        # (and not a foot-local mesh/geom field) keeps the ring reward consistent.
        thigh_common=jax.random.uniform(keys[6],(),minval=.94,maxval=1.06)
        thigh_per_leg=jax.random.uniform(keys[7],(4,),minval=.98,maxval=1.02)
        body_pos=sys.body_pos.at[LEG_BODY_IDS].multiply((thigh_common*thigh_per_leg)[:,None])
        # Foot contact softness: a real series spring-damper at the ground contact,
        # i.e. actual leg compliance rather than a kinematic proxy. Timeconst is
        # capped at .045 (not the .12 that would model very soft rubber) because
        # beyond that the standing ring_bottom penalty starts firing at rest -- see
        # bottom_allowance in walk_config.py.
        solref=sys.geom_solref.at[FOOT_GEOM_IDS,0].set(jax.random.uniform(keys[8],(4,),minval=.02,maxval=.045)) \
                              .at[FOOT_GEOM_IDS,1].set(jax.random.uniform(keys[9],(4,),minval=.8,maxval=1.2))
        solimp=sys.geom_solimp.at[FOOT_GEOM_IDS,0].set(jax.random.uniform(keys[10],(4,),minval=.010,maxval=.025))
        # Servo torque varies with battery voltage/temperature by well over +-15%
        # in practice; forcerange is symmetric so a positive scale keeps it so.
        forcerange=sys.actuator_forcerange*jax.random.uniform(keys[11],(12,1),minval=.85,maxval=1.15)
        armature=sys.dof_armature.at[_JOINT_DOFS].multiply(jax.random.uniform(keys[12],(12,),minval=.7,maxval=1.3))
        return friction,gain,bias,mass,inertia,com,body_pos,solref,solimp,forcerange,armature
    values=sample(rng)
    fields=('geom_friction','actuator_gainprm','actuator_biasprm','body_mass','body_inertia','body_ipos',
            'body_pos','geom_solref','geom_solimp','actuator_forcerange','dof_armature')
    axes=jax.tree.map(lambda _:None,sys).tree_replace({k:0 for k in fields})
    return sys.tree_replace(dict(zip(fields,values))),axes
