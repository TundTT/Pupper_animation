"""Nominal, symmetric walking randomization; no inherited leg-lift CoM bias."""
import jax
from jax import numpy as jp

def domain_randomize(sys,rng):
    @jax.vmap
    def sample(key):
        keys=jax.random.split(key,5)
        friction=sys.geom_friction.at[:,0].set(jax.random.uniform(keys[0],(),minval=.4,maxval=1.2))
        kp=sys.actuator_gainprm[:,0]*jax.random.uniform(keys[1],(12,),minval=.8,maxval=1.2)
        kd=-sys.actuator_biasprm[:,2]*jax.random.uniform(keys[2],(12,),minval=.8,maxval=1.2)
        gain=sys.actuator_gainprm.at[:,0].set(kp)
        # Preserve offset control when changing kp: zero ctrl must still target home.
        bias=sys.actuator_biasprm.at[:,0].set(kp*sys.qpos0[7:]).at[:,1].set(-kp).at[:,2].set(-kd)
        factor=jax.random.uniform(keys[3],(),minval=.9,maxval=1.1)
        mass=sys.body_mass*factor; inertia=sys.body_inertia*factor
        com=sys.body_ipos.at[1].add(jax.random.uniform(keys[4],(3,),minval=-.005,maxval=.005))
        return friction,gain,bias,mass,inertia,com
    values=sample(rng)
    fields=('geom_friction','actuator_gainprm','actuator_biasprm','body_mass','body_inertia','body_ipos')
    axes=jax.tree.map(lambda _:None,sys).tree_replace({k:0 for k in fields})
    return sys.tree_replace(dict(zip(fields,values))),axes
