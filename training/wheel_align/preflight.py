"""CPU validation only: no optimizer, gradient update or training invocation."""
import argparse
import functools
import json
import jax
from jax import numpy as jp
import numpy as np
from .env import AlignEnv
from . import configs as c,contract as ct
from .randomize import domain_randomize_wheeled
from .hybrid_wrappers import wrap_for_training

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jit-step',action='store_true');args=p.parse_args()
    env=AlignEnv(noise=False)
    state=jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.obs.block_until_ready();assert state.obs.shape==(82,)
    assert np.isfinite(np.asarray(state.obs)).all()
    # Shape tracing catches the randomized batched task/autoreset contract without learning.
    randomize=functools.partial(domain_randomize_wheeled,rng=jax.random.split(jax.random.PRNGKey(2),2))
    wrapper=wrap_for_training(AlignEnv(training=True),episode_length=6656,randomization_fn=randomize)
    shape=jax.eval_shape(wrapper.reset,jax.random.split(jax.random.PRNGKey(3),2))
    output=jax.eval_shape(wrapper.step,shape,jax.ShapeDtypeStruct((2,8),jp.float32))
    assert output.obs.shape==(2,82)
    if args.jit_step:
        state=env.select_command(state,jp.asarray(1))
        step=jax.jit(env.step)
        for _ in range(3):state=step(state,jp.zeros(8));state.obs.block_until_ready()
        assert np.isfinite(np.asarray(state.obs)).all()
        assert float(state.done)==0
    print(json.dumps(dict(status='PASS',observations=82,actions=8,control_dt=c.CONTROL_DT,
        collision_pairs='cylinder-floor; conservative sphere-wheel/body',
        randomized_batch_shape=list(output.obs.shape),jit_steps=3 if args.jit_step else 0,
        optimizer_steps=0),indent=2))

if __name__=='__main__':main()
