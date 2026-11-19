"""Approximate physical-target preservation when widening action command scales."""
import copy
import numpy as np
from jax import numpy as jp


def rescale_action_head(params, old_scale, new_scale, history):
    ratio=np.asarray(old_scale)/np.asarray(new_scale)
    result=list(copy.deepcopy(params))
    indices=np.concatenate([np.arange(24,36)+36*i for i in range(history)])
    factors=jp.array(np.tile(ratio,history))
    stats=result[0]
    result[0]=stats.replace(mean=jp.asarray(stats.mean).at[indices].multiply(factors),
                            std=jp.asarray(stats.std).at[indices].multiply(factors),
                            summed_variance=jp.asarray(stats.summed_variance).at[indices].multiply(factors**2))
    head=list(result[1]['params'].values())[-1]
    # Scaling pre-tanh means preserves small physical targets approximately;
    # it is not exact near saturation. Re-evaluate before continuing training.
    head['kernel']=jp.asarray(head['kernel']).at[:,:12].multiply(jp.array(ratio))
    head['bias']=jp.asarray(head['bias']).at[:12].multiply(jp.array(ratio))
    # PPO's scale head uses softplus. For its small exploration scales, adding
    # log(ratio) approximately keeps exploration in the same physical units.
    head['bias']=head['bias'].at[12:].add(jp.log(jp.array(ratio)))
    return tuple(result)
