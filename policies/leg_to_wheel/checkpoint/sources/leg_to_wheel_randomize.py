"""Walking DR plus independent unconverted / 5–10 mm shorter limbs for mixed stance.

This is a kinematic proxy for changed reach, not a deformable-foot simulation.
Only upstream body_pos changes; foot-local geometry and ring calibration agree.
"""
import jax
from jax import numpy as jp
from workspace.walk_randomize import domain_randomize as walk_randomize, LEG_BODY_IDS


def shorten_legs(sys, shortening):
    """Shorten each nominal upstream segment by a distance in metres."""
    positions = sys.body_pos[LEG_BODY_IDS]
    length = jp.linalg.norm(positions, axis=-1)
    positions = positions * (1.-jp.asarray(shortening)/length)[:, None]
    return sys.tree_replace({'body_pos':sys.body_pos.at[LEG_BODY_IDS].set(positions)})


def domain_randomize(sys, rng, shortening_range=(0., .010), converted_probability=.5):
    keys = jax.vmap(lambda key: jax.random.split(key, 2))(rng)
    randomized, axes = walk_randomize(sys, keys[:, 0])
    # Explicit converted/unconverted mixtures make the real support patterns
    # common, rather than rare corners of a four-dimensional uniform draw.
    # The mask stays hidden from the policy, preserving the observation ABI.
    lo, hi = shortening_range
    def sample_reach(key):
        mask = jax.random.bernoulli(jax.random.fold_in(key, 1), converted_probability, (4,))
        short = jax.random.uniform(key, (4,), minval=(lo+hi)/2, maxval=hi)
        return jp.where(mask, short, lo)
    shortening = jax.vmap(sample_reach)(keys[:, 1])
    positions = randomized.body_pos[:, LEG_BODY_IDS]
    lengths = jp.linalg.norm(positions, axis=-1)
    positions = positions * (1.-shortening/lengths)[..., None]
    return randomized.tree_replace({'body_pos':randomized.body_pos.at[:, LEG_BODY_IDS].set(positions)}), axes
