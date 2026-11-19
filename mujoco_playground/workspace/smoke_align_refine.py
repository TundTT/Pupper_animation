"""Physical reset/step and sensor-only latch checks before refinement training."""

import json
from pathlib import Path
import jax
import numpy as np
from jax import numpy as jp
from ml_collections import ConfigDict
from workspace.wheel_align_refine_env import (
    PupperWheelAlignRefineEnv,
    refinement_config,
)
from workspace.wheel_align_critic import PupperWheelAlignCriticEnv

cfg = refinement_config(
    ConfigDict(
        json.loads(
            Path(
                "workspace/output/direct_round25_dwell_feedback/config.json"
            ).read_text()
        )
    )
)
cfg.direct.reference_reset_path = (
    "workspace/output/direct_round33_reset_reference/reset_states.npz"
)
cfg.direct.reference_reset_probability = 1.0
cfg.automatic_commands = False
env = PupperWheelAlignRefineEnv(cfg)
reset = jax.jit(env.reset)
step = jax.jit(env.step)
states = []
for i in range(4):
    s = reset(jax.random.PRNGKey(i))
    states.append(s)
    assert s.obs.shape == (224,) and np.isfinite(s.obs).all()
    assert 0 < int(s.info["reset_horizon"]) <= 1800
    np.testing.assert_array_equal(s.info["initial_completed"], s.info["completed"])
    np.testing.assert_array_equal(s.info["settle_paid"], s.info["rotation_verified"])
    for _ in range(3):
        s = step(s, s.info["last_policy_action"])
        assert (
            np.isfinite(s.reward)
            and np.isfinite(np.array(list(s.metrics.values()))).all()
        )
    print(
        "reference",
        i,
        "command",
        int(s.info["command"]),
        "phase",
        s.info["phase"].tolist(),
        flush=True,
    )
s = states[0]
info = dict(s.info)
info["command"] = jp.int32(2)
ps = s.pipeline_state.replace(
    q=s.pipeline_state.q.at[env._wheel_qpos].set(info["wheel_zero"]),
    qd=s.pipeline_state.qd.at[env._wheel_dof].set(0),
)
a = jp.ones(12) * 0.2
for _ in range(5):
    out = env.wheel_hold_action(ps, info, a)
assert bool(info["encoder_held"][0])
np.testing.assert_array_equal(out[env._pos_rows], a[env._pos_rows])
# Changing simulator-only signals cannot alter the encoder latch or target.
other = dict(info)
other["phase"] = jp.ones(4, dtype=jp.int32) * 4
other["completed"] = ~info["completed"]
other["init_xy"] += 100
np.testing.assert_array_equal(
    env.wheel_hold_action(ps, dict(info), a), env.wheel_hold_action(ps, other, a)
)
# Restore the same physical angle through large unwrapped encoder revolutions.
turned = ps.replace(q=ps.q.at[env._wheel_qpos].add(20 * jp.pi))
target = env.position_targets(info, env.wheel_hold_action(turned, dict(info), a))
np.testing.assert_allclose(
    target[env._wheel_rows][0], turned.q[env._wheel_qpos][0], atol=1e-5
)
# Held wheel remains restoring after another button selects a different leg.
info["command"] = jp.int32(1)
disturbed = ps.replace(q=ps.q.at[env._wheel_qpos[0]].add(0.2))
out = env.wheel_hold_action(disturbed, info, a)
np.testing.assert_allclose(
    env.position_targets(info, out)[env._wheel_rows[0]],
    info["wheel_zero"][0],
    atol=1e-6,
)


class Critic(PupperWheelAlignCriticEnv, PupperWheelAlignRefineEnv):
    pass


critic = Critic(cfg)
c = jax.jit(critic.reset)(jax.random.PRNGKey(7))
c = jax.jit(critic.step)(c, c.info["last_policy_action"])
assert c.obs["sensors"].shape == (224,) and c.obs["critic"].shape == (51,)
cfg.direct.reference_reset_probability = 0.0
clean = PupperWheelAlignRefineEnv(cfg)
c = jax.jit(clean.reset)(jax.random.PRNGKey(9))
assert int(c.info["step"]) == 0 and int(c.info["command"]) == 0
print("ALL REFINEMENT SMOKE CHECKS PASSED", flush=True)
