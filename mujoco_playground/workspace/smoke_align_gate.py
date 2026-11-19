"""Sensor-only wheel-enable controller and mixed supervision smoke checks."""

import json
from pathlib import Path
import jax
import numpy as np
from jax import numpy as jp
from ml_collections import ConfigDict
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from workspace import configs
from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
from workspace.supervised_align_ppo import feedback_action_loss

cfg = ConfigDict(
    json.loads(
        Path("workspace/output/direct_round37_gate_clone/config.json").read_text()
    )
)
cfg.automatic_commands = False
cfg.align.initial_completed_max = 0
cfg.direct.capture_finished_reference = True
env = PupperWheelAlignGateEnv(cfg)
s = jax.jit(env.reset)(jax.random.PRNGKey(42))
s = env.set_command(s, jp.int32(2))
wheel = env._wheel_rows
q = s.pipeline_state.q[env._wheel_qpos]
info = dict(s.info)
a = jp.zeros(12).at[wheel].set(-0.8)
old = env.wheel_hold_action(s.pipeline_state, info, a)
np.testing.assert_allclose(env.position_targets(info, old)[wheel], q, atol=1e-6)
a = a.at[wheel].set(0.8)
for _ in range(3):
    out = env.wheel_hold_action(s.pipeline_state, info, a)
physical = env.position_targets(info, out)[wheel]
assert (
    abs(float(physical[0] - q[0])) > 1e-4 and abs(float(physical[0] - q[0])) <= 0.06001
)
np.testing.assert_allclose(physical[1:], q[1:], atol=1e-6)
np.testing.assert_array_equal(out[env._pos_rows], a[env._pos_rows])
other = dict(info)
other["phase"] = jp.ones(4, dtype=jp.int32) * 4
other["init_xy"] += 100
other["rotation_verified"] = ~other["rotation_verified"]
np.testing.assert_array_equal(
    env.wheel_hold_action(s.pipeline_state, dict(info), a),
    env.wheel_hold_action(s.pipeline_state, other, a),
)
step = jax.jit(env.step)
for _ in range(8):
    s = step(s, jp.zeros(12).at[wheel].set(-0.8))
    assert (
        s.obs.shape == (224,)
        and np.isfinite(s.reward)
        and np.isfinite(np.array(list(s.metrics.values()))).all()
    )
    np.testing.assert_allclose(
        s.obs[23 + wheel], s.info["issued_action"][wheel], atol=1e-6
    )
    assert np.all(np.asarray(s.info["last_policy_action"][wheel]) == np.float32(-0.8))
params = model.load_params(
    "workspace/output/wheel-align-direct_2026-09-08_18-10-16/mjx_params_1003520"
)
net = networks.make_ppo_networks(
    224,
    12,
    policy_hidden_layer_sizes=(512,) * 3,
    value_hidden_layer_sizes=(512,) * 3,
    activation=jax.nn.elu,
    preprocess_observations_fn=rs.normalize,
)
with np.load("workspace/output/direct_round37_gate_reference/demonstrations.npz") as d:
    x = jp.asarray(d["obs"][:4])[None]
    y = jp.asarray(d["actions"][:4])[None]
mem = jp.broadcast_to(x[:, 0, 23:35, None].swapaxes(1, 2), (1, 51, 12))
fn = lambda w: feedback_action_loss(
    net.policy_network, params[0], w, x, y, mem, (0, 1, 10, 50), jp.ones(12)
)
score, grad = jax.jit(jax.value_and_grad(fn))(params[1])
assert np.isfinite(score)
g = grad["params"]["hidden_3"]["bias"][wheel]
assert np.all(np.abs(g) > 0)
# The planned goal must stay fixed through the final motor-settling interval,
# even when the learned enable becomes negative before the encoder settles.
info = dict(s.info)
info['encoder_held'] = jp.zeros(4, dtype=bool)
info['goal_commanded'] = jp.zeros(4, dtype=bool)
info['spin_reference'] = info['wheel_zero'] - .02
info['enable_ticks'] = jp.ones(4, dtype=jp.int32)*3
ps = s.pipeline_state.replace(q=s.pipeline_state.q.at[env._wheel_qpos].set(info['wheel_zero']-.4),
    qd=s.pipeline_state.qd.at[env._wheel_dof].set(1.))
env.wheel_hold_action(ps,info,jp.zeros(12).at[wheel].set(.8))
assert bool(info['goal_commanded'][0]) and not bool(info['encoder_held'][0])
out=env.wheel_hold_action(ps,info,jp.zeros(12).at[wheel].set(-.8))
np.testing.assert_allclose(env.position_targets(info,out)[wheel[0]],info['wheel_zero'][0],atol=1e-6)
print("ALL GATE SMOKE CHECKS PASSED", flush=True)
