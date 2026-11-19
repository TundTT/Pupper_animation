"""Synthetic update tests: KL rejection/backtracking and wheel supervision."""

import functools
from types import SimpleNamespace
import jax
import numpy as np
import optax
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import losses
from workspace.supervised_align_ppo import (
    guarded_candidate,
    train,
    feedback_action_loss,
    phase_balanced_sequences,
    separate_actor_critic_update,
)
from workspace.wheel_align_training import make_alignment_networks
from workspace import configs

toy = losses.PPONetworkParams(policy=jp.float32(0.0), value=jp.float32(0.0))
toy_optimizer = optax.adam(0.1)
toy_opt = {
    "policy": toy_optimizer.init(toy.policy),
    "value": toy_optimizer.init(toy.value),
}
toy_grad = losses.PPONetworkParams(policy=jp.float32(1.0), value=jp.float32(1.0))
changed, changed_opt, check = separate_actor_critic_update(
    toy, toy_opt, toy_grad, toy_optimizer, lambda p: jp.where(p == 0.0, 0.0, 1e6), 0.02
)
assert float(changed.policy) == 0.0 and float(check["accepted"]) == 0.0
assert float(changed.value) < 0.0 and float(check["critic_update_finite"]) == 1.0
for a, b in zip(
    jax.tree.leaves(toy_opt["policy"]), jax.tree.leaves(changed_opt["policy"])
):
    np.testing.assert_array_equal(a, b)
assert int(changed_opt["value"][0].count) == 1
print(
    "PASS: rejected actor proposal restores actor optimizer while independent critic still learns"
)

p = model.load_params("workspace/output/align_direct_recovery/cloned_params")
net = make_alignment_networks(
    220,
    12,
    policy_hidden_layer_sizes=(128,) * 3,
    value_hidden_layer_sizes=(128,) * 3,
    activation=jax.nn.elu,
    preprocess_observations_fn=rs.normalize,
)
x = jp.broadcast_to(p[0].mean, (16, 220))
old = losses.PPONetworkParams(policy=p[1], value=p[2])
dist = net.parametric_action_distribution
base = net.policy_network.apply(p[0], old.policy, x)


def kl(candidate):
    logits = net.policy_network.apply(p[0], candidate.policy, x)
    return jp.mean(dist.create_dist(logits).kl_divergence(dist.create_dist(base)))


def shifted(amount):
    actor = jax.tree.map(lambda a: a, old.policy)
    actor["params"]["hidden_3"]["bias"] = (
        actor["params"]["hidden_3"]["bias"].at[:12].add(amount)
    )
    return old.replace(policy=actor)


check = jax.jit(functools.partial(guarded_candidate, kl_fn=kl, budget=0.02))
a, opt, accepted, k = check(old, shifted(1.0), jp.int32(7), jp.int32(8))
assert not accepted and opt == 7
for first, second in zip(jax.tree.leaves(a), jax.tree.leaves(old)):
    np.testing.assert_array_equal(first, second)
a, opt, accepted, k = check(old, shifted(0.002), jp.int32(7), jp.int32(8))
assert accepted and opt == 8 and k <= 0.02 and k > 0
print(
    "PASS: excessive proposal rejected with optimizer restored; smaller proposal backtracked into KL budget"
)

# A proposal can leave startup behavior unchanged while jumping on a later
# phase. Both populations must constrain the update.
reference_x = x + p[0].std * 0.5


def hidden(features):
    h = rs.normalize(features, p[0])
    for i in range(3):
        layer = old.policy["params"][f"hidden_{i}"]
        h = jax.nn.elu(h @ layer["kernel"] + layer["bias"])
    return h


h0, h1 = hidden(x), hidden(reference_x)
row = int(jp.argmax(jp.abs(h1[0] - h0[0])))
amount = 1.0 / (h1[0, row] - h0[0, row])
actor = jax.tree.map(lambda a: a, old.policy)
out = actor["params"]["hidden_3"]
out["kernel"] = out["kernel"].at[row, 2].add(amount)
out["bias"] = out["bias"].at[2].add(-amount * h0[0, row])
candidate = old.replace(policy=actor)
reference_base = net.policy_network.apply(p[0], old.policy, reference_x)


def reference_kl(weights):
    logits = net.policy_network.apply(p[0], weights.policy, reference_x)
    return jp.mean(
        dist.create_dist(logits).kl_divergence(dist.create_dist(reference_base))
    )


assert kl(candidate) < 0.02 and reference_kl(candidate) > 0.02
guard = jax.jit(
    functools.partial(
        guarded_candidate,
        kl_fn=lambda weights: jp.maximum(kl(weights), reference_kl(weights)),
        budget=0.02,
    )
)
kept, _, _, _ = guard(old, candidate, jp.int32(7), jp.int32(8))
assert reference_kl(kept) <= 0.02 and kl(kept) <= 0.02
print("PASS: reference-phase KL guard catches a jump invisible on startup observations")

target = jp.zeros((16, 12)).at[:, jp.array(configs.WHEEL_ACTUATOR_ROWS)].set(0.2)


def bc(weights):
    logits = net.policy_network.apply(p[0], weights, x)
    return jp.mean((jp.tanh(logits[:, :12]) - target) ** 2)


grad = jax.grad(bc)(old.policy)["params"]["hidden_3"]["bias"]
assert np.all(np.abs(np.asarray(grad)[configs.WHEEL_ACTUATOR_ROWS]) > 1e-6)
print("PASS: explicit supervision reaches all four wheel mean outputs")
features = jp.tile(x[:, None, :], (1, 6, 1))
labels = jp.tile(target[:, None, :], (1, 6, 1))
memory = x.reshape((16, 4, 55))[:, :, 23:35]


def feedback(weights):
    return feedback_action_loss(
        net.policy_network,
        p[0],
        weights,
        features,
        labels,
        memory,
        (0, 1, 2, 3),
        jp.ones(12),
    )


value, gradient = jax.jit(jax.value_and_grad(feedback))(old.policy)
assert np.isfinite(value) and all(
    np.isfinite(v).all() for v in jax.tree.leaves(gradient)
)
assert np.all(
    np.abs(
        np.asarray(gradient["params"]["hidden_3"]["bias"])[configs.WHEEL_ACTUATOR_ROWS]
    )
    > 1e-6
)
print(
    "PASS: sequence supervision differentiates through self-fed outputs and reaches all wheel means"
)
sx, sy, sm = phase_balanced_sequences(
    "workspace/output/direct_round21_sparse_data/demonstrations.npz",
    (0, 1, 10, 50),
    1800,
    length=4,
    per_phase=2,
)
np.testing.assert_array_equal(
    sm[:, jp.array([0, 1, 10, 50])], sx[:, 0].reshape((-1, 4, 56))[:, :, 23:35]
)
assert sx.shape[1:] == (4, 224) and sy.shape[1:] == (4, 12)
print(
    "PASS: sampled sparse-history sequences contain the recorded previous-action context"
)
kwargs = dict(configs.get_wheel_align_direct_config().ppo)
factory = functools.partial(
    make_alignment_networks,
    policy_hidden_layer_sizes=(128,) * 3,
    value_hidden_layer_sizes=(128,) * 3,
    activation=jax.nn.elu,
)
try:
    train(
        environment=SimpleNamespace(observation_size=220, action_size=12),
        eval_env=None,
        network_factory=factory,
        randomization_fn=None,
        restore_params=p,
        wrap_env_fn=None,
        progress_fn=None,
        policy_params_fn=lambda *args: {"audit/all_four_success_fraction": 0.0},
        supervision_path="must_not_be_opened",
        action_scale=configs.get_wheel_align_direct_config().action_scale,
        **kwargs,
    )
except RuntimeError as err:
    assert "PPO refused" in str(err)
else:
    raise AssertionError("PPO did not reject zero-success clone")
print("PASS: zero-success clone refused before data loading or environment stepping")
print("ALL PROTECTED PPO SMOKE CHECKS PASSED")
