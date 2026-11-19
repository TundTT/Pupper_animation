"""Check critic isolation, warm-start preservation, real steps and Brax PPO loss."""

import functools
import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jp
from brax.io import model
from brax.training import types
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import losses, networks
from ml_collections import ConfigDict
from workspace.wheel_align_critic import PupperWheelAlignCriticEnv, with_task_critic
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import domain_randomize_wheel_align
from workspace.wheel_align_training import wrap_for_alignment


def main():
    path = Path("workspace/output/direct_round29_precision_exploration/cloned_params")
    cfg = ConfigDict(json.loads((path.parent / "config.json").read_text()))
    cfg.direct.lift_clearance_upper = 0.010
    cfg.reward_config.scales.hold = -1.0
    public = PupperWheelAlignDirectEnv(cfg)
    env = PupperWheelAlignCriticEnv(cfg)
    assert env.observation_size == {"sensors":224,"critic":51}
    state = jax.jit(env.reset)(jax.random.PRNGKey(51))
    reference = jax.jit(public.reset)(jax.random.PRNGKey(51))
    assert state.obs["sensors"].shape == (224,) and state.obs["critic"].shape == (51,)
    np.testing.assert_array_equal(state.obs["sensors"], reference.obs)
    norm, actor, old_value = model.load_params(path)
    layers = tuple(cfg.policy.hidden_layer_sizes)
    base = networks.make_ppo_networks(
        224,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    net, value = with_task_critic(base, old_value, 224, 51, layers)
    logits = net.policy_network.apply(norm, actor, state.obs)
    altered = state.replace(
        obs={"sensors": state.obs["sensors"], "critic": state.obs["critic"] + 100.0}
    )
    np.testing.assert_array_equal(
        logits, net.policy_network.apply(norm, actor, altered.obs)
    )
    np.testing.assert_array_equal(
        logits, base.policy_network.apply(norm, actor, state.obs["sensors"])
    )
    np.testing.assert_allclose(
        net.value_network.apply(norm, value, state.obs),
        base.value_network.apply(norm, old_value, state.obs["sensors"]),
        atol=1e-5,
    )
    step = jax.jit(env.step)
    action = state.info["last_policy_action"]
    actual, changed = step(state, action), step(altered, action)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(changed)):
        np.testing.assert_array_equal(a, b)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(actual))
    print(
        "PASS:51 critic features cannot affect224 actor inputs,12 means,or actual physics; warm critic preserved",
        flush=True,
    )
    features = jax.tree.map(lambda a: jp.broadcast_to(a, (2, 20) + a.shape), state.obs)
    batch_logits = net.policy_network.apply(norm, actor, features)
    distribution = net.parametric_action_distribution
    raw = distribution.sample_no_postprocessing(batch_logits, jax.random.PRNGKey(52))
    transition = types.Transition(
        observation=features,
        action=distribution.postprocess(raw),
        reward=jp.full((2, 20), 0.02),
        discount=jp.ones((2, 20)),
        next_observation=features,
        extras={
            "state_extras": {"truncation": jp.zeros((2, 20))},
            "policy_extras": {
                "raw_action": raw,
                "log_prob": distribution.log_prob(batch_logits, raw),
                "distribution_params": batch_logits,
            },
        },
    )
    params = losses.PPONetworkParams(policy=actor, value=value)

    def objective(p):
        return losses.compute_ppo_loss(
            p,
            norm,
            transition,
            jax.random.PRNGKey(53),
            net,
            discounting=0.995,
            entropy_cost=0.001,
        )[0]

    score, grad = jax.jit(jax.value_and_grad(objective))(params)
    assert np.isfinite(float(score)) and all(
        np.isfinite(x).all() for x in jax.tree.leaves(grad)
    )
    assert np.linalg.norm(grad.value["params"]["hidden_0"]["kernel"][224:]) > 0
    print(
        "PASS: Brax PPO dictionary transitions produce finite gradients; new critic inputs receive learning signal",
        flush=True,
    )
    wrapped = wrap_for_alignment(
        env,
        cfg.episode_length,
        1,
        functools.partial(
            domain_randomize_wheel_align,
            rng=jax.random.split(jax.random.PRNGKey(54), 2),
        ),
    )
    batch = jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(55), 2))
    batch = jax.jit(wrapped.step)(batch, batch.info["last_policy_action"])
    assert batch.obs["critic"].shape == (2, 51)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(batch))
    print("ALL TASK-STATE CRITIC SMOKE CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
