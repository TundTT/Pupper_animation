"""Brax PPO with frozen preprocessing, action supervision and enforced KL bounds.

Uses Brax's networks, transitions, PPO loss, evaluation and environment wrappers.
This scoped training loop never changes the deployed actor or actuation mapping.
"""

import functools
import time
from pathlib import Path

import jax
import numpy as np
import optax
from brax.training import acting
from brax.training.acme import running_statistics
from brax.training.agents.ppo import losses, networks
from jax import numpy as jp
from workspace import configs
from workspace.wheel_align_critic import PupperWheelAlignCriticEnv, with_task_critic


def guarded_candidate(old, candidate, old_opt, candidate_opt, kl_fn, budget=0.02):
    """Backtrack the proposed step; reject and restore optimizer on failure.

    kl_fn measures divergence from the rollout behavior, not from the previous
    minibatch. This prevents many individually small steps evading the budget.
    """

    def attempt(carry, _):
        chosen, accepted, factor, measured = carry
        trial = jax.tree.map(lambda a, b: a + factor * (b - a), old, candidate)
        kl = kl_fn(trial)
        take = ~accepted & jp.isfinite(kl) & (kl <= budget)
        chosen = jax.tree.map(lambda a, b: jp.where(take, b, a), chosen, trial)
        measured = jp.where(take, kl, measured)
        return (chosen, accepted | take, factor * 0.5, measured), None

    (chosen, accepted, _, measured), _ = jax.lax.scan(
        attempt, (old, jp.array(False), jp.float32(1.0), kl_fn(old)), None, length=8
    )
    opt = jax.tree.map(lambda a, b: jp.where(accepted, b, a), old_opt, candidate_opt)
    return chosen, opt, accepted, measured


def separate_actor_critic_update(params, opt, grad, optimizer, kl_fn, budget):
    """Guard policy updates without suppressing independent value learning."""
    pu, po = optimizer.update(grad.policy, opt["policy"], params.policy)
    vu, vo = optimizer.update(grad.value, opt["value"], params.value)
    pc = optax.apply_updates(params.policy, pu)
    vc = optax.apply_updates(params.value, vu)
    proposed = kl_fn(pc)
    policy, po, accepted, measured = guarded_candidate(
        params.policy, pc, opt["policy"], po, kl_fn, budget
    )
    finite = jp.all(jp.stack([jp.all(jp.isfinite(a)) for a in jax.tree.leaves(vc)]))
    value = jax.tree.map(lambda a, b: jp.where(finite, b, a), params.value, vc)
    vo = jax.tree.map(lambda a, b: jp.where(finite, b, a), opt["value"], vo)
    return (
        params.replace(policy=policy, value=value),
        {"policy": po, "value": vo},
        dict(
            accepted=accepted.astype(jp.float32),
            guard_kl=measured,
            proposed_kl=proposed,
            critic_update_finite=finite.astype(jp.float32),
        ),
    )


def phase_balanced_demonstrations(path, per_phase=8192):
    with np.load(Path(path)) as data:
        group = np.asarray(data["group"])
        rng = np.random.default_rng(319)
        ids = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(group == g),
                    per_phase,
                    replace=np.sum(group == g) < per_phase,
                )
                for g in np.unique(group)
            ]
        )
        return jp.asarray(data["obs"][ids]), jp.asarray(data["actions"][ids])


def phase_balanced_sequences(
    path, history_taps, episode_length, length=16, per_phase=1024
):
    """Contiguous teacher corrections with real previous-action context."""
    with np.load(Path(path)) as data:
        obs, actions, episode, group = (
            data[key] for key in ("obs", "actions", "episode", "group")
        )
    order = np.argsort(episode, kind="stable")
    obs, actions, episode, group = (a[order] for a in (obs, actions, episode, group))
    names, counts = np.unique(episode, return_counts=True)
    complete = np.isin(episode, names[counts == episode_length])
    starts = np.arange(len(episode) - length + 1)
    starts = starts[
        (episode[starts] == episode[starts + length - 1]) & complete[starts]
    ]
    rng = np.random.default_rng(721)
    balanced = []
    for g in np.unique(group):
        choices = starts[group[starts] == g]
        if not len(choices):
            raise ValueError(f"No contiguous supervision sequence for phase group {g}")
        balanced.append(
            rng.choice(choices, per_phase, replace=len(choices) < per_phase)
        )
    starts = np.concatenate(balanced)
    ids = starts[:, None] + np.arange(length)[None, :]
    if tuple(history_taps) == (0, 1, 2, 3):
        memory = obs[starts].reshape((-1, 4, obs.shape[-1] // 4))[:, :, 23:35]
    else:
        first = np.maximum.accumulate(
            np.where(
                np.r_[True, episode[1:] != episode[:-1]], np.arange(len(episode)), 0
            )
        )
        past = np.maximum(
            starts[:, None] - np.arange(max(history_taps) + 1)[None, :],
            first[starts][:, None],
        )
        memory = obs[:, 23:35][past]
    return jp.asarray(obs[ids]), jp.asarray(actions[ids]), jp.asarray(memory)


def feedback_action_loss(
    policy_network, normalizer, weights, features, targets, memory, history_taps, scale
):
    """Supervise12 means through the same previous-output feedback used at deployment."""
    obs_size = features.shape[-1]
    frame_size = obs_size // len(history_taps)

    def advance(memory, data):
        frame, target = data
        frame = (
            frame.reshape((-1, len(history_taps), frame_size))
            .at[:, :, 23:35]
            .set(memory[:, jp.array(history_taps), :])
            .reshape((-1, obs_size))
        )
        logits = policy_network.apply(normalizer, weights, frame)[..., :12]
        prediction = jp.tanh(logits)
        if target.shape[-1] == 16:
            # Gate supervision carries12 physical reference commands plus4
            # binary enable labels. Previous wheel features are issued physical
            # commands, while the eight ordinary joints feed back predictions.
            wheel = jp.array(configs.WHEEL_ACTUATOR_ROWS)
            feedback = prediction.at[:, wheel].set(target[:, wheel])
            error = ((prediction - target[:, :12]) * scale).at[:, wheel].set(0.)
            score = jp.mean(jp.square(error)) + optax.sigmoid_binary_cross_entropy(
                2.*logits[:, wheel], target[:, 12:]).mean()
        elif target.shape[-1] == 12:
            feedback = prediction
            score = jp.mean(jp.square((prediction - target) * scale))
        else:
            raise ValueError('Expected12 position labels or12 plus4 gate labels')
        memory = jp.concatenate((feedback[:, None, :], memory[:, :-1, :]), axis=1)
        return memory, score

    _, scores = jax.lax.scan(
        advance, memory, (features.swapaxes(0, 1), targets.swapaxes(0, 1))
    )
    return scores.mean()


def train(
    *,
    environment,
    eval_env,
    num_timesteps,
    num_envs,
    num_evals,
    episode_length,
    action_repeat,
    batch_size,
    num_minibatches,
    num_updates_per_batch,
    unroll_length,
    learning_rate,
    entropy_cost,
    discounting,
    normalize_observations,
    seed,
    network_factory,
    randomization_fn,
    restore_params,
    wrap_env_fn,
    progress_fn,
    policy_params_fn,
    supervision_path,
    action_scale,
    supervision_weight=0.25,
    supervision_error_scales=(0.01, 0.015, 0.05) * 4,
    supervision_final_weight=0.025,
    supervision_anneal_steps=4_000_000,
    kl_budget=0.02,
):
    if restore_params is None or not normalize_observations:
        raise ValueError(
            "Supervised alignment requires cloned params and fixed normalization"
        )
    if num_envs != batch_size * num_minibatches:
        raise ValueError("Use num_envs == batch_size * num_minibatches")
    normalizer = restore_params[0]  # deliberately never updated, including warmup
    sensor_size = (environment.observation_size["sensors"]
        if isinstance(environment,PupperWheelAlignCriticEnv) else environment.observation_size)
    net = network_factory(
        sensor_size,
        environment.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    if isinstance(environment, PupperWheelAlignCriticEnv):
        net, value_params = with_task_critic(
            net,
            restore_params[2],
            sensor_size,
            environment.critic_observation_size,
            tuple(environment._config.policy.hidden_layer_sizes),
        )
        restore_params = (normalizer, restore_params[1], value_params)
    make_policy = networks.make_inference_fn(net)
    params = losses.PPONetworkParams(policy=restore_params[1], value=restore_params[2])
    baseline = policy_params_fn(0, make_policy, restore_params)
    if baseline is None or baseline["audit/all_four_success_fraction"] <= 0:
        raise RuntimeError(
            "PPO refused: the cloned actor must first pass a nonzero full-sequence audit"
        )
    if baseline["audit/survival_fraction"] < 1.0:
        raise RuntimeError("PPO refused: cloned baseline fell in its readiness audit")
    demo_obs, _ = phase_balanced_demonstrations(supervision_path)
    history_taps = environment._history_taps
    supervision_length = 64 if max(history_taps) >= 16 else 16
    supervision_batch_size = 1024 // supervision_length
    sequence_obs, sequence_action, sequence_memory = phase_balanced_sequences(
        supervision_path, history_taps, episode_length, length=supervision_length
    )
    scale = jp.asarray(action_scale) / jp.asarray(supervision_error_scales)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
    opt = {
        "policy": optimizer.init(params.policy),
        "value": optimizer.init(params.value),
    }
    key = jax.random.PRNGKey(seed)
    key, dr_key, reset_key, eval_key = jax.random.split(key, 4)
    wrapped = wrap_env_fn(
        environment,
        episode_length,
        action_repeat,
        functools.partial(randomization_fn, rng=jax.random.split(dr_key, num_envs)),
    )
    state = jax.jit(wrapped.reset)(jax.random.split(reset_key, num_envs))
    evaluation = wrap_env_fn(
        eval_env,
        episode_length,
        action_repeat,
        functools.partial(randomization_fn, rng=jax.random.split(eval_key, 128)),
    )
    evaluator = acting.Evaluator(
        evaluation, make_policy, 128, episode_length, action_repeat, eval_key
    )
    progress_fn(0, evaluator.run_evaluation(restore_params, {}))
    distribution = net.parametric_action_distribution

    def objective(p, data, rng, bc_x, bc_y, bc_memory, weight):
        loss, metrics = losses.compute_ppo_loss(
            p,
            normalizer,
            data,
            rng,
            net,
            entropy_cost=entropy_cost,
            discounting=discounting,
            clipping_epsilon=0.2,
        )
        bc = feedback_action_loss(
            net.policy_network,
            normalizer,
            p.policy,
            bc_x,
            bc_y,
            bc_memory,
            history_taps,
            scale,
        )
        metrics.update(supervision_loss=bc, supervision_weight=weight)
        return loss + weight * bc, metrics

    @jax.jit
    def iteration(state, params, opt, key, steps):
        key, roll_key, guard_key, update_key = jax.random.split(key, 4)
        policy = make_policy((normalizer, params.policy, params.value))
        state, data = acting.generate_unroll(
            wrapped,
            state,
            policy,
            roll_key,
            unroll_length,
            extra_fields=("truncation",),
        )
        data = jax.tree.map(lambda a: jp.swapaxes(a, 0, 1), data)
        sensor_obs = (
            data.observation["sensors"]
            if isinstance(data.observation, dict)
            else data.observation
        )
        flat_obs = sensor_obs.reshape((-1, sensor_size))
        guard_obs = flat_obs[jax.random.randint(guard_key, (4096,), 0, len(flat_obs))]
        behavior = net.policy_network.apply(normalizer, params.policy, guard_obs)
        reference_behavior = net.policy_network.apply(
            normalizer, params.policy, demo_obs
        )
        reference_ids = jax.random.randint(
            jax.random.fold_in(guard_key, 1), (4096,), 0, len(demo_obs)
        )
        reference_obs = demo_obs[reference_ids]
        reference_logits = reference_behavior[reference_ids]
        initial_params, initial_opt = params, opt
        weight = supervision_weight + (
            supervision_final_weight - supervision_weight
        ) * jp.minimum(steps / supervision_anneal_steps, 1.0)

        def kl_fn(p):
            logits = net.policy_network.apply(normalizer, p.policy, guard_obs)
            rollout_kl = jp.mean(
                distribution.create_dist(logits).kl_divergence(
                    distribution.create_dist(behavior)
                )
            )
            logits = net.policy_network.apply(normalizer, p.policy, reference_obs)
            reference_kl = jp.mean(
                distribution.create_dist(logits).kl_divergence(
                    distribution.create_dist(reference_logits)
                )
            )
            return jp.maximum(rollout_kl, reference_kl)

        def epoch(carry, _):
            p, opt, key = carry
            key, shuffle_key = jax.random.split(key)
            permutation = jax.random.permutation(shuffle_key, num_envs).reshape(
                (num_minibatches, batch_size)
            )

            def minibatch(carry, ids):
                p, opt, key = carry
                key, loss_key, bc_key = jax.random.split(key, 3)
                batch = jax.tree.map(lambda a: a[ids], data)
                bc_ids = jax.random.randint(
                    bc_key, (supervision_batch_size,), 0, len(sequence_obs)
                )
                (_, metrics), grad = jax.value_and_grad(objective, has_aux=True)(
                    p,
                    batch,
                    loss_key,
                    sequence_obs[bc_ids],
                    sequence_action[bc_ids],
                    sequence_memory[bc_ids],
                    weight,
                )
                actor_kl = lambda policy: kl_fn(p.replace(policy=policy))
                p, opt, update_metrics = separate_actor_critic_update(
                    p, opt, grad, optimizer, actor_kl, 0.9 * kl_budget
                )
                metrics.update(update_metrics, learning_rate=jp.float32(learning_rate))
                return (p, opt, key), metrics

            return jax.lax.scan(minibatch, (p, opt, key), permutation)

        (params, opt, _), metrics = jax.lax.scan(
            epoch, (params, opt, update_key), None, length=num_updates_per_batch
        )
        # A final full-rollout check makes the budget an enforced global bound,
        # rather than relying on the representative guard set alone.
        old_logits = data.extras["policy_extras"]["distribution_params"]
        new_logits = net.policy_network.apply(
            normalizer, params.policy, data.observation
        )
        full_kl = jp.mean(
            distribution.create_dist(new_logits).kl_divergence(
                distribution.create_dist(old_logits)
            )
        )
        reference_after = net.policy_network.apply(normalizer, params.policy, demo_obs)
        full_reference_kl = jp.mean(
            distribution.create_dist(reference_after).kl_divergence(
                distribution.create_dist(reference_behavior)
            )
        )
        accept = (
            jp.isfinite(full_kl)
            & (full_kl <= kl_budget)
            & jp.isfinite(full_reference_kl)
            & (full_reference_kl <= kl_budget)
        )
        params = params.replace(
            policy=jax.tree.map(
                lambda a, b: jp.where(accept, b, a),
                initial_params.policy,
                params.policy,
            )
        )
        opt["policy"] = jax.tree.map(
            lambda a, b: jp.where(accept, b, a), initial_opt["policy"], opt["policy"]
        )
        metrics = jax.tree.map(jp.mean, metrics)
        metrics.update(
            full_batch_accepted=accept.astype(jp.float32),
            full_batch_proposed_kl=full_kl,
            full_batch_applied_kl=jp.where(accept, full_kl, 0.0),
            full_reference_proposed_kl=full_reference_kl,
            full_reference_applied_kl=jp.where(accept, full_reference_kl, 0.0),
        )
        return state, params, opt, key, metrics

    interval = max(num_timesteps // max(num_evals - 1, 1), 1)
    next_eval = interval
    steps = 0
    start = time.monotonic()
    latest = {}
    interval_metrics = []
    while steps < num_timesteps:
        state, params, opt, key, metrics = iteration(
            state, params, opt, key, jp.float32(steps)
        )
        steps += num_envs * unroll_length * action_repeat
        latest = {f"training/{k}": float(v) for k, v in jax.device_get(metrics).items()}
        if latest["training/critic_update_finite"] != 1.0:
            raise RuntimeError(
                "Non-finite critic update rejected; stopping to diagnose numerical failure"
            )
        interval_metrics.append(latest)
        if steps >= next_eval or steps >= num_timesteps:
            latest = {
                key: float(np.mean([row[key] for row in interval_metrics]))
                for key in latest
            }
            latest["training/max_applied_kl"] = max(
                row["training/full_batch_applied_kl"] for row in interval_metrics
            )
            latest["training/max_reference_applied_kl"] = max(
                row["training/full_reference_applied_kl"] for row in interval_metrics
            )
            interval_metrics.clear()
            checkpoint = (normalizer, params.policy, params.value)
            policy_params_fn(steps, make_policy, checkpoint)
            latest["training/sps"] = steps / (time.monotonic() - start)
            progress_fn(steps, evaluator.run_evaluation(checkpoint, latest))
            next_eval += interval
    return make_policy, (normalizer, params.policy, params.value), latest
