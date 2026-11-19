"""Phase-balanced behavior cloning into an ordinary deployable 12-output MLP."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
import optax
from brax.io import model
from brax.training.acme import running_statistics
from jax import numpy as jp

from workspace import configs
from workspace.wheel_align_training import make_alignment_networks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("workspace/output/align_direct_reference"),
    )
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--init-params", type=Path)
    parser.add_argument("--augment-holds", action="store_true")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precision-scale", type=float, default=1.0)
    parser.add_argument("--wheel-weight", type=float, default=1.0)
    parser.add_argument(
        "--action-memory-noise",
        type=float,
        default=0.0,
        help="training-only radian perturbation of all previous motor targets",
    )
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()
    cfg = configs.get_wheel_align_direct_config()
    if (args.reference / "config.json").exists():
        cfg.direct.observation_frame_size = json.loads(
            (args.reference / "config.json").read_text()
        )["direct"]["observation_frame_size"]
    else:
        with np.load(args.reference / "demonstrations.npz") as data:
            cfg.direct.observation_frame_size = (
                data["obs"].shape[1] // cfg.observation_history
            )
    frame_size = int(cfg.direct.observation_frame_size)
    cfg.policy.hidden_layer_sizes = (args.width,) * 3
    cfg.direct.supervision_path = str(args.reference / "demonstrations.npz")
    cfg.direct.reference_path = str(args.reference / "poses.npz")
    reference_metadata = json.loads((args.reference / "manifest.json").read_text())
    if "history_taps" in reference_metadata:
        cfg.direct.history_taps = tuple(reference_metadata["history_taps"])
    for key, value in reference_metadata.get("student_sensor_noise", {}).items():
        cfg.dr[key] = value
    output_dir = args.output or args.reference
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(
            entity="QuadMorph",
            project="wheel-leg lift and align triangle base",
            name=output_dir.name,
            config={
                **cfg.to_dict(),
                "cloning": vars(args)
                | {
                    "reference": str(args.reference),
                    "output": str(args.output),
                    "init_params": str(args.init_params),
                },
            },
        )
    (output_dir / "wandb_run.json").write_text(
        json.dumps({"id": wandb_run.id, "url": wandb_run.url})
    )
    with np.load(args.reference / "demonstrations.npz") as data:
        observations, targets, groups, episodes = (
            data[k] for k in ("obs", "actions", "group", "episode")
        )
    validation_episodes = np.unique(episodes)[::5]
    validation = np.isin(episodes, validation_episodes)
    train = ~validation
    # Equal sampling of command/phase groups; no time index enters the network.
    rng = np.random.default_rng(5)
    balanced = []
    for group in np.unique(groups):
        indices = np.where(train & (groups == group))[0]
        if not len(indices):
            raise RuntimeError(f"No training examples for command/phase group {group}")
        balanced.append(rng.choice(indices, size=8192, replace=len(indices) < 8192))
    balanced = np.concatenate(balanced)
    obs, target = jp.asarray(observations[balanced]), jp.asarray(targets[balanced])
    valid_obs, valid_target = (
        jp.asarray(observations[validation]),
        jp.asarray(targets[validation]),
    )
    observation_size = frame_size * cfg.observation_history
    normalizer = running_statistics.update(
        running_statistics.init_state(jp.zeros(observation_size), std_eps=1e-4), obs
    )
    net = make_alignment_networks(
        observation_size,
        12,
        initial_std=0.03,
        policy_hidden_layer_sizes=(args.width,) * 3,
        value_hidden_layer_sizes=(args.width,) * 3,
        activation=jax.nn.elu,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = net.policy_network.init(jax.random.PRNGKey(1))
    # Exploration on a stationary wheel should not overwhelm the holding torque.
    output = policy["params"]["hidden_3"]
    output["bias"] = (
        output["bias"]
        .at[12 + jp.array(configs.WHEEL_ACTUATOR_ROWS)]
        .set(jp.log(jp.expm1(0.01 - 0.001)))
    )
    value = net.value_network.init(jax.random.PRNGKey(2))
    if args.init_params:
        normalizer, policy, value = model.load_params(args.init_params)
    optimizer = optax.adam(args.learning_rate)
    opt = optimizer.init(policy)
    scale = jp.asarray(cfg.action_scale)
    error_scale = jp.array((0.10, 0.15, 0.10) * 4) * args.precision_scale
    error_scale = error_scale.at[jp.array(configs.WHEEL_ACTUATOR_ROWS)].divide(
        jp.sqrt(args.wheel_weight)
    )
    balanced_groups = jp.asarray(groups[balanced])
    with open(Path(cfg.direct.reference_path).with_name("manifest.json")) as f:
        successful_episodes = json.load(f)["successful_episodes"]
    balanced_success = jp.asarray(np.isin(episodes[balanced], successful_episodes))

    def augment(x, y, group, known_success, key):
        frame = x.reshape((-1, cfg.observation_history, frame_size))
        command = jp.argmax(frame[:, 0, 6:11], axis=-1)
        active = jp.array([-1, 1, 0, 2, 3])[command]
        hold = (jp.arange(4)[None, :] != active[:, None]) | ((group % 5) == 4)[:, None]
        # Failed corrective trajectories can contain previously selected but
        # unfinished legs. Only successful teacher episodes justify goal=0.
        hold &= known_success[:, None]
        key1, key2, key3 = jax.random.split(key, 3)
        delta = jax.random.uniform(key1, hold.shape, minval=-0.4, maxval=0.4) * hold
        wheel = jp.array(configs.WHEEL_ACTUATOR_ROWS)
        frame = frame.at[:, :, 11 + wheel].add(delta[:, None, :])
        error = jp.arctan2(frame[:, :, 39:43], frame[:, :, 43:47]) - delta[:, None, :]
        frame = (
            frame.at[:, :, 39:43].set(jp.sin(error)).at[:, :, 43:47].set(jp.cos(error))
        )
        previous = jp.clip(
            frame[:, :, 23 + wheel]
            + 0.03 * jax.random.normal(key2, hold.shape)[:, None, :] * hold[:, None, :],
            -1.0,
            1.0,
        )
        frame = frame.at[:, :, 23 + wheel].set(previous)
        frame = frame.at[:, :, 35:39].add(
            0.05 * jax.random.normal(key3, hold.shape)[:, None, :] * hold[:, None, :]
        )
        # Successful demonstration segments imply an inactive selected leg has
        # finished rotating. Its fixed position goal is the calibrated default;
        # an unselected leg retains the measured mode-entry encoder anchor.
        target_hold = jp.where(frame[:, 0, 47:51] > 0.5, 0.0, frame[:, 0, 51:55])
        y = y.at[:, wheel].set(jp.where(hold, target_hold, y[:, wheel]))
        return frame.reshape(x.shape), y

    def loss(params, x, y):
        logits = net.policy_network.apply(normalizer, params, x)
        predicted = jp.tanh(logits[..., :12])
        return jp.mean(jp.square((predicted - y) * scale / error_scale))

    @jax.jit
    def update(policy, opt, key):
        key, augment_key, memory_key = jax.random.split(key, 3)
        ids = jax.random.randint(key, (args.batch_size,), 0, obs.shape[0])
        x, y = obs[ids], target[ids]
        if args.augment_holds:
            x, y = augment(
                x, y, balanced_groups[ids], balanced_success[ids], augment_key
            )
        if args.action_memory_noise:
            frame = x.reshape((-1, cfg.observation_history, frame_size))
            noise = (
                jax.random.normal(memory_key, (len(ids), 1, 12))
                * args.action_memory_noise
                / scale
            )
            frame = frame.at[:, :, 23:35].set(
                jp.clip(frame[:, :, 23:35] + noise, -1.0, 1.0)
            )
            x = frame.reshape(x.shape)
        score, grad = jax.value_and_grad(loss)(policy, x, y)
        changes, opt = optimizer.update(grad, opt, policy)
        return optax.apply_updates(policy, changes), opt, score

    @jax.jit
    def evaluate(policy):
        prediction = jp.tanh(
            net.policy_network.apply(normalizer, policy, valid_obs)[..., :12]
        )
        errors = (prediction - valid_target) * scale
        return loss(policy, valid_obs, valid_target), jp.sqrt(
            jp.mean(errors**2, axis=0)
        )

    best = float("inf")
    history = []
    key = jax.random.PRNGKey(3)
    for step in range(1, args.steps + 1):
        key, subkey = jax.random.split(key)
        policy, opt, score = update(policy, opt, subkey)
        if step % 500 == 0 or step == args.steps:
            validation_loss, error = jax.device_get(evaluate(policy))
            record = dict(
                step=step,
                train_loss=float(score),
                validation_loss=float(validation_loss),
                joint_target_rmse_rad=error.tolist(),
            )
            history.append(record)
            print(json.dumps(record), flush=True)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "cloning/train_loss": record["train_loss"],
                        "cloning/validation_loss": record["validation_loss"],
                    },
                    step=step,
                )
            if validation_loss < best:
                best = float(validation_loss)
                model.save_params(
                    output_dir / "cloned_params", (normalizer, policy, value)
                )
            if step % 2000 == 0:
                model.save_params(
                    output_dir / f"cloned_params_{step}", (normalizer, policy, value)
                )
    model.save_params(output_dir / "cloned_params_last", (normalizer, policy, value))
    (output_dir / "cloning.json").write_text(
        json.dumps(
            dict(
                history=history,
                training_episodes=np.unique(episodes[train]).tolist(),
                validation_episodes=validation_episodes.tolist(),
                balanced_training_samples=len(balanced),
                validation_samples=int(validation.sum()),
                best_validation_loss=best,
                observation_size=observation_size,
                action_size=12,
                width=args.width,
                learning_rate=args.learning_rate,
                precision_scale=args.precision_scale,
                hold_augmentation=args.augment_holds,
                init_params=str(args.init_params),
            ),
            indent=2,
        )
    )
    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(
        "Saved teacher-initialized actor and observation statistics:",
        output_dir / "cloned_params",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
