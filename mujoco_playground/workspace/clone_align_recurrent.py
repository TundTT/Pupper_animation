"""Train the existing MLP through its deployed previous-action feedback loop.

No new recurrent state is introduced: the actor still receives the same four
frames and emits twelve position targets. Training feeds its predictions back
into last-action slots instead of supplying perfect teacher actions forever.
"""

import argparse, json
from pathlib import Path
import jax, numpy as np, optax
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from workspace import configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--init-params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, default=16)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument(
        "--phase-weight",
        type=float,
        default=0.0,
        help="training-only phase classification from the actor hidden representation",
    )
    p.add_argument(
        "--error-scales",
        type=float,
        nargs=3,
        default=(0.10, 0.15, 0.05),
        metavar=("ABDUCTION", "HIP", "WHEEL"),
        help="physical radian errors of equal loss per joint type",
    )
    p.add_argument("--augment-holds", action="store_true")
    p.add_argument('--wheel-gates', action='store_true',
                   help='learn8 position targets and4 binary wheel-enable outputs from16-column supervision')
    p.add_argument('--gate-capture-goal', action='store_true')
    args = p.parse_args()
    cfg = json.loads((args.init_params.parent / "config.json").read_text())
    if args.wheel_gates:
        cfg['direct'].update(encoder_gate=True, encoder_hold=True,
            reference_reset_probability=0., reference_reset_path='',
            control_convention='eight_position_targets_four_wheel_enables')
        cfg['direct']['capture_finished_reference'] = args.gate_capture_goal
    elif args.gate_capture_goal:
        p.error('--gate-capture-goal requires --wheel-gates')
    frame_size = cfg["direct"]["observation_frame_size"]
    obs_size = frame_size * 4
    cfg["direct"]["supervision_path"] = str(args.reference / "demonstrations.npz")
    cfg["direct"]["reference_path"] = str(args.reference / "poses.npz")
    reference_metadata = json.loads((args.reference / "manifest.json").read_text())
    if "history_taps" in reference_metadata:
        cfg["direct"]["history_taps"] = reference_metadata["history_taps"]
    history_taps = (
        tuple(cfg["direct"]["history_taps"])
        if "history_taps" in cfg["direct"]
        else (0, 1, 2, 3)
    )
    for key, value in reference_metadata.get("student_sensor_noise", {}).items():
        cfg["dr"][key] = value
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    run = None
    if args.use_wandb:
        import wandb

        run = wandb.init(
            entity="QuadMorph",
            project="wheel-leg lift and align triangle base",
            name=args.output.name,
            config=cfg,
        )
        (args.output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url})
        )
    with np.load(args.reference / "demonstrations.npz") as d:
        obs, labels, episodes, groups = (
            d[k] for k in ["obs", "actions", "episode", "group"]
        )
    if labels.shape[1] != (16 if args.wheel_gates else 12):
        raise ValueError('Supervision layout does not match wheel-gate mode')
    order = np.argsort(episodes, kind="stable")
    sorted_episodes = episodes[order]
    first_in_episode = jp.asarray(
        np.maximum.accumulate(
            np.where(
                np.r_[True, sorted_episodes[1:] != sorted_episodes[:-1]],
                np.arange(len(order)),
                0,
            )
        )
    )
    starts = np.arange(len(order) - args.length + 1)
    starts = starts[
        sorted_episodes[starts] == sorted_episodes[starts + args.length - 1]
    ]
    validation_episodes = np.unique(episodes)[::5]
    valid = np.isin(sorted_episodes[starts], validation_episodes)
    rng = np.random.default_rng(319)

    def balanced(mask, count):
        selected = []
        for group in np.unique(groups):
            ids = starts[mask & (groups[order[starts]] == group)]
            if len(ids):
                selected.append(rng.choice(ids, count, replace=len(ids) < count))
        return np.concatenate(selected)

    train_starts = jp.array(balanced(~valid, 8192))
    valid_starts = jp.array(balanced(valid, 32))
    x, y = jp.asarray(obs[order]), jp.asarray(labels[order])
    offset = jp.arange(args.length)
    phase_groups = jp.asarray(groups[order])
    with open(Path(cfg["direct"]["reference_path"]).with_name("manifest.json")) as f:
        successful_ids = json.load(f)["successful_episodes"]
    known_success = jp.asarray(np.isin(episodes[order], successful_ids))
    norm, actor, critic = model.load_params(args.init_params)
    layers = tuple(cfg["policy"]["hidden_layer_sizes"])
    net = networks.make_ppo_networks(
        obs_size,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    if args.phase_weight:
        actor = {
            "actor": actor,
            "phase_kernel": jax.random.normal(jax.random.PRNGKey(43), (layers[-1], 5))
            * 0.01,
            "phase_bias": jp.zeros(5),
        }

    def deployable(weights):
        return weights["actor"] if args.phase_weight else weights

    def phase_logits(weights, features):
        hidden = rs.normalize(features, norm)
        for i in range(len(layers)):
            layer = weights["actor"]["params"][f"hidden_{i}"]
            hidden = jax.nn.elu(hidden @ layer["kernel"] + layer["bias"])
        return hidden @ weights["phase_kernel"] + weights["phase_bias"]

    scale = jp.array(cfg["action_scale"]) / jp.tile(jp.array(args.error_scales), 4)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate)
    )
    opt = optimizer.init(actor)

    def loss(weights, starts, augment_key=None):
        ids = starts[:, None] + offset[None, :]
        features = x[ids]
        targets = y[ids]
        if args.augment_holds and augment_key is not None:
            frame = features.reshape((-1, args.length, 4, frame_size))
            command = jp.argmax(frame[:, :, 0, 6:11], axis=-1)
            active = jp.array([-1, 1, 0, 2, 3])[command]
            hold = (
                (jp.arange(4)[None, None, :] != active[:, :, None])
                | ((phase_groups[ids] % 5) == 4)[:, :, None]
            ) & known_success[ids][:, :, None]
            delta = (
                jax.random.uniform(
                    augment_key, (len(starts), 1, 4), minval=-0.2, maxval=0.2
                )
                * hold
            )
            wheel = jp.array(configs.WHEEL_ACTUATOR_ROWS)
            frame = frame.at[:, :, :, 11 + wheel].add(delta[:, :, None, :])
            angle = (
                jp.arctan2(frame[:, :, :, 39:43], frame[:, :, :, 43:47])
                - delta[:, :, None, :]
            )
            frame = (
                frame.at[:, :, :, 39:43]
                .set(jp.sin(angle))
                .at[:, :, :, 43:47]
                .set(jp.cos(angle))
            )
            features = frame.reshape(features.shape)
        if history_taps == (0, 1, 2, 3):
            memory = features[:, 0].reshape((-1, 4, frame_size))[:, :, 23:35]
        else:
            past = jp.maximum(
                starts[:, None] - jp.arange(max(history_taps) + 1)[None, :],
                first_in_episode[starts][:, None],
            )
            memory = x[past].reshape(
                (len(starts), max(history_taps) + 1, 4, frame_size)
            )[:, :, 0, 23:35]

        def advance(memory, data):
            frame, target, phase = data
            frame = (
                frame.reshape((-1, 4, frame_size))
                .at[:, :, 23:35]
                .set(memory[:, jp.array(history_taps), :])
                .reshape((-1, obs_size))
            )
            logits = net.policy_network.apply(norm, deployable(weights), frame)[..., :12]
            prediction = jp.tanh(logits)
            feedback = prediction
            if args.wheel_gates:
                feedback = feedback.at[:, jp.array(configs.WHEEL_ACTUATOR_ROWS)].set(
                    target[:, jp.array(configs.WHEEL_ACTUATOR_ROWS)])
            next_memory = jp.concatenate(
                (feedback[:, None, :], memory[:, :-1, :]), axis=1
            )
            error = (prediction - target[:, :12]) * scale
            if args.wheel_gates:
                error = error.at[:, jp.array(configs.WHEEL_ACTUATOR_ROWS)].set(0.)
            score = jp.mean(error ** 2)
            if args.wheel_gates:
                score += optax.sigmoid_binary_cross_entropy(
                    2.*logits[:, jp.array(configs.WHEEL_ACTUATOR_ROWS)], target[:, 12:]).mean()
            if args.phase_weight:
                score += (
                    args.phase_weight
                    * optax.softmax_cross_entropy_with_integer_labels(
                        phase_logits(weights, frame), phase
                    ).mean()
                )
            return next_memory, score

        _, scores = jax.lax.scan(
            advance,
            memory,
            (
                features.swapaxes(0, 1),
                targets.swapaxes(0, 1),
                (phase_groups[ids] % 5).swapaxes(0, 1),
            ),
        )
        return scores.mean()

    @jax.jit
    def update(actor, opt, key):
        starts = train_starts[
            jax.random.randint(key, (args.batch_size,), 0, len(train_starts))
        ]
        key, augment_key = jax.random.split(key)
        score, grad = jax.value_and_grad(loss)(actor, starts, augment_key)
        updates, opt = optimizer.update(grad, opt, actor)
        return optax.apply_updates(actor, updates), opt, score

    evaluate = jax.jit(lambda a: loss(a, valid_starts))
    phase_accuracy = (
        jax.jit(
            lambda a: jp.mean(
                jp.argmax(phase_logits(a, x[valid_starts]), axis=-1)
                == phase_groups[valid_starts] % 5
            )
        )
        if args.phase_weight
        else None
    )
    best = float("inf")
    history = []
    key = jax.random.PRNGKey(902)
    for step in range(1, args.steps + 1):
        key, k = jax.random.split(key)
        actor, opt, score = update(actor, opt, k)
        if step % 500 == 0:
            score = float(score)
            validation = float(evaluate(actor))
            record = {"step": step, "loss": score, "validation": validation}
            if phase_accuracy is not None:
                record["phase_accuracy"] = float(phase_accuracy(actor))
            history.append(record)
            print(json.dumps(record), flush=True)
            if run is not None:
                run.log(record, step=step)
            if validation < best:
                best = validation
                model.save_params(
                    args.output / "cloned_params", (norm, deployable(actor), critic)
                )
            if step % 2000 == 0:
                model.save_params(
                    args.output / f"cloned_params_{step}",
                    (norm, deployable(actor), critic),
                )
    model.save_params(
        args.output / "cloned_params_last", (norm, deployable(actor), critic)
    )
    if args.phase_weight:
        predicted_phase = np.asarray(
            jp.argmax(phase_logits(actor, x[valid_starts]), axis=-1)
        )
        actual_phase = np.asarray(phase_groups[valid_starts] % 5)
        confusion = np.bincount(
            actual_phase * 5 + predicted_phase, minlength=25
        ).reshape((5, 5))
        (args.output / "phase_confusion.json").write_text(
            json.dumps(
                {
                    "labels": ["idle", "lift", "rotate", "lower", "hold"],
                    "rows": "actual",
                    "columns": "predicted",
                    "validation_start_frames": confusion.tolist(),
                    "training_only_head": True,
                },
                indent=2,
            )
        )
    (args.output / "cloning.json").write_text(
        json.dumps(
            {
                "length": args.length,
                "init": str(args.init_params),
                "history": history,
                "self_fed_previous_actions": True,
                "training_only_phase_weight": args.phase_weight,
                "error_scales": args.error_scales,
            },
            indent=2,
        )
    )
    if run is not None:
        run.finish()
    print("RECURRENT-FEEDBACK CLONING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
