"""Append observable command age to recorded frames; expand a cloned input layer.

Uses button history only, never simulated contact/phase labels. Added input
weights start at zero, so expansion preserves the original actor exactly.
"""

import argparse, json, shutil
from pathlib import Path
import jax, numpy as np
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.reference / "demonstrations.npz") as data:
        arrays = {k: data[k] for k in data.files}
    old = arrays["obs"].reshape((-1, 4, 55))
    age = np.zeros(old.shape[:2], dtype=np.float32)
    for ep in np.unique(arrays["episode"]):
        ids = np.flatnonzero(arrays["episode"] == ep)
        command = np.argmax(old[ids, 0, 6:11], axis=-1)
        since = np.zeros(len(ids), dtype=np.float32)
        for i in range(1, len(ids)):
            since[i] = (
                0 if command[i] != command[i - 1] else min(since[i - 1] + 0.02, 1.0)
            )
        for h in range(4):
            age[ids, h] = since[np.maximum(np.arange(len(ids)) - h, 0)]
    arrays["obs"] = np.concatenate((old, age[..., None]), axis=-1).reshape((-1, 224))
    np.savez_compressed(args.output / "demonstrations.npz", **arrays)
    for name in ("poses.npz", "manifest.json"):
        shutil.copy2(args.reference / name, args.output / name)
    metadata = json.loads((args.output / "manifest.json").read_text())
    metadata.update(
        observation_size=224, command_age_source="button history, capped at one second"
    )
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2))
    cfg = json.loads((args.params.parent / "config.json").read_text())
    cfg["direct"]["observation_frame_size"] = 56
    cfg["direct"]["supervision_path"] = str(args.output / "demonstrations.npz")
    cfg["direct"].update(
        freeze_normalization=True,
        kl_budget=0.02,
        supervision_weight=5.0,
        supervision_final_weight=0.5,
        supervision_anneal_steps=8_000_000,
    )
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    params = model.load_params(args.params)
    norm, actor, critic = params
    rows = np.concatenate([np.arange(h * 56, h * 56 + 55) for h in range(4)])

    def expanded(stats, value):
        return jp.full((224,), value, dtype=stats.dtype).at[rows].set(stats)

    count = float(norm.count.hi) * 2**32 + float(norm.count.lo)
    newnorm = norm.replace(
        mean=expanded(norm.mean, 0.0),
        std=expanded(norm.std, 1.0),
        summed_variance=expanded(norm.summed_variance, count),
    )

    def expand_weights(weights):
        new = jax.tree.map(lambda a: a, weights)
        w = weights["params"]["hidden_0"]["kernel"]
        new["params"]["hidden_0"]["kernel"] = (
            jp.zeros((224, w.shape[1]), dtype=w.dtype).at[rows].set(w)
        )
        return new

    newactor, newcritic = expand_weights(actor), expand_weights(critic)
    layers = tuple(cfg["policy"]["hidden_layer_sizes"])

    def net(size):
        return networks.make_ppo_networks(
            size,
            12,
            policy_hidden_layer_sizes=layers,
            value_hidden_layer_sizes=layers,
            activation=jax.nn.elu,
            preprocess_observations_fn=rs.normalize,
        )

    before = net(220).policy_network.apply(
        norm, actor, jp.array(old[:128].reshape((-1, 220)))
    )
    after = net(224).policy_network.apply(
        newnorm, newactor, jp.array(arrays["obs"][:128])
    )
    np.testing.assert_allclose(before, after, atol=1e-5)
    model.save_params(args.output / "expanded_params", (newnorm, newactor, newcritic))
    print(
        "PASS: input expansion preserves actor outputs; button-age family saved",
        args.output,
        flush=True,
    )


if __name__ == "__main__":
    main()
