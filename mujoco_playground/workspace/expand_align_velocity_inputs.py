"""Add encoder velocity inputs without changing a warm-start actor's function."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from jax import numpy as jp
from workspace import configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads((args.params.parent / "config.json").read_text())
    if cfg["direct"]["observation_frame_size"] != 56:
        raise ValueError("Velocity expansion expects the 56-feature parent")
    with np.load(args.reference / "demonstrations.npz") as data:
        obs = data["obs"]
    if obs.shape[1] != 256:
        raise ValueError("Reference data must have all 64 sensor features")
    norm, actor, critic = model.load_params(args.params)
    rows = np.concatenate([np.arange(h * 64, h * 64 + 56) for h in range(4)])
    empirical_mean = obs.mean(axis=0)
    empirical_std = np.maximum(obs.std(axis=0), 1e-4)
    count = float(norm.count.hi) * 2**32 + float(norm.count.lo)
    newnorm = norm.replace(
        mean=jp.array(empirical_mean).at[rows].set(norm.mean),
        std=jp.array(empirical_std).at[rows].set(norm.std),
        summed_variance=jp.array(empirical_std**2 * count)
        .at[rows]
        .set(norm.summed_variance),
    )

    def expand(weights):
        new = jax.tree.map(lambda a: a, weights)
        kernel = weights["params"]["hidden_0"]["kernel"]
        new["params"]["hidden_0"]["kernel"] = (
            jp.zeros((256, kernel.shape[1])).at[rows].set(kernel)
        )
        return new

    newactor, newcritic = expand(actor), expand(critic)
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

    before = net(224).policy_network.apply(norm, actor, jp.array(obs[:128, rows]))
    after = net(256).policy_network.apply(newnorm, newactor, jp.array(obs[:128]))
    np.testing.assert_allclose(before, after, atol=1e-5)
    cfg["direct"]["observation_frame_size"] = 64
    for key in ("position_velocity_scale", "position_velocity_noise"):
        cfg["direct"][key] = configs.get_wheel_align_direct_config().direct[key]
    cfg["direct"]["reference_path"] = str(args.reference / "poses.npz")
    cfg["direct"]["supervision_path"] = str(args.reference / "demonstrations.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    model.save_params(args.output / "expanded_params", (newnorm, newactor, newcritic))
    print(
        "PASS: 224-to-256 encoder-input expansion preserves actor outputs", flush=True
    )


if __name__ == "__main__":
    main()
