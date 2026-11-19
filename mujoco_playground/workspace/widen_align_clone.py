"""Function-preserving widening of ordinary deployed ELU MLP layers."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from jax import numpy as jp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    cfg = json.loads((args.params.parent / "config.json").read_text())
    layers = tuple(cfg["policy"]["hidden_layer_sizes"])
    obs_size = cfg["direct"]["observation_frame_size"] * cfg["observation_history"]
    norm, actor, critic = model.load_params(args.params)
    rng = np.random.default_rng(527)

    def widen(weights):
        result = jax.tree.map(lambda x: x, weights)
        for i in range(len(layers) + 1):
            original = weights["params"][f"hidden_{i}"]
            kernel = np.asarray(original["kernel"])
            row = (
                np.arange(kernel.shape[0])
                if i == 0
                else np.tile(np.arange(kernel.shape[0]), 2)
            )
            col = (
                np.arange(kernel.shape[1])
                if i == len(layers)
                else np.tile(np.arange(kernel.shape[1]), 2)
            )
            enlarged = kernel[row[:, None], col[None, :]] / (1 if i == 0 else 2)
            if i:
                # Equal-and-opposite outgoing perturbations cancel for the
                # duplicated activations, but break optimization symmetry.
                perturb = rng.normal(0.0, 1e-3, (kernel.shape[0], len(col)))
                enlarged += np.concatenate((perturb, -perturb), axis=0)
            result["params"][f"hidden_{i}"] = {
                "kernel": jp.array(enlarged, dtype=original["kernel"].dtype),
                "bias": original["bias"][col],
            }
        return result

    newactor, newcritic = widen(actor), widen(critic)

    def network(widths):
        return networks.make_ppo_networks(
            obs_size,
            12,
            policy_hidden_layer_sizes=widths,
            value_hidden_layer_sizes=widths,
            activation=jax.nn.elu,
            preprocess_observations_fn=rs.normalize,
        )

    features = norm.mean + norm.std * jax.random.normal(
        jax.random.PRNGKey(42), (128, obs_size)
    )
    oldnet, newnet = network(layers), network(tuple(2 * n for n in layers))
    before = oldnet.policy_network.apply(norm, actor, features)
    after = newnet.policy_network.apply(norm, newactor, features)
    np.testing.assert_allclose(before, after, atol=2e-5)
    cfg["policy"]["hidden_layer_sizes"] = [2 * n for n in layers]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    model.save_params(args.output / "expanded_params", (norm, newactor, newcritic))
    (args.output / "expansion.json").write_text(
        json.dumps(
            dict(
                parent=str(args.params),
                max_logit_error=float(jp.max(jp.abs(before - after))),
                hidden_layers=cfg["policy"]["hidden_layer_sizes"],
            ),
            indent=2,
        )
    )
    print(
        "PASS: widening preserves actor outputs and observation statistics; twelve-output MLP contract retained",
        flush=True,
    )


if __name__ == "__main__":
    main()
