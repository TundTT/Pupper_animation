"""Paired stochastic audits of training-only exploration, holding actor means fixed."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from ml_collections import ConfigDict
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import domain_randomize_wheel_align
from workspace.wheel_align_evaluate import make_evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = ConfigDict(json.loads((args.params.parent / "config.json").read_text()))
    env = PupperWheelAlignDirectEnv(cfg)
    layers = tuple(cfg.policy.hidden_layer_sizes)
    net = networks.make_ppo_networks(
        env.observation_size,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    make_policy = networks.make_inference_fn(net)

    def stochastic(params, deterministic):
        del deterministic
        return make_policy(params, deterministic=False)

    evaluate = make_evaluator(
        cfg,
        stochastic,
        domain_randomize_wheel_align,
        count=64,
        seed=901,
        env_type=PupperWheelAlignDirectEnv,
    )
    original = model.load_params(args.params)
    results = []
    for name, std in [("original", None), ("precision", (0.015, 0.0075, 0.005))]:
        params = jax.tree.map(lambda a: a, original)
        if std is not None:
            head = params[1]["params"][f"hidden_{len(layers)}"]
            head["kernel"] = head["kernel"].at[:, 12:].set(0.0)
            head["bias"] = (
                head["bias"]
                .at[12:]
                .set(jp.log(jp.expm1(jp.tile(jp.array(std), 4) - 0.001)))
            )
            # This changes only discarded Gaussian scale outputs. Every deployed
            # mean-layer weight, normalizer statistic and native target is identical.
            a, b = original[1]["params"], params[1]["params"]
            for key in a:
                end = 12 if key == f"hidden_{len(layers)}" else None
                np.testing.assert_array_equal(
                    a[key]["kernel"][..., :end], b[key]["kernel"][..., :end]
                )
                np.testing.assert_array_equal(
                    a[key]["bias"][:end], b[key]["bias"][:end]
                )
        result = {
            "variant": name,
            "stochastic_actions": True,
            "std_per_leg": std,
            **evaluate(params),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
