"""Calibrate training Gaussian scales with enforced KL; freeze all deployed means."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from workspace.supervised_align_ppo import (
    guarded_candidate,
    phase_balanced_demonstrations,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--std', type=float, nargs=3, default=(.015,.0075,.005),
                   help='pre-tanh Gaussian std for abduction, hip, wheel output')
    args = p.parse_args()
    cfg = json.loads((args.params.parent / "config.json").read_text())
    norm, actor, critic = model.load_params(args.params)
    original = jax.tree.map(lambda a: a, actor)
    target = jax.tree.map(lambda a: a, actor)
    layers = tuple(cfg["policy"]["hidden_layer_sizes"])
    head_name = f"hidden_{len(layers)}"
    head = target["params"][head_name]
    if min(args.std) <= .001:
        p.error('Std must exceed the distribution minimum .001')
    std = jp.tile(jp.array(args.std), 4)
    head["kernel"] = head["kernel"].at[:, 12:].set(0.0)
    head["bias"] = head["bias"].at[12:].set(jp.log(jp.expm1(std - 0.001)))
    obs, _ = phase_balanced_demonstrations(cfg["direct"]["supervision_path"])
    net = networks.make_ppo_networks(
        obs.shape[-1],
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    distribution = net.parametric_action_distribution

    @jax.jit
    def stage(current):
        before = distribution.create_dist(net.policy_network.apply(norm, current, obs))

        def kl(candidate):
            after = distribution.create_dist(
                net.policy_network.apply(norm, candidate, obs)
            )
            return jp.mean(after.kl_divergence(before))

        changed, _, accepted, measured = guarded_candidate(
            current, target, jp.float32(0.0), jp.float32(0.0), kl, budget=0.02
        )
        remaining = jp.max(jp.abs(changed["params"][head_name]["bias"] - head["bias"]))
        return changed, accepted, measured, remaining

    history = []
    for index in range(256):
        actor, accepted, kl, remaining = stage(actor)
        if not bool(accepted) or not np.isfinite(float(kl)) or float(kl) > 0.02:
            raise RuntimeError("Exploration calibration failed its enforced KL bound")
        history.append(
            {"stage": index, "applied_kl": float(kl), "remaining": float(remaining)}
        )
        if float(remaining) < 1e-6:
            break
    else:
        raise RuntimeError(
            "Exploration calibration failed to converge within256 guarded stages"
        )
    for key in original["params"]:
        end = 12 if key == head_name else None
        for field in ("kernel", "bias"):
            np.testing.assert_array_equal(
                original["params"][key][field][..., :end],
                actor["params"][key][field][..., :end],
            )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    model.save_params(args.output / "cloned_params", (norm, actor, critic))
    result = dict(
        parent=str(args.params),
        stages=history,
        mean_weights_identical=True,
        normalizer_identical=True,
        std_per_leg=list(args.std),
        output_units=('position_offset,position_offset,rotation_enable'
                      if cfg['direct'].get('encoder_gate', False) else 'position_offset'),
        ppo_environment_steps=0,
    )
    (args.output / "calibration.json").write_text(json.dumps(result, indent=2))
    print(
        f"PASS: {len(history)} guarded scale-only stages, max KL {max(x['applied_kl'] for x in history):.6f}; all deployed mean weights and normalization unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
