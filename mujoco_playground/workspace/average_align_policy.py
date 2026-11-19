"""Offline checkpoint averaging; inference remains a single unchanged dense actor."""

import argparse
import json
from pathlib import Path
import jax
import numpy as np
from brax.io import model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("params", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    loaded = [model.load_params(x) for x in a.params]
    cfgs = [json.loads((x.parent / "config.json").read_text()) for x in a.params]
    cfg = cfgs[0]
    for c in cfgs[1:]:
        for key in ("action_scale", "position_control_kp", "wheel_kv"):
            if c[key] != cfg[key]:
                raise ValueError(f"Action contract differs: {key}")
        for key in ("observation_frame_size", "history_taps"):
            if c["direct"][key] != cfg["direct"][key]:
                raise ValueError(f"Sensor contract differs: {key}")
    norm, actor, critic = loaded[0]
    for other in loaded[1:]:
        for x, y in zip(jax.tree.leaves(norm), jax.tree.leaves(other[0]), strict=True):
            np.testing.assert_array_equal(x, y)
        if jax.tree.structure(actor) != jax.tree.structure(other[1]):
            raise ValueError("Actor structures differ")
    actor = jax.tree.map(
        lambda *x: np.mean(np.stack(x), axis=0), *[x[1] for x in loaded]
    )
    cfg["direct"].update(
        encoder_hold=True,
        reference_reset_probability=0.0,
        reference_reset_path="",
        control_convention="default_plus_scaled_position_with_encoder_hold",
    )
    a.output.mkdir(parents=True, exist_ok=True)
    model.save_params(a.output / "averaged_params", (norm, actor, critic))
    (a.output / "config.json").write_text(json.dumps(cfg, indent=2))
    (a.output / "averaging.json").write_text(
        json.dumps(
            {
                "sources": [str(x) for x in a.params],
                "normalizer_bit_identical": True,
                "weights": "uniform",
                "runtime": "single original dense actor",
            },
            indent=2,
        )
    )
    print(a.output / "averaged_params")


if __name__ == "__main__":
    main()
