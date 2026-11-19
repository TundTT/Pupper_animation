"""Select among supervised snapshots by closed-loop behavior, reusing one JIT."""

import argparse, json, shutil
from pathlib import Path
import jax
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from ml_collections import ConfigDict
from workspace.wheel_align_direct_env import PupperWheelAlignDirectEnv
from workspace.wheel_align_env import domain_randomize_wheel_align
from workspace.wheel_align_evaluate import make_evaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--seed", type=int, default=901)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cfg = ConfigDict(json.loads((args.run / "config.json").read_text()))
    env_type = PupperWheelAlignDirectEnv
    if cfg.direct.get('encoder_gate', False):
        from workspace.wheel_align_gate_env import PupperWheelAlignGateEnv
        env_type = PupperWheelAlignGateEnv
        cfg.direct.reference_reset_probability = 0.
    elif cfg.direct.get('encoder_hold', False):
        from workspace.wheel_align_refine_env import PupperWheelAlignRefineEnv
        env_type = PupperWheelAlignRefineEnv
        cfg.direct.reference_reset_probability = 0.
    env = env_type(cfg)
    layers = tuple(cfg.policy.hidden_layer_sizes)
    net = networks.make_ppo_networks(
        env.observation_size,
        12,
        policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers,
        activation=jax.nn.elu,
        preprocess_observations_fn=rs.normalize,
    )
    evaluate = make_evaluator(
        cfg,
        networks.make_inference_fn(net),
        domain_randomize_wheel_align,
        count=args.count,
        seed=args.seed,
        env_type=env_type,
    )
    paths = [args.run / "cloned_params"] + sorted(
        [
            p
            for p in args.run.glob("cloned_params_*")
            if p.name.rsplit("_", 1)[-1].isdigit()
        ],
        key=lambda p: int(p.name.rsplit("_", 1)[-1]),
    )
    if args.limit:
        paths = paths[-args.limit :]
    best = None
    for path in paths:
        metrics = evaluate(model.load_params(path))
        score = (
            metrics["audit/survival_fraction"] == 1.0,
            metrics["audit/all_four_success_fraction"],
            metrics["audit/mean_completed_legs"],
            -metrics["audit/drift_p95"],
            -metrics["audit/held_error_p95"],
        )
        row = {"checkpoint": str(path), **metrics}
        print(json.dumps(row), flush=True)
        with (args.run / "snapshot_audits.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        if best is None or score > best:
            best = score
            shutil.copy2(path, args.run / "cloned_params_audit_best")
            (args.run / "audit_best.json").write_text(json.dumps(row, indent=2))
    print("CLOSED-LOOP SNAPSHOT SELECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
