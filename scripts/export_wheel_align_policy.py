#!/usr/bin/env python3
"""Export the supported 3-action wheel-align actor, with Brax parity checks.

Run with combine/mujoco_playground/.venv/bin/python. Reads the checkpoint's
saved config and frozen constants, never today's wheel-drive defaults.
This is an RTNeural network export; the ROS joint controller still needs an
alignment adapter. See WHEEL_ALIGN_DEPLOYMENT.md.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks


# Exact concatenation order in the frozen PupperWheelAlignEnv._get_obs.
OBSERVATION_BLOCKS = [
    ("body_angular_velocity", 3, "rad/s, body frame"),
    ("projected_gravity", 3, "body frame; world gravity [0,0,-1]"),
    ("joint_position_or_wheel_speed", 12,
     "position rows: q-default; wheel rows: qd*forward_sign/wheel_velocity_normalizer"),
    ("last_policy_action", 3, "previous clipped 3-action correction, initially zero"),
    ("command_one_hot", 5, "command_states order"),
    ("target_error_sin", 4, "sin(wrap(target-q_wheel))*tracked"),
    ("target_error_cos", 4, "cos(wrap(target-q_wheel))*tracked"),
    ("phase_one_hot", 20, "leg-major flatten of [4,5], phase_states order"),
    ("tracked", 4, "0 or 1"),
    ("completed", 4, "0 or 1"),
    ("wheel_clearance", 4, "lowest tilted cylinder point above ground, metres / 0.01"),
    ("wheel_normal_load", 4, "sum of wheel contact normal forces, newtons / 10"),
    ("body_displacement_xy", 2, "world XY minus episode-start XY, metres / 0.05"),
    ("body_linear_velocity_xy", 2, "m/s, body frame"),
    ("heading", 2, "[cross_z(initial,current), dot(initial,current)] of unit XY forward"),
    ("wheel_reference_error_sin", 4, "sin(wheel_reference-q_wheel)"),
    ("wheel_reference_error_cos", 4, "cos(wheel_reference-q_wheel)"),
    ("lift_reference", 4, "rad"),
    ("stance_shift", 2, "metres / 0.02"),
    ("phase_age", 4, "min(age_in_control_steps/prepare_steps,1)"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_network(params):
    mean = np.asarray(params[0].mean, dtype=np.float64)
    std = np.asarray(params[0].std, dtype=np.float64)
    if mean.shape != (376,) or std.shape != mean.shape:
        raise ValueError("Expected 376-element running observation statistics")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid checkpoint normalization statistics")
    actor = params[1]["params"]
    names = ["hidden_0", "hidden_1", "hidden_2"]
    if set(actor) != set(names):
        raise ValueError(f"Unexpected actor layers: {list(actor)}")
    layers = []
    for index, (name, shape) in enumerate(zip(names, [(376, 128), (128, 128), (128, 6)])):
        kernel = np.asarray(actor[name]["kernel"], dtype=np.float64)
        bias = np.asarray(actor[name]["bias"], dtype=np.float64)
        if kernel.shape != shape or bias.shape != (shape[1],):
            raise ValueError(f"Unexpected dimensions for {name}")
        if index == 0:
            # Brax normalize is exactly (obs-mean)/std: no extra epsilon or clip.
            # Keep RTNeural's input-by-output kernel orientation.
            bias = bias - (mean / std) @ kernel
            kernel = kernel / std[:, None]
        if index == 2:
            # Deterministic NormalTanh policy: tanh(location), discard scale head.
            kernel, bias = kernel[:, :3], bias[:3]
        layers.append({
            "type": "dense", "activation": "tanh" if index == 2 else "elu",
            "shape": [None, len(bias)],
            "weights": [kernel.astype(np.float32).tolist(), bias.astype(np.float32).tolist()],
        })
    return {"in_shape": [None, 376], "layers": layers}


def metadata(run, checkpoint, config):
    source = run / "training_source"
    spec = importlib.util.spec_from_file_location("frozen_alignment_configs", source / "configs.py")
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    blocks, offset = [], 0
    for name, size, transform in OBSERVATION_BLOCKS:
        blocks.append(dict(name=name, offset=offset, size=size, transform=transform))
        offset += size
    if offset != 94:
        raise ValueError("Observation contract must contain 94 entries per frame")
    align = config["align"]
    wheel_rows = constants.WHEEL_ACTUATOR_ROWS
    return {
        "behavior": "wheel_align",
        "deployment_status": "network_export_only_requires_alignment_controller",
        "observation_history": 4,
        "single_observation_size": 94,
        "observation_layout": blocks,
        "observation_history_order": "newest_first",
        "observation_history_reset": "first measured frame followed by three zero frames",
        "observation_clip": [-100.0, 100.0],
        "observation_normalization": "checkpoint mean/std folded into layer 0; do not normalize twice",
        "command_states": list(constants.COMMAND_STATES),
        "phase_states": ["idle", "lift", "rotate", "lower", "hold"],
        "leg_order": list(constants.FOOT_ROW_BY_LEG),
        "joint_names": list(constants.JOINT_NAMES),
        "wheel_joint_rows": list(wheel_rows),
        "wheel_forward_sign": constants.WHEEL_FORWARD_SIGN.tolist(),
        "wheel_velocity_normalizer": float(constants.WHEEL_MAX_SPEED),
        "policy_action_size": 3,
        "policy_action_names": align["action_names"],
        "policy_action_range": [-1.0, 1.0],
        "policy_action_scale": [align["clearance_adjust"], align["balance_adjust"], align["balance_adjust"]],
        "policy_action_units": ["metres", "metres", "metres"],
        "action_mapping": "clearance_target + action[0]*clearance_adjust; support_shift + action[1:]*balance_adjust; then stateful feedback controller",
        "ctrl_dt": config["ctrl_dt"],
        "feedback_controller": {
            "required": True,
            "source": "training_source/wheel_align_env.py",
            "config": align,
            "default_joint_pos": constants.DEFAULT_POSE.tolist(),
            "position_joint_rows": list(constants.POSITION_ACTUATOR_ROWS),
            "position_joint_lower_limits": constants.JOINT_LOWER_LIMITS[constants.POSITION_ACTUATOR_ROWS].tolist(),
            "position_joint_upper_limits": constants.JOINT_UPPER_LIMITS[constants.POSITION_ACTUATOR_ROWS].tolist(),
            "hip_lift_sign": constants.HIP_LIFT_SIGN,
            "kps": [align["wheel_position_kp"] if i in wheel_rows else config["position_control_kp"] for i in range(12)],
            "kds": [align["wheel_position_kd"] if i in wheel_rows else config["dof_damping"] for i in range(12)],
            "joint_control": "12 position targets; wheel targets recentered as q + atan2(sin(goal-q),cos(goal-q)) every physics step, no wheel angle limits",
            "support_mapping": "-pinv(wheel_geom_jacobian[:, leg_abduction_hip_dofs]) @ body_xy_axes at model home; see frozen env __init__",
            "calibration": "latch each leg's calibration_home once, target=home+pi; retries preserve target",
            "external_commands": "disable automatic_commands; set_command accepts indices 0..4",
        },
        "training_config": config,
        "provenance": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "config_sha256": sha256(run / "config.json"),
            "source_sha256": {p.name: sha256(p) for p in sorted(source.glob("*.py"))},
            "best_checkpoint": json.loads((run / "best_checkpoint.json").read_text()),
            "robustness_comparison": json.loads((run / "robustness_comparison.json").read_text()),
        },
    }


def verify(params, payload, reference_out):
    """Compare serialized float32 weights with Brax's original inference path."""
    net = networks.make_ppo_networks(
        376, 3, policy_hidden_layer_sizes=(128, 128), activation=jax.nn.elu,
        preprocess_observations_fn=running_statistics.normalize,
    )
    policy = networks.make_inference_fn(net)(params, deterministic=True)
    rng = np.random.default_rng(1901)
    mean, std = np.asarray(params[0].mean), np.asarray(params[0].std)
    observations = np.clip(mean + rng.normal(size=(512, 376)) * std * 2, -100, 100).astype(np.float32)
    # Zero-history startup and every command index, alongside clip-boundary probes.
    for index in range(5):
        observations[index, 94:] = 0
        observations[index, 21:26] = np.eye(5, dtype=np.float32)[index]
    observations[5] = 0
    observations[6:8] = np.array([-100, 100])[:, None]
    expected = np.asarray(policy(jax.numpy.asarray(observations), jax.random.PRNGKey(0))[0])
    actual = observations
    for layer in payload["layers"]:
        kernel, bias = [np.asarray(x, dtype=np.float32) for x in layer["weights"]]
        actual = actual @ kernel + bias
        if layer["activation"] == "elu":
            negative = actual < 0
            actual[negative] = np.expm1(actual[negative])
        else:
            actual = np.tanh(actual)
    error = float(np.max(np.abs(expected - actual)))
    if not np.isfinite(actual).all() or error > 3e-4:
        raise ValueError(f"Serialized network differs from Brax: max absolute error {error}")
    if reference_out:
        np.savetxt(reference_out, np.concatenate([observations, expected], axis=1), delimiter=",", fmt="%.9g")
    print(f"PASS: {len(observations)} Brax/JSON inference cases, max absolute action error {error:.9g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-out", type=Path, help="optional CSV for independent RTNeural parity check")
    args = parser.parse_args()
    run = args.params.parent
    config = json.loads((run / "config.json").read_text())
    if (config["observation_history"] != 4 or config["policy"]["activation"] != "elu"
            or config["policy"]["hidden_layer_sizes"] != [128, 128]
            or not config["ppo"]["normalize_observations"]
            or config["align"]["action_names"] != ["clearance_trim", "body_shift_x", "body_shift_y"]):
        raise ValueError("Unsupported wheel-align contract; review the exporter for this run")
    params = model.load_params(str(args.params))
    payload = {**export_network(params), **metadata(run, args.params, config)}
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    verify(params, json.loads(serialized), args.reference_out)
    args.out.write_text(serialized)
    print(f"Wrote {args.out}: 376 -> 128 ELU -> 128 ELU -> 3 tanh")


if __name__ == "__main__":
    main()
