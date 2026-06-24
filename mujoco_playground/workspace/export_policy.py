"""Export a trained leg-lift policy to the robot's neural_controller JSON.

Adapts pupperv3-mjx `export.convert_params`: it folds observation normalization
into the first dense layer and drops the value head, producing the same dense-MLP
JSON the `neural_controller` ros2_control plugin loads via `model_path`.

The leg-lift observation differs from the locomotion policy: instead of a velocity
command it carries a 5-way one-hot "which leg is up" command. That layout is written
into the JSON so the on-robot observation builder and the O-button state machine can
be matched to it. See README.md — wiring the command into neural_controller is the
key deployment task.
"""

import argparse
import json
import os

import numpy as np
from brax.io import model
from jax import numpy as jp

from workspace import configs


def _fold_in_normalization(A, b, mean, std):
    A_prime = A / std[:, np.newaxis]
    b_prime = (b - (A.T @ (mean / std)[:, np.newaxis]).T)[0]
    return A_prime, b_prime


def convert_params(params, activation: str, final_activation: str = "tanh") -> dict:
    """Convert brax PPO params -> dense-MLP layer dicts (normalization folded in)."""
    mean, std = params[0].mean, params[0].std
    params_dict = params[1]["params"]
    layers = []
    input_size = None
    for i, (_, layer_params) in enumerate(params_dict.items()):
        is_first = i == 0
        is_final = i == len(params_dict) - 1
        bias = layer_params["bias"]
        kernel = layer_params["kernel"]
        if is_first:
            kernel, bias = _fold_in_normalization(A=kernel, b=bias, mean=mean, std=std)
            input_size = kernel.shape[0]
        if is_final:
            # PPO policy head outputs [mean, std]; keep only the mean half.
            bias, _ = jp.split(bias, 2, axis=-1)
            kernel, _ = jp.split(kernel, 2, axis=-1)
        layers.append({
            "type": "dense",
            "activation": activation if not is_final else final_activation,
            "shape": [None, len(bias)],
            "weights": [kernel.tolist(), bias.tolist()],
        })
    return {"in_shape": [None, input_size], "layers": layers}


def main() -> None:
    p = argparse.ArgumentParser(description="Export leg-lift policy to neural_controller JSON.")
    p.add_argument("--params", required=True, help="path to brax mjx_params from train.py")
    p.add_argument("--out", default=None, help="output .json path")
    args = p.parse_args()

    config = configs.get_config()
    params = model.load_params(args.params)
    net = convert_params(params, activation=config.policy.activation)

    final = {
        **net,
        # --- how to drive the policy on-robot ---
        "action_scale": config.action_scale,
        "kp": config.position_control_kp,
        "kd": config.dof_damping,
        "default_joint_pos": configs.DEFAULT_POSE.tolist(),
        "joint_upper_limits": configs.JOINT_UPPER_LIMITS.tolist(),
        "joint_lower_limits": configs.JOINT_LOWER_LIMITS.tolist(),
        "observation_history": config.observation_history,
        # --- leg-lift specifics (NOT present in the locomotion policy) ---
        "behavior": "leg_lift",
        "observation_layout": ["ang_vel[3]", "gravity[3]", "command_one_hot[5]", "joint_pos_minus_default[12]", "last_action[12]"],
        "command_states": configs.COMMAND_STATES,  # index 0 = stand
        # The O button advances this clockwise sequence of command indices; each
        # press lowers the current leg and raises the next.
        "button_sequence": ["stand", "front_l", "front_r", "back_r", "back_l"],
    }

    out = args.out or os.path.join(os.path.dirname(args.params), "policy_leg_lift.json")
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Wrote {out}")
    print("Deploy: copy into neural_controller/launch/, add a controller instance in "
          "config.yaml (model_path), spawn in launch.py, bind the O button + command "
          "state machine in joy_util_node.")


if __name__ == "__main__":
    main()
