"""Export a trained policy to the robot's neural_controller JSON.

Adapts pupperv3-mjx `export.convert_params`: it folds observation normalization
into the first dense layer and drops the value head, producing the same dense-MLP
JSON the `neural_controller` ros2_control plugin loads via `model_path`.

Two behaviors:
  --behavior wheel     (default on this branch) the wheeled velocity-command policy
  --behavior leg_lift  the legacy leg-lift policy (kept for reference; its env
                       lives on `master`)

Wheeled export notes
--------------------
* WHEEL_FORWARD_SIGN is FOLDED INTO the exported per-joint action_scale. The
  controller computes `action * action_scale` for velocity-type joints with no
  further sign handling, so a negative scale on the right-hand wheels reproduces
  the mirrored-axis correction with ZERO C++ change. Do not also apply the sign
  on the robot or it will cancel out.
* action_types must be "velocity" on the wheel joints and "position" on the rest;
  the controller already supports this per-joint (see its parameters yaml).
* kp MUST be 0 on the wheel joints. The real control board computes
  torque = (pos_cmd - pos_state)*kp + (vel_cmd - vel_state)*kd, so kp=0 turns it
  into torque = kd*(vel_cmd - vel_state) -- algebraically identical to MuJoCo's
  <velocity kv> actuator, with kd playing the role of kv (configs wheel_kv).
  A nonzero kp would fight the wheel against a stale position target.
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


def _wheel_payload(config) -> dict:
    """Deployment metadata for the wheeled velocity-command policy."""
    # Per-joint action scale with the mirrored-wheel sign folded in (see module
    # docstring). Position rows are untouched.
    action_scale = np.array(config.action_scale, dtype=float).copy()
    action_scale[configs.WHEEL_ACTUATOR_ROWS] *= configs.WHEEL_FORWARD_SIGN

    action_types = ["position"] * 12
    for r in configs.WHEEL_ACTUATOR_ROWS:
        action_types[r] = "velocity"

    # kp=0 on the wheels turns the control board's PD law into a pure velocity
    # servo; kd there is the velocity gain (MuJoCo's kv).
    kps = [float(config.position_control_kp)] * 12
    kds = [float(config.dof_damping)] * 12
    for r in configs.WHEEL_ACTUATOR_ROWS:
        kps[r] = 0.0
        kds[r] = float(config.wheel_kv)

    return {
        "behavior": "wheel",
        "action_scale": action_scale.tolist(),
        "action_types": action_types,
        # NOTE: the controller does NOT read these. It reads scalar "kp"/"kd" only
        # (set_param_from_json_scalar, broadcast to all 12 joints), so per-joint
        # gains have to live in config.yaml. They are emitted here as the values to
        # paste there, and this export deliberately omits scalar "kp"/"kd" -- if it
        # emitted them they would overwrite config.yaml's per-joint arrays and put
        # a nonzero kp back on the wheels.
        "kps": kps,
        "kds": kds,
        # Init phase: neural_controller drives EVERY joint to default_joint_pos
        # under POSITION control with init_kps before handing over to the policy.
        # On a wheel that means "rotate to angle 0 with kp=7.5" -- the wheels would
        # spin on activation. init_kps must be 0 on the wheel rows so they are only
        # damped, never position-servoed, during init.
        "init_kps": [0.0 if i in set(configs.WHEEL_ACTUATOR_ROWS) else 7.5 for i in range(12)],
        "init_kds": [0.1 if i in set(configs.WHEEL_ACTUATOR_ROWS) else 0.25 for i in range(12)],
        "default_joint_pos": configs.DEFAULT_POSE.tolist(),
        "joint_upper_limits": configs.JOINT_UPPER_LIMITS.tolist(),
        "joint_lower_limits": configs.JOINT_LOWER_LIMITS.tolist(),
        "observation_history": config.observation_history,
        "observation_layout": [
            "ang_vel[3]", "gravity[3]", "cmd_xyyaw_vel[3]",
            "joint_pos_minus_default_OR_wheel_vel_normalized[12]", "last_action[12]",
        ],
        # The wheel rows of the joint block are NOT joint angles: they carry
        # (wheel joint velocity / wheel_velocity_normalizer), sign-corrected the
        # same way the action is. A free-spinning wheel's angle is unbounded and
        # wraps, so feeding it would be meaningless. The stock controller leaves
        # those slots at 0 for velocity-type joints -- it needs a small patch to
        # populate them, see README's deployment section.
        "wheel_joint_rows": list(configs.WHEEL_ACTUATOR_ROWS),
        "wheel_velocity_normalizer": float(configs.WHEEL_MAX_SPEED),
        "wheel_forward_sign": configs.WHEEL_FORWARD_SIGN.tolist(),
        "command_ranges": {
            "lin_vel_x": list(config.lin_vel_x_range),
            "lin_vel_y": list(config.lin_vel_y_range),
            "ang_vel_yaw": list(config.ang_vel_yaw_range),
        },
    }


def _leg_lift_payload(config) -> dict:
    """Legacy leg-lift metadata (that env lives on `master`)."""
    return {
        "behavior": "leg_lift",
        "action_scale": config.action_scale,
        "kp": config.position_control_kp,
        "kd": config.dof_damping,
        "default_joint_pos": configs.DEFAULT_POSE.tolist(),
        "joint_upper_limits": configs.JOINT_UPPER_LIMITS.tolist(),
        "joint_lower_limits": configs.JOINT_LOWER_LIMITS.tolist(),
        "observation_history": config.observation_history,
        "observation_layout": ["ang_vel[3]", "gravity[3]", "command_one_hot[5]", "joint_pos_minus_default[12]", "last_action[12]"],
        "command_states": configs.COMMAND_STATES,
        "button_sequence": ["stand", "front_l", "front_r", "back_r", "back_l"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export a policy to neural_controller JSON.")
    p.add_argument("--params", required=True, help="path to brax mjx_params from train.py")
    p.add_argument("--out", default=None, help="output .json path")
    p.add_argument("--behavior", default="wheel", choices=["wheel", "leg_lift"])
    args = p.parse_args()

    if args.behavior == "wheel":
        config = configs.get_wheel_config()
        payload = _wheel_payload(config)
        default_name = "policy_wheel.json"
    else:
        config = configs.get_config()
        payload = _leg_lift_payload(config)
        default_name = "policy_leg_lift.json"

    params = model.load_params(args.params)
    net = convert_params(params, activation=config.policy.activation)

    expected_in = config.observation_history * 33 if args.behavior == "wheel" else None
    if expected_in is not None and net["in_shape"][1] != expected_in:
        raise ValueError(
            f"exported in_shape {net['in_shape'][1]} != observation_history "
            f"({config.observation_history}) * single_obs (33) = {expected_in}. "
            "The params were probably trained with a different observation_history "
            "than configs.get_wheel_config() currently declares."
        )

    final = {**net, **payload}
    out = args.out or os.path.join(os.path.dirname(args.params), default_name)
    with open(out, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Wrote {out}  (behavior={args.behavior}, in_shape={net['in_shape']})")


if __name__ == "__main__":
    main()
