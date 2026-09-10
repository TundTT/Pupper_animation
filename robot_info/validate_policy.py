#!/usr/bin/env python3
"""Validate an exported RTNeural policy against the robot contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


VECTOR_SIZE = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="RTNeural policy JSON")
    parser.add_argument(
        "--profile",
        help="Hardware profile; defaults to model hardware_profile metadata",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("robot_contract.json"),
        help="Robot contract JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat compatibility warnings as failures",
    )
    return parser.parse_args()


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{path} is a Git LFS pointer, not a policy JSON")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, raw


def last_dimension(value: Any) -> int | None:
    if not isinstance(value, list) or not value:
        return None
    dimension = value[-1]
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        return None
    return dimension


def numeric_vector(
    model: dict[str, Any],
    key: str,
    errors: list[str],
    warnings: list[str],
    *,
    required: bool = True,
    allow_scalar: bool = False,
) -> list[float] | None:
    if key not in model:
        (errors if required else warnings).append(f"missing `{key}` metadata")
        return None
    value = model[key]
    if allow_scalar and isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            errors.append(f"`{key}` scalar must be finite")
            return None
        warnings.append(f"`{key}` uses a legacy scalar; export 12 explicit values")
        return [float(value)] * VECTOR_SIZE
    if not isinstance(value, list) or len(value) != VECTOR_SIZE:
        errors.append(f"`{key}` must contain exactly {VECTOR_SIZE} values")
        return None
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        errors.append(f"`{key}` values must be finite numbers")
        return None
    return [float(item) for item in value]


def check_shape_and_layers(
    model: dict[str, Any],
    contract: dict[str, Any],
    behavior: str,
    errors: list[str],
    warnings: list[str],
) -> tuple[int | None, int | None]:
    behavior_contract = contract["policy"]["behaviors"][behavior]
    command_states = model.get("command_states")
    if behavior == "leg_lift":
        if (
            not isinstance(command_states, list)
            or not command_states
            or any(not isinstance(item, str) or not item for item in command_states)
        ):
            errors.append("leg-lift policy needs non-empty string `command_states`")
            frame_size = None
        else:
            if len(set(command_states)) != len(command_states):
                errors.append("`command_states` values must be unique")
            frame_size = behavior_contract["single_frame_fixed_size"] + len(command_states)
    else:
        frame_size = behavior_contract["single_frame_size"]

    history = model.get("observation_history")
    if isinstance(history, bool) or not isinstance(history, int) or history < 1:
        errors.append("`observation_history` must be a positive integer")
        history = None

    input_size = last_dimension(model.get("in_shape"))
    if input_size is None:
        errors.append("`in_shape` must end with an integer input width")
    elif frame_size is not None and history is not None:
        expected = frame_size * history
        if input_size != expected:
            errors.append(
                f"input width {input_size} does not equal frame {frame_size} * history {history}"
            )

    layers = model.get("layers")
    output_size = None
    if not isinstance(layers, list) or not layers:
        errors.append("`layers` must be a non-empty list")
        return input_size, output_size

    allowed = set(contract["policy"]["allowed_activations"])
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            errors.append(f"layer {index} must be an object")
            continue
        activation = layer.get("activation")
        if activation not in allowed:
            errors.append(f"layer {index} uses unsupported activation `{activation}`")

    final_layer = layers[-1]
    if isinstance(final_layer, dict):
        output_size = last_dimension(final_layer.get("shape"))
        if output_size != contract["policy"]["output_size"]:
            errors.append(
                f"policy output width must be {contract['policy']['output_size']}, got {output_size}"
            )
        expected_activation = contract["policy"]["recommended_output_activation"]
        if final_layer.get("activation") != expected_activation:
            errors.append(f"final activation must be `{expected_activation}`")

    if "observation_layout" not in model:
        warnings.append("missing `observation_layout` metadata")
    elif model["observation_layout"] != behavior_contract["observation_layout"]:
        warnings.append("`observation_layout` text differs from the canonical contract layout")
    return input_size, output_size


def check_identity(
    model: dict[str, Any],
    contract: dict[str, Any],
    profile: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> list[str]:
    expected_names = contract["canonical_joint_order"]
    joint_names = model.get("joint_names")
    if joint_names is None:
        warnings.append("missing `joint_names`; compatibility relies on external YAML")
    elif joint_names != expected_names:
        errors.append("`joint_names` does not match canonical joint order")

    expected_types = profile["action_types"]
    action_types = model.get("action_types")
    if action_types is None:
        warnings.append("missing `action_types`; compatibility relies on external YAML")
        return expected_types
    if action_types != expected_types:
        errors.append("`action_types` does not match the selected hardware profile")
        return expected_types
    return action_types


def check_action_envelope(
    model: dict[str, Any],
    profile: dict[str, Any],
    action_types: list[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    action_scale = numeric_vector(
        model, "action_scale", errors, warnings, allow_scalar=True
    )
    defaults = numeric_vector(model, "default_joint_pos", errors, warnings)
    lower = numeric_vector(model, "joint_lower_limits", errors, warnings)
    upper = numeric_vector(model, "joint_upper_limits", errors, warnings)
    if any(value is None for value in (action_scale, defaults, lower, upper)):
        return

    assert action_scale is not None
    assert defaults is not None
    assert lower is not None
    assert upper is not None
    for index, name in enumerate(action_types):
        if lower[index] > upper[index]:
            errors.append(f"joint {index} policy lower limit exceeds upper limit")
            continue
        if name == "position":
            if not lower[index] <= defaults[index] <= upper[index]:
                errors.append(f"joint {index} default position is outside policy limits")
            reach = abs(action_scale[index])
            target_low = max(defaults[index] - reach, lower[index])
            target_high = min(defaults[index] + reach, upper[index])
            hardware_low = profile["hardware_position_min"][index]
            hardware_high = profile["hardware_position_max"][index]
            if target_low < hardware_low or target_high > hardware_high:
                errors.append(
                    f"joint {index} target envelope [{target_low:g}, {target_high:g}] "
                    f"exceeds hardware [{hardware_low:g}, {hardware_high:g}]"
                )
        elif name == "velocity":
            velocity_max = profile["hardware_velocity_max"][index]
            if abs(action_scale[index]) > velocity_max:
                errors.append(
                    f"joint {index} velocity scale {abs(action_scale[index]):g} "
                    f"exceeds hardware maximum {velocity_max:g}"
                )
        else:
            errors.append(f"joint {index} has unsupported action type `{name}`")


def check_scalar_gain(
    model: dict[str, Any],
    profile: dict[str, Any],
    key: str,
    maximum_key: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    if key not in model:
        warnings.append(f"missing scalar `{key}` metadata")
        return
    value = model[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        errors.append(f"`{key}` must be a finite scalar for the current controller ABI")
        return
    for index, maximum in enumerate(profile[maximum_key]):
        if float(value) < 0.0 or float(value) > maximum:
            errors.append(
                f"`{key}` joint {index} value {float(value):g} is outside [0, {maximum:g}]"
            )


def check_mixed_gains(
    model: dict[str, Any],
    profile: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    if "kp" in model or "kd" in model:
        errors.append("mixed-gain profile must not use scalar `kp` or `kd` JSON overrides")
    for key, reference_key, maximum_key in (
        ("kps", "reference_policy_kp", "hardware_kp_max"),
        ("kds", "reference_policy_kd", "hardware_kd_max"),
    ):
        values = numeric_vector(model, key, errors, warnings)
        if values is None:
            continue
        if values != profile[reference_key]:
            errors.append(f"`{key}` does not match the selected profile reference gains")
        for index, value in enumerate(values):
            if value < 0.0 or value > profile[maximum_key][index]:
                errors.append(
                    f"`{key}` joint {index} value {value:g} is outside "
                    f"[0, {profile[maximum_key][index]:g}]"
                )


def check_behavior_metadata(
    model: dict[str, Any],
    contract: dict[str, Any],
    behavior: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    low_present = "command_low" in model
    high_present = "command_high" in model
    if low_present != high_present:
        errors.append("`command_low` and `command_high` must be declared together")
    elif low_present:
        low = model["command_low"]
        high = model["command_high"]
        valid = (
            isinstance(low, list)
            and isinstance(high, list)
            and len(low) == 3
            and len(high) == 3
            and all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in low + high
            )
        )
        if not valid:
            errors.append("command bounds must be finite three-element vectors")
        elif any(float(low[i]) > 0.0 or float(high[i]) < 0.0 for i in range(3)):
            errors.append("command bounds must contain zero")
        elif any(float(low[i]) > float(high[i]) for i in range(3)):
            errors.append("command lower bounds must not exceed upper bounds")
    elif behavior in {"locomotion", "wheel"}:
        warnings.append("missing deployed `command_low` and `command_high` metadata")

    if "orientation_command" in model:
        orientation = model["orientation_command"]
        if (
            not isinstance(orientation, list)
            or len(orientation) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in orientation
            )
        ):
            errors.append("`orientation_command` must be a finite three-element vector")
        elif abs(sum(float(value) ** 2 for value in orientation) - 1.0) > 1e-6:
            errors.append("`orientation_command` must be a unit vector")

    if behavior == "wheel":
        wheel_contract = contract["policy"]["behaviors"]["wheel"]
        for key in ("wheel_joint_rows", "wheel_forward_sign"):
            if model.get(key) != wheel_contract[key]:
                errors.append(f"wheel policy `{key}` does not match robot contract")
        normalizer = model.get("wheel_velocity_normalizer")
        if isinstance(normalizer, bool) or not isinstance(normalizer, (int, float)):
            errors.append("wheel policy needs numeric `wheel_velocity_normalizer`")
        elif not math.isfinite(float(normalizer)) or float(normalizer) <= 0.0:
            errors.append("`wheel_velocity_normalizer` must be finite and positive")


def check_provenance(
    model: dict[str, Any],
    contract: dict[str, Any],
    profile_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    declared_profile = model.get("hardware_profile")
    if declared_profile is None:
        warnings.append("missing `hardware_profile` metadata")
    elif declared_profile != profile_name:
        errors.append(
            f"model hardware profile `{declared_profile}` differs from `{profile_name}`"
        )

    declared_schema = model.get("robot_contract_schema_version")
    if declared_schema is None:
        warnings.append("missing `robot_contract_schema_version` metadata")
    elif declared_schema != contract["schema_version"]:
        errors.append(
            f"model contract schema {declared_schema} differs from {contract['schema_version']}"
        )

    for key in ("source_commit", "training_model_sha256", "training_control_dt"):
        if key not in model:
            warnings.append(f"missing `{key}` provenance metadata")

    training_hash = model.get("training_model_sha256")
    if training_hash is not None and (
        not isinstance(training_hash, str)
        or len(training_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in training_hash)
    ):
        errors.append("`training_model_sha256` must be a 64-digit hexadecimal hash")

    training_dt = model.get("training_control_dt")
    if training_dt is not None and (
        isinstance(training_dt, bool)
        or not isinstance(training_dt, (int, float))
        or not math.isfinite(float(training_dt))
        or float(training_dt) <= 0.0
    ):
        errors.append("`training_control_dt` must be finite and positive")


def main() -> int:
    args = parse_args()
    try:
        contract, _ = read_json(args.contract)
        model, model_bytes = read_json(args.model)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    behaviors = contract["policy"]["behaviors"]
    behavior = model.get("behavior", "locomotion")
    if "behavior" not in model:
        warnings.append("missing `behavior`; controller will assume legacy `locomotion`")
    if not isinstance(behavior, str) or behavior not in behaviors:
        errors.append(f"unsupported behavior `{behavior}`")
        behavior = "locomotion"

    profile_name = args.profile or model.get("hardware_profile")
    profiles = contract["profiles"]
    if not profile_name:
        print("FAIL: select `--profile` or embed `hardware_profile`", file=sys.stderr)
        return 2
    if not isinstance(profile_name, str) or profile_name not in profiles:
        print(f"FAIL: unknown hardware profile `{profile_name}`", file=sys.stderr)
        return 2
    profile = profiles[profile_name]
    if behavior not in profile["compatible_behaviors"]:
        errors.append(f"behavior `{behavior}` is incompatible with profile `{profile_name}`")

    input_size, output_size = check_shape_and_layers(
        model, contract, behavior, errors, warnings
    )
    action_types = check_identity(model, contract, profile, errors, warnings)
    check_action_envelope(model, profile, action_types, errors, warnings)
    if "reference_policy_kp" in profile:
        check_mixed_gains(model, profile, errors, warnings)
    else:
        check_scalar_gain(model, profile, "kp", "hardware_kp_max", errors, warnings)
        check_scalar_gain(model, profile, "kd", "hardware_kd_max", errors, warnings)
    check_behavior_metadata(model, contract, behavior, errors, warnings)
    check_provenance(model, contract, profile_name, errors, warnings)

    digest = hashlib.sha256(model_bytes).hexdigest()
    status = "FAIL" if errors or (args.strict and warnings) else "PASS"
    print(f"{status}: {args.model}")
    print(f"  sha256: {digest}")
    print(f"  behavior/profile: {behavior}/{profile_name}")
    print(f"  input/output: {input_size}/{output_size}")
    for warning in warnings:
        print(f"  WARN: {warning}")
    for error in errors:
        print(f"  ERROR: {error}")
    if args.strict and warnings:
        print("  ERROR: --strict treats warnings as failures")
    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
