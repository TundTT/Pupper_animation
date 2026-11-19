"""Supervise calibrated position goals during verified unloaded rotation.

This changes training labels, not runtime actuation. All stance/lift targets and
all wheel labels outside the measured ROTATE phase are preserved.
"""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from workspace import configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    with np.load(args.reference / "demonstrations.npz") as data:
        arrays = {key: data[key] for key in data.files}
    group = arrays["group"]
    command = group // 5
    active = np.array([-1, 1, 0, 2, 3])[command]
    selected = (command > 0) & (group % 5 == 2)
    samples = np.flatnonzero(selected)
    rows = np.array(configs.WHEEL_ACTUATOR_ROWS)[active[selected]]
    before = arrays["actions"].copy()
    arrays["actions"][samples, rows] = 0.0
    changed = np.zeros(before.shape, dtype=bool)
    changed[samples, rows] = True
    np.testing.assert_array_equal(before[~changed], arrays["actions"][~changed])
    assert np.all(arrays["actions"][samples, rows] == 0.0)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "demonstrations.npz", **arrays)
    shutil.copy2(args.reference / "poses.npz", args.output / "poses.npz")
    manifest = json.loads((args.reference / "manifest.json").read_text())
    manifest.update(
        wheel_goal_labels=True,
        source_reference=str(args.reference),
        changed_rotation_labels=len(samples),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"PASS: {len(samples)} active unloaded wheel labels set to calibrated local goal; all other labels unchanged",
        flush=True,
    )


if __name__ == "__main__":
    main()
