"""Build sparse real-sensor history from complete recorded control sequences."""

import argparse
import json
import shutil
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--taps", type=int, nargs=4, default=(0, 1, 10, 50))
    args = p.parse_args()
    if tuple(sorted(set(args.taps))) != tuple(args.taps) or args.taps[0] != 0:
        p.error("Four increasing taps must start at zero")
    with np.load(args.reference / "demonstrations.npz") as data:
        arrays = {key: data[key] for key in data.files}
    frame_size = arrays["obs"].shape[1] // 4
    current = arrays["obs"].reshape((-1, 4, frame_size))[:, 0]
    selected = []
    replacement = []
    retained = []
    for episode in np.unique(arrays["episode"]):
        ids = np.flatnonzero(arrays["episode"] == episode)
        # Legacy recorders omitted dead/unreachable-label frames. Only full
        # 1800-tick recordings establish that no time samples are missing.
        if len(ids) != 1800:
            continue
        indices = np.maximum(
            np.arange(len(ids))[:, None] - np.array(args.taps)[None, :], 0
        )
        replacement.append(current[ids[indices]].reshape((-1, frame_size * 4)))
        selected.append(ids)
        retained.append(int(episode))
    if len(retained) < 32:
        raise RuntimeError("Need at least 32 complete sequences for sparse history")
    selected = np.concatenate(selected)
    arrays = {key: value[selected] for key, value in arrays.items()}
    arrays["obs"] = np.concatenate(replacement)
    manifest = json.loads((args.reference / "manifest.json").read_text())
    manifest["successful_episodes"] = sorted(
        set(manifest["successful_episodes"]) & set(retained)
    )
    if len(manifest["successful_episodes"]) < 16:
        raise RuntimeError("Too few complete successful teacher examples retained")
    manifest.update(
        source_reference=str(args.reference),
        history_taps=args.taps,
        retained_episodes=retained,
        samples=len(selected),
        phase_counts={
            str(int(group)): int((arrays["group"] == group).sum())
            for group in np.unique(arrays["group"])
        },
    )
    manifest.pop("initial_rotation_delta_bins", None)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "demonstrations.npz", **arrays)
    shutil.copy2(args.reference / "poses.npz", args.output / "poses.npz")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"PASS: {len(retained)} contiguous sequences, {len(selected)} samples, history taps {args.taps}",
        flush=True,
    )


if __name__ == "__main__":
    main()
