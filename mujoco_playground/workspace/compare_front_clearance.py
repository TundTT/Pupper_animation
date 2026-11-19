"""Plot measured per-foot clearance and contact sliding from matched gait traces."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--command', type=float, nargs=3, default=[.2, 0, 0])
    parser.add_argument('--clearance_mm',type=float,default=4.)
    args = parser.parse_args()
    datasets = [np.load(args.baseline), np.load(args.candidate)]
    fig, axes = plt.subplots(4, 2, figsize=(13, 9), sharex='col')
    feet = ['Front right', 'Front left', 'Back right', 'Back left']
    for data, label, color in zip(datasets, ['Baseline', 'Candidate'], ['#bb5533', '#187ca8']):
        matches = np.flatnonzero(np.all(np.isclose(data['commands'], args.command), axis=1))
        if not len(matches):
            raise ValueError(f'No matching command in {label}')
        index = matches[0]  # identical first reset seed, no seed cherry-picking
        z = data['z'][:, index].reshape(-1, 4)
        contact = data['contact'][:, index].reshape(-1, 4)
        speed = data['tangential_speed'][:, index].reshape(-1, 4)
        time = (np.arange(len(z)) + 1) * .004
        window = (time >= 2) & (time <= 4)
        for foot in range(4):
            axes[foot, 0].plot(time[window], 1000*z[window, foot], color=color, label=label, lw=1)
            axes[foot, 1].plot(time[window], (speed*contact)[window, foot], color=color, label=label, lw=1)
    for i, name in enumerate(feet):
        axes[i, 0].axhline(0, color='black', lw=.6)
        axes[i, 0].axhline(args.clearance_mm, color='gray', lw=.6, linestyle='--')
        axes[i, 0].set_ylabel(f'{name}\nClearance (mm)')
        axes[i, 1].set_ylabel('Contact sliding (m/s)')
        for ax in axes[i]: ax.grid(alpha=.15)
    axes[0, 0].legend(); axes[0, 1].legend()
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    fig.suptitle(f'Command {args.command}; first matched reset seed; 250 Hz measurements')
    fig.tight_layout()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)


if __name__ == '__main__':
    main()
