#!/usr/bin/env python3
"""
One subplot per CL method, one coloured line per environment/task.

* Directory layout, --metric switch, baseline normalisation and every other
  CLI flag are identical to plot_avg.py.
* Colours are auto-generated; the first task is blue, the next green … (husl).

Additional features:
1. X-axis ticks plotted on every subplot.
2. Vertical dividing lines between tasks.
3. Top x-axis labels reading "Task 1", "Task 2", …, colored to match lines for each method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# plotting defaults ---------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.grid"] = False

CRIT = {0.9: 1.833, 0.95: 1.96, 0.99: 2.576}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--algo', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--arch', nargs='+', default=['easy_levels', 'medium_levels', 'hard_levels'])
    p.add_argument('--strategy', required=True)
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--steps_per_task', type=float, default=1e7)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--confidence', type=float, default=0.95, choices=[0.9, 0.95, 0.99])
    p.add_argument('--plot_name', default=None)
    return p.parse_args()


def load_series(fp: Path) -> np.ndarray:
    if fp.suffix == '.json':
        return np.array(json.loads(fp.read_text()), dtype=float)
    if fp.suffix == '.npz':
        return np.load(fp)['data'].astype(float)
    raise ValueError(f'Unsupported suffix: {fp.suffix}')


def collect_env_curves(base: Path, algo: str, method: str, strat: str,
                       seq_len: int, seeds: List[int], arch: str):
    folder = base / algo / method / f"{strat}_{seq_len}" / "TI" / arch
    env_names, per_env_seed = [], []

    # discover envs
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists():
            continue
        # files = sorted(f for f in sd.glob(f"*_reward.*") if "training" not in f.name)
        files = sorted(
            (f for f in sd.glob("*_reward.*") if "training" not in f.name),
            key=lambda p: int(re.match(r"(\d+)_", p.name).group(1))
        )
        if not files:
            continue
        suffix = "_reward"
        env_names = [f.name.split('_', 1)[1].rsplit(suffix, 1)[0] for f in files]
        per_env_seed = [[] for _ in env_names]
    if not env_names: raise RuntimeError(f'No data for {method}')

    # gather
    for seed in seeds:
        sd = folder / f"seed_{seed}"
        if not sd.exists(): continue
        for idx, env in enumerate(env_names):
            fp = list(sd.glob(f"{idx}_*_reward.json"))[0]
            arr = load_series(fp)
            per_env_seed[idx].append(arr)

    T_max = max(max(map(len, curves)) for curves in per_env_seed if curves)
    curves = []
    for env_curves in per_env_seed:
        if env_curves:
            stacked = np.vstack([np.pad(a, (0, T_max - len(a)), constant_values=np.nan)
                                 for a in env_curves])
        else:
            stacked = np.full((1, T_max), np.nan)
        curves.append(stacked)

    return env_names, curves


def smooth_and_ci(data: np.ndarray, sigma: float, conf: float):
    mean = gaussian_filter1d(np.nanmean(data, axis=0), sigma=sigma)
    sd = gaussian_filter1d(np.nanstd(data, axis=0), sigma=sigma)
    n_eff = np.sum(~np.isnan(data), axis=0) 
    ci = CRIT[conf] * sd / np.sqrt(n_eff)
    # ci = CRIT[conf] * sd / np.sqrt(data.shape[0])
    return mean, ci


def plot():
    args = parse_args()
    data_root = Path(__file__).resolve().parent.parent / args.data_root

    total = args.seq_len * args.steps_per_task
    colours = sns.color_palette("hls", args.seq_len)
    boundaries = [i * args.steps_per_task for i in range(args.seq_len + 1)]
    mids = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(args.seq_len)]

    archs = args.arch
    fig_h = 2.5 * len(archs) if len(archs) > 1 else 2.8
    fig, axes = plt.subplots(len(archs), 1, sharex=False, sharey=True, figsize=(12, fig_h))
    if len(archs) == 1: axes = [axes]

    for m_idx, arch in enumerate(archs):
        ax = axes[m_idx]
        envs, curves = collect_env_curves(data_root, args.algo, args.method, args.strategy, args.seq_len, args.seeds, arch)

        ax.set_xticks(boundaries)
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
        for b in boundaries: ax.axvline(b, linestyle='--', linewidth=0.5, color='gray')

        for i, curve in enumerate(curves):
            mean, ci = smooth_and_ci(curve, args.sigma, args.confidence)
            x = np.linspace(0, total, len(mean))
            ax.plot(x, mean, color=colours[i])
            ax.fill_between(x, mean - ci, mean + ci, alpha=0.1, color=colours[i])

        ax.set_xlim(0, total)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Normalized Score")
        ax.set_title(arch, fontsize=11)

        twin = ax.twiny()
        twin.set_xlim(ax.get_xlim())
        twin.set_xticks(mids)
        labels = [f"Task {i + 1}" for i in range(args.seq_len)]
        twin.set_xticklabels(labels, fontsize=10)
        twin.tick_params(axis='x', length=0)
        for idx, label in enumerate(twin.get_xticklabels()):
            label.set_color(colours[idx])

    axes[-1].set_xlabel('Environment Steps')
    plt.tight_layout()
    out = Path(__file__).resolve().parent.parent / 'plots'
    out.mkdir(exist_ok=True)
    name = args.plot_name or f"per_task_norm_reward"
    plt.savefig(out / f"{name}.png")
    plt.savefig(out / f"{name}.pdf")
    plt.show()


if __name__ == '__main__':
    plot()
