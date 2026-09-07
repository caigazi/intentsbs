"""Render the saved local-observation state around authoritative label flips."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from run_v2_local_observability_audit import (
    load_samples,
    observation_distance,
    opposite_chirality,
)


def draw(ax, sample, phase: str, observation_delta: float) -> None:
    ax.axhline(0.0, color="0.88", lw=0.7)
    ax.axvline(0.0, color="0.88", lw=0.7)
    ax.scatter([0.0], [0.0], marker="*", s=85, color="black", zorder=5)
    ax.arrow(0.0, 0.0, 0.35 * sample.ego[5], 0.35 * sample.ego[6],
             color="#2ca02c", width=0.008, length_includes_head=True)
    ax.arrow(0.0, 0.0, 0.35 * sample.action[0], 0.35 * sample.action[1],
             color="#d62728", width=0.012, length_includes_head=True)
    for edge in sample.edges:
        x, y = edge[:2]
        ax.scatter([x], [y], marker="o", s=55, color="#1f77b4", zorder=4)
        ax.arrow(x, y, 0.25 * edge[2], 0.25 * edge[3], color="#ff7f0e",
                 width=0.006, length_includes_head=True)
        ax.arrow(x, y, 0.25 * edge[4], 0.25 * edge[5], color="#9467bd",
                 width=0.006, length_includes_head=True)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("goal-forward (normalized)", fontsize=7)
    ax.set_ylabel("goal-side (normalized)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"{phase}: {sample.case}, step {sample.step}, agent {sample.agent}\n"
        f"Teacher (alpha,beta)=({sample.action[0]:.3f},{sample.action[1]:+.3f}); "
        f"pair obs distance={observation_delta:.3f}",
        fontsize=8,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples, _ = load_samples(args.reports)
    tracks = {}
    for sample in samples:
        tracks.setdefault((sample.case, sample.agent), []).append(sample)
    flips = []
    for track in tracks.values():
        track.sort(key=lambda item: item.step)
        for left, right in zip(track, track[1:]):
            if opposite_chirality(left, right):
                flips.append((left, right, observation_distance(left, right)))

    fig, axes = plt.subplots(len(flips), 2, figsize=(10, 3.0 * len(flips)),
                             constrained_layout=True)
    if len(flips) == 1:
        axes = [axes]
    for row, (left, right, distance) in enumerate(flips):
        draw(axes[row][0], left, "before", distance)
        draw(axes[row][1], right, "after", distance)
    fig.legend(handles=[
        Line2D([0], [0], marker="*", color="black", lw=0, label="ego"),
        Line2D([0], [0], marker="o", color="#1f77b4", lw=0,
               label="sensed neighbor"),
        Line2D([0], [0], color="#2ca02c", label="own previous Intent"),
        Line2D([0], [0], color="#d62728", label="new Teacher label"),
        Line2D([0], [0], color="#ff7f0e", label="neighbor relative velocity"),
        Line2D([0], [0], color="#9467bd", label="neighbor previous Intent"),
    ], loc="upper center", ncol=3, fontsize=8)
    fig.suptitle(
        "Authoritative-label chirality flips in the legal Student local frame",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    print(args.output)


if __name__ == "__main__":
    main()
