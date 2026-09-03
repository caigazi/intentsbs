"""Local-radius neighbor discovery for distributed simulation.

The controller still receives only measurements inside ``radius``.  The grid
is simulator infrastructure that avoids testing every global robot pair; it is
equivalent to a physical range sensor returning its local detections.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def radius_neighbor_pairs(positions: np.ndarray, radius: float) -> list[tuple[int, int]]:
    """Return every unordered pair at Euclidean distance at most ``radius``."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")

    cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    keys = np.floor(positions / radius).astype(np.int64)
    for agent, key in enumerate(keys):
        cells[(int(key[0]), int(key[1]))].append(agent)

    radius_sq = float(radius * radius)
    pairs: list[tuple[int, int]] = []
    for i, key in enumerate(keys):
        cx, cy = int(key[0]), int(key[1])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cells.get((cx + dx, cy + dy), ()):
                    if j <= i:
                        continue
                    delta = positions[i] - positions[j]
                    if float(delta @ delta) <= radius_sq + 1e-12:
                        pairs.append((i, j))
    pairs.sort()
    return pairs


def radius_neighbor_lists(positions: np.ndarray, radius: float) -> list[np.ndarray]:
    """Return sorted local neighbor ids for every robot."""
    neighbors: list[list[int]] = [[] for _ in range(len(positions))]
    for i, j in radius_neighbor_pairs(positions, radius):
        neighbors[i].append(j)
        neighbors[j].append(i)
    return [np.asarray(sorted(value), dtype=np.int32) for value in neighbors]
