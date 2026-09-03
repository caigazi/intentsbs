"""Small no-obstacle scene families used before the obstacle phase."""

from __future__ import annotations

import numpy as np


def mixed_crossing_scene(seed: int, n_agents: int
                         ) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate 2--8 robot variable-role conflict scenes."""
    if not 2 <= n_agents <= 8:
        raise ValueError("mixed crossing scenes support 2--8 robots")
    rng = np.random.default_rng(seed + 101 * n_agents)
    topology = ("radial", "offset_radial", "permuted", "two_stream")[seed % 4]
    center = np.array([2.0, 2.0]) + rng.uniform(-0.10, 0.10, size=2)
    radius = rng.uniform(0.82, 1.04)
    angles = (rng.uniform(-np.pi, np.pi)
              + 2.0 * np.pi * np.arange(n_agents) / n_agents)
    exact = topology == "radial"
    if not exact:
        angles += rng.uniform(-0.08, 0.08, size=n_agents)
    radial_scale = (np.ones(n_agents) if exact else
                    rng.uniform(0.91, 1.09, size=n_agents))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    starts = center + radius * radial_scale[:, None] * directions

    if topology in {"radial", "offset_radial"}:
        goal_angles = angles + np.pi
        if topology == "offset_radial":
            goal_angles += rng.uniform(-0.14, 0.14, size=n_agents)
        goals = center + radius * np.column_stack(
            (np.cos(goal_angles), np.sin(goal_angles)))
    elif topology == "permuted":
        offsets = np.linspace(-0.32, 0.32, n_agents)
        offsets = offsets[rng.permutation(n_agents)]
        goal_angles = angles + np.pi + offsets
        goal_scale = rng.uniform(0.90, 1.10, size=n_agents)
        goals = center + radius * goal_scale[:, None] * np.column_stack(
            (np.cos(goal_angles), np.sin(goal_angles)))
    else:
        stream_side = np.where(np.arange(n_agents) % 2 == 0, 1.0, -1.0)
        goal_angles = angles + np.pi + stream_side * rng.uniform(
            0.16, 0.30, size=n_agents)
        goals = center + radius * np.column_stack(
            (np.cos(goal_angles), np.sin(goal_angles)))

    route = goals - starts
    unit = route / np.maximum(np.linalg.norm(route, axis=1, keepdims=True), 1e-9)
    speed = (np.zeros(n_agents) if exact else
             rng.uniform(0.0, 0.14, size=n_agents))
    return np.column_stack((starts, unit * speed[:, None])), goals, topology


def headon_scene() -> tuple[np.ndarray, np.ndarray, str]:
    state = np.array([
        [1.35, 2.0, 0.0, 0.0],
        [2.65, 2.0, 0.0, 0.0],
    ])
    goals = np.array([[2.65, 2.0], [1.35, 2.0]])
    return state, goals, "headon"


def random_local_scene(seed: int, n_agents: int, area_size: float = 4.0,
                       minimum_separation: float = 0.22
                       ) -> tuple[np.ndarray, np.ndarray, str]:
    """GCBF-style random starts/goals, including many benign interactions."""
    if n_agents < 2:
        raise ValueError("random local scenes require at least two robots")
    rng = np.random.default_rng(seed + 17 * n_agents)

    def sample_points() -> np.ndarray:
        points: list[np.ndarray] = []
        for _ in range(n_agents):
            for _attempt in range(50_000):
                point = rng.uniform(0.15, area_size - 0.15, size=2)
                if all(np.linalg.norm(point - other) > minimum_separation
                       for other in points):
                    points.append(point)
                    break
            else:
                raise RuntimeError("could not sample separated random points")
        return np.asarray(points)

    starts = sample_points()
    goals = sample_points()
    return np.column_stack((starts, np.zeros((n_agents, 2)))), goals, "random"
