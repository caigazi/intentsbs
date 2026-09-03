"""Small, dependency-free geometry helpers for oriented rectangles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    center: np.ndarray
    size: np.ndarray
    theta: float


def world_to_local(points: np.ndarray, rect: Rectangle) -> np.ndarray:
    delta = np.asarray(points, dtype=float) - rect.center
    c, s = np.cos(rect.theta), np.sin(rect.theta)
    return delta @ np.array([[c, -s], [s, c]], dtype=float)


def local_to_world(points: np.ndarray, rect: Rectangle) -> np.ndarray:
    c, s = np.cos(rect.theta), np.sin(rect.theta)
    return np.asarray(points, dtype=float) @ np.array([[c, s], [-s, c]], dtype=float) + rect.center


def closest_point(point: np.ndarray, rect: Rectangle) -> np.ndarray:
    """Closest point on/in a rectangle; inside points map to themselves."""
    local = world_to_local(np.asarray(point)[None, :], rect)[0]
    half = rect.size / 2.0
    clipped = np.clip(local, -half, half)
    return local_to_world(clipped[None, :], rect)[0]


def signed_distance(point: np.ndarray, rect: Rectangle) -> float:
    """Euclidean signed distance: negative inside, positive outside."""
    q = np.abs(world_to_local(np.asarray(point)[None, :], rect)[0]) - rect.size / 2.0
    outside = np.linalg.norm(np.maximum(q, 0.0))
    inside = min(max(q[0], q[1]), 0.0)
    return float(outside + inside)


def vertices(rect: Rectangle) -> np.ndarray:
    half = rect.size / 2.0
    local = np.array([[half[0], half[1]], [-half[0], half[1]],
                      [-half[0], -half[1]], [half[0], -half[1]]])
    return local_to_world(local, rect)
