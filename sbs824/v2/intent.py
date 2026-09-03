"""Model-independent decoding and tracking of the two-dimensional Intent."""

from __future__ import annotations

import numpy as np

from ..simulation import Config
from .protocol import SyncEventProtocol


IDENTITY_INTENT = np.array([1.0, 0.0])


def decode_world_velocity(state: np.ndarray, goals: np.ndarray,
                          base_velocity: np.ndarray,
                          parameters: np.ndarray,
                          protocol: SyncEventProtocol) -> np.ndarray:
    """Decode local ``(alpha,beta)`` into the communicated world velocity."""
    state = np.asarray(state, dtype=float)
    goals = np.asarray(goals, dtype=float)
    base_velocity = np.asarray(base_velocity, dtype=float)
    params = np.clip(np.asarray(parameters, dtype=float), -1.0, 1.0)
    goal_delta = goals - state[:, :2]
    forward = goal_delta / np.maximum(
        np.linalg.norm(goal_delta, axis=1, keepdims=True), 1e-9)
    side = np.column_stack((-forward[:, 1], forward[:, 0]))
    base_speed = np.linalg.norm(base_velocity, axis=1, keepdims=True)
    return (params[:, :1] * base_speed * forward
            + params[:, 1:] * protocol.lateral_speed * side)


def rate_limit(current: np.ndarray, target: np.ndarray,
               cfg: Config, protocol: SyncEventProtocol) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - current
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    limit = protocol.max_intent_accel * cfg.dt
    updated = current + delta * np.minimum(
        1.0, limit / np.maximum(norm, 1e-12))
    speed = np.linalg.norm(updated, axis=1, keepdims=True)
    return updated * np.minimum(
        1.0, cfg.max_speed / np.maximum(speed, 1e-12))


def tracking_force(state: np.ndarray, gain: np.ndarray, cfg: Config,
                   agent_ids: np.ndarray, world_intent: np.ndarray,
                   protocol: SyncEventProtocol,
                   base_force: np.ndarray) -> np.ndarray:
    """Track a world velocity Intent through the same LQR backend."""
    result = np.asarray(base_force, dtype=float).copy()
    ids = np.asarray(agent_ids, dtype=int)
    if not len(ids):
        return result
    waypoint = state[ids, :2] + protocol.intent_lookahead * world_intent[ids]
    target = np.column_stack((waypoint, world_intent[ids]))
    error = target - state[ids]
    norm = np.linalg.norm(error, axis=1, keepdims=True)
    error_max = np.abs(error / np.maximum(norm, 1e-9) * cfg.sense_radius)
    error = np.clip(error, -error_max, error_max)
    result[ids] = np.clip(error @ gain.T, -cfg.max_force, cfg.max_force)
    return result
