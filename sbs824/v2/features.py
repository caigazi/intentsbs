"""Deployable local GNN observations with exactly a 2-float neighbor message."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import Config
from ..spatial import radius_neighbor_lists
from .protocol import IntentMode, SyncEventProtocol


@dataclass(frozen=True)
class FeatureBatch:
    self_features: np.ndarray
    edge_features: np.ndarray
    edge_mask: np.ndarray
    neighbor_ids: np.ndarray


def build_local_features(state: np.ndarray, goals: np.ndarray,
                         base_velocity: np.ndarray,
                         world_messages: np.ndarray,
                         own_mode: np.ndarray,
                         own_evidence: np.ndarray,
                         cfg: Config,
                         protocol: SyncEventProtocol) -> FeatureBatch:
    """Build observations without neighbor mode/evidence/goal information."""
    protocol.validate(cfg)
    state = np.asarray(state, dtype=float)
    goals = np.asarray(goals, dtype=float)
    base_velocity = np.asarray(base_velocity, dtype=float)
    messages = np.asarray(world_messages, dtype=float)
    mode = np.asarray(own_mode)
    evidence = np.asarray(own_evidence, dtype=bool)
    n = cfg.n_agents
    if state.shape != (n, 4) or goals.shape != (n, 2):
        raise ValueError("state/goals shapes do not match cfg.n_agents")
    if base_velocity.shape != (n, 2) or messages.shape != (n, 2):
        raise ValueError("velocity/message shapes do not match cfg.n_agents")
    if mode.shape != (n,) or evidence.shape != (n,):
        raise ValueError("own mode/evidence shapes do not match cfg.n_agents")

    goal_delta = goals - state[:, :2]
    goal_distance = np.linalg.norm(goal_delta, axis=1)
    forward = goal_delta / np.maximum(goal_distance[:, None], 1e-9)
    side = np.column_stack((-forward[:, 1], forward[:, 0]))
    neighbors = radius_neighbor_lists(state[:, :2], protocol.sensing_radius)
    slots = max(1, max((len(ids) for ids in neighbors), default=0))
    self_features = np.zeros((n, 9), dtype=np.float32)
    edge_features = np.zeros((n, slots, 9), dtype=np.float32)
    edge_mask = np.zeros((n, slots), dtype=bool)
    neighbor_ids = np.full((n, slots), -1, dtype=np.int32)

    for ego, local_ids in enumerate(neighbors):
        def project(vector):
            return np.array([forward[ego] @ vector, side[ego] @ vector])

        self_features[ego] = np.concatenate((
            [goal_distance[ego] / cfg.area_size],
            project(state[ego, 2:]) / cfg.max_speed,
            project(base_velocity[ego]) / cfg.max_speed,
            project(messages[ego]) / cfg.max_speed,
            [float(mode[ego] == IntentMode.ACTIVE),
             float(mode[ego] == IntentMode.RELEASE)],
        ))
        ordered = sorted(local_ids, key=lambda j: float(
            np.linalg.norm(state[int(j), :2] - state[ego, :2])))
        for slot, raw_neighbor in enumerate(ordered):
            neighbor = int(raw_neighbor)
            rel_p = state[neighbor, :2] - state[ego, :2]
            rel_v = state[neighbor, 2:] - state[ego, 2:]
            distance = float(np.linalg.norm(rel_p))
            speed2 = float(rel_v @ rel_v)
            closing = float(-rel_p @ rel_v / max(distance, 1e-9))
            ttc = float(np.clip(
                -rel_p @ rel_v / max(speed2, 1e-9),
                0.0, protocol.prediction_horizon))
            edge_features[ego, slot] = np.concatenate((
                project(rel_p) / protocol.sensing_radius,
                project(rel_v) / cfg.max_speed,
                project(messages[neighbor]) / cfg.max_speed,
                [distance / protocol.sensing_radius,
                 closing / cfg.max_speed, ttc],
            ))
            edge_mask[ego, slot] = True
            neighbor_ids[ego, slot] = neighbor
    return FeatureBatch(self_features, edge_features, edge_mask, neighbor_ids)
