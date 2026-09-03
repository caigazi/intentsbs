"""Strictly local trigger candidates and audit-only shadow predictors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..simulation import Config
from ..spatial import radius_neighbor_lists
from .protocol import SyncEventProtocol
from .sampled_safety import sampled_wang_step


@dataclass(frozen=True)
class TriggerResult:
    evidence: np.ndarray
    edges: tuple[tuple[int, int], ...]
    components: tuple[tuple[int, ...], ...]


def _ego_candidate(ego_position: np.ndarray, ego_reference: np.ndarray,
                   relative_positions: np.ndarray,
                   neighbor_references: np.ndarray,
                   protocol: SyncEventProtocol) -> np.ndarray:
    """Evaluate one ego using only locally measurable/message information."""
    rel_p = np.asarray(relative_positions, dtype=float)
    neighbor_ref = np.asarray(neighbor_references, dtype=float)
    if rel_p.shape != neighbor_ref.shape or rel_p.ndim != 2 or rel_p.shape[1] != 2:
        raise ValueError("local relative positions/references must be (K, 2)")
    _ = np.asarray(ego_position, dtype=float)  # documents the local API
    rel_v = neighbor_ref - np.asarray(ego_reference, dtype=float)
    distance = np.linalg.norm(rel_p, axis=1)
    speed2 = np.sum(rel_v * rel_v, axis=1)
    closing = -np.sum(rel_p * rel_v, axis=1) / np.maximum(distance, 1e-9)
    raw_t_ca = -np.sum(rel_p * rel_v, axis=1) / np.maximum(speed2, 1e-9)
    t_ca = np.clip(raw_t_ca, 0.0, protocol.prediction_horizon)
    closest = np.linalg.norm(rel_p + t_ca[:, None] * rel_v, axis=1)
    sensed = distance <= protocol.sensing_radius + 1e-12
    approaching = closing > protocol.minimum_closing_speed
    risky = ((distance < protocol.trigger_distance)
             | ((raw_t_ca > 0.0)
                & (protocol.prediction_horizon > 0.0)
                & (closest < protocol.trigger_distance)))
    return sensed & approaching & risky


def ttc_candidates(state: np.ndarray, ego_base_velocity: np.ndarray,
                   neighbor_world_messages: np.ndarray | None,
                   cfg: Config, protocol: SyncEventProtocol) -> TriggerResult:
    """Global simulator wrapper composed exclusively of ego-local evaluations."""
    protocol.validate(cfg)
    state = np.asarray(state, dtype=float)
    ego_base = np.asarray(ego_base_velocity, dtype=float)
    messages = (ego_base if neighbor_world_messages is None else
                np.asarray(neighbor_world_messages, dtype=float))
    if (state.shape != (cfg.n_agents, 4)
            or ego_base.shape != (cfg.n_agents, 2)
            or messages.shape != (cfg.n_agents, 2)):
        raise ValueError("state/base/message shapes do not match cfg.n_agents")
    neighbors = radius_neighbor_lists(state[:, :2], protocol.sensing_radius)
    directed: set[tuple[int, int]] = set()
    for ego, ids in enumerate(neighbors):
        if not len(ids):
            continue
        local = _ego_candidate(
            state[ego, :2], ego_base[ego],
            state[ids, :2] - state[ego, :2], messages[ids], protocol)
        directed.update((ego, int(j)) for j in ids[local])
    # The geometry is symmetric in exact arithmetic.  Unioning directed local
    # detections makes the simulator robust to floating-point edge cases while
    # preserving the information available at either endpoint.
    edges = tuple(sorted({tuple(sorted(edge)) for edge in directed}))
    evidence = np.zeros(cfg.n_agents, dtype=bool)
    for left, right in edges:
        evidence[left] = evidence[right] = True
    return TriggerResult(evidence, edges, _components(cfg.n_agents, edges))


def _components(n_agents: int, edges: tuple[tuple[int, int], ...]
                ) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(n_agents)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[tuple[int, ...]] = []
    unseen = {i for i, values in enumerate(adjacency) if values}
    while unseen:
        root = min(unseen)
        stack, found = [root], set()
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(adjacency[node] - found)
        unseen -= found
        result.append(tuple(sorted(found)))
    return tuple(result)


def _integrate_nominal_step(state: np.ndarray, force: np.ndarray,
                            cfg: Config) -> np.ndarray:
    """Integrate one unconstrained tick with constant force."""
    nxt = np.asarray(state, dtype=float).copy()
    velocity = nxt[:, 2:].copy()
    acceleration = np.asarray(force, dtype=float) / cfg.mass
    nxt[:, :2] += (
        velocity * cfg.dt + 0.5 * acceleration * cfg.dt * cfg.dt)
    nxt[:, 2:] = velocity + acceleration * cfg.dt
    nxt[:, 2:] = np.clip(nxt[:, 2:], -cfg.max_speed, cfg.max_speed)
    return nxt


def shadow_progress_ratio(ego_local_id: int, local_state: np.ndarray,
                          reference_velocity: np.ndarray, cfg: Config,
                          horizon_steps: int,
                          safety_substeps: int | None = None) -> float:
    """Audit a local multi-neighbor Base+Wang progress ratio.

    This uses only the ego state, sensed relative states and the communicated
    world velocity references.  It is an audit primitive, not yet a frozen
    online trigger.
    """
    local_state = np.asarray(local_state, dtype=float)
    reference = np.asarray(reference_velocity, dtype=float)
    if local_state.ndim != 2 or local_state.shape[1] != 4:
        raise ValueError("local_state must have shape (K, 4)")
    if reference.shape != (len(local_state), 2):
        raise ValueError("reference_velocity must have shape (K, 2)")
    if not 0 <= ego_local_id < len(local_state):
        raise ValueError("ego_local_id is outside local_state")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")

    direction = reference[ego_local_id]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return 1.0
    direction = direction / norm
    nominal = local_state.copy()
    safe = local_state.copy()
    latch = np.zeros(len(local_state), dtype=bool)
    substeps = cfg.wang_safety_substeps if safety_substeps is None else safety_substeps
    if substeps < 1:
        raise ValueError("safety_substeps must be positive")
    local_cfg = replace(cfg, n_agents=len(local_state), n_obstacles=0)
    nominal_progress = 0.0
    safe_progress = 0.0
    for _ in range(horizon_steps):
        nominal_force = np.clip(
            cfg.mass * (reference - nominal[:, 2:]) / cfg.dt,
            -cfg.max_force, cfg.max_force)
        safe_force = np.clip(
            cfg.mass * (reference - safe[:, 2:]) / cfg.dt,
            -cfg.max_force, cfg.max_force)
        nominal_before = nominal[ego_local_id, :2].copy()
        safe_before = safe[ego_local_id, :2].copy()
        nominal = _integrate_nominal_step(nominal, nominal_force, local_cfg)
        safety = sampled_wang_step(
            safe, safe_force, [], local_cfg, latch,
            integration_substeps=substeps)
        safe = safety.next_state
        nominal_progress += max(
            0.0, float(direction @ (nominal[ego_local_id, :2]
                                    - nominal_before)))
        safe_progress += max(
            0.0, float(direction @ (safe[ego_local_id, :2] - safe_before)))
    return float(np.clip(safe_progress / (nominal_progress + 1e-9), 0.0, 2.0))


def local_shadow_ratios(state: np.ndarray, reference_velocity: np.ndarray,
                        candidates: TriggerResult, cfg: Config,
                        horizon_steps: int,
                        safety_substeps: int | None = None) -> np.ndarray:
    """Compute one multi-neighbor shadow ratio per candidate ego."""
    state = np.asarray(state, dtype=float)
    reference = np.asarray(reference_velocity, dtype=float)
    adjacency = [set() for _ in range(cfg.n_agents)]
    for left, right in candidates.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    ratios = np.ones(cfg.n_agents, dtype=float)
    for ego, neighbors in enumerate(adjacency):
        if not neighbors:
            continue
        ids = np.asarray([ego, *sorted(neighbors)], dtype=int)
        ratios[ego] = shadow_progress_ratio(
            0, state[ids], reference[ids], cfg, horizon_steps,
            safety_substeps=safety_substeps)
    return ratios


def pairwise_shadow_ratios(state: np.ndarray, reference_velocity: np.ndarray,
                           candidates: TriggerResult, cfg: Config,
                           horizon_steps: int,
                           safety_substeps: int | None = None) -> dict[tuple[int, int], tuple[float, float]]:
    """Audit pair-only ratios; multi-neighbor effects are intentionally absent."""
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for left, right in candidates.edges:
        ids = np.asarray([left, right])
        local_state = np.asarray(state)[ids]
        local_ref = np.asarray(reference_velocity)[ids]
        result[(left, right)] = (
            shadow_progress_ratio(0, local_state, local_ref, cfg,
                                  horizon_steps, safety_substeps),
            shadow_progress_ratio(1, local_state, local_ref, cfg,
                                  horizon_steps, safety_substeps),
        )
    return result
