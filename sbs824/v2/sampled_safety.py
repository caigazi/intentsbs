"""Sampled-data Wang safety execution shared by runtime and Teacher.

The safety controller is evaluated exactly once per 30-ms control tick. The
normal QP force is held over that interval. Eq. (17) is different: maximum
braking is applied only until zero speed and then switches to zero force.
Integration samples audit the continuous trajectory; they are not extra QP
updates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Rectangle
from ..simulation import Config
from ..spatial import radius_neighbor_pairs
from ..wang_safety import (solve_wang_braking_qp, wang_pair_barrier)


@dataclass(frozen=True)
class SampledSafetyStep:
    next_state: np.ndarray
    action: np.ndarray
    certified_brake_agents: tuple[int, ...]
    hard_feasible: bool
    minimum_center_distance: float
    out_of_certificate_pair_samples: int
    qp_solves: int = 1


def _minimum_center(state: np.ndarray) -> float:
    if len(state) < 2:
        return float("inf")
    delta = state[:, None, :2] - state[None, :, :2]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def _integrate_hybrid_braking(
        state: np.ndarray, force: np.ndarray, brake_mask: np.ndarray,
        cfg: Config, interval: float) -> np.ndarray:
    """Integrate Eq. (17): maximum brake to zero, then remain stopped."""
    physical = np.asarray(state, dtype=float).copy()
    velocity = physical[:, 2:].copy()
    acceleration = np.asarray(force, dtype=float) / cfg.mass
    brake_mask = np.asarray(brake_mask, dtype=bool)

    normal = ~brake_mask
    if np.any(normal):
        physical[normal, :2] += (
            velocity[normal] * interval
            + 0.5 * acceleration[normal] * interval * interval)
        physical[normal, 2:] = np.clip(
            velocity[normal] + acceleration[normal] * interval,
            -cfg.max_speed, cfg.max_speed)

    if np.any(brake_mask):
        ids = np.flatnonzero(brake_mask)
        brake_velocity = velocity[ids]
        speed = np.linalg.norm(brake_velocity, axis=1)
        alpha = cfg.max_force / cfg.mass
        moving = speed > 1e-12
        direction = np.zeros_like(brake_velocity)
        direction[moving] = brake_velocity[moving] / speed[moving, None]
        brake_acceleration = -alpha * direction
        active_time = np.minimum(interval, speed / alpha)
        physical[ids, :2] += (
            brake_velocity * active_time[:, None]
            + 0.5 * brake_acceleration * active_time[:, None] ** 2)
        next_velocity = (
            brake_velocity + brake_acceleration * active_time[:, None])
        stopped = speed <= alpha * interval + 1e-12
        next_velocity[stopped] = 0.0
        physical[ids, 2:] = next_velocity
    return physical


def sampled_wang_step(state: np.ndarray, reference_force: np.ndarray,
                      obstacles: list[Rectangle], cfg: Config,
                      braking_latch: np.ndarray,
                      *, integration_substeps: int
                      ) -> SampledSafetyStep:
    """Apply one 33-Hz Wang decision with exact hybrid Eq. (17) braking.

    ``integration_substeps`` changes only numerical integration and intersample
    certificate auditing. It never changes the number of QP solves.
    """
    if integration_substeps < 1:
        raise ValueError("integration_substeps must be positive")
    physical = np.asarray(state, dtype=float).copy()
    solved = solve_wang_braking_qp(
        physical, reference_force, obstacles, cfg,
        braking_latch=braking_latch)
    action = np.asarray(solved.action, dtype=float)
    initial_velocity = physical[:, 2:].copy()
    brake_mask = np.zeros(cfg.n_agents, dtype=bool)
    brake_mask[solved.certified_brake_agents] = True
    sub_dt = cfg.dt / integration_substeps
    minimum = _minimum_center(physical)
    outside = len(solved.out_of_certificate_pairs)

    for _ in range(integration_substeps):
        physical = _integrate_hybrid_braking(
            physical, action, brake_mask, cfg, sub_dt)
        minimum = min(minimum, _minimum_center(physical))
        for i, j in radius_neighbor_pairs(
                physical[:, :2], cfg.sense_radius):
            if wang_pair_barrier(physical[i], physical[j], cfg) < (
                    -cfg.certificate_tolerance):
                outside += 1

    effective_action = (
        cfg.mass * (physical[:, 2:] - initial_velocity) / cfg.dt)
    return SampledSafetyStep(
        next_state=physical,
        action=effective_action,
        certified_brake_agents=tuple(solved.certified_brake_agents),
        hard_feasible=bool(solved.hard_feasible),
        minimum_center_distance=minimum,
        out_of_certificate_pair_samples=outside,
    )
