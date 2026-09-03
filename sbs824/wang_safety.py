"""Wang--Ames--Egerstedt feasible safety certificates (T-RO 2017).

The robot--robot implementation below follows Eqs. (15)--(17): ``u`` in the
paper is acceleration, whereas the simulator QP variable is force.  The force
conversion therefore appears exactly once in :func:`wang_pair_force_row`.

The static-convex-obstacle row is the corresponding braking-segment geometry:
the complete maximum-braking segment is enclosed by a ball around its midpoint
and that ball is kept outside the convex obstacle.  It is infrastructure, not
claimed as a contribution of IntentComm.
"""

from __future__ import annotations

import numpy as np

from .component_qp import ComponentQPResult
from .geometry import Rectangle, closest_point
from .simulation import Config, solve_qp_diagnostics
from .spatial import radius_neighbor_lists, radius_neighbor_pairs


def _max_acceleration(cfg: Config) -> float:
    return cfg.max_force / cfg.mass


def _scale_consistent_gamma(cfg: Config, safe_distance: float) -> float:
    """Raw Eq. (16) gamma induced by h_bar = h / safe_distance**2."""
    return cfg.wang_gamma / safe_distance ** 4


def wang_pair_barrier(state_i: np.ndarray, state_j: np.ndarray,
                      cfg: Config) -> float:
    """Eq. (15), with the corrected ``4 alpha_j`` second denominator."""
    alpha_i = alpha_j = _max_acceleration(cfg)
    p_i, v_i = state_i[:2], state_i[2:]
    p_j, v_j = state_j[:2], state_j[2:]
    n_i, n_j = float(np.linalg.norm(v_i)), float(np.linalg.norm(v_j))
    q_i = p_i + n_i * v_i / (4.0 * alpha_i)
    q_j = p_j + n_j * v_j / (4.0 * alpha_j)
    radius = (cfg.wang_pair_safe_radius_factor * cfg.car_radius
              + n_i * n_i / (4.0 * alpha_i)
              + n_j * n_j / (4.0 * alpha_j))
    return float((q_i - q_j) @ (q_i - q_j) - radius * radius)


def wang_pair_acceleration_halfspace(
        state_i: np.ndarray, state_j: np.ndarray, cfg: Config
) -> tuple[np.ndarray, float, float]:
    """Return paper Eq. (16) as ``A_hat @ acceleration_i <= b_hat``.

    This is the published decentralized row for ego ``i``.  It is not a
    responsibility split applied to a different joint CBF.
    """
    alpha_i = alpha_j = _max_acceleration(cfg)
    p_i, v_i = state_i[:2], state_i[2:]
    p_j, v_j = state_j[:2], state_j[2:]
    delta_p = p_i - p_j
    delta_v = v_i - v_j
    n_i, n_j = float(np.linalg.norm(v_i)), float(np.linalg.norm(v_j))
    h_hat = wang_pair_barrier(state_i, state_j, cfg)

    # The paper explicitly defines lim_{||v_i||->0} A_hat_ij = 0.
    if n_i <= 1e-10:
        a_hat = np.zeros(2)
    else:
        dot_ij = float(v_i @ v_j)
        dot_pi = float(delta_p @ v_i)
        d_safe = cfg.wang_pair_safe_radius_factor * cfg.car_radius
        a_hat = (
            (n_i * n_j / (8.0 * alpha_i * alpha_j)) * v_j
            + (dot_ij * n_j / (8.0 * alpha_i * alpha_j * n_i)) * v_i
            - (dot_pi / (2.0 * alpha_i * n_i)) * v_i
            - (n_i / (2.0 * alpha_i)) * delta_p
            + (d_safe / alpha_i) * v_i
            + (n_j * n_j / (4.0 * alpha_i * alpha_j)) * v_i
        )
    gamma = _scale_consistent_gamma(
        cfg, cfg.wang_pair_safe_radius_factor * cfg.car_radius)
    b_hat = (float(delta_p @ delta_v)
             + n_i / (2.0 * alpha_i) * float(delta_v @ v_i)
             + 0.5 * gamma * h_hat ** 3)
    return a_hat, b_hat, h_hat


def wang_pair_force_row(state_i: np.ndarray, state_j: np.ndarray,
                        cfg: Config) -> tuple[np.ndarray, float, float]:
    """Eq. (16) converted to the simulator convention ``C @ force >= d``."""
    a_hat, b_hat, h_hat = wang_pair_acceleration_halfspace(state_i, state_j, cfg)
    return -a_hat / cfg.mass, -b_hat, h_hat


def obstacle_braking_barrier(state_i: np.ndarray, obstacle: Rectangle,
                             cfg: Config) -> float:
    """Braking-midpoint viable-set value for one static convex obstacle."""
    alpha = _max_acceleration(cfg)
    p, v = state_i[:2], state_i[2:]
    speed = float(np.linalg.norm(v))
    q = p + speed * v / (4.0 * alpha)
    closest = closest_point(q, obstacle)
    delta = q - closest
    radius = (cfg.wang_obstacle_safe_radius_factor * cfg.car_radius
              + speed * speed / (4.0 * alpha))
    return float(delta @ delta - radius * radius)


def obstacle_force_row(state_i: np.ndarray, obstacle: Rectangle,
                       cfg: Config) -> tuple[np.ndarray, float, float]:
    """Static-obstacle analogue of Eq. (16), in ``C @ force >= d`` form."""
    alpha = _max_acceleration(cfg)
    p, v = state_i[:2], state_i[2:]
    speed = float(np.linalg.norm(v))
    q = p + speed * v / (4.0 * alpha)
    closest = closest_point(q, obstacle)
    delta = q - closest
    d_safe = cfg.wang_obstacle_safe_radius_factor * cfg.car_radius
    h_hat = obstacle_braking_barrier(state_i, obstacle, cfg)
    if speed <= 1e-10:
        a_hat = np.zeros(2)
    else:
        jacobian = (speed * np.eye(2) + np.outer(v, v) / speed) / (4.0 * alpha)
        radius = d_safe + speed * speed / (4.0 * alpha)
        grad_v = 2.0 * jacobian @ delta - radius * v / alpha
        a_hat = -grad_v
    # Projection onto a convex set has zero first-order contribution along
    # the outward residual, so the position drift is 2*delta dot velocity.
    gamma = _scale_consistent_gamma(
        cfg, cfg.wang_obstacle_safe_radius_factor * cfg.car_radius)
    b_hat = 2.0 * float(delta @ v) + gamma * h_hat ** 3
    return -a_hat / cfg.mass, -b_hat, h_hat


def certified_braking_force(velocity: np.ndarray, cfg: Config) -> np.ndarray:
    """Continuous Eq. (17) command before its zero-speed switch.

    Eq. (17) applies maximum braking until the velocity reaches zero and then
    switches the force to zero.  Shortening the force so that a low-speed
    robot stops only at the end of a sampled interval is not equivalent: it
    lengthens the stopping segment and invalidates Lemma VI.1.
    """
    speed = float(np.linalg.norm(velocity))
    if speed <= 1e-10:
        return np.zeros(2)
    return -cfg.max_force * velocity / speed


def solve_wang_braking_qp(state: np.ndarray, u_ref: np.ndarray,
                          obstacles: list[Rectangle], cfg: Config,
                          braking_latch: np.ndarray | None = None
                          ) -> ComponentQPResult:
    """Independent local Eq. (16) QPs with the certified Eq. (17) mode.

    ``braking_latch`` implements the proof text following Eq. (17): after an
    infeasible normal QP, an agent keeps maximum-braking until it has stopped;
    only then may it return to the normal QP.  Passing ``None`` is convenient
    for one-shot unit tests but deployment should preserve the latch.
    """
    n = cfg.n_agents
    if braking_latch is None:
        braking_latch = np.zeros(n, dtype=bool)
    if braking_latch.shape != (n,):
        raise ValueError("braking_latch must have shape (n_agents,)")
    neighbors = radius_neighbor_lists(state[:, :2], cfg.sense_radius)
    action = np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)
    brake_agents: list[int] = []
    hard_infeasible: list[list[int]] = []
    numerical: list[list[int]] = []
    out_pairs: set[tuple[int, int]] = set()
    out_obstacles: list[tuple[int, int]] = []
    pair_rows = 0
    obstacle_rows = 0

    for i, local_neighbors in enumerate(neighbors):
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for j in local_neighbors:
            row, bound, h_hat = wang_pair_force_row(state[i], state[j], cfg)
            rows.append(row)
            rhs.append(bound)
            pair_rows += 1
            if h_hat < -cfg.certificate_tolerance:
                out_pairs.add(tuple(sorted((i, int(j)))))
        for obstacle_id, obstacle in enumerate(obstacles):
            # Query using the actual obstacle distance; obstacles outside the
            # local sensing disk do not enter this robot's controller.
            if np.linalg.norm(state[i, :2] - closest_point(state[i, :2], obstacle)) > cfg.sense_radius:
                continue
            row, bound, h_hat = obstacle_force_row(state[i], obstacle, cfg)
            rows.append(row)
            rhs.append(bound)
            obstacle_rows += 1
            if h_hat < -cfg.certificate_tolerance:
                out_obstacles.append((i, obstacle_id))

        if not rows:
            braking_latch[i] = False
            continue
        solution = solve_qp_diagnostics(
            u_ref[i], np.asarray(rows), np.asarray(rhs), cfg)
        speed = float(np.linalg.norm(state[i, 2:]))
        if braking_latch[i] and speed > 1e-8:
            action[i] = certified_braking_force(state[i, 2:], cfg)
            brake_agents.append(i)
        elif solution.hard_feasible and solution.numerically_valid:
            braking_latch[i] = False
            action[i] = solution.control
        else:
            braking_latch[i] = True
            action[i] = certified_braking_force(state[i, 2:], cfg)
            brake_agents.append(i)
            if not solution.hard_feasible:
                hard_infeasible.append([i])
            if not solution.numerically_valid:
                numerical.append([i])

    # Lemma VI.1 certifies a pair only when *both* endpoints brake.  Therefore
    # a braking request is propagated over the currently sensed connected
    # component (one synchronous emergency bit per edge).  Braking only the
    # infeasible ego while its neighbor keeps accelerating is not certified.
    brake_set = set(brake_agents)
    changed = True
    pair_edges = radius_neighbor_pairs(state[:, :2], cfg.sense_radius)
    while changed:
        changed = False
        for i, j in pair_edges:
            if (i in brake_set) != (j in brake_set):
                brake_set.update((i, j))
                changed = True
    for i in sorted(brake_set):
        action[i] = certified_braking_force(state[i, 2:], cfg)
        braking_latch[i] = float(np.linalg.norm(state[i, 2:])) > 1e-8
    brake_agents = sorted(brake_set)

    # A certified brake is not an uncertified fallback, hence hard_feasible
    # here refers to existence of the normal Eq. (16) action only.
    return ComponentQPResult(
        action=action,
        hard_feasible=not hard_infeasible and not numerical,
        infeasible_components=[],
        component_sizes=[len(value) for value in neighbors if len(value) > 0],
        pair_rows=pair_rows,
        obstacle_rows=obstacle_rows,
        hard_infeasible_components=hard_infeasible,
        numerically_invalid_components=numerical,
        certified_brake_agents=brake_agents,
        out_of_certificate_pairs=sorted(out_pairs),
        out_of_certificate_obstacles=out_obstacles,
    )
