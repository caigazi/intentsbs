"""Development-only conflict-component joint ECBF-QP safety oracle.

This backend preserves the original coupled pair constraint and therefore has
no fixed responsibility split.  It is intentionally centralized *within each
currently sensed conflict component*.  It is used to deconfound liveness
experiments from an artificial rho split; it is not the final distributed
deployment algorithm or a claimed contribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from .geometry import Rectangle
from .simulation import Config, _local_constraints, solve_qp_diagnostics
from .spatial import radius_neighbor_pairs


@dataclass
class ComponentQPResult:
    action: np.ndarray
    hard_feasible: bool
    infeasible_components: list[list[int]]
    component_sizes: list[int]
    pair_rows: int
    obstacle_rows: int
    # Keep the fallback cause explicit.  A hard-infeasible local polytope and
    # a numerically invalid projection are different backend failures and must
    # not be merged in the experiment logs.
    hard_infeasible_components: list[list[int]] = field(default_factory=list)
    numerically_invalid_components: list[list[int]] = field(default_factory=list)
    # Eq. (17)-style braking is a certified hybrid mode, not an unclassified
    # optimizer failure.  Legacy solvers leave this empty.
    certified_brake_agents: list[int] = field(default_factory=list)
    # Initial/current states outside the braking-viable set invalidate the
    # continuous-time certificate and therefore remain separately visible.
    out_of_certificate_pairs: list[tuple[int, int]] = field(default_factory=list)
    out_of_certificate_obstacles: list[tuple[int, int]] = field(default_factory=list)


def _braking_pair_row(state: np.ndarray, i: int, j: int,
                      cfg: Config) -> tuple[np.ndarray, float]:
    """Return the input-aware joint row a_i and rhs for pair (i,j)."""
    rel_p = state[i, :2] - state[j, :2]
    distance = float(np.linalg.norm(rel_p))
    normal = rel_p / max(distance, 1e-12)
    rel_v = state[i, 2:] - state[j, 2:]
    radial_speed = float(normal @ rel_v)
    declared_pair_safe = cfg.pair_safe_radius_factor * cfg.car_radius
    sample_guard = 2.0 * np.sqrt(2.0) * cfg.max_speed * cfg.dt
    gap = max(distance - declared_pair_safe - sample_guard, 1e-9)
    a_max = cfg.max_force / cfg.mass
    braking = float(np.sqrt(4.0 * a_max * gap))
    h = braking + radial_speed
    tangential = rel_v - radial_speed * normal
    geometric_drift = float(tangential @ tangential / max(distance, 1e-12))
    braking_drift = float((2.0 * a_max / max(braking, 1e-12)) * radial_speed)
    rhs = -h - braking_drift - geometric_drift
    return normal / cfg.mass, rhs


def _sampled_braking_pair_row(state: np.ndarray, i: int, j: int,
                               cfg: Config) -> tuple[np.ndarray, float]:
    """One-step discrete braking-CBF row for the forward-Euler plant.

    At control step k, p[k+1] is already fixed by the current velocity while
    u[k] changes v[k+1].  We therefore evaluate the braking barrier at the
    exactly predicted next relative position and impose

        h[k+1] >= (1 - gamma*dt) h[k].

    The resulting inequality is linear in the two controls and avoids the
    singular worst-case distance guard used by the legacy continuous row.
    """
    rel_p = state[i, :2] - state[j, :2]
    rel_v = state[i, 2:] - state[j, 2:]
    distance = float(np.linalg.norm(rel_p))
    normal = rel_p / max(distance, 1e-12)
    declared_pair_safe = cfg.pair_safe_radius_factor * cfg.car_radius
    a_max = cfg.max_force / cfg.mass
    gap = distance - declared_pair_safe
    radial_speed = float(normal @ rel_v)
    h_now = np.sqrt(max(0.0, 4.0 * a_max * gap)) + radial_speed

    rel_p_next = rel_p + cfg.dt * rel_v
    distance_next = float(np.linalg.norm(rel_p_next))
    normal_next = rel_p_next / max(distance_next, 1e-12)
    # If the next sampled position is already unsafe, u[k] cannot repair it
    # under forward Euler because u[k] only affects v[k+1].  Return an
    # explicitly impossible row so the hard-infeasible state stays visible.
    if distance_next <= declared_pair_safe + 1e-10:
        row = cfg.dt * normal_next / cfg.mass
        max_joint_lhs = 2.0 * cfg.max_force * float(np.abs(row).sum())
        return row, max_joint_lhs + 1.0

    gap_next = distance_next - declared_pair_safe
    radial_next_without_control = float(normal_next @ rel_v)
    h_next_without_control = (
        np.sqrt(4.0 * a_max * gap_next) + radial_next_without_control)
    gamma = 1.0
    rhs = (1.0 - gamma * cfg.dt) * h_now - h_next_without_control
    return cfg.dt * normal_next / cfg.mass, float(rhs)


def _pair_row(state: np.ndarray, i: int, j: int, cfg: Config,
              mode: str) -> tuple[np.ndarray, float]:
    if mode == "legacy_guard":
        return _braking_pair_row(state, i, j, cfg)
    if mode == "sampled_discrete":
        return _sampled_braking_pair_row(state, i, j, cfg)
    raise ValueError(f"unknown pair row mode: {mode}")


def solve_distributed_braking_qp(state: np.ndarray, u_ref: np.ndarray,
                                 obstacles: list[Rectangle], cfg: Config,
                                 responsibility: float = 0.5,
                                 pair_row_mode: str = "legacy_guard") -> ComponentQPResult:
    """Conventional local DecCBF with braking barrier and hybrid stop backup.

    This keeps one independent 2-D QP per robot.  A symmetric 0.5 share is the
    default classical cooperative split; the function exists as a candidate
    deployment backend, not as a new contribution.
    """
    n = cfg.n_agents
    rows: list[list[np.ndarray]] = [[] for _ in range(n)]
    rhs: list[list[float]] = [[] for _ in range(n)]
    pair_rows = 0
    degrees = np.zeros(n, dtype=int)
    for i, j in radius_neighbor_pairs(state[:, :2], cfg.sense_radius):
        a_i, b_ij = _pair_row(state, i, j, cfg, pair_row_mode)
        rows[i].append(a_i)
        rhs[i].append(responsibility * b_ij)
        rows[j].append(-a_i)
        rhs[j].append((1.0 - responsibility) * b_ij)
        pair_rows += 1
        degrees[i] += 1
        degrees[j] += 1
    obstacle_rows = 0
    for i in range(n):
        local = _local_constraints(i, state, obstacles, cfg, agent_neighbors=())
        for row in np.flatnonzero(local.kinds == 1):
            rows[i].append(local.a[row])
            rhs[i].append(float(local.b[row]))
            obstacle_rows += 1

    action = np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)
    infeasible: list[list[int]] = []
    hard_infeasible: list[list[int]] = []
    numerically_invalid: list[list[int]] = []
    for i in range(n):
        if not rows[i]:
            continue
        solution = solve_qp_diagnostics(
            u_ref[i], np.asarray(rows[i]), np.asarray(rhs[i]), cfg)
        if solution.hard_feasible and solution.numerically_valid:
            action[i] = solution.control
        else:
            action[i] = np.clip(-cfg.mass * state[i, 2:] / cfg.dt,
                                -cfg.max_force, cfg.max_force)
            infeasible.append([i])
            if not solution.hard_feasible:
                hard_infeasible.append([i])
            if not solution.numerically_valid:
                numerically_invalid.append([i])
    return ComponentQPResult(
        action=action, hard_feasible=not infeasible,
        infeasible_components=infeasible,
        component_sizes=[int(value) for value in degrees if value > 0],
        pair_rows=pair_rows, obstacle_rows=obstacle_rows,
        hard_infeasible_components=hard_infeasible,
        numerically_invalid_components=numerically_invalid,
    )


def _components(n_agents: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency = [set() for _ in range(n_agents)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    unseen = set(range(n_agents))
    result: list[list[int]] = []
    while unseen:
        root = min(unseen)
        stack, component = [root], []
        unseen.remove(root)
        while stack:
            i = stack.pop()
            component.append(i)
            for j in adjacency[i]:
                if j in unseen:
                    unseen.remove(j)
                    stack.append(j)
        result.append(sorted(component))
    return result


def _project_qp(reference: np.ndarray, a: np.ndarray, b: np.ndarray,
                max_force: float) -> tuple[np.ndarray, bool]:
    """Euclidean projection onto A u >= b and the actuator box."""
    clipped = np.clip(reference, -max_force, max_force)
    if not len(b):
        return clipped, True
    if np.min(a @ clipped - b) >= -1e-8:
        return clipped, True

    bounds = [(-max_force, max_force)] * len(reference)
    feasible = linprog(np.zeros(len(reference)), A_ub=-a, b_ub=-b,
                       bounds=bounds, method="highs")
    if not feasible.success:
        return clipped, False

    def objective(value: np.ndarray) -> float:
        delta = value - reference
        return 0.5 * float(delta @ delta)

    def jacobian(value: np.ndarray) -> np.ndarray:
        return value - reference

    result = minimize(
        objective, feasible.x, jac=jacobian, method="SLSQP",
        bounds=Bounds(-max_force, max_force),
        constraints=LinearConstraint(a, b, np.inf),
        options={"ftol": 1e-10, "maxiter": 200, "disp": False},
    )
    value = np.asarray(result.x if result.success else feasible.x, dtype=float)
    valid = bool(np.min(a @ value - b) >= -1e-7
                 and np.max(np.abs(value)) <= max_force + 1e-7)
    return value, valid


def solve_component_qp(state: np.ndarray, u_ref: np.ndarray,
                       obstacles: list[Rectangle], cfg: Config,
                       pair_row_mode: str = "legacy_guard") -> ComponentQPResult:
    n = cfg.n_agents
    declared_pair_safe = cfg.pair_safe_radius_factor * cfg.car_radius
    # One-step sampled-data guard: under the component-wise velocity box, two
    # robots can close by at most 2*sqrt(2)*v_max*dt before a newly computed
    # acceleration affects position in the forward-Euler model.
    sample_guard = 2.0 * np.sqrt(2.0) * cfg.max_speed * cfg.dt
    pair_safe = declared_pair_safe + sample_guard
    # Input-aware braking barrier (Wang--Ames--Egerstedt family):
    # h = sqrt(4*a_max*(distance-d_safe)) + radial_relative_speed.
    # Each robot can guarantee at least max_force/m acceleration along any
    # radial direction, hence the pair can generate twice that separation
    # acceleration.  Unlike the old distance-only HOCBF, this safe set excludes
    # states that are already too close/fast to brake under the input bound.
    a_max = cfg.max_force / cfg.mass
    # A one-second reciprocal class-K rate is deliberately slower than the
    # old omega=4 HOCBF.  With sampled control, a large gamma permits h to fall
    # too quickly between updates and defeats the braking-set interpretation.
    gamma = 1.0
    pair_data: list[tuple[int, int, np.ndarray, float]] = []
    edges: list[tuple[int, int]] = []
    for i, j in radius_neighbor_pairs(state[:, :2], cfg.sense_radius):
        a_i, rhs = _pair_row(state, i, j, cfg, pair_row_mode)
        pair_data.append((i, j, a_i, rhs))
        edges.append((i, j))

    obstacle_data: dict[int, list[tuple[np.ndarray, float]]] = {i: [] for i in range(n)}
    for i in range(n):
        local = _local_constraints(i, state, obstacles, cfg, agent_neighbors=())
        for row in np.flatnonzero(local.kinds == 1):
            obstacle_data[i].append((local.a[row], float(local.b[row])))

    action = np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)
    infeasible: list[list[int]] = []
    component_sizes: list[int] = []
    for component in _components(n, edges):
        has_obstacle = any(obstacle_data[i] for i in component)
        component_pairs = [item for item in pair_data
                           if item[0] in component and item[1] in component]
        if not component_pairs and not has_obstacle:
            continue
        component_sizes.append(len(component))
        local_id = {agent: k for k, agent in enumerate(component)}
        rows, rhs = [], []
        for i, j, a_i, b_ij in component_pairs:
            row = np.zeros(2 * len(component))
            row[2 * local_id[i]:2 * local_id[i] + 2] = a_i
            row[2 * local_id[j]:2 * local_id[j] + 2] = -a_i
            rows.append(row)
            rhs.append(b_ij)
        for i in component:
            for a_i, b_i in obstacle_data[i]:
                row = np.zeros(2 * len(component))
                row[2 * local_id[i]:2 * local_id[i] + 2] = a_i
                rows.append(row)
                rhs.append(b_i)
        reference = u_ref[component].reshape(-1)
        value, ok = _project_qp(reference, np.asarray(rows), np.asarray(rhs), cfg.max_force)
        if ok:
            action[component] = value.reshape(-1, 2)
        else:
            # Explicit emergency action; infeasibility remains visible and is
            # never converted into a misleading soft-safety success.
            action[component] = np.clip(-cfg.mass * state[component, 2:] / cfg.dt,
                                        -cfg.max_force, cfg.max_force)
            infeasible.append(component)
    return ComponentQPResult(
        action=action, hard_feasible=not infeasible,
        infeasible_components=infeasible, component_sizes=component_sizes,
        pair_rows=len(pair_data),
        obstacle_rows=sum(len(value) for value in obstacle_data.values()),
        hard_infeasible_components=infeasible,
        numerically_invalid_components=[],
    )
