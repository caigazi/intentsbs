"""Pure NumPy distributed double-integrator + local analytic CBF-QP.

The simulation world knows all rectangles for collision checking.  A robot's
controller receives only agents and obstacle surfaces within ``sense_radius``.
There is no map, global path planner, or centralized optimization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .geometry import Rectangle, closest_point, signed_distance
from .spatial import radius_neighbor_lists, radius_neighbor_pairs


@dataclass(frozen=True)
class Config:
    n_agents: int = 8
    n_obstacles: int = 8
    seed: int = 0
    controller: str = "cbf_qp"  # nominal | cbf_qp
    scenario: str = "random"  # random | head_on | bottleneck | ring
    area_size: float = 4.0
    steps: int = 256
    adaptive_stop: bool = False
    adaptive_min_steps: int = 256
    adaptive_still_steps: int = 333
    adaptive_speed_epsilon: float = 0.01
    adaptive_goal_dwell_steps: int = 33
    adaptive_goal_speed_epsilon: float = 0.05
    dt: float = 0.03
    mass: float = 0.1
    car_radius: float = 0.05
    pair_safe_radius_factor: float = 4.0
    # Conservative ego-wise sufficient condition used by the current empirical
    # safety baseline. A 0.5 symmetric split remains an explicit ablation.
    pair_responsibility: float = 1.0
    obstacle_safe_radius_factor: float = 6.0
    # Wang's D_s is the actual forbidden center distance, not the wider
    # GCBF+ training safe-mask margin.  Keep it separate from legacy margins.
    # 2r is the physical collision boundary; the additional 0.05r (2.5 mm)
    # is an explicit sampled-data margin and is reported separately.
    wang_pair_safe_radius_factor: float = 2.05
    wang_obstacle_safe_radius_factor: float = 1.0
    sense_radius: float = 0.5
    # Liveness coordination may start before the hard safety filter becomes
    # active.  It must remain no larger than the locally sensed radius.
    intent_trigger_distance: float = 0.28
    # Seeing a neighbor is not itself a coordination event.  After predictive
    # conflict evidence disappears, hold the last intent briefly to prevent
    # chatter before returning to the nominal controller.
    intent_release_hold_steps: int = 10
    intent_min_closing_speed: float = 1e-3
    # Set to zero for a pure range-triggered coordination ablation.  Positive
    # values use relative-motion closest-approach prediction over this horizon.
    intent_prediction_horizon: float = 1.0
    # Offline-teacher option for exact symmetric curriculum scenes.  It never
    # changes the deployed student interface or safety filter.
    teacher_force_shared_intent: bool = False
    obstacle_size_min: float = 0.1
    obstacle_size_max: float = 0.5
    max_force: float = 1.0
    max_speed: float = 0.5
    # Safety constraints are not the place to impose the learned coordinator's
    # top-k communication budget. Keep every locally sensed constraint in the
    # QP, then apply ``influence_top_k`` only to learning/logging inputs.
    # 0 means unlimited and is required for every safety claim.  A positive
    # value exists only for explicitly labelled computational stress tests;
    # it drops safety rows and must never be used as a certified controller.
    max_constraints: int = 0
    # Match the official GCBF+ local graph in the main method: every robot
    # inside the 0.5 m sensing disk contributes a message.  Positive Top-K
    # values remain available only as an explicit communication ablation.
    influence_top_k: int = 0
    obstacle_clearance: float = 0.20
    obstacle_border_margin: float = 0.15
    cbf_omega: float = 4.0
    cbf_zeta: float = 1.0
    # Dimensionless class-K coefficient.  Eq. (15) has units of distance^2;
    # applying gamma to h directly is not scale invariant.  The implementation
    # uses h_bar=h/D_safe^2, equivalent to gamma_raw=gamma_bar/D_safe^4.
    wang_gamma: float = 1.0
    certificate_tolerance: float = 1e-9
    # Eq. (15)--(17) is continuous-time.  This controls only the internal
    # safety-filter/integration rate while Intent/LQR remain at ``dt``.
    wang_safety_substeps: int = 16
    slack_penalty: float = 200.0
    qp_iterations: int = 100
    reach_radius: float = 0.1
    stall_window_steps: int = 33
    stall_progress_epsilon: float = 0.02
    stall_nominal_threshold: float = 0.15
    stall_intervention_threshold: float = 0.15
    sbs_progress_ratio_threshold: float = 0.20
    sbs_nominal_progress_epsilon: float = 0.05
    sbs_active_fraction: float = 0.80
    sbs_causal_progress_epsilon: float = 0.02
    dual_activity_threshold: float = 1e-3


@dataclass
class EpisodeResult:
    row: dict
    positions: np.ndarray
    goals: np.ndarray
    obstacles: list[Rectangle]
    reached_ever: np.ndarray
    collision_ever: np.ndarray
    agent_collision_ever: np.ndarray
    obstacle_collision_ever: np.ndarray
    interventions: np.ndarray
    slacks: np.ndarray
    sbs_ever: np.ndarray
    sbs_final_class: np.ndarray
    sbs_online_class: np.ndarray
    agent_constraint_pressure: np.ndarray
    obstacle_constraint_pressure: np.ndarray
    states: np.ndarray
    agent_correction: np.ndarray
    agent_constraint_load: np.ndarray
    agent_cancellation_index: np.ndarray
    qp_hard_feasible: np.ndarray
    local_agent_degree: np.ndarray
    local_constraint_count: np.ndarray


@dataclass(frozen=True)
class LocalConstraintSet:
    """One ego robot's local constraints with cause labels kept intact."""

    a: np.ndarray
    b: np.ndarray
    kinds: np.ndarray  # 0 = robot, 1 = obstacle
    sources: np.ndarray


@dataclass(frozen=True)
class QPSolution:
    """QP output plus KKT diagnostics for the original safety constraints."""

    control: np.ndarray
    slack: np.ndarray
    dual: np.ndarray
    hard_feasible: bool
    numerically_valid: bool


def _recover_nonnegative_dual(normals: np.ndarray, delta: np.ndarray,
                              active: np.ndarray) -> np.ndarray:
    """Recover one sparse nonnegative KKT multiplier vector in 2-D."""
    dual = np.zeros(len(normals))
    ids = np.flatnonzero(active)
    if len(ids) == 0 or np.linalg.norm(delta) < 1e-12:
        return dual
    choices: list[tuple[float, float, tuple[int, ...], np.ndarray]] = []
    for idx in ids:
        normal = normals[idx]
        denom = float(normal @ normal)
        if denom <= 1e-14:
            continue
        value = float(normal @ delta / denom)
        if value >= -1e-9:
            value = max(0.0, value)
            reconstruction = value * normal
            choices.append((float(np.linalg.norm(reconstruction - delta)), value ** 2,
                            (int(idx),), np.array([value])))
    for left_pos, left in enumerate(ids):
        for right in ids[left_pos + 1:]:
            matrix = np.stack((normals[left], normals[right]), axis=1)
            if abs(float(np.linalg.det(matrix))) <= 1e-12:
                continue
            values = np.linalg.solve(matrix, delta)
            if np.all(values >= -1e-9):
                values = np.maximum(values, 0.0)
                choices.append((float(np.linalg.norm(matrix @ values - delta)),
                                float(values @ values), (int(left), int(right)), values))
    if not choices:
        return dual
    _, _, support, values = min(choices, key=lambda item: (item[0], item[1]))
    dual[list(support)] = values
    return dual


def _dare_gain(dt: float, mass: float) -> np.ndarray:
    """Iterative DARE solution matching GCBF+'s Q=5I, R=I LQR."""
    a = np.eye(4)
    a[0, 2] = dt
    a[1, 3] = dt
    b = np.zeros((4, 2))
    b[2, 0] = dt / mass
    b[3, 1] = dt / mass
    q, r = 5.0 * np.eye(4), np.eye(2)
    x = q.copy()
    for _ in range(10_000):
        gain = np.linalg.solve(r + b.T @ x @ b, b.T @ x @ a)
        nxt = a.T @ x @ a - a.T @ x @ b @ gain + q
        if np.max(np.abs(nxt - x)) < 1e-12:
            x = nxt
            break
        x = nxt
    return np.linalg.solve(r + b.T @ x @ b, b.T @ x @ a)


def _sample_rectangles(rng: np.random.Generator, cfg: Config) -> list[Rectangle]:
    """Sample separated convex rectangles.

    A conservative circumscribed-circle test prevents overlapping rectangles
    from accidentally forming U-shaped compound traps.  This reduces O-SBS
    confounding without pretending that a single convex obstacle can never
    create a myopic navigation minimum.
    """
    result: list[Rectangle] = []
    for _ in range(cfg.n_obstacles):
        for _attempt in range(50_000):
            size = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max, size=2)
            radius = 0.5 * float(np.linalg.norm(size))
            margin = cfg.obstacle_border_margin + radius
            if 2.0 * margin >= cfg.area_size:
                raise ValueError("Obstacle margin is too large for the arena")
            candidate = Rectangle(
                center=rng.uniform(margin, cfg.area_size - margin, size=2),
                size=size,
                theta=float(rng.uniform(0.0, 2.0 * np.pi)),
            )
            separated = all(
                np.linalg.norm(candidate.center - rect.center)
                > radius + 0.5 * np.linalg.norm(rect.size) + cfg.obstacle_clearance
                for rect in result
            )
            if separated:
                result.append(candidate)
                break
        else:
            raise RuntimeError("Could not sample separated convex obstacles")
    return result


def _valid_point(point: np.ndarray, existing: Iterable[np.ndarray], obstacles: list[Rectangle],
                 pair_min_dist: float, obstacle_min_dist: float) -> bool:
    if any(np.linalg.norm(point - other) <= pair_min_dist for other in existing):
        return False
    return all(signed_distance(point, rect) > obstacle_min_dist for rect in obstacles)


def _sample_points(rng: np.random.Generator, count: int, cfg: Config,
                   obstacles: list[Rectangle]) -> np.ndarray:
    points: list[np.ndarray] = []
    # Initial states must start inside the declared CBF safe set.  Using only
    # the physical collision diameter here makes a conservative CBF infeasible
    # at t=0 and confounds later liveness statistics with scene-generation
    # violations.
    pair_min_dist = cfg.pair_safe_radius_factor * cfg.car_radius
    obstacle_min_dist = cfg.obstacle_safe_radius_factor * cfg.car_radius
    for _ in range(count):
        for _attempt in range(50_000):
            candidate = rng.uniform(0.0, cfg.area_size, size=2)
            if _valid_point(candidate, points, obstacles, pair_min_dist, obstacle_min_dist):
                points.append(candidate)
                break
        else:
            raise RuntimeError("Could not sample a collision-free scene")
    return np.asarray(points)


def generate_scene(cfg: Config) -> tuple[np.ndarray, np.ndarray, list[Rectangle]]:
    rng = np.random.default_rng(cfg.seed)
    if cfg.scenario != "random" and cfg.n_agents % 2:
        raise ValueError("The deterministic conflict scenarios require an even number of agents")
    if cfg.scenario == "head_on":
        pairs = cfg.n_agents // 2
        ys = np.linspace(0.25, cfg.area_size - 0.25, pairs)
        ys = np.clip(ys + rng.uniform(-0.008, 0.008, size=pairs), 0.22, cfg.area_size - 0.22)
        left = np.column_stack((np.full(pairs, 1.10), ys))
        right = np.column_stack((np.full(pairs, 2.90), ys))
        starts = np.vstack((left, right))
        goals = np.vstack((right, left))
        return starts, goals, []
    if cfg.scenario == "ring":
        angles = np.linspace(0.0, 2.0 * np.pi, cfg.n_agents, endpoint=False)
        angles += rng.uniform(-0.004, 0.004, size=cfg.n_agents)
        center = np.array([cfg.area_size / 2.0, cfg.area_size / 2.0])
        starts = center + 1.35 * np.column_stack((np.cos(angles), np.sin(angles)))
        goals = center - 1.35 * np.column_stack((np.cos(angles), np.sin(angles)))
        return starts, goals, []
    if cfg.scenario == "bottleneck":
        gap = 0.24
        wall_height = (cfg.area_size - gap) / 2.0
        obstacles = [
            Rectangle(np.array([2.0, wall_height / 2.0]), np.array([0.18, wall_height]), 0.0),
            Rectangle(np.array([2.0, cfg.area_size - wall_height / 2.0]), np.array([0.18, wall_height]), 0.0),
        ]
        pairs = cfg.n_agents // 2
        xs = [1.55, 1.20, 0.85, 0.50]
        ys = [1.55, 1.85, 2.15, 2.45]
        candidates = np.array([[x, y] for x in xs for y in ys], dtype=float)
        if pairs > len(candidates):
            raise ValueError("bottleneck supports at most 32 agents")
        left = candidates[:pairs]
        right = left.copy()
        right[:, 0] = cfg.area_size - left[:, 0]
        return np.vstack((left, right)), np.vstack((right, left)), obstacles
    if cfg.scenario != "random":
        raise ValueError(f"Unknown scenario: {cfg.scenario}")
    obstacles = _sample_rectangles(rng, cfg)
    starts = _sample_points(rng, cfg.n_agents, cfg, obstacles)
    goals = _sample_points(rng, cfg.n_agents, cfg, obstacles)
    return starts, goals, obstacles


def nominal_lqr(state: np.ndarray, goals: np.ndarray, gain: np.ndarray, cfg: Config) -> np.ndarray:
    goal_state = np.column_stack((goals, np.zeros((cfg.n_agents, 2))))
    error = goal_state - state
    norm = np.linalg.norm(error, axis=1, keepdims=True)
    error_max = np.abs(error / np.maximum(norm, 1e-9) * cfg.sense_radius)
    error = np.clip(error, -error_max, error_max)
    return np.clip(error @ gain.T, -cfg.max_force, cfg.max_force)


def _local_constraints(agent_id: int, state: np.ndarray, obstacles: list[Rectangle],
                       cfg: Config,
                       agent_neighbors: Iterable[int] | None = None) -> LocalConstraintSet:
    """Return the closest local constraints A u >= b for one robot."""
    pos, vel = state[agent_id, :2], state[agent_id, 2:]
    candidates: list[tuple[float, int, int, np.ndarray, float]] = []
    # Match GCBF+'s conservative safe-mask margins (collision is at 2r/r).
    pair_safe = cfg.pair_safe_radius_factor * cfg.car_radius
    k_d = 2.0 * cfg.cbf_zeta * cfg.cbf_omega
    k_p = cfg.cbf_omega ** 2

    # Agent-agent constraints. ``pair_responsibility`` is an explicit backend
    # assumption: 0.5 reproduces a symmetric split; the current default 1.0
    # makes every ego robot assume that its neighbour contributes nothing.
    neighbor_ids = (range(cfg.n_agents) if agent_neighbors is None
                    else agent_neighbors)
    for j in neighbor_ids:
        if j == agent_id:
            continue
        rel_p = pos - state[j, :2]
        distance = float(np.linalg.norm(rel_p))
        if distance > cfg.sense_radius:
            continue
        rel_v = vel - state[j, 2:]
        h = rel_p @ rel_p - pair_safe ** 2
        h_dot = 2.0 * rel_p @ rel_v
        drift = 2.0 * rel_v @ rel_v
        common = drift + k_d * h_dot + k_p * h
        a = (2.0 / cfg.mass) * rel_p
        # Conservative full ego responsibility; 0.5 is an explicit ablation.
        b = -cfg.pair_responsibility * common
        candidates.append((distance - pair_safe, 0, j, a, float(b)))

    # Obstacle constraints: controller sees only a local surface return.
    for obstacle_id, rect in enumerate(obstacles):
        surface = closest_point(pos, rect)
        rel_p = pos - surface
        distance = signed_distance(pos, rect)
        if distance > cfg.sense_radius:
            continue
        # Starts are outside.  A tiny outward normal keeps the constraint
        # numerically defined if discrete integration enters the boundary.
        norm = np.linalg.norm(rel_p)
        if norm < 1e-8:
            local_dir = pos - rect.center
            rel_p = local_dir / max(np.linalg.norm(local_dir), 1e-8) * 1e-6
        obstacle_safe = cfg.obstacle_safe_radius_factor * cfg.car_radius
        h = rel_p @ rel_p - obstacle_safe ** 2
        h_dot = 2.0 * rel_p @ vel
        # ``rel_p`` is the vector to the closest *surface*, not to a fixed
        # point obstacle.  Tangential velocity along a flat rectangle face
        # does not increase clearance.  Using 2||v||^2 here would therefore
        # create a false positive drift term and can let a diagonal trajectory
        # cross the face.  The normal component is conservative at corners.
        normal_speed_sq = float((rel_p @ vel) ** 2 / max(rel_p @ rel_p, 1e-12))
        drift = 2.0 * normal_speed_sq
        common = drift + k_d * h_dot + k_p * h
        a = (2.0 / cfg.mass) * rel_p
        b = -common
        candidates.append((distance - cfg.car_radius, 1, obstacle_id, a, float(b)))

    candidates.sort(key=lambda item: item[0])
    selected = (candidates if cfg.max_constraints <= 0
                else candidates[:cfg.max_constraints])
    if not selected:
        return LocalConstraintSet(np.zeros((0, 2)), np.zeros(0),
                                  np.zeros(0, dtype=np.int8), np.zeros(0, dtype=int))
    return LocalConstraintSet(
        np.stack([item[3] for item in selected]),
        np.array([item[4] for item in selected]),
        np.array([item[1] for item in selected], dtype=np.int8),
        np.array([item[2] for item in selected], dtype=int),
    )


def _active_agent_neighbors(state: np.ndarray, u_ref: np.ndarray, cfg: Config) -> np.ndarray:
    """Whether each ego robot has a locally active agent-agent CBF constraint."""
    active = np.zeros(cfg.n_agents, dtype=bool)
    pair_safe = cfg.pair_safe_radius_factor * cfg.car_radius
    k_d = 2.0 * cfg.cbf_zeta * cfg.cbf_omega
    k_p = cfg.cbf_omega ** 2
    for i, j in radius_neighbor_pairs(state[:, :2], cfg.sense_radius):
        rel_p = state[i, :2] - state[j, :2]
        rel_v = state[i, 2:] - state[j, 2:]
        h = rel_p @ rel_p - pair_safe ** 2
        h_dot = 2.0 * rel_p @ rel_v
        common = 2.0 * rel_v @ rel_v + k_d * h_dot + k_p * h
        a_i = (2.0 / cfg.mass) * rel_p
        a_j = -a_i
        b = -cfg.pair_responsibility * common
        if b - a_i @ u_ref[i] > 1e-6 or b - a_j @ u_ref[j] > 1e-6:
            active[i] = active[j] = True
    return active


def solve_qp_diagnostics(u_ref: np.ndarray, a: np.ndarray, b: np.ndarray,
                         cfg: Config) -> QPSolution:
    """Solve the two-dimensional local CBF-QP.

    The hard feasible problem is solved exactly (up to floating point) by
    enumerating projections and vertices of its 2-D feasible polygon.  Only
    when the hard constraints and actuator box are mutually infeasible do we
    fall back to the penalized slack problem.  The decision dimension remains
    two regardless of team size.
    """
    if len(b) == 0:
        control = np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)
        return QPSolution(control, np.zeros(0), np.zeros(0), True,
                          bool(np.all(np.isfinite(control))))

    # C u >= d includes the actuator box, expressed as four half-spaces.
    box_a = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    box_b = np.full(4, -cfg.max_force)
    c = np.vstack((a, box_a))
    d = np.concatenate((b, box_b))
    candidates: list[np.ndarray] = [np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)]

    # Orthogonal projection of u_ref onto each active boundary.
    for normal, offset in zip(c, d):
        denom = float(normal @ normal)
        if denom > 1e-14:
            candidates.append(u_ref + ((offset - normal @ u_ref) / denom) * normal)

    # Every vertex in 2-D is an intersection of two boundary lines.
    for left in range(len(d)):
        for right in range(left + 1, len(d)):
            matrix = np.stack((c[left], c[right]))
            det = float(np.linalg.det(matrix))
            if abs(det) > 1e-12:
                candidates.append(np.linalg.solve(matrix, np.array([d[left], d[right]])))

    feasible = [candidate for candidate in candidates
                if np.all(np.isfinite(candidate))
                and np.all(c @ candidate >= d - 1e-9)]
    if feasible:
        costs = [float(np.sum((candidate - u_ref) ** 2)) for candidate in feasible]
        u = feasible[int(np.argmin(costs))]
        active = np.abs(c @ u - d) <= 2e-8
        all_dual = _recover_nonnegative_dual(c, u - u_ref, active)
        return QPSolution(u, np.maximum(0.0, b - a @ u), all_dual[:len(b)],
                          True, bool(np.all(np.isfinite(all_dual))))

    # Hard infeasibility: deterministic soft fallback with actuator projection.
    u = np.clip(u_ref.copy(), -cfg.max_force, cfg.max_force)
    spectral = float(np.linalg.norm(a, ord=2) ** 2)
    step = 1.0 / (1.0 + cfg.slack_penalty * spectral + 1e-12)
    for _ in range(cfg.qp_iterations):
        violation = np.maximum(0.0, b - a @ u)
        grad = u - u_ref - cfg.slack_penalty * (a.T @ violation)
        nxt = np.clip(u - step * grad, -cfg.max_force, cfg.max_force)
        if np.max(np.abs(nxt - u)) < 1e-9:
            u = nxt
            break
        u = nxt
    slack = np.maximum(0.0, b - a @ u)
    converged = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(slack)))
    dual = cfg.slack_penalty * slack
    return QPSolution(u, slack, dual, False, converged)


def solve_soft_qp(u_ref: np.ndarray, a: np.ndarray, b: np.ndarray,
                  cfg: Config) -> tuple[np.ndarray, np.ndarray, bool]:
    """Backward-compatible wrapper used by older scripts and tests."""
    solution = solve_qp_diagnostics(u_ref, a, b, cfg)
    return solution.control, solution.slack, solution.numerically_valid


def _collision_and_clearance(state: np.ndarray, obstacles: list[Rectangle],
                             cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pos = state[:, :2]
    collision = np.zeros(cfg.n_agents, dtype=bool)
    agent_collision = np.zeros(cfg.n_agents, dtype=bool)
    obstacle_collision = np.zeros(cfg.n_agents, dtype=bool)
    minimum = np.inf
    # Every collision pair lies inside sense_radius.  The grid therefore gives
    # exactly the same collision flags without a global all-pairs scan.
    local_pairs = radius_neighbor_pairs(pos, cfg.sense_radius)
    for i, j in local_pairs:
        clearance = float(np.linalg.norm(pos[i] - pos[j]) - 2.0 * cfg.car_radius)
        minimum = min(minimum, clearance)
        if clearance < 0.0:
            collision[i] = collision[j] = True
            agent_collision[i] = agent_collision[j] = True
    if not local_pairs:
        # Exact minimum is larger than this censoring threshold.  Safety and
        # fallback metrics only need the lower bound in that case.
        minimum = min(minimum, cfg.sense_radius - 2.0 * cfg.car_radius)
    for i in range(cfg.n_agents):
        for rect in obstacles:
            clearance = signed_distance(pos[i], rect) - cfg.car_radius
            minimum = min(minimum, clearance)
            if clearance < 0.0:
                collision[i] = True
                obstacle_collision[i] = True
    return (collision, agent_collision, obstacle_collision,
            float(minimum if np.isfinite(minimum) else cfg.area_size))


def _sbs_metrics(q_nom: np.ndarray, q_safe: np.ndarray,
                 agent_pressure: np.ndarray, obstacle_pressure: np.ndarray,
                 q_without_agents: np.ndarray, q_without_obstacles: np.ndarray,
                 reached: np.ndarray, collisions: np.ndarray,
                 cfg: Config) -> dict:
    """Classify temporal O/A/M-SBS evidence for every robot.

    The local counterfactuals re-solve the same ego QP after removing one
    constraint family.  They are instantaneous attribution evidence, not a
    full rollout-level causal proof; the latter remains a Phase-1 validation.
    """
    w = cfg.stall_window_steps
    if len(q_nom) < w:
        empty = np.zeros(cfg.n_agents, dtype=bool)
        return {"window_fraction": 0.0, "agent_ever": empty, "episode": 0,
                "first_time": -1.0, "final_class": np.zeros(cfg.n_agents, dtype=np.int8),
                "online_class": np.zeros((len(q_nom) + 1, cfg.n_agents), dtype=np.int8),
                "a_ever": empty, "o_ever": empty, "m_ever": empty,
                "a_persistent": empty, "o_persistent": empty, "m_persistent": empty,
                "x_persistent": empty}
    generic_flags, agent_flags, obstacle_flags = [], [], []
    first = None
    for t in range(len(q_nom) - w + 1):
        nominal_progress = q_nom[t:t + w].sum(axis=0) * cfg.dt
        safe_progress = q_safe[t:t + w].sum(axis=0) * cfg.dt
        ratio = safe_progress / np.maximum(nominal_progress, 1e-9)
        remained_safe = ~collisions[t + 1:t + w + 1].any(axis=0)
        not_finished = ~reached[t:t + w + 1].any(axis=0)
        generic = ((nominal_progress > cfg.sbs_nominal_progress_epsilon)
                   & (ratio < cfg.sbs_progress_ratio_threshold)
                   & remained_safe & not_finished)
        agent_relief = ((q_without_agents[t:t + w] - q_safe[t:t + w]).sum(axis=0)
                        * cfg.dt)
        obstacle_relief = ((q_without_obstacles[t:t + w] - q_safe[t:t + w]).sum(axis=0)
                           * cfg.dt)
        agent = (generic
                 & (agent_pressure[t:t + w].mean(axis=0) >= cfg.sbs_active_fraction)
                 & (agent_relief > cfg.sbs_causal_progress_epsilon))
        obstacle = (generic
                    & (obstacle_pressure[t:t + w].mean(axis=0) >= cfg.sbs_active_fraction)
                    & (obstacle_relief > cfg.sbs_causal_progress_epsilon))
        generic_flags.append(generic)
        agent_flags.append(agent)
        obstacle_flags.append(obstacle)
        if first is None and agent.any():
            first = t + w
    generic_a = np.asarray(generic_flags)
    agent_a = np.asarray(agent_flags)
    obstacle_a = np.asarray(obstacle_flags)
    a_ever = agent_a.any(axis=0)
    o_ever = obstacle_a.any(axis=0)
    first_generic = np.argmax(generic_a, axis=0)
    first_agent = np.argmax(agent_a, axis=0)
    first_obstacle = np.argmax(obstacle_a, axis=0)
    first_generic[~generic_a.any(axis=0)] = len(generic_a) + 1
    first_agent[~a_ever] = len(agent_a) + 1
    first_obstacle[~o_ever] = len(obstacle_a) + 1

    final_generic = generic_a[-1]
    final_agent = agent_a[-1]
    final_obstacle = obstacle_a[-1]
    # M-SBS requires obstacle-causal evidence to precede agent-causal evidence.
    # A generic low-progress window alone is not evidence of an obstacle root.
    mixed = final_agent & (first_obstacle < first_agent)
    persistent_a = final_agent & ~mixed & ~final_obstacle
    persistent_o = final_generic & final_obstacle & ~final_agent
    persistent_m = final_generic & mixed
    persistent_x = final_generic & final_agent & final_obstacle & ~mixed
    final_class = np.zeros(cfg.n_agents, dtype=np.int8)
    final_class[persistent_o] = 1  # O-SBS
    final_class[persistent_a] = 2  # A-SBS
    final_class[persistent_m] = 3  # M-SBS: obstacle cause precedes agent cause
    final_class[persistent_x] = 4  # unresolved simultaneous/joint attribution
    # Past-only label for visualization/runtime diagnostics. Window t is
    # displayed at its right endpoint t+w, so final labels never leak backward.
    window_class = np.zeros_like(agent_a, dtype=np.int8)
    mixed_root = first_obstacle < first_agent
    window_m = generic_a & agent_a & mixed_root[None, :]
    window_x = generic_a & agent_a & obstacle_a & ~window_m
    window_a = generic_a & agent_a & ~obstacle_a & ~window_m
    window_o = generic_a & obstacle_a & ~agent_a
    window_class[window_o] = 1
    window_class[window_a] = 2
    window_class[window_m] = 3
    window_class[window_x] = 4
    online_class = np.zeros((len(q_nom) + 1, cfg.n_agents), dtype=np.int8)
    online_class[w:w + len(window_class)] = window_class
    m_ever = a_ever & (first_obstacle < first_agent)
    agent_associated = a_ever
    return {
        "window_fraction": float(agent_a.mean()),
        "agent_ever": agent_associated,
        "episode": int(agent_associated.any()),
        "first_time": -1.0 if first is None else first * cfg.dt,
        "final_class": final_class,
        "online_class": online_class,
        "a_ever": a_ever & ~m_ever,
        "o_ever": o_ever,
        "m_ever": m_ever,
        "a_persistent": persistent_a,
        "o_persistent": persistent_o,
        "m_persistent": persistent_m,
        "x_persistent": persistent_x,
    }


def run_episode(cfg: Config) -> EpisodeResult:
    starts, goals, obstacles = generate_scene(cfg)
    gain = _dare_gain(cfg.dt, cfg.mass)
    state = np.column_stack((starts, np.zeros((cfg.n_agents, 2))))

    positions = [state[:, :2].copy()]
    states = [state.copy()]
    distances = [np.linalg.norm(state[:, :2] - goals, axis=1)]
    reached_history = [distances[0] < cfg.reach_radius]
    collision_ever = np.zeros(cfg.n_agents, dtype=bool)
    agent_collision_ever = np.zeros(cfg.n_agents, dtype=bool)
    obstacle_collision_ever = np.zeros(cfg.n_agents, dtype=bool)
    nominal_history, intervention_history, slack_history = [], [], []
    q_nominal_history, q_safe_history = [], []
    agent_pressure_history, obstacle_pressure_history = [], []
    agent_correction_history, agent_load_history, cancellation_history = [], [], []
    qp_hard_feasible_history = []
    local_agent_degree_history, local_constraint_count_history = [], []
    q_without_agents_history, q_without_obstacles_history = [], []
    collision_history = [np.zeros(cfg.n_agents, dtype=bool)]
    qp_failures = 0
    min_clearance = np.inf
    active_constraints = 0
    still_unfinished_steps = 0
    goal_dwell_steps = 0
    termination_reason = "max_steps"

    for _ in range(cfg.steps):
        u_ref = nominal_lqr(state, goals, gain, cfg)
        goal_delta = goals - state[:, :2]
        goal_dir = goal_delta / np.maximum(np.linalg.norm(goal_delta, axis=1, keepdims=True), 1e-9)
        v_nominal_next = np.clip(state[:, 2:] + u_ref / cfg.mass * cfg.dt,
                                 -cfg.max_speed, cfg.max_speed)
        q_nominal_history.append(np.maximum(0.0, np.sum(goal_dir * v_nominal_next, axis=1)))
        action = u_ref.copy()
        step_slack = np.zeros(cfg.n_agents)
        step_agent_pressure = np.zeros(cfg.n_agents, dtype=bool)
        step_obstacle_pressure = np.zeros(cfg.n_agents, dtype=bool)
        step_agent_correction = np.zeros((cfg.n_agents, 2))
        step_agent_load = np.zeros(cfg.n_agents)
        step_cancellation = np.zeros(cfg.n_agents)
        step_hard_feasible = np.ones(cfg.n_agents, dtype=bool)
        step_agent_degree = np.zeros(cfg.n_agents, dtype=np.int16)
        step_constraint_count = np.zeros(cfg.n_agents, dtype=np.int16)
        action_without_agents = u_ref.copy()
        action_without_obstacles = u_ref.copy()
        if cfg.controller == "cbf_qp":
            local_neighbors = radius_neighbor_lists(state[:, :2], cfg.sense_radius)
            for i in range(cfg.n_agents):
                constraints = _local_constraints(
                    i, state, obstacles, cfg, agent_neighbors=local_neighbors[i])
                a, b = constraints.a, constraints.b
                active_constraints += len(b)
                step_constraint_count[i] = len(b)
                step_agent_degree[i] = int(np.count_nonzero(constraints.kinds == 0))
                solution = solve_qp_diagnostics(u_ref[i], a, b, cfg)
                action[i], slack, ok = (solution.control, solution.slack,
                                        solution.numerically_valid)
                step_hard_feasible[i] = solution.hard_feasible
                step_slack[i] = float(np.max(slack, initial=0.0))
                qp_failures += int(not ok)
                if len(b):
                    influential = solution.dual > cfg.dual_activity_threshold
                    step_agent_pressure[i] = bool(np.any(influential & (constraints.kinds == 0)))
                    step_obstacle_pressure[i] = bool(np.any(influential & (constraints.kinds == 1)))

                    aa = constraints.kinds == 0
                    if np.any(aa):
                        per_constraint = solution.dual[aa, None] * a[aa]
                        # KKT total/cancellation diagnostics use every active
                        # AA row.  ``influence_top_k`` is reserved for the
                        # future edge encoder input and must not under-count a
                        # crowd of individually moderate constraints.
                        correction = per_constraint.sum(axis=0)
                        load = float(np.linalg.norm(per_constraint, axis=1).sum())
                        step_agent_correction[i] = correction
                        step_agent_load[i] = load
                        step_cancellation[i] = (0.0 if load <= 1e-12 else
                                                float(np.clip(1.0 - np.linalg.norm(correction) / load,
                                                              0.0, 1.0)))

                    keep_obstacles = constraints.kinds == 1
                    keep_agents = constraints.kinds == 0
                    if np.any(keep_obstacles):
                        action_without_agents[i], _, ok_oa = solve_soft_qp(
                            u_ref[i], a[keep_obstacles], b[keep_obstacles], cfg)
                    else:
                        action_without_agents[i], ok_oa = u_ref[i], True
                    if np.all(keep_agents):
                        # Removing a nonexistent obstacle family is exactly the
                        # original solved action; do not solve the same QP twice.
                        action_without_obstacles[i], ok_oo = action[i], True
                    elif np.any(keep_agents):
                        action_without_obstacles[i], _, ok_oo = solve_soft_qp(
                            u_ref[i], a[keep_agents], b[keep_agents], cfg)
                    else:
                        action_without_obstacles[i], ok_oo = u_ref[i], True
                    qp_failures += int(not ok_oa) + int(not ok_oo)
        elif cfg.controller != "nominal":
            raise ValueError(f"Unknown controller: {cfg.controller}")

        v_without_agents = np.clip(
            state[:, 2:] + action_without_agents / cfg.mass * cfg.dt,
            -cfg.max_speed, cfg.max_speed)
        v_without_obstacles = np.clip(
            state[:, 2:] + action_without_obstacles / cfg.mass * cfg.dt,
            -cfg.max_speed, cfg.max_speed)
        q_without_agents_history.append(
            np.maximum(0.0, np.sum(goal_dir * v_without_agents, axis=1)))
        q_without_obstacles_history.append(
            np.maximum(0.0, np.sum(goal_dir * v_without_obstacles, axis=1)))
        agent_pressure_history.append(step_agent_pressure)
        obstacle_pressure_history.append(step_obstacle_pressure)
        agent_correction_history.append(step_agent_correction)
        agent_load_history.append(step_agent_load)
        cancellation_history.append(step_cancellation)
        qp_hard_feasible_history.append(step_hard_feasible)
        local_agent_degree_history.append(step_agent_degree)
        local_constraint_count_history.append(step_constraint_count)

        action = np.clip(action, -cfg.max_force, cfg.max_force)
        nominal_history.append(np.linalg.norm(u_ref, axis=1))
        intervention_history.append(np.linalg.norm(action - u_ref, axis=1))
        slack_history.append(step_slack)

        # GCBF+ DoubleIntegrator uses forward Euler and clips velocity.
        state[:, :2] += state[:, 2:] * cfg.dt
        state[:, 2:] += action / cfg.mass * cfg.dt
        state[:, 2:] = np.clip(state[:, 2:], -cfg.max_speed, cfg.max_speed)

        collision, agent_collision, obstacle_collision, clearance = _collision_and_clearance(
            state, obstacles, cfg)
        collision_ever |= collision
        agent_collision_ever |= agent_collision
        obstacle_collision_ever |= obstacle_collision
        collision_history.append(collision.copy())
        min_clearance = min(min_clearance, clearance)
        positions.append(state[:, :2].copy())
        states.append(state.copy())
        distances.append(np.linalg.norm(state[:, :2] - goals, axis=1))
        # Task completion is absorbing for liveness accounting even though the
        # physical trajectory is allowed to continue for a fixed-horizon audit.
        reached_history.append(reached_history[-1] | (distances[-1] < cfg.reach_radius))
        new_goal_delta = goals - state[:, :2]
        new_goal_dir = new_goal_delta / np.maximum(np.linalg.norm(new_goal_delta, axis=1, keepdims=True), 1e-9)
        q_safe_history.append(np.maximum(0.0, np.sum(new_goal_dir * state[:, 2:], axis=1)))

        if cfg.adaptive_stop and len(nominal_history) >= cfg.adaptive_min_steps:
            current_distance = np.linalg.norm(state[:, :2] - goals, axis=1)
            current_speed = np.linalg.norm(state[:, 2:], axis=1)
            dwelling = ((current_distance < cfg.reach_radius)
                        & (current_speed < cfg.adaptive_goal_speed_epsilon))
            goal_dwell_steps = goal_dwell_steps + 1 if dwelling.all() else 0
            if goal_dwell_steps >= cfg.adaptive_goal_dwell_steps:
                termination_reason = "all_goal_dwell"
                break
            unfinished = current_distance >= cfg.reach_radius
            unfinished_moving = np.linalg.norm(
                state[unfinished, 2:], axis=1) > cfg.adaptive_speed_epsilon
            # If all robots are currently inside the goal balls but have not
            # slowed enough to dwell, they are not an unresolved still case.
            still_unfinished_steps = (0 if not unfinished.any() or unfinished_moving.any()
                                      else still_unfinished_steps + 1)
            if still_unfinished_steps >= cfg.adaptive_still_steps:
                termination_reason = "unresolved_still"
                break

    positions_a = np.asarray(positions)
    distances_a = np.asarray(distances)
    reached_a = np.asarray(reached_history)
    nominal_a = np.asarray(nominal_history)
    interventions_a = np.asarray(intervention_history)
    slacks_a = np.asarray(slack_history)
    reached_ever = reached_a.any(axis=0)
    actual_steps = len(nominal_history)
    initial_distance = distances_a[0]
    progress = np.clip((initial_distance - distances_a[-1]) / np.maximum(initial_distance, 1e-9), -1.0, 1.0)
    attribution = _sbs_metrics(
        np.asarray(q_nominal_history), np.asarray(q_safe_history),
        np.asarray(agent_pressure_history), np.asarray(obstacle_pressure_history),
        np.asarray(q_without_agents_history), np.asarray(q_without_obstacles_history),
        reached_a, np.asarray(collision_history), cfg)
    sbs_window_fraction = attribution["window_fraction"]
    sbs_ever = attribution["agent_ever"]
    sbs_episode = attribution["episode"]
    sbs_first_time = attribution["first_time"]

    row = {
        **asdict(cfg),
        "actual_steps": actual_steps,
        "actual_seconds": actual_steps * cfg.dt,
        "termination_reason": termination_reason,
        "safe_agent_rate": float((~collision_ever).mean()),
        "collision_free_episode": int(not collision_ever.any()),
        "agent_collision_rate": float(agent_collision_ever.mean()),
        "obstacle_collision_rate": float(obstacle_collision_ever.mean()),
        "reach_agent_rate": float(reached_ever.mean()),
        "all_reached": int(reached_ever.all()),
        "final_goal_agent_rate": float((distances_a[-1] < cfg.reach_radius).mean()),
        "all_goal_dwell": int(termination_reason == "all_goal_dwell"),
        "success_agent_rate": float((reached_ever & ~collision_ever).mean()),
        "mean_normalized_progress": float(progress.mean()),
        "stall_window_fraction": sbs_window_fraction,
        "sbs_window_fraction": sbs_window_fraction,
        "sbs_agent_trigger_rate": float(sbs_ever.mean()),
        "sbs_episode_triggered": sbs_episode,
        "sbs_first_trigger_time": sbs_first_time,
        "a_sbs_ever_agent_rate": float(attribution["a_ever"].mean()),
        "o_sbs_ever_agent_rate": float(attribution["o_ever"].mean()),
        "m_sbs_ever_agent_rate": float(attribution["m_ever"].mean()),
        "a_sbs_persistent_agent_rate": float(attribution["a_persistent"].mean()),
        "o_sbs_persistent_agent_rate": float(attribution["o_persistent"].mean()),
        "m_sbs_persistent_agent_rate": float(attribution["m_persistent"].mean()),
        "x_sbs_persistent_agent_rate": float(attribution["x_persistent"].mean()),
        "a_sbs_persistent_episode": int(attribution["a_persistent"].any()),
        "o_sbs_persistent_episode": int(attribution["o_persistent"].any()),
        "m_sbs_persistent_episode": int(attribution["m_persistent"].any()),
        "x_sbs_persistent_episode": int(attribution["x_persistent"].any()),
        "agent_pressure_fraction": float(np.asarray(agent_pressure_history).mean()),
        "obstacle_pressure_fraction": float(np.asarray(obstacle_pressure_history).mean()),
        "mean_intervention": float(interventions_a.mean()),
        "p95_intervention": float(np.quantile(interventions_a, 0.95)),
        "mean_max_slack": float(slacks_a.mean()),
        "max_slack": float(slacks_a.max(initial=0.0)),
        "qp_failure_count": qp_failures,
        "qp_hard_infeasible_count": int((~np.asarray(qp_hard_feasible_history)).sum()),
        "mean_local_constraints": float(active_constraints / (actual_steps * cfg.n_agents)),
        "mean_local_agent_degree": float(np.asarray(local_agent_degree_history).mean()),
        "p95_local_agent_degree": float(np.quantile(local_agent_degree_history, 0.95)),
        "max_local_agent_degree": int(np.max(local_agent_degree_history, initial=0)),
        "minimum_clearance": float(min_clearance),
    }
    states_a = np.asarray(states)
    return EpisodeResult(
        row=row, positions=positions_a, goals=goals, obstacles=obstacles,
        reached_ever=reached_ever, collision_ever=collision_ever,
        agent_collision_ever=agent_collision_ever,
        obstacle_collision_ever=obstacle_collision_ever,
        interventions=interventions_a, slacks=slacks_a, sbs_ever=sbs_ever,
        sbs_final_class=attribution["final_class"],
        sbs_online_class=attribution["online_class"],
        agent_constraint_pressure=np.asarray(agent_pressure_history),
        obstacle_constraint_pressure=np.asarray(obstacle_pressure_history),
        states=states_a,
        agent_correction=np.asarray(agent_correction_history),
        agent_constraint_load=np.asarray(agent_load_history),
        agent_cancellation_index=np.asarray(cancellation_history),
        qp_hard_feasible=np.asarray(qp_hard_feasible_history),
        local_agent_degree=np.asarray(local_agent_degree_history),
        local_constraint_count=np.asarray(local_constraint_count_history))
