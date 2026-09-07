"""Component-local adaptive CEM Teacher with auditable search provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

import jax.numpy as jnp

from ..simulation import Config, nominal_lqr
from .intent import decode_world_velocity, rate_limit, tracking_force
from .jax_rollout import batched_candidate_costs
from .protocol import IntentMode, SyncEventProtocol
from .runtime import PreparedStep, RuntimeState
from .sampled_safety import sampled_wang_step


@dataclass(frozen=True)
class TeacherBudget:
    """Development/search-diagnostic budget.

    This flexible budget deliberately remains available for coverage and
    numerical diagnostics.  It is not an authoritative data-Teacher contract.
    Formal labels must use :class:`AuthoritativeComponentTeacher`.
    """
    strong_samples: int = 12
    strong_iterations: int = 3
    light_samples: int = 6
    light_iterations: int = 1
    horizon_steps: int = 40
    max_reuse_ticks: int = 3
    elite_fraction: float = 0.25
    covariance_floor_alpha: float = 0.04
    covariance_floor_beta: float = 0.05
    new_std_alpha: float = 0.34
    new_std_beta: float = 0.44
    tie_fraction: float = 0.01
    exact_shortlist_size: int = 32
    jax_evaluation_batch_size: int = 256
    planning_integration_substeps: int = 32
    candidate_backend: str = "jax_x64"


@dataclass(frozen=True)
class AuthoritativeTeacherBudget:
    """Non-negotiable budget floor for formal v2 Teacher labels."""

    samples: int = 4096
    iterations: int = 3
    restarts: int = 4
    horizon_steps: int = 40
    elite_fraction: float = 0.25
    tie_fraction: float = 0.01
    exact_shortlist_size: int = 32
    jax_evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        if self.samples < 4096:
            raise ValueError("authoritative Teacher requires >=4096 samples")
        if self.iterations < 3:
            raise ValueError("authoritative Teacher requires >=3 iterations")
        if self.restarts < 4:
            raise ValueError("authoritative Teacher requires >=4 restarts")
        if self.horizon_steps != 40:
            raise ValueError("authoritative Teacher requires the 40-step horizon")

    def search_budget(self, protocol: SyncEventProtocol) -> TeacherBudget:
        """Build the internal cold-search budget without a light path."""
        return TeacherBudget(
            strong_samples=self.samples,
            strong_iterations=self.iterations,
            light_samples=self.samples,
            light_iterations=self.iterations,
            horizon_steps=self.horizon_steps,
            max_reuse_ticks=0,
            elite_fraction=self.elite_fraction,
            tie_fraction=self.tie_fraction,
            exact_shortlist_size=self.exact_shortlist_size,
            jax_evaluation_batch_size=self.jax_evaluation_batch_size,
            planning_integration_substeps=protocol.integration_substeps,
            candidate_backend="jax_x64")


@dataclass(frozen=True)
class TeacherFeedback:
    certified_brake_fraction: float = 0.0
    out_of_certificate_pair_samples: int = 0
    progress_ratio: float = 1.0

    @property
    def requires_search(self) -> bool:
        # All feedback fields are diagnostics.  Conflict naturally lowers
        # progress, while an actually out-of-certificate runtime state is an
        # invalid rollout that CEM cannot retroactively repair.  Search timing
        # comes only from component changes and the periodic cache scheduler.
        return False

    @property
    def requires_strong(self) -> bool:
        return False


@dataclass
class BranchDistribution:
    members: tuple[int, ...]
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ComponentCache:
    members: tuple[int, ...]
    positive: BranchDistribution
    negative: BranchDistribution
    last_parameters: np.ndarray
    ticks_since_search: int = 0


@dataclass(frozen=True)
class TeacherDecision:
    parameters: np.ndarray
    source: np.ndarray
    teacher_cost: np.ndarray
    incumbent_cost: np.ndarray
    validated: np.ndarray
    component_id: np.ndarray
    positive_branch_cost: np.ndarray
    negative_branch_cost: np.ndarray
    branch_near_tie: np.ndarray


class ComponentCEMTeacher:
    """Flexible development CEM; never a formal data Teacher by itself.

    Reuse performs no rollout.  Full horizon evaluations occur only on search
    ticks, so reported rollout counts equal the actual expensive calls.
    """

    def __init__(self, protocol: SyncEventProtocol,
                 budget: TeacherBudget | None = None, seed: int = 0,
                 *, audit_exact_candidates: bool = False):
        self.protocol = protocol
        self.budget = budget or TeacherBudget()
        self.rng = np.random.default_rng(seed)
        self.audit_exact_candidates = bool(audit_exact_candidates)
        self.search_audit: list[dict] = []
        self.audit_exact_rollouts = 0
        self.caches: list[ComponentCache] = []
        self.calls = 0
        self.strong_searches = 0
        self.light_searches = 0
        self.reuse_decisions = 0
        self.objective_rollouts = 0
        self.candidate_rollouts = 0
        self.approximate_rollouts = 0
        self.exact_validation_rollouts = 0
        self.simulated_control_steps = 0
        self.simulated_safety_qp_solves = 0
        self.simulated_integration_samples = 0

    @staticmethod
    def _active_components(prepared: PreparedStep) -> list[tuple[int, ...]]:
        active = set(np.flatnonzero(
            prepared.mode == IntentMode.ACTIVE).astype(int).tolist())
        result: list[tuple[int, ...]] = []
        assigned: set[int] = set()
        for component in prepared.trigger.components:
            members = tuple(sorted(active.intersection(component)))
            if members:
                result.append(members)
                assigned.update(members)
        result.extend((agent,) for agent in sorted(active - assigned))
        return result

    def _new_branch(self, members: tuple[int, ...], previous: np.ndarray,
                    sign: float) -> BranchDistribution:
        ids = np.asarray(members, dtype=int)
        mean = np.empty((len(ids), 2), dtype=float)
        mean[:, 0] = previous[ids, 0]
        mean[:, 1] = np.abs(previous[ids, 1])
        # A branch stores beta magnitude; sign is applied to candidates.
        mean[:, 1] = np.maximum(mean[:, 1], 0.08)
        std = np.tile(
            [self.budget.new_std_alpha, self.budget.new_std_beta],
            (len(ids), 1)).astype(float)
        _ = sign
        return BranchDistribution(members, mean, std)

    def _project_branch(self, old: BranchDistribution,
                        members: tuple[int, ...], previous: np.ndarray,
                        sign: float) -> BranchDistribution:
        branch = self._new_branch(members, previous, sign)
        old_index = {agent: index for index, agent in enumerate(old.members)}
        for new_index, agent in enumerate(members):
            if agent in old_index:
                branch.mean[new_index] = old.mean[old_index[agent]]
                branch.std[new_index] = old.std[old_index[agent]]
        return branch

    def _match_caches(self, components: list[tuple[int, ...]],
                      previous: np.ndarray
                      ) -> list[tuple[ComponentCache, str]]:
        unused = set(range(len(self.caches)))
        matched: list[tuple[ComponentCache, str]] = []
        for members in components:
            candidates = []
            current = set(members)
            for index in unused:
                old = set(self.caches[index].members)
                overlap = len(current & old)
                union = len(current | old)
                if overlap:
                    candidates.append((overlap / union, overlap, -index, index))
            if candidates:
                old_index = max(candidates)[-1]
                old = self.caches[old_index]
                unused.remove(old_index)
                exact = old.members == members
                overlap_fraction = len(current & set(old.members)) / max(
                    len(current), len(old.members))
                last_parameters = previous[np.asarray(members)].copy()
                old_parameter_index = {
                    agent: index for index, agent in enumerate(old.members)}
                for new_index, agent in enumerate(members):
                    if agent in old_parameter_index:
                        last_parameters[new_index] = old.last_parameters[
                            old_parameter_index[agent]]
                cache = ComponentCache(
                    members=members,
                    positive=self._project_branch(
                        old.positive, members, previous, +1.0),
                    negative=self._project_branch(
                        old.negative, members, previous, -1.0),
                    last_parameters=last_parameters,
                    ticks_since_search=(old.ticks_since_search if exact else 0),
                )
                status = ("exact" if exact else
                          ("minor" if overlap_fraction >= 0.5 else "major"))
                matched.append((cache, status))
            else:
                ids = np.asarray(members, dtype=int)
                matched.append((ComponentCache(
                    members=members,
                    positive=self._new_branch(members, previous, +1.0),
                    negative=self._new_branch(members, previous, -1.0),
                    last_parameters=previous[ids].copy(),
                ), "new"))
        return matched

    @staticmethod
    def _minimum_center(state: np.ndarray) -> float:
        if len(state) < 2:
            return float("inf")
        delta = state[:, None, :2] - state[None, :, :2]
        distance = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(distance, np.inf)
        return float(distance.min())

    @staticmethod
    def _branch_costs_near_tie(incumbent_cost: float,
                               positive_cost: float,
                               negative_cost: float,
                               tie_fraction: float) -> bool:
        """Compare branch loss against the useful improvement being recovered."""
        best = min(positive_cost, negative_cost)
        improvement_scale = max(abs(incumbent_cost - best), 1.0)
        return bool(
            abs(positive_cost - negative_cost)
            <= tie_fraction * improvement_scale)

    def _rollout_cost(self, runtime: RuntimeState, goals: np.ndarray,
                      gain: np.ndarray, cfg: Config,
                      parameters: np.ndarray, active_ids: np.ndarray,
                      release_ids: np.ndarray, focus_ids: np.ndarray,
                      *, integration_substeps: int | None = None) -> float:
        """Run the exact v2 decoder/tracker/Wang stack for one candidate."""
        self.objective_rollouts += 1
        substeps = (self.protocol.integration_substeps
                    if integration_substeps is None else
                    int(integration_substeps))
        if substeps < 1:
            raise ValueError("Teacher integration_substeps must be positive")
        if substeps == self.protocol.integration_substeps:
            self.exact_validation_rollouts += 1
        else:
            self.approximate_rollouts += 1
        self.simulated_control_steps += self.budget.horizon_steps
        self.simulated_safety_qp_solves += self.budget.horizon_steps
        self.simulated_integration_samples += (
            self.budget.horizon_steps * substeps)
        state = runtime.physical.copy()
        world = runtime.world_intent.copy()
        latch = runtime.braking_latch.copy()
        previous_force = runtime.previous_safe_force.copy()
        initial_distance = np.linalg.norm(
            state[focus_ids, :2] - goals[focus_ids], axis=1)
        distance_integral = 0.0
        brake = 0
        outside = 0
        minimum_center = self._minimum_center(state)
        active_or_release = np.zeros(cfg.n_agents, dtype=bool)
        active_or_release[active_ids] = True
        active_or_release[release_ids] = True
        for horizon_step in range(self.budget.horizon_steps):
            base = nominal_lqr(state, goals, gain, cfg)
            base_velocity = np.clip(
                state[:, 2:] + base / cfg.mass * cfg.dt,
                -cfg.max_speed, cfg.max_speed)
            target = base_velocity.copy()
            if len(active_ids):
                decoded = decode_world_velocity(
                    state, goals, base_velocity, parameters, self.protocol)
                target[active_ids] = decoded[active_ids]
            world[~active_or_release] = base_velocity[~active_or_release]
            if np.any(active_or_release):
                world[active_or_release] = rate_limit(
                    world[active_or_release], target[active_or_release], cfg,
                    self.protocol)
            reference_force = tracking_force(
                state, gain, cfg, np.flatnonzero(active_or_release), world,
                self.protocol, base)
            if self.protocol.action_smooth_weight > 0.0 and (
                    runtime.step > 0 or horizon_step > 0):
                weight = self.protocol.action_smooth_weight
                reference_force = ((reference_force + weight * previous_force)
                                   / (1.0 + weight))
            safety = sampled_wang_step(
                state, reference_force, [], cfg, latch,
                integration_substeps=substeps)
            brake += len(safety.certified_brake_agents)
            outside += safety.out_of_certificate_pair_samples
            state = safety.next_state
            minimum_center = min(
                minimum_center, safety.minimum_center_distance)
            previous_force = safety.action.copy()
            distance_integral += float(np.linalg.norm(
                state[focus_ids, :2] - goals[focus_ids], axis=1).sum()) * cfg.dt

        final_distance = np.linalg.norm(
            state[focus_ids, :2] - goals[focus_ids], axis=1)
        progress = float((initial_distance - final_distance).sum())
        focus_params = parameters[focus_ids]
        previous_params = runtime.parameters[focus_ids]
        effort = float(np.sum(
            (focus_params[:, 0] - 1.0) ** 2
            + 0.15 * focus_params[:, 1] ** 2))
        change = float(np.sum((focus_params - previous_params) ** 2))
        public_violation = max(
            0.0, self.protocol.hard_center_distance - minimum_center)
        return (1e9 * float(public_violation > 0.0)
                + 2e7 * outside
                + 400.0 * brake
                + 120.0 * float(final_distance.sum())
                + 3.0 * distance_integral
                - 180.0 * progress
                + 0.05 * effort
                + 0.5 * change
                + 1e8 * public_violation)

    def _jax_candidate_costs(
            self, runtime: RuntimeState, goals: np.ndarray,
            gain: np.ndarray, cfg: Config, candidates: np.ndarray,
            active_ids: np.ndarray, release_ids: np.ndarray,
            focus_ids: np.ndarray, *, integration_substeps: int,
            candidate_rollouts: bool) -> np.ndarray:
        """Evaluate a candidate batch with the x64 JAX runtime objective."""
        candidates = np.asarray(candidates, dtype=float)
        if candidates.ndim != 3 or candidates.shape[1:] != (cfg.n_agents, 2):
            raise ValueError(
                "candidates must have shape (batch, n_agents, 2)")
        if len(candidates) == 0:
            raise ValueError("candidate batch must not be empty")
        substeps = int(integration_substeps)
        if substeps < 1:
            raise ValueError("Teacher integration_substeps must be positive")
        active_mask = np.zeros(cfg.n_agents, dtype=bool)
        release_mask = np.zeros(cfg.n_agents, dtype=bool)
        focus_mask = np.zeros(cfg.n_agents, dtype=bool)
        active_mask[np.asarray(active_ids, dtype=int)] = True
        release_mask[np.asarray(release_ids, dtype=int)] = True
        focus_mask[np.asarray(focus_ids, dtype=int)] = True
        batch_size = int(self.budget.jax_evaluation_batch_size)
        if batch_size < 1:
            raise ValueError("jax_evaluation_batch_size must be positive")
        device_costs = []
        for start in range(0, len(candidates), batch_size):
            chunk = candidates[start:start + batch_size]
            logical_size = len(chunk)
            if logical_size < batch_size:
                chunk = np.concatenate((
                    chunk,
                    np.repeat(chunk[-1:], batch_size - logical_size, axis=0),
                ))
            chunk_costs = batched_candidate_costs(
                jnp.asarray(runtime.physical),
                jnp.asarray(runtime.world_intent),
                jnp.asarray(runtime.braking_latch),
                jnp.asarray(runtime.previous_safe_force),
                jnp.asarray(chunk), jnp.asarray(runtime.parameters),
                jnp.asarray(goals), jnp.asarray(gain),
                jnp.asarray(active_mask), jnp.asarray(release_mask),
                jnp.asarray(focus_mask), n_agents=cfg.n_agents,
                horizon_steps=self.budget.horizon_steps,
                integration_substeps=substeps,
                runtime_step=runtime.step, dt=cfg.dt, mass=cfg.mass,
                sense_radius=cfg.sense_radius,
                max_force=cfg.max_force, max_speed=cfg.max_speed,
                safe_distance=(
                    cfg.wang_pair_safe_radius_factor * cfg.car_radius),
                hard_distance=self.protocol.hard_center_distance,
                wang_gamma=cfg.wang_gamma,
                lateral_speed=self.protocol.lateral_speed,
                max_intent_accel=self.protocol.max_intent_accel,
                intent_lookahead=self.protocol.intent_lookahead,
                action_smooth_weight=self.protocol.action_smooth_weight)
            device_costs.append(chunk_costs[:logical_size])
        costs = np.asarray(jnp.concatenate(device_costs))
        rollouts = len(candidates)
        self.objective_rollouts += rollouts
        if candidate_rollouts:
            self.candidate_rollouts += rollouts
        if substeps == self.protocol.integration_substeps:
            self.exact_validation_rollouts += rollouts
        else:
            self.approximate_rollouts += rollouts
        self.simulated_control_steps += rollouts * self.budget.horizon_steps
        self.simulated_safety_qp_solves += rollouts * self.budget.horizon_steps
        self.simulated_integration_samples += (
            rollouts * self.budget.horizon_steps * substeps)
        return costs

    def evaluate_candidates_exact_jax(
            self, runtime: RuntimeState, goals: np.ndarray,
            gain: np.ndarray, cfg: Config, candidates: np.ndarray,
            active_ids: np.ndarray, release_ids: np.ndarray,
            focus_ids: np.ndarray) -> np.ndarray:
        """Evaluate final candidates with deployed 32-substep dynamics on GPU."""
        self.protocol.validate(cfg)
        if self.budget.candidate_backend != "jax_x64":
            raise ValueError("exact JAX evaluation requires jax_x64 backend")
        return self._jax_candidate_costs(
            runtime, goals, gain, cfg, candidates, active_ids, release_ids,
            focus_ids,
            integration_substeps=self.protocol.integration_substeps,
            candidate_rollouts=False)

    def _search_branch(self, cache: BranchDistribution, sign: float,
                       runtime: RuntimeState, prepared: PreparedStep,
                       goals: np.ndarray, gain: np.ndarray, cfg: Config,
                       working: np.ndarray, active_ids: np.ndarray,
                       release_ids: np.ndarray, *, strong: bool
                       ) -> tuple[np.ndarray, float]:
        samples = (self.budget.strong_samples if strong else
                   self.budget.light_samples)
        iterations = (self.budget.strong_iterations if strong else
                      self.budget.light_iterations)
        elite_count = max(2, int(round(self.budget.elite_fraction * samples)))
        ids = np.asarray(cache.members, dtype=int)
        best = working.copy()
        best_cost = float("inf")
        mean, std = cache.mean.copy(), cache.std.copy()
        exact_shortlist = []
        for iteration in range(iterations):
            latent = self.rng.normal(
                mean, std, size=(samples, len(ids), 2))
            latent[..., 0] = np.clip(latent[..., 0], -1.0, 1.0)
            latent[..., 1] = np.clip(latent[..., 1], 0.0, 1.0)
            # Always evaluate the warm-start mean as one candidate.
            latent[0] = mean
            candidates = np.repeat(working[None], samples, axis=0)
            candidates[:, ids, 0] = latent[:, :, 0]
            candidates[:, ids, 1] = sign * latent[:, :, 1]
            if self.budget.candidate_backend == "jax_x64":
                costs = self._jax_candidate_costs(
                    runtime, goals, gain, cfg, candidates, active_ids,
                    release_ids, ids,
                    integration_substeps=(
                        self.budget.planning_integration_substeps),
                    candidate_rollouts=True)
            elif self.budget.candidate_backend == "numpy":
                costs = np.asarray([
                    self._rollout_cost(
                        runtime, goals, gain, cfg, candidate, active_ids,
                        release_ids, ids,
                        integration_substeps=(
                            self.budget.planning_integration_substeps))
                    for candidate in candidates
                ])
                self.candidate_rollouts += samples
            else:
                raise ValueError(
                    f"unknown candidate backend: "
                    f"{self.budget.candidate_backend}")
            if self.audit_exact_candidates:
                exact_costs = np.asarray([
                    self._rollout_cost(
                        runtime, goals, gain, cfg, candidate, active_ids,
                        release_ids, ids,
                        integration_substeps=self.protocol.integration_substeps)
                    for candidate in candidates
                ])
                self.audit_exact_rollouts += samples
                approximate_order = np.argsort(costs, kind="stable")
                exact_order = np.argsort(exact_costs, kind="stable")
                approximate_ranks = np.empty(samples, dtype=int)
                exact_ranks = np.empty(samples, dtype=int)
                approximate_ranks[approximate_order] = np.arange(samples)
                exact_ranks[exact_order] = np.arange(samples)
                if (np.std(approximate_ranks) > 0.0
                        and np.std(exact_ranks) > 0.0):
                    rank_correlation = float(np.corrcoef(
                        approximate_ranks, exact_ranks)[0, 1])
                else:
                    rank_correlation = None
                approximate_best = int(approximate_order[0])
                exact_best = int(exact_order[0])
                exact_best_cost = float(exact_costs[exact_best])
                self.search_audit.append({
                    "members": list(cache.members),
                    "branch_sign": int(np.sign(sign)),
                    "iteration": iteration,
                    "strong": bool(strong),
                    "approximate_costs": costs.tolist(),
                    "exact_costs": exact_costs.tolist(),
                    "member_parameters": candidates[:, ids].tolist(),
                    "rank_correlation": rank_correlation,
                    "approximate_best_index": approximate_best,
                    "exact_best_index": exact_best,
                    "exact_rank_of_approximate_best": int(
                        exact_ranks[approximate_best]),
                    "approximate_rank_of_exact_best": int(
                        approximate_ranks[exact_best]),
                    "approximate_best_exact_regret": float(
                        (exact_costs[approximate_best] - exact_best_cost)
                        / max(abs(exact_best_cost), 1.0)),
                })
            order = np.argsort(costs)
            if self.budget.candidate_backend == "jax_x64":
                keep = min(self.budget.exact_shortlist_size, samples)
                exact_shortlist.extend(candidates[order[:keep]].copy())
            if costs[order[0]] < best_cost:
                best_cost = float(costs[order[0]])
                best = working.copy()
                selected = latent[order[0]]
                best[ids, 0] = selected[:, 0]
                best[ids, 1] = sign * selected[:, 1]
            elite = latent[order[:elite_count]]
            mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
            std = 0.25 * std + 0.75 * elite.std(axis=0)
            std[:, 0] = np.maximum(std[:, 0],
                                   self.budget.covariance_floor_alpha)
            std[:, 1] = np.maximum(std[:, 1],
                                   self.budget.covariance_floor_beta)
        if exact_shortlist:
            shortlist = np.asarray(exact_shortlist)
            exact_costs = self.evaluate_candidates_exact_jax(
                runtime, goals, gain, cfg, shortlist, active_ids,
                release_ids, ids)
            exact_best = int(np.argmin(exact_costs))
            best = shortlist[exact_best].copy()
            best_cost = float(exact_costs[exact_best])
        cache.mean, cache.std = mean, std
        return best, best_cost

    def decide(self, runtime: RuntimeState, prepared: PreparedStep,
               goals: np.ndarray, gain: np.ndarray, cfg: Config,
               feedback: TeacherFeedback | None = None) -> TeacherDecision:
        self.protocol.validate(cfg)
        self.calls += 1
        feedback = feedback or TeacherFeedback()
        working = runtime.parameters.copy()
        active_ids = np.flatnonzero(prepared.mode == IntentMode.ACTIVE)
        release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
        source = np.full(cfg.n_agents, "bypass", dtype="U8")
        cost = np.full(cfg.n_agents, np.nan)
        incumbent = np.full(cfg.n_agents, np.nan)
        validated = np.zeros(cfg.n_agents, dtype=bool)
        component_id = np.full(cfg.n_agents, -1, dtype=np.int32)
        positive_branch_cost = np.full(cfg.n_agents, np.nan)
        negative_branch_cost = np.full(cfg.n_agents, np.nan)
        branch_near_tie = np.zeros(cfg.n_agents, dtype=bool)
        components = self._active_components(prepared)
        matched = self._match_caches(components, runtime.parameters)
        next_caches: list[ComponentCache] = []

        for comp_index, (cache, structure_status) in enumerate(matched):
            ids = np.asarray(cache.members, dtype=int)
            component_id[ids] = comp_index
            structure_changed = structure_status != "exact"
            major_change = structure_status in {"new", "major"}
            must_search = (structure_changed or feedback.requires_search
                           or cache.ticks_since_search
                           >= self.budget.max_reuse_ticks)
            if not must_search:
                working[ids] = cache.last_parameters
                source[ids] = "reuse"
                cache.ticks_since_search += 1
                self.reuse_decisions += 1
                next_caches.append(cache)
                continue

            # A one-member expansion/contraction keeps the projected CEM
            # distribution and receives a cheap light update.  Only a new or
            # largely different component discards enough context to justify
            # the strong budget.
            strong = major_change or feedback.requires_strong
            if strong:
                self.strong_searches += 1
                source[ids] = "strong"
            else:
                self.light_searches += 1
                source[ids] = "light"
            positive, _positive_approx = self._search_branch(
                cache.positive, +1.0, runtime, prepared, goals, gain, cfg,
                working, active_ids, release_ids, strong=strong)
            negative, _negative_approx = self._search_branch(
                cache.negative, -1.0, runtime, prepared, goals, gain, cfg,
                working, active_ids, release_ids, strong=strong)
            # Approximate rollouts only rank CEM candidates.  Every selected
            # branch is re-evaluated with the exact deployed 32-substep safety
            # dynamics before it can become a label.
            if self.budget.candidate_backend == "jax_x64":
                shortlist_costs = self.evaluate_candidates_exact_jax(
                    runtime, goals, gain, cfg,
                    np.stack((working, positive, negative)), active_ids,
                    release_ids, ids)
                incumbent_value, positive_cost, negative_cost = (
                    float(value) for value in shortlist_costs)
            else:
                incumbent_value = self._rollout_cost(
                    runtime, goals, gain, cfg, working, active_ids,
                    release_ids, ids,
                    integration_substeps=self.protocol.integration_substeps)
                positive_cost = self._rollout_cost(
                    runtime, goals, gain, cfg, positive, active_ids,
                    release_ids, ids,
                    integration_substeps=self.protocol.integration_substeps)
                negative_cost = self._rollout_cost(
                    runtime, goals, gain, cfg, negative, active_ids,
                    release_ids, ids,
                    integration_substeps=self.protocol.integration_substeps)
            near_tie = self._branch_costs_near_tie(
                incumbent_value, positive_cost, negative_cost,
                self.budget.tie_fraction)
            positive_branch_cost[ids] = positive_cost
            negative_branch_cost[ids] = negative_cost
            branch_near_tie[ids] = near_tie
            if near_tie:
                prior = float(np.mean(runtime.parameters[ids, 1]))
                answer, answer_cost = ((negative, negative_cost)
                                       if prior < -0.03
                                       else (positive, positive_cost))
            else:
                answer, answer_cost = ((positive, positive_cost)
                                       if positive_cost < negative_cost
                                       else (negative, negative_cost))
            if answer_cost > incumbent_value:
                answer, answer_cost = working.copy(), incumbent_value
            working[ids] = answer[ids]
            cache.last_parameters = working[ids].copy()
            cache.ticks_since_search = 0
            cost[ids] = answer_cost
            incumbent[ids] = incumbent_value
            validated[ids] = True
            next_caches.append(cache)

        self.caches = next_caches
        return TeacherDecision(
            parameters=working,
            source=source,
            teacher_cost=cost,
            incumbent_cost=incumbent,
            validated=validated,
            component_id=component_id,
            positive_branch_cost=positive_branch_cost,
            negative_branch_cost=negative_branch_cost,
            branch_near_tie=branch_near_tie,
        )

    def statistics(self) -> dict:
        return {
            "budget": asdict(self.budget),
            "calls": self.calls,
            "strong_searches": self.strong_searches,
            "light_searches": self.light_searches,
            "reuse_decisions": self.reuse_decisions,
            "objective_rollouts": self.objective_rollouts,
            "candidate_rollouts": self.candidate_rollouts,
            "approximate_rollouts": self.approximate_rollouts,
            "exact_validation_rollouts": self.exact_validation_rollouts,
            "audit_exact_rollouts": self.audit_exact_rollouts,
            "incumbent_and_branch_validation_rollouts": (
                self.objective_rollouts - self.candidate_rollouts),
            "simulated_control_steps": self.simulated_control_steps,
            "simulated_safety_qp_solves": self.simulated_safety_qp_solves,
            "simulated_integration_samples": (
                self.simulated_integration_samples),
        }


class AuthoritativeComponentTeacher:
    """Fresh-per-ACTIVE-tick exact-ranking Teacher for formal v2 labels.

    Every ACTIVE tick creates independent CEM instances. No light budget,
    reuse, quick fallback, approximate safety ranking, or old CEM distribution
    can enter this path.
    """

    tier = "authoritative_v2"

    def __init__(self, protocol: SyncEventProtocol,
                 budget: AuthoritativeTeacherBudget | None = None,
                 seed: int = 0):
        self.protocol = protocol
        self.budget = budget or AuthoritativeTeacherBudget()
        self.search_budget = self.budget.search_budget(protocol)
        if (self.search_budget.planning_integration_substeps
                != protocol.integration_substeps):
            raise ValueError(
                "authoritative ranking must use deployed integration substeps")
        if self.search_budget.candidate_backend != "jax_x64":
            raise ValueError("authoritative Teacher requires JAX x64 ranking")
        self.rng = np.random.default_rng(seed)
        self.calls = 0
        self.searches = 0
        self.restart_searches = 0
        self.selected_restart_counts = np.zeros(
            self.budget.restarts, dtype=np.int64)
        self.admissible_restart_counts: list[int] = []
        self.last_restart_costs: list[float] = []
        self.restart_cost_history: list[list[float]] = []
        self.internal_statistics: list[dict] = []

    @staticmethod
    def _component_ids(n_agents: int,
                       components: tuple[tuple[int, ...], ...]) -> np.ndarray:
        result = np.full(n_agents, -1, dtype=np.int32)
        for index, component in enumerate(components):
            result[np.asarray(component, dtype=int)] = index
        return result

    @staticmethod
    def _empty_decision(runtime: RuntimeState, cfg: Config,
                        components: tuple[tuple[int, ...], ...],
                        source_name: str) -> TeacherDecision:
        source = np.full(cfg.n_agents, "bypass", dtype="U8")
        active_ids = np.asarray(
            sorted(agent for component in components for agent in component),
            dtype=int)
        source[active_ids] = source_name
        nan = np.full(cfg.n_agents, np.nan)
        return TeacherDecision(
            parameters=runtime.parameters.copy(),
            source=source,
            teacher_cost=nan.copy(),
            incumbent_cost=nan.copy(),
            validated=np.zeros(cfg.n_agents, dtype=bool),
            component_id=AuthoritativeComponentTeacher._component_ids(
                cfg.n_agents, components),
            positive_branch_cost=nan.copy(),
            negative_branch_cost=nan.copy(),
            branch_near_tie=np.zeros(cfg.n_agents, dtype=bool))

    def decide(self, runtime: RuntimeState, prepared: PreparedStep,
               goals: np.ndarray, gain: np.ndarray, cfg: Config,
               feedback: TeacherFeedback | None = None) -> TeacherDecision:
        """Return one fresh multi-restart exact label."""
        self.protocol.validate(cfg)
        self.calls += 1
        components = tuple(ComponentCEMTeacher._active_components(prepared))
        if not components:
            return self._empty_decision(runtime, cfg, components, "bypass")

        active_ids = np.flatnonzero(prepared.mode == IntentMode.ACTIVE)
        release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
        focus_ids = np.flatnonzero(prepared.refresh_mask)
        trials: list[tuple[float, float, ComponentCEMTeacher,
                           TeacherDecision]] = []
        for _ in range(self.budget.restarts):
            restart_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
            teacher = ComponentCEMTeacher(
                self.protocol, budget=self.search_budget, seed=restart_seed)
            decision = teacher.decide(
                runtime, prepared, goals, gain, cfg, feedback)
            exact = teacher.evaluate_candidates_exact_jax(
                runtime, goals, gain, cfg,
                np.stack((decision.parameters, runtime.parameters)),
                active_ids, release_ids, focus_ids)
            trials.append((
                float(exact[0]), float(exact[1]), teacher, decision))
            self.internal_statistics.append(teacher.statistics())
            self.restart_searches += 1

        best_cost = min(trial[0] for trial in trials)
        self.last_restart_costs = [trial[0] for trial in trials]
        self.restart_cost_history.append(self.last_restart_costs.copy())
        common_incumbent = trials[0][1]
        improvement_scale = max(common_incumbent - best_cost, 1.0)
        admissible = [
            index for index, trial in enumerate(trials)
            if trial[0] - best_cost
            <= self.budget.tie_fraction * improvement_scale]
        # Canonicalize only inside the already validated quality band.  This
        # chooses an actual CEM action (never an unsafe mean/template) using
        # previous Intent, which is part of the Student-visible observation.
        selected = min(admissible, key=lambda index: (
            float(np.sum((
                trials[index][3].parameters[focus_ids]
                - runtime.parameters[focus_ids]) ** 2)),
            trials[index][0], index))
        selected_cost, incumbent_cost, _, decision = trials[selected]
        self.selected_restart_counts[selected] += 1
        self.admissible_restart_counts.append(len(admissible))
        source = decision.source.astype("U8", copy=True)
        source[focus_ids] = "auth"
        cost = decision.teacher_cost.copy()
        incumbent = decision.incumbent_cost.copy()
        validated = decision.validated.copy()
        cost[focus_ids] = selected_cost
        incumbent[focus_ids] = incumbent_cost
        validated[focus_ids] = True
        self.searches += 1
        return TeacherDecision(
            parameters=decision.parameters.copy(),
            source=source,
            teacher_cost=cost,
            incumbent_cost=incumbent,
            validated=validated,
            component_id=decision.component_id.copy(),
            positive_branch_cost=decision.positive_branch_cost.copy(),
            negative_branch_cost=decision.negative_branch_cost.copy(),
            branch_near_tie=decision.branch_near_tie.copy())

    def statistics(self) -> dict:
        totals = {
            name: int(sum(row[name] for row in self.internal_statistics))
            for name in (
                "objective_rollouts", "candidate_rollouts",
                "exact_validation_rollouts", "approximate_rollouts",
                "simulated_control_steps", "simulated_safety_qp_solves",
                "simulated_integration_samples")
        }
        return {
            "tier": self.tier,
            "budget": asdict(self.budget),
            "internal_search_budget": asdict(self.search_budget),
            "calls": self.calls,
            "searches": self.searches,
            "restart_searches": self.restart_searches,
            "selected_restart_counts": self.selected_restart_counts.tolist(),
            "admissible_restart_counts": self.admissible_restart_counts,
            "restart_cost_history": self.restart_cost_history,
            **totals,
        }
