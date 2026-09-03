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
    planning_integration_substeps: int = 4
    candidate_backend: str = "jax_x64"


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


class ComponentCEMTeacher:
    """Offline Teacher; the deployed Student remains purely local.

    Reuse performs no rollout.  Full horizon evaluations occur only on search
    ticks, so reported rollout counts equal the actual expensive calls.
    """

    def __init__(self, protocol: SyncEventProtocol,
                 budget: TeacherBudget | None = None, seed: int = 0):
        self.protocol = protocol
        self.budget = budget or TeacherBudget()
        self.rng = np.random.default_rng(seed)
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
        for _ in range(iterations):
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
                active_mask = np.zeros(cfg.n_agents, dtype=bool)
                release_mask = np.zeros(cfg.n_agents, dtype=bool)
                focus_mask = np.zeros(cfg.n_agents, dtype=bool)
                active_mask[active_ids] = True
                release_mask[release_ids] = True
                focus_mask[ids] = True
                costs = np.asarray(batched_candidate_costs(
                    jnp.asarray(runtime.physical),
                    jnp.asarray(runtime.world_intent),
                    jnp.asarray(runtime.braking_latch),
                    jnp.asarray(runtime.previous_safe_force),
                    jnp.asarray(candidates), jnp.asarray(runtime.parameters),
                    jnp.asarray(goals), jnp.asarray(gain),
                    jnp.asarray(active_mask), jnp.asarray(release_mask),
                    jnp.asarray(focus_mask), n_agents=cfg.n_agents,
                    horizon_steps=self.budget.horizon_steps,
                    integration_substeps=(
                        self.budget.planning_integration_substeps),
                    runtime_step=runtime.step, dt=cfg.dt, mass=cfg.mass,
                    sense_radius=cfg.sense_radius,
                    max_force=cfg.max_force, max_speed=cfg.max_speed,
                    safe_distance=(cfg.wang_pair_safe_radius_factor
                                   * cfg.car_radius),
                    hard_distance=self.protocol.hard_center_distance,
                    wang_gamma=cfg.wang_gamma,
                    lateral_speed=self.protocol.lateral_speed,
                    max_intent_accel=self.protocol.max_intent_accel,
                    intent_lookahead=self.protocol.intent_lookahead,
                    action_smooth_weight=(
                        self.protocol.action_smooth_weight)))
                # These batched calls are real approximate objective rollouts.
                self.objective_rollouts += samples
                self.approximate_rollouts += samples
                self.simulated_control_steps += (
                    samples * self.budget.horizon_steps)
                self.simulated_safety_qp_solves += (
                    samples * self.budget.horizon_steps)
                self.simulated_integration_samples += (
                    samples * self.budget.horizon_steps
                    * self.budget.planning_integration_substeps)
            elif self.budget.candidate_backend == "numpy":
                costs = np.asarray([
                    self._rollout_cost(
                        runtime, goals, gain, cfg, candidate, active_ids,
                        release_ids, ids,
                        integration_substeps=(
                            self.budget.planning_integration_substeps))
                    for candidate in candidates
                ])
            else:
                raise ValueError(
                    f"unknown candidate backend: "
                    f"{self.budget.candidate_backend}")
            self.candidate_rollouts += samples
            order = np.argsort(costs)
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
            incumbent_value = self._rollout_cost(
                runtime, goals, gain, cfg, working, active_ids,
                release_ids, ids,
                integration_substeps=self.protocol.integration_substeps)
            positive, _positive_approx = self._search_branch(
                cache.positive, +1.0, runtime, prepared, goals, gain, cfg,
                working, active_ids, release_ids, strong=strong)
            negative, _negative_approx = self._search_branch(
                cache.negative, -1.0, runtime, prepared, goals, gain, cfg,
                working, active_ids, release_ids, strong=strong)
            # Approximate rollouts only rank CEM candidates.  Every selected
            # branch is re-evaluated with the exact deployed 32-substep safety
            # dynamics before it can become a label.
            positive_cost = self._rollout_cost(
                runtime, goals, gain, cfg, positive, active_ids,
                release_ids, ids,
                integration_substeps=self.protocol.integration_substeps)
            negative_cost = self._rollout_cost(
                runtime, goals, gain, cfg, negative, active_ids,
                release_ids, ids,
                integration_substeps=self.protocol.integration_substeps)
            scale = max(1.0, min(abs(positive_cost), abs(negative_cost)))
            if abs(positive_cost - negative_cost) <= self.budget.tie_fraction * scale:
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
            "incumbent_and_branch_validation_rollouts": (
                self.objective_rollouts - self.candidate_rollouts),
            "simulated_control_steps": self.simulated_control_steps,
            "simulated_safety_qp_solves": self.simulated_safety_qp_solves,
            "simulated_integration_samples": (
                self.simulated_integration_samples),
        }
