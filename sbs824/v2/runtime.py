"""One clean execution path shared by v2 Teacher and Student rollouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import Config, nominal_lqr
from .features import FeatureBatch, build_local_features
from .intent import (IDENTITY_INTENT, decode_world_velocity, rate_limit,
                     tracking_force)
from .protocol import (IntentMode, SyncEventProtocol, transition_modes,
                       update_mask)
from .sampled_safety import sampled_wang_step
from .trigger import TriggerResult, ttc_candidates


@dataclass
class RuntimeState:
    physical: np.ndarray
    parameters: np.ndarray
    world_intent: np.ndarray
    mode: np.ndarray
    clear_counts: np.ndarray
    braking_latch: np.ndarray
    previous_safe_force: np.ndarray
    step: int = 0


@dataclass(frozen=True)
class RuntimeTrace:
    features: FeatureBatch
    evidence_mask: np.ndarray
    mode: np.ndarray
    component_id: np.ndarray
    optimized_mask: np.ndarray
    label_mask: np.ndarray
    applied_mask: np.ndarray
    world_intent_before: np.ndarray
    world_intent_after: np.ndarray
    reference_force: np.ndarray
    safe_force: np.ndarray
    next_physical: np.ndarray
    certified_brake_agents: int
    out_of_certificate_pair_samples: int
    safety_qp_solves: int
    minimum_center_distance: float


@dataclass(frozen=True)
class PreparedStep:
    base_force: np.ndarray
    base_velocity: np.ndarray
    trigger: TriggerResult
    mode: np.ndarray
    clear_counts: np.ndarray
    refresh_mask: np.ndarray
    features: FeatureBatch


def initialize_runtime(physical: np.ndarray, goals: np.ndarray,
                       gain: np.ndarray, cfg: Config,
                       protocol: SyncEventProtocol) -> RuntimeState:
    protocol.validate(cfg)
    physical = np.asarray(physical, dtype=float).copy()
    if physical.shape != (cfg.n_agents, 4):
        raise ValueError("physical state must have shape (n_agents, 4)")
    base = nominal_lqr(physical, goals, gain, cfg)
    base_velocity = np.clip(
        physical[:, 2:] + base / cfg.mass * cfg.dt,
        -cfg.max_speed, cfg.max_speed)
    return RuntimeState(
        physical=physical,
        parameters=np.tile(IDENTITY_INTENT, (cfg.n_agents, 1)),
        world_intent=base_velocity,
        mode=np.full(cfg.n_agents, IntentMode.BYPASS, dtype=np.int8),
        clear_counts=np.zeros(cfg.n_agents, dtype=int),
        braking_latch=np.zeros(cfg.n_agents, dtype=bool),
        previous_safe_force=np.zeros((cfg.n_agents, 2), dtype=float),
    )


def _component_ids(n_agents: int, trigger: TriggerResult) -> np.ndarray:
    result = np.full(n_agents, -1, dtype=np.int32)
    for component_id, component in enumerate(trigger.components):
        result[list(component)] = component_id
    return result


def prepare_step(runtime: RuntimeState, goals: np.ndarray, gain: np.ndarray,
                 cfg: Config, protocol: SyncEventProtocol,
                 trigger: TriggerResult | None = None) -> PreparedStep:
    """Compute the one shared evidence/mode/feature/mask decision context."""
    protocol.validate(cfg)
    state = np.asarray(runtime.physical, dtype=float)
    goals = np.asarray(goals, dtype=float)
    base = nominal_lqr(state, goals, gain, cfg)
    base_velocity = np.clip(
        state[:, 2:] + base / cfg.mass * cfg.dt,
        -cfg.max_speed, cfg.max_speed)
    trigger = (ttc_candidates(
        state, base_velocity, runtime.world_intent, cfg, protocol)
        if trigger is None else trigger)
    release_complete = (
        np.linalg.norm(runtime.world_intent - base_velocity, axis=1)
        <= protocol.release_speed_tolerance)
    mode, clear = transition_modes(
        runtime.mode, runtime.clear_counts, trigger.evidence,
        release_complete, protocol)
    refresh = update_mask(mode, runtime.step, protocol)
    features = build_local_features(
        state, goals, base_velocity, runtime.world_intent, mode,
        trigger.evidence, cfg, protocol)
    return PreparedStep(base, base_velocity, trigger, mode, clear, refresh,
                        features)


def apply_prepared_step(runtime: RuntimeState, prepared: PreparedStep,
                        goals: np.ndarray, gain: np.ndarray, cfg: Config,
                        protocol: SyncEventProtocol,
                        proposed_parameters: np.ndarray
                        ) -> tuple[RuntimeState, RuntimeTrace]:
    """Apply a Teacher/Student proposal under a previously prepared context."""
    state = np.asarray(runtime.physical, dtype=float)
    proposal = np.asarray(proposed_parameters, dtype=float)
    if proposal.shape != (cfg.n_agents, 2):
        raise ValueError("proposed_parameters must have shape (n_agents, 2)")
    mode = prepared.mode
    refresh = prepared.refresh_mask
    parameters = runtime.parameters.copy()
    parameters[refresh] = np.clip(proposal[refresh], -1.0, 1.0)
    parameters[mode == IntentMode.BYPASS] = IDENTITY_INTENT
    base = prepared.base_force
    base_velocity = prepared.base_velocity
    features = prepared.features
    world_before = runtime.world_intent.copy()
    target = base_velocity.copy()
    active = mode == IntentMode.ACTIVE
    release = mode == IntentMode.RELEASE
    if np.any(active):
        decoded = decode_world_velocity(
            state, goals, base_velocity, parameters, protocol)
        target[active] = decoded[active]
    world_after = base_velocity.copy()
    controlled = active | release
    if np.any(controlled):
        world_after[controlled] = rate_limit(
            runtime.world_intent[controlled], target[controlled], cfg,
            protocol)

    ids = np.flatnonzero(controlled)
    reference_force = tracking_force(
        state, gain, cfg, ids, world_after, protocol, base)
    if protocol.action_smooth_weight > 0.0 and runtime.step > 0:
        weight = protocol.action_smooth_weight
        reference_force = ((reference_force
                            + weight * runtime.previous_safe_force)
                           / (1.0 + weight))

    latch = runtime.braking_latch.copy()
    safety = sampled_wang_step(
        state, reference_force, [], cfg, latch,
        integration_substeps=protocol.integration_substeps)
    physical = safety.next_state

    next_runtime = RuntimeState(
        physical=physical,
        parameters=parameters,
        world_intent=world_after,
        mode=mode,
        clear_counts=prepared.clear_counts,
        braking_latch=latch,
        previous_safe_force=(cfg.mass
                             * (physical[:, 2:] - state[:, 2:]) / cfg.dt),
        step=runtime.step + 1,
    )
    trace = RuntimeTrace(
        features=features,
        evidence_mask=prepared.trigger.evidence.copy(),
        mode=mode.copy(),
        component_id=_component_ids(cfg.n_agents, prepared.trigger),
        optimized_mask=refresh.copy(),
        label_mask=refresh.copy(),
        applied_mask=refresh.copy(),
        world_intent_before=world_before,
        world_intent_after=world_after.copy(),
        reference_force=reference_force.copy(),
        safe_force=safety.action.copy(),
        next_physical=physical.copy(),
        certified_brake_agents=len(safety.certified_brake_agents),
        out_of_certificate_pair_samples=(
            safety.out_of_certificate_pair_samples),
        safety_qp_solves=safety.qp_solves,
        minimum_center_distance=safety.minimum_center_distance,
    )
    return next_runtime, trace


def step_runtime(runtime: RuntimeState, goals: np.ndarray, gain: np.ndarray,
                 cfg: Config, protocol: SyncEventProtocol,
                 proposed_parameters: np.ndarray,
                 trigger: TriggerResult | None = None
                 ) -> tuple[RuntimeState, RuntimeTrace]:
    """Prepare and execute one 30-ms v2 step."""
    prepared = prepare_step(runtime, goals, gain, cfg, protocol, trigger)
    return apply_prepared_step(
        runtime, prepared, goals, gain, cfg, protocol,
        proposed_parameters)
