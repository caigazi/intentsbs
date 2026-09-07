"""Development audit of online Teacher continuity and scheduled-search quality.

The audit follows one closed-loop v2 trajectory, records only Student-visible
features and Teacher actions, and periodically compares the online label with
fresh authoritative searches on the identical state.  It is not a training
dataset and must not be used as one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import (apply_prepared_step, initialize_runtime,
                               prepare_step)
from sbs824.v2.scenes import headon_scene, mixed_crossing_scene
from sbs824.v2.teacher import ComponentCEMTeacher, TeacherBudget


def load_case(name: str):
    if name == "n2_headon":
        state, goals, topology = headon_scene()
        return state, goals, topology, 120002
    if name == "n5_partial":
        state, goals, topology = mixed_crossing_scene(94501, 5)
        return state, goals, topology, 94501
    if name == "n8_two_stream":
        state, goals, topology = mixed_crossing_scene(94803, 8)
        return state, goals, topology, 94803
    raise ValueError(name)


def local_observation_vector(features, ego: int, n_agents: int) -> np.ndarray:
    """Flatten one permutation-stable local observation with zero padding."""
    result = np.zeros(9 + (n_agents - 1) * 10, dtype=np.float32)
    result[:9] = features.self_features[ego]
    cursor = 9
    for slot in np.flatnonzero(features.edge_mask[ego]):
        result[cursor:cursor + 9] = features.edge_features[ego, slot]
        result[cursor + 9] = 1.0
        cursor += 10
    return result


def exact_pair(evaluator: ComponentCEMTeacher, runtime, prepared, goals,
               gain, cfg, proposal: np.ndarray) -> tuple[float, float]:
    active_ids = np.flatnonzero(prepared.mode == 1)
    release_ids = np.flatnonzero(prepared.mode == 2)
    focus_ids = np.flatnonzero(prepared.refresh_mask)
    costs = evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg,
        np.stack((proposal, runtime.parameters)), active_ids, release_ids,
        focus_ids)
    return float(costs[0]), float(costs[1])


def fresh_reference(runtime, prepared, goals, gain, cfg,
                    budget: TeacherBudget, seed: int,
                    restarts: int) -> tuple[float, np.ndarray, list[float]]:
    costs: list[float] = []
    actions: list[np.ndarray] = []
    for restart in range(restarts):
        search_seed = seed + 1_000_003 * restart
        teacher = ComponentCEMTeacher(
            SYNC_EVENT_V2, budget=budget, seed=search_seed)
        decision = teacher.decide(runtime, prepared, goals, gain, cfg)
        cost, _ = exact_pair(
            teacher, runtime, prepared, goals, gain, cfg,
            decision.parameters)
        costs.append(cost)
        actions.append(decision.parameters.copy())
    best = int(np.argmin(costs))
    return costs[best], actions[best], costs


def continuity_summary(records: list[dict], n_agents: int) -> dict:
    per_agent: dict[str, dict] = {}
    all_action_jumps: list[float] = []
    all_observation_jumps: list[float] = []
    close_pair_action_jumps: list[float] = []
    for agent in range(n_agents):
        rows = [row for row in records if agent in row["active_ids"]]
        if not rows:
            continue
        observations = np.asarray([
            row["local_observations"][str(agent)] for row in rows])
        actions = np.asarray([
            row["parameters"][agent] for row in rows])
        observation_jumps = np.linalg.norm(
            np.diff(observations, axis=0), axis=1)
        action_jumps = np.linalg.norm(np.diff(actions, axis=0), axis=1)
        beta_signs = np.sign(np.where(
            np.abs(actions[:, 1]) > 0.03, actions[:, 1], 0.0))
        sign_flips = int(np.count_nonzero(
            (beta_signs[1:] * beta_signs[:-1]) < 0.0))
        all_action_jumps.extend(action_jumps.tolist())
        all_observation_jumps.extend(observation_jumps.tolist())

        nearest = []
        if len(rows) > 1:
            distances = np.linalg.norm(
                observations[:, None] - observations[None, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            for left in range(len(rows)):
                right = int(np.argmin(distances[left]))
                action_distance = float(np.linalg.norm(
                    actions[left] - actions[right]))
                nearest.append({
                    "left_step": rows[left]["runtime_step"],
                    "right_step": rows[right]["runtime_step"],
                    "observation_l2": float(distances[left, right]),
                    "action_l2": action_distance,
                })
                if distances[left, right] <= 0.10:
                    close_pair_action_jumps.append(action_distance)
        per_agent[str(agent)] = {
            "active_records": len(rows),
            "maximum_consecutive_observation_l2": (
                float(np.max(observation_jumps))
                if len(observation_jumps) else 0.0),
            "maximum_consecutive_action_l2": (
                float(np.max(action_jumps)) if len(action_jumps) else 0.0),
            "beta_sign_flips": sign_flips,
            "nearest_observation_pairs": nearest,
        }
    return {
        "per_agent": per_agent,
        "maximum_consecutive_observation_l2": (
            max(all_observation_jumps, default=0.0)),
        "maximum_consecutive_action_l2": max(all_action_jumps, default=0.0),
        "close_observation_threshold_l2": 0.10,
        "close_observation_pair_count": len(close_pair_action_jumps),
        "maximum_close_pair_action_l2": (
            max(close_pair_action_jumps, default=0.0)),
    }


def run(case: str, output: Path, active_ticks: int,
        authoritative_samples: int, authoritative_iterations: int,
        reference_restarts: int, seed: int,
        online_full_search: bool, reset_search_cache: bool,
        online_restarts: int) -> dict:
    state, goals, topology, scene_seed = load_case(case)
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(state, goals, gain, cfg, SYNC_EVENT_V2)
    budget_kwargs = {
        "strong_samples": authoritative_samples,
        "strong_iterations": authoritative_iterations,
    }
    if online_full_search:
        budget_kwargs.update({
            "light_samples": authoritative_samples,
            "light_iterations": authoritative_iterations,
        })
    budget = TeacherBudget(**budget_kwargs)
    online = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=budget, seed=seed)
    records: list[dict] = []
    first_active_step = None
    started = perf_counter()
    while runtime.step < cfg.steps and len(records) < active_ticks:
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        if not prepared.refresh_mask.any():
            runtime, _ = apply_prepared_step(
                runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2,
                runtime.parameters)
            continue
        if first_active_step is None:
            first_active_step = runtime.step
        cache_reset = False
        if reset_search_cache and online.caches:
            current_components = sorted(online._active_components(prepared))
            cached_components = sorted(cache.members for cache in online.caches)
            periodic_search_due = any(
                cache.ticks_since_search >= budget.max_reuse_ticks
                for cache in online.caches)
            if (current_components != cached_components
                    or periodic_search_due):
                online.caches = []
                cache_reset = True
        trial_teachers = [online]
        trial_decisions = [
            online.decide(runtime, prepared, goals, gain, cfg)]
        primary_sources = sorted(set(
            trial_decisions[0].source[
                prepared.refresh_mask].astype(str).tolist()))
        searched = any(
            value in {"strong", "light"} for value in primary_sources)
        if searched:
            for restart in range(1, online_restarts):
                restart_teacher = ComponentCEMTeacher(
                    SYNC_EVENT_V2, budget=budget,
                    seed=seed + 7_000_021 * (len(records) + 1) + restart)
                trial_teachers.append(restart_teacher)
                trial_decisions.append(restart_teacher.decide(
                    runtime, prepared, goals, gain, cfg))
        trial_exact = [exact_pair(
            teacher, runtime, prepared, goals, gain, cfg,
            decision.parameters)
            for teacher, decision in zip(trial_teachers, trial_decisions)]
        selected_trial = int(np.argmin([value[0] for value in trial_exact]))
        online = trial_teachers[selected_trial]
        decision = trial_decisions[selected_trial]
        online_cost, incumbent_cost = trial_exact[selected_trial]
        sources = sorted(set(
            decision.source[prepared.refresh_mask].astype(str).tolist()))
        # Probe every actual search tick. Reuse labels are judged through their
        # next scheduled search, avoiding an expensive redundant search each tick.
        should_probe = any(value in {"strong", "light"} for value in sources)
        reference_cost = None
        reference_action = None
        reference_costs = None
        improvement_regret = None
        cost_relative_regret = None
        if should_probe:
            reference_cost, reference_action, reference_costs = fresh_reference(
                runtime, prepared, goals, gain, cfg, budget,
                seed + 10_000_019 * (len(records) + 1), reference_restarts)
            improvement_regret = (
                (online_cost - reference_cost)
                / max(incumbent_cost - reference_cost, 1.0))
            cost_relative_regret = max(
                online_cost - reference_cost, 0.0) / max(
                    abs(reference_cost), 1.0)
        active_ids = np.flatnonzero(prepared.refresh_mask).astype(int)
        records.append({
            "runtime_step": runtime.step,
            "active_ids": active_ids.tolist(),
            "sources": sources,
            "cache_reset_before_search": cache_reset,
            "online_restart_exact_costs": [
                value[0] for value in trial_exact],
            "selected_online_restart": selected_trial,
            "parameters": decision.parameters.tolist(),
            "online_exact_cost": online_cost,
            "incumbent_exact_cost": incumbent_cost,
            "reference_best_exact_cost": reference_cost,
            "reference_restart_exact_costs": reference_costs,
            "reference_parameters": (
                None if reference_action is None else reference_action.tolist()),
            "online_improvement_normalized_regret": improvement_regret,
            "online_cost_relative_regret": cost_relative_regret,
            "local_observations": {
                str(agent): local_observation_vector(
                    prepared.features, agent, cfg.n_agents).tolist()
                for agent in active_ids
            },
        })
        runtime, trace = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2,
            decision.parameters)
        if (trace.out_of_certificate_pair_samples > 0
                or trace.minimum_center_distance
                < SYNC_EVENT_V2.hard_center_distance):
            break

    probed = [row for row in records
              if row["online_improvement_normalized_regret"] is not None]
    regrets = [row["online_improvement_normalized_regret"] for row in probed]
    cost_regrets = [row["online_cost_relative_regret"] for row in probed]
    report = {
        "audit_kind": "teacher_temporal_distillability_development_audit",
        "formal_dataset": False,
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "first_active_step": first_active_step,
        "active_ticks_requested": active_ticks,
        "active_ticks_recorded": len(records),
        "protocol": SYNC_EVENT_V2.manifest(),
        "online_budget": online.statistics()["budget"],
        "reference_restarts": reference_restarts,
        "online_full_search": online_full_search,
        "reset_search_cache": reset_search_cache,
        "online_restarts": online_restarts,
        "search_ticks_probed": len(probed),
        "maximum_online_improvement_normalized_regret": (
            max(regrets, default=None)),
        "maximum_online_cost_relative_regret": (
            max(cost_regrets, default=None)),
        "search_ticks_above_one_percent_regret": int(sum(
            value > 0.01 for value in regrets)),
        "substantive_search_ticks_above_one_percent": int(sum(
            improvement > 0.01 and relative > 0.01
            for improvement, relative in zip(regrets, cost_regrets))),
        "continuity": continuity_summary(records, cfg.n_agents),
        "records": records,
        "teacher_statistics": online.statistics(),
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "This audit separates online cache/light-search continuity from "
            "fresh authoritative-search quality on the same state. Student "
            "training remains blocked if scheduled online labels have large "
            "regret or locally similar observations receive incompatible actions."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"),
        default="n5_partial")
    parser.add_argument("--active-ticks", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--reference-restarts", type=int, default=2)
    parser.add_argument(
        "--online-full-search", action="store_true",
        help=("use the authoritative population/iterations for scheduled "
              "online light searches; development A/B only"))
    parser.add_argument(
        "--reset-search-cache", action="store_true",
        help=("discard the cached CEM distribution when component structure "
              "changes or a periodic search is due; development A/B only"))
    parser.add_argument("--online-restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=194501)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.active_ticks < 2:
        parser.error("--active-ticks must be at least two")
    if args.samples < 2 or args.iterations < 1:
        parser.error("invalid authoritative search budget")
    if args.reference_restarts < 1:
        parser.error("--reference-restarts must be positive")
    if args.online_restarts < 1:
        parser.error("--online-restarts must be positive")
    report = run(
        args.case, args.output, args.active_ticks, args.samples,
        args.iterations, args.reference_restarts, args.seed,
        args.online_full_search, args.reset_search_cache,
        args.online_restarts)
    print(json.dumps({
        "case": report["case"],
        "active_ticks_recorded": report["active_ticks_recorded"],
        "search_ticks_probed": report["search_ticks_probed"],
        "maximum_online_improvement_normalized_regret": (
            report["maximum_online_improvement_normalized_regret"]),
        "maximum_online_cost_relative_regret": (
            report["maximum_online_cost_relative_regret"]),
        "search_ticks_above_one_percent_regret": (
            report["search_ticks_above_one_percent_regret"]),
        "substantive_search_ticks_above_one_percent": (
            report["substantive_search_ticks_above_one_percent"]),
        "wall_seconds": report["wall_seconds"],
        "report": str(args.output / "REPORT.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
