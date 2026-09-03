"""Replay saved CEM populations through multiple JAX fidelity levels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

from run_v2_teacher_distillability import capture_first_active
from sbs824.v2.jax_rollout import batched_candidate_costs
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2_DEV


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=int)
    result[order] = np.arange(len(values))
    return result


def rank_metrics(approximate: np.ndarray, exact: np.ndarray,
                 spans: list[tuple[int, int]]) -> dict:
    correlations = []
    exact_best_approximate_ranks = []
    winner_regrets = []
    for start, stop in spans:
        local_approximate = approximate[start:stop]
        local_exact = exact[start:stop]
        approximate_ranks = ranks(local_approximate)
        exact_ranks = ranks(local_exact)
        if (np.std(approximate_ranks) > 0.0
                and np.std(exact_ranks) > 0.0):
            correlations.append(float(np.corrcoef(
                approximate_ranks, exact_ranks)[0, 1]))
        approximate_best = int(np.argmin(local_approximate))
        exact_best = int(np.argmin(local_exact))
        exact_best_approximate_ranks.append(
            int(approximate_ranks[exact_best]))
        best_cost = float(local_exact[exact_best])
        winner_regrets.append(float(
            (local_exact[approximate_best] - best_cost)
            / max(abs(best_cost), 1.0)))
    candidate_count = spans[0][1] - spans[0][0]
    return {
        "population_count": len(spans),
        "candidates_per_population": candidate_count,
        "mean_rank_correlation": float(np.mean(correlations)),
        "median_rank_correlation": float(np.median(correlations)),
        "exact_best_is_approximate_top1": int(sum(
            value == 0 for value in exact_best_approximate_ranks)),
        "exact_best_in_approximate_top3": int(sum(
            value < 3 for value in exact_best_approximate_ranks)),
        "exact_best_in_approximate_top6": int(sum(
            value < 6 for value in exact_best_approximate_ranks)),
        "exact_best_in_approximate_top8": int(sum(
            value < 8 for value in exact_best_approximate_ranks)),
        "maximum_approximate_rank_of_exact_best": int(max(
            exact_best_approximate_ranks)),
        "median_approximate_winner_exact_regret": float(np.median(
            winner_regrets)),
        "maximum_approximate_winner_exact_regret": float(max(
            winner_regrets)),
    }


def evaluate(runtime, prepared, goals, gain, cfg,
             candidates: np.ndarray, horizon_steps: int,
             integration_substeps: int) -> np.ndarray:
    active = prepared.mode == IntentMode.ACTIVE
    release = prepared.mode == IntentMode.RELEASE
    focus = prepared.refresh_mask
    return np.asarray(batched_candidate_costs(
        jnp.asarray(runtime.physical),
        jnp.asarray(runtime.world_intent),
        jnp.asarray(runtime.braking_latch),
        jnp.asarray(runtime.previous_safe_force),
        jnp.asarray(candidates),
        jnp.asarray(runtime.parameters),
        jnp.asarray(goals), jnp.asarray(gain),
        jnp.asarray(active), jnp.asarray(release), jnp.asarray(focus),
        n_agents=cfg.n_agents, horizon_steps=horizon_steps,
        integration_substeps=integration_substeps,
        runtime_step=runtime.step, dt=cfg.dt, mass=cfg.mass,
        sense_radius=cfg.sense_radius, max_force=cfg.max_force,
        max_speed=cfg.max_speed,
        safe_distance=cfg.wang_pair_safe_radius_factor * cfg.car_radius,
        hard_distance=SYNC_EVENT_V2_DEV.hard_center_distance,
        wang_gamma=cfg.wang_gamma,
        lateral_speed=SYNC_EVENT_V2_DEV.lateral_speed,
        max_intent_accel=SYNC_EVENT_V2_DEV.max_intent_accel,
        intent_lookahead=SYNC_EVENT_V2_DEV.intent_lookahead,
        action_smooth_weight=SYNC_EVENT_V2_DEV.action_smooth_weight))


def run(source_report: Path, substeps: list[int], output: Path) -> dict:
    started = perf_counter()
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if source.get("audit_kind") != (
            "teacher_search_sampling_vs_surrogate_diagnostic"):
        raise ValueError("source is not a Teacher search diagnostic report")
    case = source["case"]
    horizon_steps = int(source["budget"]["horizon_steps"])
    runtime, prepared, goals, gain, cfg, topology, scene_seed = (
        capture_first_active(case))
    if runtime.step != source["snapshot_step"]:
        raise RuntimeError("reconstructed snapshot step differs from report")

    candidates_parts = []
    exact_parts = []
    spans = []
    cursor = 0
    expected_members = tuple(np.flatnonzero(prepared.refresh_mask).tolist())
    for seed_report in source["seed_reports"]:
        for record in seed_report["population_records"]:
            members = tuple(record["members"])
            if members != expected_members:
                raise RuntimeError(
                    "this sweep requires one population over all ACTIVE agents")
            member_parameters = np.asarray(record["member_parameters"],
                                           dtype=float)
            population = np.repeat(
                runtime.parameters[None], len(member_parameters), axis=0)
            population[:, np.asarray(members, dtype=int)] = member_parameters
            candidates_parts.append(population)
            exact_parts.append(np.asarray(record["exact_costs"], dtype=float))
            spans.append((cursor, cursor + len(population)))
            cursor += len(population)
    candidates = np.concatenate(candidates_parts, axis=0)
    exact = np.concatenate(exact_parts, axis=0)

    fidelity = {}
    for value in substeps:
        compile_started = perf_counter()
        approximate = evaluate(
            runtime, prepared, goals, gain, cfg, candidates,
            horizon_steps, value)
        compile_and_first = perf_counter() - compile_started
        steady_started = perf_counter()
        steady = evaluate(
            runtime, prepared, goals, gain, cfg, candidates,
            horizon_steps, value)
        steady_seconds = perf_counter() - steady_started
        np.testing.assert_array_equal(approximate, steady)
        metrics = rank_metrics(approximate, exact, spans)
        scale = np.maximum(np.abs(exact), 1.0)
        metrics.update({
            "compile_and_first_seconds": compile_and_first,
            "steady_seconds": steady_seconds,
            "maximum_relative_cost_error": float(np.max(
                np.abs(approximate - exact) / scale)),
            "median_relative_cost_error": float(np.median(
                np.abs(approximate - exact) / scale)),
        })
        fidelity[str(value)] = metrics

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "audit_kind": "teacher_surrogate_fidelity_sweep",
        "formal_dataset": False,
        "source_report": str(source_report),
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "snapshot_step": runtime.step,
        "candidate_count": len(candidates),
        "population_count": len(spans),
        "horizon_steps": horizon_steps,
        "exact_reference_substeps": 32,
        "fidelity": fidelity,
        "wall_seconds": perf_counter() - started,
    }
    (output / "REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "case": case,
        "candidate_count": len(candidates),
        "wall_seconds": result["wall_seconds"],
        "report": str(output / "REPORT.json"),
    }), flush=True)
    return result


def parse_substeps(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("substeps must be positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--substeps", type=parse_substeps,
                        default=parse_substeps("4,8,16,32"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_report, args.substeps, args.output)


if __name__ == "__main__":
    main()
