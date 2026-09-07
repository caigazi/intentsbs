"""Search the exact Teacher objective inside the shared Student action space."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_v2_label_equivalence_projection_audit import (
    canonical_observation_key,
    equivalence_groups,
    project_shared_labels,
)
from run_v2_label_rollout_manifest import runtime_from_trace
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import prepare_step
from sbs824.v2.teacher import (
    AuthoritativeComponentTeacher,
    ComponentCEMTeacher,
)


def partition_component(record: dict, members: tuple[int, ...]) -> list[list[int]]:
    slots = len(record["parameters"]) - 1
    grouped: dict[bytes, list[int]] = {}
    for agent in members:
        key = canonical_observation_key(
            record["local_observations"][str(agent)], slots)
        grouped.setdefault(key, []).append(int(agent))
    return list(grouped.values())


def search_shared(evaluator, rng, runtime, goals, gain, cfg, working,
                  active_ids, release_ids, groups, initial):
    budget = evaluator.budget
    samples = budget.strong_samples
    elite_count = max(2, int(round(budget.elite_fraction * samples)))
    mean = np.asarray([
        np.mean(initial[np.asarray(group, dtype=int)], axis=0)
        for group in groups], dtype=float)
    std = np.tile(
        [budget.new_std_alpha, budget.new_std_beta], (len(groups), 1))
    best = working.copy()
    best_cost = float("inf")
    for _ in range(budget.strong_iterations):
        latent = rng.normal(mean, std, size=(samples, len(groups), 2))
        latent[..., 0] = np.clip(latent[..., 0], -1.0, 1.0)
        latent[..., 1] = np.clip(latent[..., 1], -1.0, 1.0)
        latent[0] = mean
        candidates = np.repeat(working[None], samples, axis=0)
        for group_index, group in enumerate(groups):
            ids = np.asarray(group, dtype=int)
            candidates[:, ids, 0] = latent[:, group_index, 0, None]
            candidates[:, ids, 1] = latent[:, group_index, 1, None]
        focus_ids = np.asarray(
            sorted(agent for group in groups for agent in group), dtype=int)
        costs = evaluator.evaluate_candidates_exact_jax(
            runtime, goals, gain, cfg, candidates,
            active_ids, release_ids, focus_ids)
        order = np.argsort(costs, kind="stable")
        if float(costs[order[0]]) < best_cost:
            best_cost = float(costs[order[0]])
            best = candidates[order[0]].copy()
        elite = latent[order[:elite_count]]
        mean = 0.25 * mean + 0.75 * elite.mean(axis=0)
        std = 0.25 * std + 0.75 * elite.std(axis=0)
        std[:, 0] = np.maximum(std[:, 0], budget.covariance_floor_alpha)
        std[:, 1] = np.maximum(std[:, 1], budget.covariance_floor_beta)
    return best, best_cost


def constrained_restart(evaluator, rng, runtime, prepared, goals, gain, cfg,
                        record):
    working = runtime.parameters.copy()
    initial = np.asarray(record["parameters"], dtype=float)
    active_ids = np.flatnonzero(prepared.mode == IntentMode.ACTIVE)
    release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
    # The deployed Student is shared across the whole swarm.  Therefore equal
    # local graph inputs must share one action even when the agents happen to
    # belong to different Teacher connected components.  Optimizing each
    # component separately silently violated that constraint.
    groups = partition_component(record, tuple(int(i) for i in active_ids))
    candidate, candidate_cost = search_shared(
        evaluator, rng, runtime, goals, gain, cfg, working,
        active_ids, release_ids, groups, initial)
    incumbent_cost = float(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, working[None],
        active_ids, release_ids, active_ids)[0])
    if candidate_cost < incumbent_cost:
        working[active_ids] = candidate[active_ids]
    focus_ids = np.flatnonzero(prepared.refresh_mask)
    cost = float(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, working[None],
        active_ids, release_ids, focus_ids)[0])
    return working, cost


def audit_state(path: Path, seed: int) -> dict | None:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["records"][0]
    if record.get("accepted_for_training") is False:
        return None
    _, changed = project_shared_labels(
        record["parameters"], equivalence_groups(record))
    if not changed:
        return None
    trace = np.load(Path(report["source_trace"]))
    step = int(record["runtime_step"])
    n_agents = int(report["n_agents"])
    goals = np.asarray(trace["goals"], dtype=float)
    cfg = make_v2_config(
        n_agents=n_agents, n_obstacles=0, seed=int(report["seed"]), steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = runtime_from_trace(trace, step)
    prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
    authoritative = AuthoritativeComponentTeacher(SYNC_EVENT_V2, seed=seed)
    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=authoritative.search_budget, seed=seed)
    rng = np.random.default_rng(seed)
    trials = [constrained_restart(
        evaluator, rng, runtime, prepared, goals, gain, cfg, record)
        for _ in range(authoritative.budget.restarts)]
    active_ids = np.flatnonzero(prepared.mode == IntentMode.ACTIVE)
    release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
    focus_ids = np.flatnonzero(prepared.refresh_mask)
    student = np.asarray(record["student_parameters"], dtype=float)
    final_candidates = np.stack((
        *[trial[0] for trial in trials], student, runtime.parameters,
        np.asarray(record["parameters"], dtype=float)))
    final_costs = np.asarray(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, final_candidates,
        active_ids, release_ids, focus_ids), dtype=float)
    constrained_index = int(np.argmin(final_costs[:-1]))
    constrained_cost = float(final_costs[constrained_index])
    original_cost = float(final_costs[-1])
    incumbent_cost = float(final_costs[-2])
    scale = max(incumbent_cost - original_cost, 1.0)
    regret = (constrained_cost - original_cost) / scale
    global_groups = partition_component(record, tuple(int(i) for i in active_ids))
    equal_input_gap = max((
        float(np.max(np.linalg.norm(
            final_candidates[constrained_index][np.asarray(group)][:, None, :]
            - final_candidates[constrained_index][np.asarray(group)][None, :, :],
            axis=2)))
        for group in global_groups if len(group) > 1), default=0.0)
    if equal_input_gap > 1e-9:
        raise RuntimeError(
            "constrained search produced unequal actions for equal Student inputs: "
            f"{equal_input_gap}")
    sources = ([f"constrained_restart_{index}" for index in range(len(trials))]
               + ["student", "incumbent"])
    return {
        "report": str(path),
        "case": report["case"],
        "n_agents": n_agents,
        "runtime_step": step,
        "equivalent_groups": changed,
        "original_teacher_exact_cost": original_cost,
        "student_compatible_exact_cost": constrained_cost,
        "student_compatible_source": sources[constrained_index],
        "incumbent_exact_cost": incumbent_cost,
        "constrained_restart_costs": [float(cost) for _, cost in trials],
        "student_compatible_improvement_regret": float(regret),
        "inside_one_percent_quality_band": bool(regret <= 0.01),
        "student_compatible_max_equal_input_gap": equal_input_gap,
        "student_compatible_parameters": final_candidates[
            constrained_index].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-summary", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9460000)
    args = parser.parse_args()
    source_paths = []
    for summary_path in args.label_summary:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol") != SYNC_EVENT_V2.manifest():
            raise RuntimeError(f"protocol manifest differs: {summary_path}")
        source_paths.extend(Path(row["report"]) for row in summary["reports"])
    started = perf_counter()
    records = []
    for index, path in enumerate(source_paths):
        result = audit_state(path, args.seed + 10_000_019 * index)
        if result is not None:
            records.append(result)
    by_n = Counter(row["n_agents"] for row in records)
    passed_by_n = Counter(
        row["n_agents"] for row in records
        if row["inside_one_percent_quality_band"])
    report = {
        "audit_kind": "student_compatible_constrained_cem_exact_audit",
        "formal_dataset": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "source_summaries": [str(path) for path in args.label_summary],
        "conflicting_state_count": len(records),
        "conflicting_states_by_n": dict(sorted(by_n.items())),
        "quality_band_pass_count": sum(
            row["inside_one_percent_quality_band"] for row in records),
        "quality_band_pass_by_n": dict(sorted(passed_by_n.items())),
        "worst_improvement_regret": max(
            (row["student_compatible_improvement_regret"] for row in records),
            default=None),
        "wall_seconds": perf_counter() - started,
        "records": records,
        "interpretation": (
            "Authoritative-size CEM searches only actions compatible with the "
            "shared deterministic Student: agents with identical local graph "
            "inputs anywhere in the swarm share alpha and beta. Student and "
            "incumbent actions are explicit non-degrading candidates."),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in report.items() if key != "records"
    }), flush=True)


if __name__ == "__main__":
    main()
