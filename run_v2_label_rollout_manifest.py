"""Apply the authoritative v2 Teacher to a rollout-selected snapshot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_v2_teacher_temporal_audit import local_observation_vector
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import RuntimeState, prepare_step
from sbs824.v2.teacher import (
    AuthoritativeComponentTeacher,
    ComponentCEMTeacher,
)


def runtime_from_trace(trace, step: int) -> RuntimeState:
    return RuntimeState(
        physical=np.asarray(trace["physical"][step], dtype=float).copy(),
        parameters=np.asarray(trace["parameters"][step], dtype=float).copy(),
        world_intent=np.asarray(trace["world_intent"][step], dtype=float).copy(),
        mode=np.asarray(trace["modes"][step], dtype=np.int8).copy(),
        clear_counts=np.asarray(trace["clear_counts"][step], dtype=int).copy(),
        braking_latch=np.asarray(
            trace["braking_latch"][step], dtype=bool).copy(),
        previous_safe_force=np.asarray(
            trace["previous_safe_force"][step], dtype=float).copy(),
        step=step,
    )


def select_validation_entries(entries: list[dict], maximum_per_scene: int
                              ) -> list[dict]:
    """Keep the latest automatically selected states, never hand-picked steps."""
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        grouped.setdefault(entry["scene_id"], []).append(entry)
    selected = []
    for scene_id in sorted(grouped):
        rows = sorted(grouped[scene_id], key=lambda row: row["snapshot_step"])
        selected.extend(rows[-maximum_per_scene:])
    return selected


def teacher_label_is_accepted(exact_costs, relative_tolerance: float = 1e-8
                              ) -> tuple[bool, str, float]:
    """Accept a DAgger correction only when it beats both available baselines."""
    exact_costs = np.asarray(exact_costs, dtype=float)
    if exact_costs.shape != (3,):
        raise ValueError("expected Teacher, Student, and incumbent exact costs")
    baseline_index = int(np.argmin(exact_costs[1:])) + 1
    baseline_source = ("student", "incumbent")[baseline_index - 1]
    baseline_cost = float(exact_costs[baseline_index])
    tolerance = relative_tolerance * max(abs(baseline_cost), 1.0)
    return (bool(exact_costs[0] <= baseline_cost + tolerance),
            baseline_source, baseline_cost)


def label_entry(entry: dict, seed: int) -> dict:
    trace_path = Path(entry["trace"])
    trace = np.load(trace_path)
    step = int(entry["snapshot_step"])
    n_agents = int(entry["n_agents"])
    goals = np.asarray(trace["goals"], dtype=float)
    cfg = make_v2_config(
        n_agents=n_agents, n_obstacles=0, seed=int(entry["seed"]), steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = runtime_from_trace(trace, step)
    prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
    active_ids = np.flatnonzero(prepared.refresh_mask).astype(int)
    if not len(active_ids):
        raise RuntimeError(
            f"manifest selected non-ACTIVE snapshot {entry['scene_id']}:{step}")
    teacher = AuthoritativeComponentTeacher(SYNC_EVENT_V2, seed=seed)
    decision = teacher.decide(runtime, prepared, goals, gain, cfg)
    student = np.asarray(trace["student_proposals"][step], dtype=float)
    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=teacher.search_budget, seed=0)
    exact = np.asarray(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg,
        np.stack((decision.parameters, student, runtime.parameters)),
        np.flatnonzero(prepared.mode == IntentMode.ACTIVE),
        np.flatnonzero(prepared.mode == IntentMode.RELEASE),
        active_ids), dtype=float)
    accepted, baseline_source, baseline_cost = teacher_label_is_accepted(exact)
    return {
        "runtime_step": step,
        "active_ids": active_ids.tolist(),
        "parameters": decision.parameters.tolist(),
        "student_parameters": student.tolist(),
        "teacher_exact_cost": float(exact[0]),
        "student_exact_cost": float(exact[1]),
        "incumbent_exact_cost": float(exact[2]),
        "accepted_for_training": accepted,
        "best_baseline_source": baseline_source,
        "best_baseline_exact_cost": baseline_cost,
        "teacher_regret_vs_best_baseline": float(
            (exact[0] - baseline_cost) / max(abs(baseline_cost), 1.0)),
        "teacher_relative_improvement_over_student": float(
            (exact[1] - exact[0]) / max(abs(exact[1]), 1.0)),
        "selection_reason": entry["selection_reason"],
        "failure_event_step": entry["failure_event_step"],
        "restart_exact_costs": teacher.last_restart_costs,
        "local_observations": {
            str(agent): local_observation_vector(
                prepared.features, agent, cfg.n_agents).tolist()
            for agent in active_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-per-scene", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8250000)
    args = parser.parse_args()
    if args.maximum_per_scene < 1:
        parser.error("--maximum-per-scene must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != SYNC_EVENT_V2.manifest():
        raise RuntimeError("rollout manifest protocol differs from runtime")
    entries = select_validation_entries(
        manifest["entries"], args.maximum_per_scene)
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    reports = []
    for index, entry in enumerate(entries):
        record = label_entry(entry, args.seed + 10_000_019 * index)
        report = {
            "audit_kind": "systematic_rollout_authoritative_labels",
            "formal_dataset": False,
            "teacher_tier": "authoritative_v2",
            "protocol": SYNC_EVENT_V2.manifest(),
            "case": entry["scene_id"],
            "n_agents": entry["n_agents"],
            "seed": entry["seed"],
            "topology": entry["topology"],
            "source_trace": entry["trace"],
            "records": [record],
        }
        case_output = args.output / (
            f"{entry['scene_id']}_step{entry['snapshot_step']}.json")
        case_output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append({
            "scene_id": entry["scene_id"],
            "n_agents": entry["n_agents"],
            "snapshot_step": entry["snapshot_step"],
            "report": str(case_output),
            "accepted_for_training": record["accepted_for_training"],
            "teacher_relative_improvement_over_student": record[
                "teacher_relative_improvement_over_student"],
        })
    accepted_reports = [
        row["report"] for row in reports if row["accepted_for_training"]]
    rejected_reports = [
        row["report"] for row in reports if not row["accepted_for_training"]]
    summary = {
        "audit_kind": "systematic_rollout_label_pipeline_validation",
        "formal_dataset": False,
        "teacher_tier": "authoritative_v2",
        "protocol": SYNC_EVENT_V2.manifest(),
        "source_manifest": str(args.manifest),
        "selection_is_automatic": True,
        "maximum_per_scene": args.maximum_per_scene,
        "reports": reports,
        "accepted_report_count": len(accepted_reports),
        "rejected_report_count": len(rejected_reports),
        "accepted_reports": accepted_reports,
        "rejected_reports": rejected_reports,
        "wall_seconds": perf_counter() - started,
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
