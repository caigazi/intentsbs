"""Audit whether Student-equivalent Teacher labels admit a safe shared projection."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from run_v2_label_rollout_manifest import runtime_from_trace
from run_v2_tiny_student_overfit import parse_observation
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import prepare_step
from sbs824.v2.teacher import AuthoritativeComponentTeacher, ComponentCEMTeacher


def canonical_observation_key(values, slots: int, decimals: int = 6) -> bytes:
    own, edges, mask = parse_observation(values, slots)
    active = edges[mask]
    if len(active):
        order = np.lexsort(tuple(
            active[:, column]
            for column in range(active.shape[1] - 1, -1, -1)))
        active = active[order]
    canonical = np.concatenate((
        own, active.reshape(-1), np.asarray([len(active)], dtype=np.float32)))
    return np.round(canonical, decimals).tobytes()


def equivalence_groups(record: dict) -> list[list[int]]:
    slots = len(record["parameters"]) - 1
    grouped: dict[bytes, list[int]] = {}
    for agent in record["active_ids"]:
        key = canonical_observation_key(
            record["local_observations"][str(agent)], slots)
        grouped.setdefault(key, []).append(int(agent))
    return [agents for agents in grouped.values() if len(agents) > 1]


def project_shared_labels(parameters, groups: list[list[int]]):
    projected = np.asarray(parameters, dtype=float).copy()
    changed_groups = []
    for agents in groups:
        actions = projected[agents]
        maximum_gap = max(
            float(np.linalg.norm(left - right))
            for left in actions for right in actions)
        if maximum_gap <= 0.1:
            continue
        projected[agents] = np.mean(actions, axis=0)
        changed_groups.append({
            "agents": agents,
            "maximum_original_action_l2": maximum_gap,
            "shared_parameters": projected[agents[0]].tolist(),
        })
    return projected, changed_groups


def audit_report(path: Path, evaluator, teacher_budget) -> dict | None:
    report = json.loads(path.read_text(encoding="utf-8"))
    record = report["records"][0]
    if record.get("accepted_for_training") is False:
        return None
    projected, changed_groups = project_shared_labels(
        record["parameters"], equivalence_groups(record))
    if not changed_groups:
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
    active_ids = np.asarray(record["active_ids"], dtype=int)
    student = np.asarray(record["student_parameters"], dtype=float)
    candidates = np.stack((
        np.asarray(record["parameters"], dtype=float), projected,
        student, runtime.parameters))
    exact = np.asarray(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, candidates,
        np.flatnonzero(prepared.mode == IntentMode.ACTIVE),
        np.flatnonzero(prepared.mode == IntentMode.RELEASE),
        active_ids), dtype=float)
    denominator = max(float(exact[3] - exact[0]), 1.0)
    regret = float((exact[1] - exact[0]) / denominator)
    return {
        "report": str(path),
        "case": report["case"],
        "n_agents": n_agents,
        "runtime_step": step,
        "changed_groups": changed_groups,
        "teacher_exact_cost": float(exact[0]),
        "shared_projection_exact_cost": float(exact[1]),
        "student_exact_cost": float(exact[2]),
        "incumbent_exact_cost": float(exact[3]),
        "shared_projection_improvement_regret": regret,
        "inside_one_percent_quality_band": bool(regret <= 0.01),
        "projected_parameters": projected.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-summary", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = AuthoritativeComponentTeacher(SYNC_EVENT_V2, seed=0)
    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=teacher.search_budget, seed=0)
    source_reports = []
    for summary_path in args.label_summary:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol") != SYNC_EVENT_V2.manifest():
            raise RuntimeError(f"protocol manifest differs: {summary_path}")
        source_reports.extend(Path(row["report"]) for row in summary["reports"])
    records = []
    for path in source_reports:
        result = audit_report(path, evaluator, teacher.search_budget)
        if result is not None:
            records.append(result)
    by_n = Counter(row["n_agents"] for row in records)
    passed_by_n = Counter(
        row["n_agents"] for row in records
        if row["inside_one_percent_quality_band"])
    report = {
        "audit_kind": "student_equivalent_label_projection_exact_cost",
        "formal_dataset": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "source_summaries": [str(path) for path in args.label_summary],
        "conflicting_state_count": len(records),
        "conflicting_states_by_n": dict(sorted(by_n.items())),
        "projection_pass_count": sum(
            row["inside_one_percent_quality_band"] for row in records),
        "projection_pass_by_n": dict(sorted(passed_by_n.items())),
        "worst_projection_improvement_regret": max(
            (row["shared_projection_improvement_regret"] for row in records),
            default=None),
        "records": records,
        "interpretation": (
            "The shared projection is the MSE projection of inconsistent "
            "Teacher labels onto the action subspace realizable by the shared "
            "permutation-invariant Student. It is accepted only when the same "
            "32-substep exact objective keeps it in the one-percent quality band."),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        key: value for key, value in report.items() if key != "records"
    }), flush=True)


if __name__ == "__main__":
    main()
