"""Train one development-only small-DAgger Student from D0 plus queried labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from run_v2_tiny_student_overfit import parse_observation, train
from sbs824.v2.protocol import SYNC_EVENT_V2
from sbs824.v2.student import IntentStudent, StudentArchitecture


def teacher_record_is_accepted(row: dict, relative_tolerance: float = 1e-8
                               ) -> bool:
    """Reject an optimizer miss instead of teaching a worse action to DAgger."""
    keys = ("teacher_exact_cost", "student_exact_cost", "incumbent_exact_cost")
    if not all(key in row for key in keys):
        return True
    baseline = min(float(row["student_exact_cost"]),
                   float(row["incumbent_exact_cost"]))
    tolerance = relative_tolerance * max(abs(baseline), 1.0)
    return bool(float(row["teacher_exact_cost"]) <= baseline + tolerance)


def load_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("teacher_tier") != "authoritative_v2":
        raise RuntimeError(f"non-authoritative report rejected: {path}")
    if report.get("formal_dataset", True):
        raise RuntimeError(f"formal dataset unexpectedly supplied: {path}")
    if report.get("protocol") != SYNC_EVENT_V2.manifest():
        raise RuntimeError(f"protocol manifest differs: {path}")
    return report


def queried_report_paths(summary_paths: list[Path]) -> list[Path]:
    paths = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("protocol") != SYNC_EVENT_V2.manifest():
            raise RuntimeError(f"query summary protocol differs: {summary_path}")
        paths.extend(Path(row["report"]) for row in summary["reports"])
    return paths


def load_compatibility_overrides(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("protocol") != SYNC_EVENT_V2.manifest():
        raise RuntimeError(f"compatibility audit protocol differs: {path}")
    if audit.get("audit_kind") != "student_compatible_constrained_cem_exact_audit":
        raise RuntimeError(f"unexpected compatibility audit: {path}")
    return {str(Path(row["report"])): row for row in audit["records"]}


def build_training_data(seed_paths: list[Path], query_paths: list[Path],
                        compatibility: dict[str, dict] | None = None):
    compatibility = compatibility or {}
    sources = [("D0", path) for path in seed_paths]
    sources.extend(("queried", path) for path in query_paths)
    loaded = [(split, path, load_report(path)) for split, path in sources]
    slots = max(
        len(row["parameters"]) - 1
        for _, _, report in loaded for row in report["records"])
    own_rows, edge_rows, mask_rows, labels, provenance = [], [], [], [], []
    rejected = []
    accepted_states = Counter()
    accepted_agents = Counter()
    canonicalized_states = Counter()
    canonicalized_agents = Counter()
    for split, path, report in loaded:
        for row in report["records"]:
            if split == "queried" and not teacher_record_is_accepted(row):
                rejected.append({
                    "reason": "teacher_worse_than_available_baseline",
                    "report": str(path),
                    "case": report["case"],
                    "runtime_step": int(row["runtime_step"]),
                    "agent_labels": len(row["active_ids"]),
                    "teacher_exact_cost": float(row["teacher_exact_cost"]),
                    "student_exact_cost": float(row["student_exact_cost"]),
                    "incumbent_exact_cost": float(row["incumbent_exact_cost"]),
                })
                continue
            parameters_value = row["parameters"]
            compatible = compatibility.get(str(path)) if split == "queried" else None
            if compatible is not None:
                if not compatible["inside_one_percent_quality_band"]:
                    rejected.append({
                        "reason": "student_compatible_search_outside_quality_band",
                        "report": str(path),
                        "case": report["case"],
                        "runtime_step": int(row["runtime_step"]),
                        "agent_labels": len(row["active_ids"]),
                        "student_compatible_improvement_regret": float(
                            compatible["student_compatible_improvement_regret"]),
                    })
                    continue
                parameters_value = compatible["student_compatible_parameters"]
                canonicalized_states[split] += 1
                canonicalized_agents[split] += len(row["active_ids"])
            accepted_states[split] += 1
            parameters = np.asarray(parameters_value, dtype=np.float32)
            for agent in row["active_ids"]:
                own, edges, mask = parse_observation(
                    row["local_observations"][str(agent)], slots)
                own_rows.append(own)
                edge_rows.append(edges)
                mask_rows.append(mask)
                labels.append(parameters[agent])
                provenance.append({
                    "split": split,
                    "report": str(path),
                    "case": report["case"],
                    "runtime_step": int(row["runtime_step"]),
                    "agent": int(agent),
                })
                accepted_agents[split] += 1
    data = {
        "self": np.asarray(own_rows, dtype=np.float32),
        "edges": np.asarray(edge_rows, dtype=np.float32),
        "mask": np.asarray(mask_rows, dtype=bool),
        "labels": np.asarray(labels, dtype=np.float32),
    }
    return (data, provenance, dict(accepted_states), dict(accepted_agents),
            rejected, dict(canonicalized_states), dict(canonicalized_agents))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--label-summary", nargs="+", type=Path, required=True)
    parser.add_argument("--compatibility-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("version") != "small_dagger_v1":
        raise RuntimeError("unsupported small-DAgger protocol")
    if not 1 <= args.round <= int(protocol["maximum_rounds"]):
        raise RuntimeError("round lies outside the frozen protocol")
    seed_paths = [Path(path) for path in protocol["seed_dataset_reports"]]
    query_paths = queried_report_paths(args.label_summary)
    compatibility = load_compatibility_overrides(args.compatibility_audit)
    (data, provenance, state_counts, agent_counts, rejected,
     canonicalized_states, canonicalized_agents) = build_training_data(
        seed_paths, query_paths, compatibility)
    architecture = StudentArchitecture()
    student_spec = protocol["student"]
    model, params, history, wall_seconds = train(
        data, int(student_spec["seed"]),
        int(student_spec["development_gradient_steps"]), 3e-3, architecture)
    fitted = np.asarray(model.apply(
        {"params": params}, jnp.asarray(data["self"]),
        jnp.asarray(data["edges"]), jnp.asarray(data["mask"])))
    fitted_l2 = np.linalg.norm(fitted - data["labels"], axis=1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(params))
    report = {
        "audit_kind": "small_dagger_development_training",
        "formal_dataset": False,
        "formal_checkpoint": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "dagger_protocol": protocol,
        "dagger_round": args.round,
        "architecture": architecture.manifest(),
        "parameter_count": int(sum(
            value.size for value in jax.tree_util.tree_leaves(params))),
        "training_sources": {
            "D0": [str(path) for path in seed_paths],
            "queried_summaries": [str(path) for path in args.label_summary],
            "compatibility_audit": (
                str(args.compatibility_audit)
                if args.compatibility_audit is not None else None),
        },
        "accepted_state_counts": state_counts,
        "accepted_agent_label_counts": agent_counts,
        "canonicalized_state_counts": canonicalized_states,
        "canonicalized_agent_label_counts": canonicalized_agents,
        "rejected_query_count": len(rejected),
        "rejected_agent_label_count": sum(
            row["agent_labels"] for row in rejected),
        "rejected_queries": rejected,
        "training_samples": len(data["labels"]),
        "training_steps": int(student_spec["development_gradient_steps"]),
        "training_loss": "logit-space MSE",
        "seed": int(student_spec["seed"]),
        "history": history,
        "final": history[-1],
        "maximum_training_action_l2": float(fitted_l2.max()),
        "mean_training_action_l2": float(fitted_l2.mean()),
        "wall_seconds": wall_seconds,
        "interpretation": (
            "Development small-DAgger checkpoint retrained from scratch. "
            "Queried states where the stochastic Teacher failed to beat both "
            "the Student and incumbent under the same exact objective abstain "
            "and do not enter supervision. Conflicting Teacher labels are "
            "replaced only by constrained-CEM actions that satisfy the shared "
            "Student symmetry and the exact one-percent quality gate."),
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_samples": report["training_samples"],
        "accepted_agent_label_counts": agent_counts,
        "canonicalized_agent_label_counts": canonicalized_agents,
        "rejected_query_count": len(rejected),
        "final": report["final"],
        "wall_seconds": wall_seconds,
        "report": str(args.output / "REPORT.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
