"""Diagnose CEM sampling coverage versus approximate-cost misranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_v2_teacher_distillability import capture_first_active
from sbs824.v2.protocol import SYNC_EVENT_V2_DEV
from sbs824.v2.teacher import ComponentCEMTeacher, TeacherBudget


def finite_component_cost(values: np.ndarray, active_ids: np.ndarray) -> float:
    selected = np.asarray(values)[active_ids]
    selected = selected[np.isfinite(selected)]
    if not len(selected):
        return float("nan")
    return float(np.min(selected))


def run(case: str, seeds: list[int], output: Path) -> dict:
    started = perf_counter()
    (runtime, prepared, goals, gain, cfg, topology,
     scene_seed) = capture_first_active(case)
    active_ids = np.flatnonzero(prepared.refresh_mask)
    budget = TeacherBudget()
    seed_reports = []
    all_records = []

    for seed in seeds:
        teacher = ComponentCEMTeacher(
            SYNC_EVENT_V2_DEV, budget=budget, seed=seed,
            audit_exact_candidates=True)
        decision = teacher.decide(runtime, prepared, goals, gain, cfg)
        records = teacher.search_audit
        if not records:
            raise RuntimeError("search diagnostic captured no candidate records")
        sampled_exact = np.asarray([
            cost for record in records for cost in record["exact_costs"]
        ])
        approximate_selected_exact = np.asarray([
            record["exact_costs"][record["approximate_best_index"]]
            for record in records
        ])
        seed_report = {
            "seed": seed,
            "chosen_parameters": decision.parameters[active_ids].tolist(),
            "chosen_exact_cost": finite_component_cost(
                decision.teacher_cost, active_ids),
            "incumbent_exact_cost": finite_component_cost(
                decision.incumbent_cost, active_ids),
            "best_exact_cost_sampled": float(np.min(sampled_exact)),
            "best_exact_cost_among_approximate_population_winners": float(
                np.min(approximate_selected_exact)),
            "mean_rank_correlation": float(np.mean([
                record["rank_correlation"] for record in records
                if record["rank_correlation"] is not None
            ])),
            "maximum_approximate_winner_exact_regret": float(max(
                record["approximate_best_exact_regret"]
                for record in records)),
            "maximum_approximate_rank_of_exact_best": int(max(
                record["approximate_rank_of_exact_best"]
                for record in records)),
            "teacher_statistics": teacher.statistics(),
            "population_records": records,
        }
        seed_reports.append(seed_report)
        all_records.extend((seed, record) for record in records)

    global_best = min(
        report["best_exact_cost_sampled"] for report in seed_reports)
    for report in seed_reports:
        scale = max(abs(global_best), 1.0)
        report["sampling_regret_to_global_best"] = float(
            (report["best_exact_cost_sampled"] - global_best) / scale)
        report["chosen_regret_to_own_best_sampled"] = float(
            (report["chosen_exact_cost"]
             - report["best_exact_cost_sampled"])
            / max(abs(report["best_exact_cost_sampled"]), 1.0))

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "audit_kind": "teacher_search_sampling_vs_surrogate_diagnostic",
        "formal_dataset": False,
        "teacher_budget_tier": "historical_strong_12x3x40",
        "authoritative_for_gate1": False,
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "snapshot_step": runtime.step,
        "active_seeds": seeds,
        "active_agent_ids": active_ids.tolist(),
        "protocol": SYNC_EVENT_V2_DEV.manifest(),
        "budget": seed_reports[0]["teacher_statistics"]["budget"],
        "global_best_exact_cost_sampled": global_best,
        "seed_reports": seed_reports,
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "Large chosen-to-own-best regret indicates surrogate ranking loss. "
            "Large seed-best-to-global regret indicates sampling or "
            "initialization coverage loss."),
    }
    (output / "REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    compact = {
        "case": case,
        "wall_seconds": result["wall_seconds"],
        "global_best_exact_cost_sampled": global_best,
        "report": str(output / "REPORT.json"),
    }
    print(json.dumps(compact), flush=True)
    return result


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"),
        default="n5_partial")
    parser.add_argument("--teacher-seeds", type=parse_seeds,
                        default=parse_seeds("0,1,2,3,4"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.teacher_seeds, args.output)


if __name__ == "__main__":
    main()
