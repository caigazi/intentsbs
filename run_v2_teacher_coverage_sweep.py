"""Compare CEM sample count against independent-restart coverage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_v2_teacher_distillability import capture_first_active
from run_v2_teacher_search_diagnostic import parse_seeds
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2
from sbs824.v2.teacher import ComponentCEMTeacher, TeacherBudget


@dataclass(frozen=True)
class CoverageStrategy:
    samples: int
    iterations: int
    restarts: int

    @property
    def name(self) -> str:
        return f"{self.samples}x{self.iterations}x{self.restarts}r"

    @property
    def candidate_budget_per_branch(self) -> int:
        return self.samples * self.iterations * self.restarts


def parse_strategies(text: str) -> list[CoverageStrategy]:
    result = []
    for token in text.split(","):
        values = token.strip().lower().replace("r", "").split("x")
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                "strategies must use samplesxiterationsxrestarts")
        strategy = CoverageStrategy(*(int(value) for value in values))
        if min(strategy.samples, strategy.iterations, strategy.restarts) < 1:
            raise argparse.ArgumentTypeError("strategy values must be positive")
        result.append(strategy)
    if not result:
        raise argparse.ArgumentTypeError("provide at least one strategy")
    if len({strategy.name for strategy in result}) != len(result):
        raise argparse.ArgumentTypeError("strategies must be unique")
    return result


def independent_seed(trial_seed: int, restart: int) -> int:
    """Keep restart zero comparable with historical single-seed runs."""
    return trial_seed if restart == 0 else trial_seed + 1_000_003 * restart


def component_costs(values: np.ndarray, component_ids: np.ndarray,
                    active_ids: np.ndarray) -> list[dict]:
    """Return one non-duplicated exact cost for every active component."""
    values = np.asarray(values)
    component_ids = np.asarray(component_ids)
    result = []
    for component_id in sorted(set(component_ids[active_ids].tolist())):
        members = active_ids[component_ids[active_ids] == component_id]
        selected = values[members]
        selected = selected[np.isfinite(selected)]
        if not len(selected):
            raise RuntimeError(
                f"component {component_id} has no finite Teacher cost")
        if not np.allclose(selected, selected[0], atol=1e-10, rtol=0.0):
            raise RuntimeError(
                f"component {component_id} members disagree on Teacher cost")
        result.append({
            "component_id": int(component_id),
            "members": members.tolist(),
            "cost": float(selected[0]),
        })
    return result


def run(case: str, trial_seeds: list[int],
        strategies: list[CoverageStrategy], output: Path,
        *, numpy_spot_checks_per_strategy: int = 1) -> dict:
    started = perf_counter()
    (runtime, prepared, goals, gain, cfg, topology,
     scene_seed) = capture_first_active(case)
    active_ids = np.flatnonzero(prepared.refresh_mask)
    release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
    strategy_reports = []

    for strategy in strategies:
        trials = []
        numpy_spot_checks = []
        for trial_seed in trial_seeds:
            restart_reports = []
            for restart in range(strategy.restarts):
                search_seed = independent_seed(trial_seed, restart)
                budget = replace(
                    TeacherBudget(),
                    strong_samples=strategy.samples,
                    strong_iterations=strategy.iterations)
                teacher = ComponentCEMTeacher(
                    SYNC_EVENT_V2, budget=budget, seed=search_seed,
                    audit_exact_candidates=False)
                decision = teacher.decide(
                    runtime, prepared, goals, gain, cfg)
                if not np.all(decision.validated[active_ids]):
                    raise RuntimeError(
                        "coverage sweep produced an unvalidated active label")
                stage_components = component_costs(
                    decision.teacher_cost, decision.component_id, active_ids)
                exact_candidates = np.stack(
                    (decision.parameters, runtime.parameters))
                joint_costs = teacher.evaluate_candidates_exact_jax(
                    runtime, goals, gain, cfg, exact_candidates,
                    active_ids, release_ids, active_ids)
                final_joint_cost, incumbent_joint_cost = (
                    float(value) for value in joint_costs)
                final_components = []
                incumbent_components = []
                for item in stage_components:
                    members = np.asarray(item["members"], dtype=int)
                    exact_costs = teacher.evaluate_candidates_exact_jax(
                        runtime, goals, gain, cfg, exact_candidates,
                        active_ids, release_ids, members)
                    final_components.append({
                        "component_id": item["component_id"],
                        "members": item["members"],
                        "cost": float(exact_costs[0]),
                    })
                    incumbent_components.append({
                        "component_id": item["component_id"],
                        "members": item["members"],
                        "cost": float(exact_costs[1]),
                    })
                if len(numpy_spot_checks) < numpy_spot_checks_per_strategy:
                    numpy_costs = np.asarray([
                        teacher._rollout_cost(
                            runtime, goals, gain, cfg, candidate,
                            active_ids, release_ids, active_ids,
                            integration_substeps=(
                                SYNC_EVENT_V2.integration_substeps))
                        for candidate in exact_candidates
                    ])
                    if not np.allclose(
                            joint_costs, numpy_costs, atol=2e-2, rtol=2e-4):
                        raise RuntimeError(
                            "32-substep JAX/NumPy exact spot check failed")
                    numpy_spot_checks.append({
                        "trial_seed": trial_seed,
                        "restart": restart,
                        "jax_costs": joint_costs.tolist(),
                        "numpy_costs": numpy_costs.tolist(),
                        "maximum_absolute_error": float(np.max(
                            np.abs(joint_costs - numpy_costs))),
                        "maximum_relative_error": float(np.max(
                            np.abs(joint_costs - numpy_costs)
                            / np.maximum(np.abs(numpy_costs), 1.0))),
                    })
                restart_reports.append({
                    "restart": restart,
                    "search_seed": search_seed,
                    "chosen_exact_cost": final_joint_cost,
                    "incumbent_exact_cost": incumbent_joint_cost,
                    "component_stage_exact_costs": stage_components,
                    "component_exact_costs": final_components,
                    "component_incumbent_exact_costs": incumbent_components,
                    "chosen_parameters": decision.parameters[
                        active_ids].tolist(),
                    "teacher_statistics": teacher.statistics(),
                })
            best = min(
                restart_reports, key=lambda item: item["chosen_exact_cost"])
            trials.append({
                "trial_seed": trial_seed,
                "best_restart": best["restart"],
                "best_search_seed": best["search_seed"],
                "chosen_exact_cost": best["chosen_exact_cost"],
                "incumbent_exact_cost": best["incumbent_exact_cost"],
                "component_stage_exact_costs": (
                    best["component_stage_exact_costs"]),
                "component_exact_costs": best["component_exact_costs"],
                "component_incumbent_exact_costs": (
                    best["component_incumbent_exact_costs"]),
                "chosen_parameters": best["chosen_parameters"],
                "restarts": restart_reports,
            })
        strategy_reports.append({
            "strategy": asdict(strategy),
            "name": strategy.name,
            "candidate_budget_per_branch": (
                strategy.candidate_budget_per_branch),
            "numpy_exact_spot_checks": numpy_spot_checks,
            "trials": trials,
        })

    pooled_best = min(
        trial["chosen_exact_cost"]
        for strategy in strategy_reports for trial in strategy["trials"])
    scale = max(abs(pooled_best), 1.0)
    for strategy in strategy_reports:
        cost_relative_regrets = []
        improvement_normalized_regrets = []
        costs = []
        for trial in strategy["trials"]:
            cost_regret = (
                trial["chosen_exact_cost"] - pooled_best) / scale
            improvement_scale = max(
                trial["incumbent_exact_cost"] - pooled_best, 1.0)
            improvement_regret = (
                trial["chosen_exact_cost"] - pooled_best) / improvement_scale
            trial["cost_relative_regret_to_pooled_best"] = float(cost_regret)
            trial["improvement_normalized_regret"] = float(improvement_regret)
            cost_relative_regrets.append(cost_regret)
            improvement_normalized_regrets.append(improvement_regret)
            costs.append(trial["chosen_exact_cost"])
        strategy["summary"] = {
            "minimum_exact_cost": float(np.min(costs)),
            "median_exact_cost": float(np.median(costs)),
            "maximum_exact_cost": float(np.max(costs)),
            "median_cost_relative_regret": float(
                np.median(cost_relative_regrets)),
            "maximum_cost_relative_regret": float(
                np.max(cost_relative_regrets)),
            "median_improvement_normalized_regret": float(
                np.median(improvement_normalized_regrets)),
            "p95_improvement_normalized_regret": float(
                np.percentile(improvement_normalized_regrets, 95)),
            "maximum_improvement_normalized_regret": float(
                np.max(improvement_normalized_regrets)),
            "fraction_within_1_percent_improvement_regret": float(
                np.mean(np.asarray(improvement_normalized_regrets) <= 0.01)),
            "fraction_within_5_percent_improvement_regret": float(
                np.mean(np.asarray(improvement_normalized_regrets) <= 0.05)),
        }

    component_keys = [
        item["component_id"]
        for item in strategy_reports[0]["trials"][0][
            "component_exact_costs"]]
    pooled_component_best = {
        component_id: min(
            item["cost"]
            for strategy in strategy_reports
            for trial in strategy["trials"]
            for item in trial["component_exact_costs"]
            if item["component_id"] == component_id)
        for component_id in component_keys
    }
    for strategy in strategy_reports:
        component_summaries = []
        for component_id in component_keys:
            costs = []
            regrets = []
            members = None
            for trial in strategy["trials"]:
                chosen = next(
                    item for item in trial["component_exact_costs"]
                    if item["component_id"] == component_id)
                incumbent = next(
                    item for item in trial["component_incumbent_exact_costs"]
                    if item["component_id"] == component_id)
                best = pooled_component_best[component_id]
                scale = max(incumbent["cost"] - best, 1.0)
                costs.append(chosen["cost"])
                regrets.append((chosen["cost"] - best) / scale)
                members = chosen["members"]
            component_summaries.append({
                "component_id": component_id,
                "members": members,
                "pooled_best_exact_cost": pooled_component_best[component_id],
                "median_exact_cost": float(np.median(costs)),
                "maximum_exact_cost": float(np.max(costs)),
                "p95_improvement_normalized_regret": float(
                    np.percentile(regrets, 95)),
                "maximum_improvement_normalized_regret": float(
                    np.max(regrets)),
                "fraction_within_1_percent_improvement_regret": float(
                    np.mean(np.asarray(regrets) <= 0.01)),
            })
        strategy["component_summaries"] = component_summaries

    output.mkdir(parents=True, exist_ok=False)
    result = {
        "audit_kind": "teacher_sampling_restart_coverage_sweep",
        "formal_dataset": False,
        "authoritative_for_gate1": False,
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "snapshot_step": runtime.step,
        "trial_seeds": trial_seeds,
        "active_agent_ids": active_ids.tolist(),
        "protocol": SYNC_EVENT_V2.manifest(),
        "fixed_budget": {
            "horizon_steps": TeacherBudget().horizon_steps,
            "planning_integration_substeps": (
                TeacherBudget().planning_integration_substeps),
            "candidate_backend": TeacherBudget().candidate_backend,
            "exact_shortlist_reranking": True,
            "authoritative_validation_backend": "jax_x64_32_substeps",
            "numpy_spot_checks_per_strategy": (
                numpy_spot_checks_per_strategy),
        },
        "pooled_best_exact_cost": pooled_best,
        "strategies": strategy_reports,
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "All candidates use 32-substep JAX ranking. Each branch winner, "
            "final combined-component decision, incumbent, and cross-restart "
            "winner is compared by the same 32-substep JAX x64 objective. "
            "NumPy is retained only for bounded numerical spot checks. Component "
            "summaries re-evaluate the final combined action rather than "
            "summing order-dependent intermediate search costs. This audit "
            "measures sampling coverage only."),
    }
    report_path = output / "REPORT.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "case": case,
        "wall_seconds": result["wall_seconds"],
        "pooled_best_exact_cost": pooled_best,
        "summaries": {
            report["name"]: report["summary"]
            for report in strategy_reports
        },
        "report": str(report_path),
    }, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"),
        default="n5_partial")
    parser.add_argument("--trial-seeds", type=parse_seeds,
                        default=parse_seeds("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument(
        "--strategies", type=parse_strategies,
        default=parse_strategies(
            "12x3x1,24x3x1,12x3x2,48x3x1,12x3x4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numpy-spot-checks-per-strategy", type=int,
                        default=1)
    args = parser.parse_args()
    if args.numpy_spot_checks_per_strategy < 0:
        parser.error("--numpy-spot-checks-per-strategy must be nonnegative")
    run(args.case, args.trial_seeds, args.strategies, args.output,
        numpy_spot_checks_per_strategy=(
            args.numpy_spot_checks_per_strategy))


if __name__ == "__main__":
    main()
