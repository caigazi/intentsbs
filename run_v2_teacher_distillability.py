"""Gate-1 audit for Teacher label stability and local equivariance.

This script deliberately writes an audit report, not a formal training dataset.
It freezes one first-ACTIVE decision snapshot, repeats independent Teacher
searches, and compares identity, rotation, permutation, and reflection variants.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import (RuntimeState, apply_prepared_step,
                               initialize_runtime, prepare_step)
from sbs824.v2.scenes import headon_scene, mixed_crossing_scene
from sbs824.v2.teacher import (AuthoritativeTeacherBudget,
                               ComponentCEMTeacher, TeacherBudget)


@dataclass(frozen=True)
class Variant:
    name: str
    matrix: np.ndarray
    order: np.ndarray

    @property
    def handedness(self) -> float:
        return float(np.linalg.det(self.matrix))


def load_case(name: str) -> tuple[np.ndarray, np.ndarray, str, int]:
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


def capture_first_active(name: str):
    state, goals, topology, scene_seed = load_case(name)
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(
        state, goals, gain, cfg, SYNC_EVENT_V2)
    for _ in range(cfg.steps):
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)
        if prepared.refresh_mask.any():
            return runtime, prepared, goals, gain, cfg, topology, scene_seed
        runtime, _ = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2,
            runtime.parameters)
    raise RuntimeError(f"{name} never reached an ACTIVE decision")


def transform_snapshot(runtime: RuntimeState, goals: np.ndarray,
                       variant: Variant) -> tuple[RuntimeState, np.ndarray]:
    matrix = np.asarray(variant.matrix, dtype=float)
    order = np.asarray(variant.order, dtype=int)
    physical = runtime.physical.copy()
    physical[:, :2] = physical[:, :2] @ matrix.T
    physical[:, 2:] = physical[:, 2:] @ matrix.T
    parameters = runtime.parameters.copy()
    if variant.handedness < 0.0:
        parameters[:, 1] *= -1.0
    transformed = RuntimeState(
        physical=physical[order].copy(),
        parameters=parameters[order].copy(),
        world_intent=(runtime.world_intent @ matrix.T)[order].copy(),
        mode=runtime.mode[order].copy(),
        clear_counts=runtime.clear_counts[order].copy(),
        braking_latch=runtime.braking_latch[order].copy(),
        previous_safe_force=(
            runtime.previous_safe_force @ matrix.T)[order].copy(),
        step=runtime.step,
    )
    return transformed, (np.asarray(goals) @ matrix.T)[order].copy()


def map_agents_back(values: np.ndarray, variant: Variant,
                    *, action: bool = False) -> np.ndarray:
    result = np.empty_like(values)
    result[variant.order] = values
    if action and variant.handedness < 0.0:
        result[:, 1] *= -1.0
    return result


def map_action_to_variant(parameters: np.ndarray,
                          variant: Variant) -> np.ndarray:
    result = np.asarray(parameters, dtype=float).copy()
    if variant.handedness < 0.0:
        result[:, 1] *= -1.0
    return result[variant.order].copy()


def feature_equivariance(reference, transformed, variant: Variant) -> dict:
    mapped_self = map_agents_back(
        transformed.self_features, variant).astype(float)
    if variant.handedness < 0.0:
        mapped_self[:, [2, 4, 6]] *= -1.0
    self_error = float(np.max(np.abs(
        mapped_self - reference.self_features.astype(float))))

    edge_error = 0.0
    compared = 0
    for new_ego, old_ego in enumerate(variant.order):
        for slot in np.flatnonzero(transformed.edge_mask[new_ego]):
            new_neighbor = int(transformed.neighbor_ids[new_ego, slot])
            old_neighbor = int(variant.order[new_neighbor])
            candidates = np.flatnonzero(
                reference.edge_mask[old_ego]
                & (reference.neighbor_ids[old_ego] == old_neighbor))
            if len(candidates) != 1:
                raise RuntimeError("transformed neighbor mapping is ambiguous")
            row = transformed.edge_features[new_ego, slot].astype(float).copy()
            if variant.handedness < 0.0:
                row[[1, 3, 5]] *= -1.0
            target = reference.edge_features[old_ego, candidates[0]].astype(float)
            edge_error = max(edge_error, float(np.max(np.abs(row - target))))
            compared += 1
    expected = int(np.count_nonzero(reference.edge_mask))
    if compared != expected:
        raise RuntimeError(
            f"edge comparison count differs: {compared} != {expected}")
    return {
        "maximum_self_feature_error": self_error,
        "maximum_edge_feature_error": edge_error,
        "directed_edges_compared": compared,
    }


def teacher_batch(runtime: RuntimeState, prepared, goals: np.ndarray,
                  gain: np.ndarray, cfg, seeds: list[int],
                  budget: TeacherBudget, variant: Variant,
                  restarts: int) -> dict:
    parameters = []
    costs = []
    incumbents = []
    validated = []
    statistics = []
    restart_reports = []
    active_ids = np.flatnonzero(prepared.mode == 1)
    release_ids = np.flatnonzero(prepared.mode == 2)
    focus_ids = np.flatnonzero(prepared.refresh_mask)
    for seed in seeds:
        trial_restarts = []
        for restart in range(restarts):
            search_seed = (seed if restart == 0
                           else seed + 1_000_003 * restart)
            teacher = ComponentCEMTeacher(
                SYNC_EVENT_V2, budget=budget, seed=search_seed)
            decision = teacher.decide(runtime, prepared, goals, gain, cfg)
            exact_pair = teacher.evaluate_candidates_exact_jax(
                runtime, goals, gain, cfg,
                np.stack((decision.parameters, runtime.parameters)),
                active_ids, release_ids, focus_ids)
            cost_row = np.full(cfg.n_agents, np.nan)
            incumbent_row = np.full(cfg.n_agents, np.nan)
            cost_row[focus_ids] = float(exact_pair[0])
            incumbent_row[focus_ids] = float(exact_pair[1])
            trial_restarts.append({
                "restart": restart,
                "search_seed": search_seed,
                "joint_exact_cost": float(exact_pair[0]),
                "incumbent_joint_exact_cost": float(exact_pair[1]),
                "raw_parameters": decision.parameters.copy(),
                "raw_component_id": decision.component_id.copy(),
                "parameters": map_agents_back(
                    decision.parameters, variant, action=True),
                "costs": map_agents_back(cost_row, variant),
                "incumbents": map_agents_back(incumbent_row, variant),
                "validated": map_agents_back(decision.validated, variant),
                "positive_branch_cost": map_agents_back(
                    decision.positive_branch_cost, variant),
                "negative_branch_cost": map_agents_back(
                    decision.negative_branch_cost, variant),
                "branch_near_tie": map_agents_back(
                    decision.branch_near_tie, variant),
                "statistics": teacher.statistics(),
            })
        minimum_cost = min(
            item["joint_exact_cost"] for item in trial_restarts)
        incumbent_cost = trial_restarts[0]["incumbent_joint_exact_cost"]
        improvement_scale = max(incumbent_cost - minimum_cost, 1.0)
        admissible = []
        for item in trial_restarts:
            item["improvement_normalized_regret"] = (
                (item["joint_exact_cost"] - minimum_cost)
                / improvement_scale)
            if (item["improvement_normalized_regret"]
                    <= budget.tie_fraction):
                mismatches = 0
                component_ids = item["raw_component_id"]
                for component_id in sorted(set(
                        component_ids[focus_ids].tolist())):
                    members = focus_ids[
                        component_ids[focus_ids] == component_id]
                    prior = float(np.mean(runtime.parameters[members, 1]))
                    desired_sign = -1.0 if prior < -0.03 else 1.0
                    selected_beta = float(np.mean(
                        item["raw_parameters"][members, 1]))
                    mismatches += int(
                        desired_sign * selected_beta <= 0.03)
                item["canonical_chirality_mismatches"] = mismatches
                admissible.append(item)
        best = min(admissible, key=lambda item: (
            item["canonical_chirality_mismatches"],
            item["joint_exact_cost"]))
        parameters.append(best["parameters"])
        costs.append(best["costs"])
        incumbents.append(best["incumbents"])
        validated.append(best["validated"])
        statistics.append(best["statistics"])
        restart_reports.append({
            "teacher_seed": seed,
            "selected_restart": best["restart"],
            "selected_search_seed": best["search_seed"],
            "joint_exact_costs": [
                item["joint_exact_cost"] for item in trial_restarts],
            "improvement_normalized_regrets": [
                item["improvement_normalized_regret"]
                for item in trial_restarts],
            "admissible_restart_count": len(admissible),
            "selected_canonical_chirality_mismatches": (
                best["canonical_chirality_mismatches"]),
            "selected_positive_branch_cost": (
                best["positive_branch_cost"].tolist()),
            "selected_negative_branch_cost": (
                best["negative_branch_cost"].tolist()),
            "selected_branch_near_tie": best["branch_near_tie"].tolist(),
        })
    return {
        "parameters": np.asarray(parameters),
        "costs": np.asarray(costs),
        "incumbents": np.asarray(incumbents),
        "validated": np.asarray(validated),
        "statistics": statistics,
        "restart_reports": restart_reports,
    }


def label_summary(batch: dict, active_ids: np.ndarray) -> dict:
    labels = batch["parameters"][:, active_ids]
    beta = labels[..., 1]
    centered = labels - labels.mean(axis=0, keepdims=True)
    pairwise = labels[:, None] - labels[None, :]
    costs = batch["costs"][:, active_ids]
    incumbents = batch["incumbents"][:, active_ids]
    improvement = (incumbents - costs) / np.maximum(np.abs(incumbents), 1.0)
    return {
        "active_agent_ids": active_ids.tolist(),
        "labels_by_seed": labels.tolist(),
        "mean_label_by_agent": labels.mean(axis=0).tolist(),
        "std_label_by_agent": labels.std(axis=0).tolist(),
        "maximum_centered_action_l2": float(np.max(
            np.linalg.norm(centered, axis=-1))),
        "maximum_pairwise_action_l2": float(np.max(
            np.linalg.norm(pairwise, axis=-1))),
        "positive_beta_counts": np.sum(beta > 0.03, axis=0).tolist(),
        "negative_beta_counts": np.sum(beta < -0.03, axis=0).tolist(),
        "near_zero_beta_counts": np.sum(np.abs(beta) <= 0.03, axis=0).tolist(),
        "relative_improvement_by_seed": np.mean(improvement, axis=1).tolist(),
        "median_relative_improvement": float(np.median(improvement)),
        "worse_than_incumbent_count": int(np.count_nonzero(costs > incumbents)),
    }


def compare_batches(reference: dict, candidate: dict,
                    active_ids: np.ndarray) -> dict:
    expected = reference["parameters"][:, active_ids]
    actual = candidate["parameters"][:, active_ids]
    same_seed = np.abs(actual - expected)
    mean_error = np.abs(actual.mean(axis=0) - expected.mean(axis=0))
    std_error = np.abs(actual.std(axis=0) - expected.std(axis=0))
    expected_cost = reference["costs"][:, active_ids]
    actual_cost = candidate["costs"][:, active_ids]
    incumbent_cost = reference["incumbents"][:, active_ids]
    best_cost = np.minimum(expected_cost, actual_cost)
    improvement_scale = np.maximum(incumbent_cost - best_cost, 1.0)
    return {
        "maximum_same_seed_action_error": float(np.max(same_seed)),
        "maximum_distribution_mean_error": float(np.max(mean_error)),
        "maximum_distribution_std_error": float(np.max(std_error)),
        "maximum_relative_cost_error": float(np.nanmax(
            np.abs(actual_cost - expected_cost)
            / np.maximum(np.abs(expected_cost), 1.0))),
        "maximum_improvement_normalized_cost_difference": float(np.nanmax(
            np.abs(actual_cost - expected_cost) / improvement_scale)),
    }


def make_fixed_candidates(runtime: RuntimeState, active_ids: np.ndarray,
                          count: int, seed: int) -> np.ndarray:
    if count < 4:
        raise ValueError("fixed-candidate audit requires at least four actions")
    rng = np.random.default_rng(seed)
    candidates = np.repeat(runtime.parameters[None], count, axis=0)
    candidates[:, active_ids] = rng.uniform(
        -1.0, 1.0, size=(count, len(active_ids), 2))
    candidates[0] = runtime.parameters
    candidates[1, active_ids] = np.array([1.0, 0.0])
    candidates[2, active_ids, 0] = 0.75
    candidates[2, active_ids, 1] = 0.60
    candidates[3, active_ids, 0] = 0.75
    candidates[3, active_ids, 1] = -0.60
    return candidates


def exact_candidate_costs(runtime: RuntimeState, prepared,
                          goals: np.ndarray, gain: np.ndarray, cfg,
                          budget: TeacherBudget,
                          candidates: np.ndarray) -> np.ndarray:
    active_ids = np.flatnonzero(prepared.mode == 1)
    release_ids = np.flatnonzero(prepared.mode == 2)
    focus_ids = np.flatnonzero(prepared.refresh_mask)
    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=budget, seed=0)
    return evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, candidates, active_ids, release_ids,
        focus_ids)


def fixed_candidate_equivariance(
        runtime: RuntimeState, reference_prepared, goals: np.ndarray,
        gain: np.ndarray, cfg, budget: TeacherBudget,
        variants: list[Variant], candidate_count: int,
        candidate_seed: int) -> tuple[dict, dict]:
    active_ids = np.flatnonzero(reference_prepared.refresh_mask)
    candidates = make_fixed_candidates(
        runtime, active_ids, candidate_count, candidate_seed)
    reference_costs = exact_candidate_costs(
        runtime, reference_prepared, goals, gain, cfg, budget, candidates)
    reports = {}
    contexts = {}
    for variant in variants:
        transformed_runtime, transformed_goals = transform_snapshot(
            runtime, goals, variant)
        transformed_prepared = prepare_step(
            transformed_runtime, transformed_goals, gain, cfg,
            SYNC_EVENT_V2)
        mapped_refresh = map_agents_back(
            transformed_prepared.refresh_mask, variant)
        if not np.array_equal(mapped_refresh, reference_prepared.refresh_mask):
            raise RuntimeError(f"{variant.name} changed the ACTIVE refresh mask")
        transformed_candidates = np.asarray([
            map_action_to_variant(parameters, variant)
            for parameters in candidates
        ])
        costs = exact_candidate_costs(
            transformed_runtime, transformed_prepared, transformed_goals,
            gain, cfg, budget, transformed_candidates)
        difference = np.abs(costs - reference_costs)
        scale = np.maximum(np.abs(reference_costs), 1.0)
        feature_report = feature_equivariance(
            reference_prepared.features, transformed_prepared.features,
            variant)
        reports[variant.name] = {
            "feature_equivariance": feature_report,
            "maximum_absolute_cost_error": float(np.max(difference)),
            "maximum_relative_cost_error": float(np.max(difference / scale)),
            "mean_relative_cost_error": float(np.mean(difference / scale)),
        }
        contexts[variant.name] = {
            "runtime": transformed_runtime,
            "prepared": transformed_prepared,
            "goals": transformed_goals,
        }
    maximum_relative = max(
        report["maximum_relative_cost_error"] for report in reports.values())
    maximum_feature = max(
        max(report["feature_equivariance"]["maximum_self_feature_error"],
            report["feature_equivariance"]["maximum_edge_feature_error"])
        for report in reports.values())
    return {
        "candidate_count": candidate_count,
        "candidate_seed": candidate_seed,
        "exact_integration_substeps": SYNC_EVENT_V2.integration_substeps,
        "variants": reports,
        "maximum_relative_cost_error": maximum_relative,
        "maximum_feature_error": maximum_feature,
        "provisional_pass": bool(
            maximum_relative <= 1e-10 and maximum_feature <= 1e-6),
    }, contexts


def cross_seed_action_cost(runtime: RuntimeState, prepared,
                           goals: np.ndarray, gain: np.ndarray, cfg,
                           budget: TeacherBudget, batch: dict,
                           active_ids: np.ndarray) -> dict:
    exact_costs = exact_candidate_costs(
        runtime, prepared, goals, gain, cfg, budget,
        batch["parameters"])
    flattened = batch["parameters"][:, active_ids].reshape(
        len(exact_costs), -1)
    action_distance = np.linalg.norm(
        flattened[:, None] - flattened[None, :], axis=-1)
    best = float(np.min(exact_costs))
    cost_relative_regret = (exact_costs - best) / max(abs(best), 1.0)
    incumbent_costs = batch["incumbents"][:, active_ids][:, 0]
    improvement_scale = np.maximum(incumbent_costs - best, 1.0)
    improvement_regret = (exact_costs - best) / improvement_scale
    mean_parameters = batch["parameters"].mean(axis=0)
    mean_action_cost = float(exact_candidate_costs(
        runtime, prepared, goals, gain, cfg, budget,
        mean_parameters[None])[0])
    common_incumbent = float(incumbent_costs[0])
    mean_action_improvement_regret = (
        (mean_action_cost - best)
        / max(common_incumbent - best, 1.0))
    medoid_index = int(np.argmin(action_distance.sum(axis=1)))
    return {
        "exact_cost_by_seed": exact_costs.tolist(),
        "cost_relative_regret_by_seed": cost_relative_regret.tolist(),
        "improvement_normalized_regret_by_seed": improvement_regret.tolist(),
        "action_l2_distance_matrix": action_distance.tolist(),
        "maximum_action_l2_distance": float(np.max(action_distance)),
        "maximum_cost_relative_regret": float(
            np.max(cost_relative_regret)),
        "median_cost_relative_regret": float(
            np.median(cost_relative_regret)),
        "maximum_improvement_normalized_regret": float(
            np.max(improvement_regret)),
        "median_improvement_normalized_regret": float(
            np.median(improvement_regret)),
        "mean_action_exact_cost": mean_action_cost,
        "mean_action_improvement_normalized_regret": float(
            mean_action_improvement_regret),
        "medoid_seed_index": medoid_index,
        "medoid_exact_cost": float(exact_costs[medoid_index]),
        "best_seed_index": int(np.argmin(exact_costs)),
        "interpretation": (
            "Large action distance with small regret indicates equivalent "
            "modes. The mean-action exact regret directly tests whether an "
            "MSE-style Student average remains a good label."),
    }


def run_audit(case: str, seeds: list[int], budget: TeacherBudget,
              output: Path, candidate_count: int,
              candidate_seed: int, *, debug_quick: bool,
              restarts: int) -> dict:
    started = perf_counter()
    (runtime, reference_prepared, goals, gain, cfg, topology,
     scene_seed) = capture_first_active(case)
    n = cfg.n_agents
    identity = np.eye(2)
    variants = [
        Variant("identity", identity, np.arange(n)),
        Variant("rotation_90", np.array([[0.0, -1.0], [1.0, 0.0]]),
                np.arange(n)),
        Variant("reverse_permutation", identity, np.arange(n)[::-1]),
        Variant("reflection_x", np.array([[1.0, 0.0], [0.0, -1.0]]),
                np.arange(n)),
    ]
    active_ids = np.flatnonzero(reference_prepared.refresh_mask)
    gate1a, contexts = fixed_candidate_equivariance(
        runtime, reference_prepared, goals, gain, cfg, budget, variants,
        candidate_count, candidate_seed)
    variant_batches = {}
    for variant in variants:
        context = contexts[variant.name]
        variant_batches[variant.name] = teacher_batch(
            context["runtime"], context["prepared"], context["goals"],
            gain, cfg, seeds, budget, variant, restarts)
    identity_batch = variant_batches["identity"]
    label_report = label_summary(identity_batch, active_ids)
    gate1b = cross_seed_action_cost(
        runtime, reference_prepared, goals, gain, cfg, budget,
        identity_batch, active_ids)
    gate1c = {
        variant.name: compare_batches(
            identity_batch, variant_batches[variant.name], active_ids)
        for variant in variants
    }

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "snapshot.npz", state=runtime.physical, goals=goals,
        parameters=runtime.parameters, world_intent=runtime.world_intent,
        mode=runtime.mode, active_ids=active_ids)
    result = {
        "audit_kind": "teacher_distillability_development_audit",
        "formal_dataset": False,
        "teacher_budget_tier": (
            "debug_quick_non_authoritative" if debug_quick
            else "authoritative_search_candidate"),
        "authoritative_for_gate1": False,
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "snapshot_step": runtime.step,
        "teacher_seeds": seeds,
        "teacher_restarts": restarts,
        "active_agent_ids": active_ids.tolist(),
        "protocol": SYNC_EVENT_V2.manifest(),
        "budget": identity_batch["statistics"][0]["budget"],
        "gate1a_objective_runtime_equivariance": gate1a,
        "gate1b_cross_seed_action_cost": gate1b,
        "gate1c_teacher_search_equivariance": gate1c,
        "identity_label_summary": label_report,
        "teacher_statistics_by_seed": identity_batch["statistics"],
        "restart_selection_by_variant": {
            name: batch["restart_reports"]
            for name, batch in variant_batches.items()
        },
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "Development audit only. Gate 1A uses exactly transformed fixed "
            "candidates. Gate 1B re-evaluates every seed label under one exact "
            "objective. Gate 1C runs the authoritative-search candidate on "
            "every transformed snapshot and maps labels back. Do not train "
            "Student until label inferability and canonical consistency pass."),
    }
    (output / "REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_seeds(text: str) -> list[int]:
    seeds = [int(value.strip()) for value in text.split(",") if value.strip()]
    if len(seeds) < 2:
        raise argparse.ArgumentTypeError("provide at least two Teacher seeds")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"),
        default="n5_partial")
    parser.add_argument("--teacher-seeds", type=parse_seeds,
                        default=parse_seeds("0,1,2,3,4"))
    parser.add_argument(
        "--debug-quick", action="store_true",
        help="non-authoritative debugging/proposal budget; never use for Gate 1")
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument("--candidate-seed", type=int, default=8241)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--restarts", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.iterations < 1:
        parser.error("--samples must be >=2 and --iterations must be positive")
    restarts = args.restarts if args.restarts is not None else (
        1 if args.debug_quick else 4)
    if restarts < 1:
        parser.error("--restarts must be positive")
    if args.debug_quick:
        budget = TeacherBudget(
            strong_samples=8, strong_iterations=2,
            light_samples=4, light_iterations=1,
            horizon_steps=24, max_reuse_ticks=3)
    else:
        authoritative = AuthoritativeTeacherBudget(
            samples=args.samples, iterations=args.iterations,
            restarts=restarts)
        budget = authoritative.search_budget(SYNC_EVENT_V2)
    result = run_audit(
        args.case, args.teacher_seeds, budget, args.output,
        args.candidate_count, args.candidate_seed,
        debug_quick=args.debug_quick, restarts=restarts)
    compact = {
        "case": result["case"],
        "snapshot_step": result["snapshot_step"],
        "active_agent_ids": result["active_agent_ids"],
        "teacher_budget_tier": result["teacher_budget_tier"],
        "teacher_restarts": result["teacher_restarts"],
        "wall_seconds": result["wall_seconds"],
        "report": str(args.output / "REPORT.json"),
    }
    print(json.dumps(compact), flush=True)


if __name__ == "__main__":
    main()
