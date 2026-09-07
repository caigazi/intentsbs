"""Cross-evaluate saved pre/post-flip actions without invoking CEM search."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from run_v2_teacher_smoke import scene
from run_v2_teacher_temporal_audit import local_observation_vector
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import apply_prepared_step, initialize_runtime, prepare_step
from sbs824.v2.teacher import ComponentCEMTeacher, TeacherBudget


def reconstruct(report: dict):
    records = {int(row["runtime_step"]): row for row in report["records"]}
    state, goals, _, scene_seed = scene(report["case"])
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(state, goals, gain, cfg, SYNC_EVENT_V2)
    snapshots = {}
    max_error = 0.0
    while runtime.step <= max(records):
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        row = records.get(runtime.step)
        if row is not None:
            actual_ids = np.flatnonzero(prepared.refresh_mask).astype(int).tolist()
            if actual_ids != row["active_ids"]:
                raise RuntimeError(f"active ids differ at step {runtime.step}")
            for agent in actual_ids:
                actual = local_observation_vector(
                    prepared.features, agent, cfg.n_agents)
                expected = np.asarray(
                    row["local_observations"][str(agent)], dtype=float)
                max_error = max(
                    max_error, float(np.max(np.abs(actual - expected))))
            snapshots[runtime.step] = (deepcopy(runtime), prepared)
            proposal = np.asarray(row["parameters"], dtype=float)
        else:
            proposal = runtime.parameters.copy()
        runtime, _ = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2, proposal)
    if max_error > 2e-6:
        raise RuntimeError(f"replay diverged: maximum feature error {max_error}")
    return records, snapshots, goals, gain, cfg, max_error


def exact_costs(evaluator, runtime, prepared, goals, gain, cfg, candidates,
                focus_ids=None):
    active_ids = np.flatnonzero(prepared.mode == 1)
    release_ids = np.flatnonzero(prepared.mode == 2)
    if focus_ids is None:
        focus_ids = np.flatnonzero(prepared.refresh_mask)
    focus_ids = np.asarray(focus_ids, dtype=int)
    return np.asarray(evaluator.evaluate_candidates_exact_jax(
        runtime, goals, gain, cfg, np.asarray(candidates, dtype=float),
        active_ids, release_ids, focus_ids), dtype=float)


def active_components(prepared):
    active = set(np.flatnonzero(prepared.refresh_mask).astype(int).tolist())
    result = []
    for component in prepared.trigger.components:
        members = sorted(active.intersection(map(int, component)))
        if len(members) >= 2:
            result.append(members)
    return result


def perturb_physical(runtime, components, fractional_delta, quantity):
    changed = deepcopy(runtime)
    physical = changed.physical.copy()
    columns = slice(0, 2) if quantity == "relative_position" else slice(2, 4)
    for members in components:
        values = physical[members, columns]
        center = values.mean(axis=0, keepdims=True)
        physical[members, columns] = center + (1.0 + fractional_delta) * (
            values - center)
    changed.physical = physical
    return changed


def replace_fields(base, donor, fields):
    changed = deepcopy(base)
    for field in fields:
        setattr(changed, field, deepcopy(getattr(donor, field)))
    return changed


def audit(report_path: Path, output: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records, snapshots, goals, gain, cfg, replay_error = reconstruct(report)
    steps = sorted(records)
    flip_events = []
    for left_step, right_step in zip(steps, steps[1:]):
        left = records[left_step]
        right = records[right_step]
        common = sorted(set(left["active_ids"]) & set(right["active_ids"]))
        flipped = [agent for agent in common
                   if (abs(left["parameters"][agent][1]) > 0.03
                       and abs(right["parameters"][agent][1]) > 0.03
                       and left["parameters"][agent][1]
                       * right["parameters"][agent][1] < 0)]
        if flipped:
            flip_events.append((left_step, right_step, flipped))
    if not flip_events:
        raise RuntimeError("report has no consecutive chirality flip")

    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=TeacherBudget(), seed=0)
    results = []
    for left_step, right_step, flipped in flip_events:
        action_left = np.asarray(records[left_step]["parameters"], dtype=float)
        action_right = np.asarray(records[right_step]["parameters"], dtype=float)
        actions = [action_left, action_right]
        event = {
            "before_step": left_step,
            "after_step": right_step,
            "flipped_agents": flipped,
            "cross_cost_rows": [],
            "component_cross_cost_rows": [],
            "hybrid_state_rows": [],
            "perturbations": [],
        }
        for state_name, step in (("before", left_step), ("after", right_step)):
            runtime, prepared = snapshots[step]
            costs = exact_costs(
                evaluator, runtime, prepared, goals, gain, cfg,
                actions + [runtime.parameters.copy()])
            best = float(min(costs[:2]))
            denominator = max(float(costs[2]) - best, 1.0)
            event["cross_cost_rows"].append({
                "state": state_name,
                "runtime_step": step,
                "cost_action_before": float(costs[0]),
                "cost_action_after": float(costs[1]),
                "cost_incumbent": float(costs[2]),
                "preferred_saved_action": (
                    "before" if costs[0] <= costs[1] else "after"),
                "before_action_improvement_regret": float(
                    (costs[0] - best) / denominator),
                "after_action_improvement_regret": float(
                    (costs[1] - best) / denominator),
                "both_actions_inside_one_percent_band": bool(
                    max(costs[:2]) - best <= 0.01 * denominator),
            })

            for component in active_components(prepared):
                if not set(component).intersection(flipped):
                    continue
                component_costs = exact_costs(
                    evaluator, runtime, prepared, goals, gain, cfg,
                    actions + [runtime.parameters.copy()], component)
                component_best = float(min(component_costs[:2]))
                component_denominator = max(
                    float(component_costs[2]) - component_best, 1.0)
                event["component_cross_cost_rows"].append({
                    "state": state_name,
                    "runtime_step": step,
                    "component": component,
                    "flipped_agents_in_component": sorted(
                        set(component).intersection(flipped)),
                    "cost_action_before": float(component_costs[0]),
                    "cost_action_after": float(component_costs[1]),
                    "cost_incumbent": float(component_costs[2]),
                    "preferred_saved_action": (
                        "before" if component_costs[0] <= component_costs[1]
                        else "after"),
                    "before_action_improvement_regret": float(
                        (component_costs[0] - component_best)
                        / component_denominator),
                    "after_action_improvement_regret": float(
                        (component_costs[1] - component_best)
                        / component_denominator),
                    "both_actions_inside_one_percent_band": bool(
                        max(component_costs[:2]) - component_best
                        <= 0.01 * component_denominator),
                })

        left_runtime, left_prepared = snapshots[left_step]
        right_runtime, right_prepared = snapshots[right_step]
        hybrid_specs = (
            ("original", ()),
            ("physical_from_other", ("physical",)),
            ("visible_history_from_other", (
                "parameters", "world_intent", "mode", "clear_counts")),
            ("hidden_safety_from_other", (
                "braking_latch", "previous_safe_force")),
            ("all_history_from_other", (
                "parameters", "world_intent", "mode", "clear_counts",
                "braking_latch", "previous_safe_force")),
        )
        for base_name, base_step, base_runtime, base_prepared, donor_runtime in (
                ("before", left_step, left_runtime, left_prepared, right_runtime),
                ("after", right_step, right_runtime, right_prepared, left_runtime)):
            for component in active_components(base_prepared):
                if not set(component).intersection(flipped):
                    continue
                baseline_observations = {
                    agent: local_observation_vector(
                        base_prepared.features, agent, cfg.n_agents)
                    for agent in component
                }
                for variant, fields in hybrid_specs:
                    changed = replace_fields(base_runtime, donor_runtime, fields)
                    changed_prepared = prepare_step(
                        changed, goals, gain, cfg, SYNC_EVENT_V2)
                    observation_l2 = max(float(np.linalg.norm(
                        local_observation_vector(
                            changed_prepared.features, agent, cfg.n_agents)
                        - baseline_observations[agent]))
                        for agent in component)
                    costs = exact_costs(
                        evaluator, changed, base_prepared, goals, gain, cfg,
                        actions, component)
                    event["hybrid_state_rows"].append({
                        "base_state": base_name,
                        "runtime_step": base_step,
                        "component": component,
                        "flipped_agents_in_component": sorted(
                            set(component).intersection(flipped)),
                        "variant": variant,
                        "replaced_fields": list(fields),
                        "maximum_legal_observation_l2_from_base": observation_l2,
                        "cost_action_before": float(costs[0]),
                        "cost_action_after": float(costs[1]),
                        "before_minus_after": float(costs[0] - costs[1]),
                        "preferred_saved_action": (
                            "before" if costs[0] <= costs[1] else "after"),
                    })

        for state_name, step in (("before", left_step), ("after", right_step)):
            runtime, prepared = snapshots[step]
            components = [component for component in active_components(prepared)
                          if set(component).intersection(flipped)]
            for component in components:
                for quantity in ("relative_position", "relative_velocity"):
                    for fractional_delta in (-0.02, -0.01, 0.0, 0.01, 0.02):
                        changed = perturb_physical(
                            runtime, [component], fractional_delta, quantity)
                        costs = exact_costs(
                            evaluator, changed, prepared, goals, gain, cfg,
                            actions, component)
                        event["perturbations"].append({
                            "state": state_name,
                            "runtime_step": step,
                            "component": component,
                            "flipped_agents_in_component": sorted(
                                set(component).intersection(flipped)),
                            "quantity": quantity,
                            "fractional_delta": fractional_delta,
                            "cost_action_before": float(costs[0]),
                            "cost_action_after": float(costs[1]),
                            "before_minus_after": float(costs[0] - costs[1]),
                            "preferred_saved_action": (
                                "before" if costs[0] <= costs[1] else "after"),
                        })
        results.append(event)

    result = {
        "audit_kind": "saved_flip_cross_exact_cost_and_local_perturbation",
        "formal_dataset": False,
        "ran_cem_search": False,
        "objective": "40-step, JAX x64, 32-substep exact sampled-safety",
        "source_report": str(report_path),
        "case": report["case"],
        "maximum_saved_feature_replay_error": replay_error,
        "events": results,
        "interpretation": (
            "Cross rows distinguish an actual preference reversal from two "
            "mutually admissible modes. Perturbations scale relative positions "
            "or velocities about each active component centroid while preserving "
            "the saved protocol/history state; they are sensitivity probes only."),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report, args.output)
    print(json.dumps({
        "case": result["case"],
        "ran_cem_search": result["ran_cem_search"],
        "maximum_saved_feature_replay_error": (
            result["maximum_saved_feature_replay_error"]),
        "cross_cost_rows": result["events"][0]["cross_cost_rows"],
        "report": str(args.output / "REPORT.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
