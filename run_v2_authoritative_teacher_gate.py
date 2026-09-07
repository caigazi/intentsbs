"""Short quality/continuity audit for the simple authoritative v2 Teacher.

Every ACTIVE tick performs one fresh four-restart search. The audit records all
restart costs, the selected exact action and local temporal continuity. It is
not a full trajectory Gate and never writes a Student training dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_v2_teacher_smoke import minimum_center, scene
from run_v2_teacher_temporal_audit import (continuity_summary, exact_pair,
                                           local_observation_vector)
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import (apply_prepared_step, initialize_runtime,
                               prepare_step)
from sbs824.v2.teacher import (AuthoritativeComponentTeacher,
                               AuthoritativeTeacherBudget,
                               ComponentCEMTeacher)


def run(case: str, output: Path, seed: int, active_ticks: int) -> dict:
    state, goals, topology, scene_seed = scene(case)
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(state, goals, gain, cfg, SYNC_EVENT_V2)
    budget = AuthoritativeTeacherBudget()
    teacher = AuthoritativeComponentTeacher(
        SYNC_EVENT_V2, budget=budget, seed=seed)
    evaluator = ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=teacher.search_budget, seed=0)
    records: list[dict] = []
    dwell = 0
    termination = "step_limit"
    minimum = minimum_center(runtime.physical)
    started = perf_counter()

    for _ in range(cfg.steps):
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        if prepared.refresh_mask.any():
            decision = teacher.decide(runtime, prepared, goals, gain, cfg)
            online_cost, incumbent_cost = exact_pair(
                evaluator, runtime, prepared, goals, gain, cfg,
                decision.parameters)
            active_ids = np.flatnonzero(
                prepared.refresh_mask).astype(int)
            sources = sorted(set(
                decision.source[active_ids].astype(str).tolist()))
            restart_costs = teacher.last_restart_costs.copy()
            restart_best = min(restart_costs)
            selected_regret = (
                (online_cost - restart_best)
                / max(incumbent_cost - restart_best, 1.0))
            records.append({
                "runtime_step": runtime.step,
                "active_ids": active_ids.tolist(),
                "sources": sources,
                "parameters": decision.parameters.tolist(),
                "online_exact_cost": online_cost,
                "incumbent_exact_cost": incumbent_cost,
                "restart_exact_costs": restart_costs,
                "selected_improvement_regret_to_restart_best": (
                    selected_regret),
                "local_observations": {
                    str(agent): local_observation_vector(
                        prepared.features, agent, cfg.n_agents).tolist()
                    for agent in active_ids
                },
            })
            proposal = decision.parameters
        else:
            proposal = runtime.parameters.copy()

        runtime, trace = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2,
            proposal)
        minimum = min(minimum, minimum_center(runtime.physical))
        if (trace.out_of_certificate_pair_samples > 0
                or trace.minimum_center_distance
                < SYNC_EVENT_V2.hard_center_distance):
            termination = "safety_invalid"
            break
        inside = np.linalg.norm(
            runtime.physical[:, :2] - goals, axis=1) < cfg.reach_radius
        arrived = inside & (np.linalg.norm(
            runtime.physical[:, 2:], axis=1)
            < cfg.adaptive_goal_speed_epsilon)
        dwell = dwell + 1 if arrived.all() else 0
        if dwell >= cfg.adaptive_goal_dwell_steps:
            termination = "all_goal_dwell"
            break
        if len(records) >= active_ticks:
            termination = "active_tick_limit"
            break

    selected_regrets = [
        row["selected_improvement_regret_to_restart_best"]
        for row in records]
    report = {
        "audit_kind": "authoritative_teacher_short_continuity_audit",
        "formal_dataset": False,
        "teacher_tier": teacher.tier,
        "case": case,
        "topology": topology,
        "scene_seed": scene_seed,
        "termination": termination,
        "success": int(termination == "all_goal_dwell"),
        "runtime_steps": runtime.step,
        "active_ticks_requested": active_ticks,
        "active_ticks": len(records),
        "minimum_center_distance": minimum,
        "protocol": SYNC_EVENT_V2.manifest(),
        "teacher": teacher.statistics(),
        "maximum_selected_regret_to_restart_best": (
            max(selected_regrets, default=None)),
        "selected_actions_outside_one_percent_band": int(sum(
            value > 0.01 for value in selected_regrets)),
        "continuity": continuity_summary(records, cfg.n_agents),
        "records": records,
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "Development audit of the deliberately simple fresh-per-tick "
            "Teacher. The selected action must remain inside the one-percent "
            "quality band of the four exact restart winners. Temporal "
            "continuity is reported separately; no full Gate is claimed."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"), required=True)
    parser.add_argument("--seed", type=int, default=8240001)
    parser.add_argument("--active-ticks", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.active_ticks < 2:
        parser.error("--active-ticks must be at least two")
    result = run(args.case, args.output, args.seed, args.active_ticks)
    print(json.dumps({
        key: result[key] for key in (
            "case", "termination", "runtime_steps", "active_ticks",
            "minimum_center_distance",
            "maximum_selected_regret_to_restart_best",
            "selected_actions_outside_one_percent_band", "wall_seconds")
    }), flush=True)


if __name__ == "__main__":
    main()
