"""Label a few saved Student failure-window snapshots with the authoritative Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from run_v2_student_full_rollout import load_checkpoint
from run_v2_teacher_smoke import scene
from run_v2_teacher_temporal_audit import local_observation_vector
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import RuntimeState, prepare_step
from sbs824.v2.teacher import (
    AuthoritativeComponentTeacher,
    ComponentCEMTeacher,
)


DEFAULT_SPECS = {
    "n5_partial": (34, 38, 42, 46, 50, 54),
    "n8_two_stream": (28, 32, 36, 40, 44, 48),
}


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


def label_case(case, steps, trace_path, model, params, output, seed):
    initial, goals, topology, scene_seed = scene(case)
    cfg = make_v2_config(
        n_agents=len(initial), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    trace = np.load(trace_path)
    if len(trace["physical"]) <= max(steps):
        raise RuntimeError(f"trace ends before requested {case} snapshot")
    records = []
    started = perf_counter()
    for offset, step in enumerate(steps):
        runtime = runtime_from_trace(trace, step)
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        active_ids = np.flatnonzero(prepared.refresh_mask).astype(int)
        if not len(active_ids):
            raise RuntimeError(f"{case} step {step} has no ACTIVE label mask")
        prediction = model.apply(
            {"params": params}, jnp.asarray(prepared.features.self_features),
            jnp.asarray(prepared.features.edge_features),
            jnp.asarray(prepared.features.edge_mask))
        prediction = np.asarray(jax.block_until_ready(prediction), dtype=float)

        teacher = AuthoritativeComponentTeacher(
            SYNC_EVENT_V2, seed=seed + 1_000_003 * offset)
        decision = teacher.decide(runtime, prepared, goals, gain, cfg)
        evaluator = ComponentCEMTeacher(
            SYNC_EVENT_V2, budget=teacher.search_budget, seed=0)
        exact = np.asarray(evaluator.evaluate_candidates_exact_jax(
            runtime, goals, gain, cfg,
            np.stack((decision.parameters, prediction, runtime.parameters)),
            np.flatnonzero(prepared.mode == IntentMode.ACTIVE),
            np.flatnonzero(prepared.mode == IntentMode.RELEASE),
            active_ids), dtype=float)
        records.append({
            "case": case,
            "runtime_step": step,
            "active_ids": active_ids.tolist(),
            "parameters": decision.parameters.tolist(),
            "student_parameters": prediction.tolist(),
            "teacher_exact_cost": float(exact[0]),
            "student_exact_cost": float(exact[1]),
            "incumbent_exact_cost": float(exact[2]),
            "teacher_relative_improvement_over_student": float(
                (exact[1] - exact[0]) / max(abs(exact[1]), 1.0)),
            "restart_exact_costs": teacher.last_restart_costs,
            "local_observations": {
                str(agent): local_observation_vector(
                    prepared.features, agent, cfg.n_agents).tolist()
                for agent in active_ids
            },
        })
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "audit_kind": "tiny_student_failure_window_authoritative_labels",
        "formal_dataset": False,
        "teacher_tier": "authoritative_v2",
        "protocol": SYNC_EVENT_V2.manifest(),
        "case": case,
        "topology": topology,
        "source_student_trace": str(trace_path),
        "snapshot_steps": list(steps),
        "records": records,
        "wall_seconds": perf_counter() - started,
        "interpretation": (
            "Development-only failure-window DAgger labels. Every snapshot "
            "uses a fresh four-restart authoritative Teacher; no reuse, light "
            "search, quick fallback, or formal dataset is involved."),
    }
    (output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8245000)
    parser.add_argument(
        "--spec", action="append", metavar="CASE:STEP,STEP,...",
        help="Override the default cases with explicit saved-trace snapshots")
    args = parser.parse_args()
    specs = DEFAULT_SPECS
    if args.spec:
        specs = {}
        for value in args.spec:
            case, separator, raw_steps = value.partition(":")
            if not separator or case not in DEFAULT_SPECS:
                raise ValueError(f"invalid --spec: {value}")
            steps = tuple(int(item) for item in raw_steps.split(","))
            if not steps or any(step < 0 for step in steps):
                raise ValueError(f"invalid --spec steps: {value}")
            specs[case] = steps
    model, params, _ = load_checkpoint(args.checkpoint, args.training_report)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for case_index, (case, steps) in enumerate(specs.items()):
        reports.append(label_case(
            case, steps, args.trace_root / f"trajectory_{case}.npz",
            model, params, args.output / case,
            args.seed + 10_000_019 * case_index))
    summary = {
        "audit_kind": "tiny_student_failure_window_authoritative_label_batch",
        "formal_dataset": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "cases": [{
            "case": report["case"],
            "snapshot_steps": report["snapshot_steps"],
            "wall_seconds": report["wall_seconds"],
            "minimum_teacher_relative_improvement_over_student": min(
                row["teacher_relative_improvement_over_student"]
                for row in report["records"]),
            "maximum_teacher_relative_improvement_over_student": max(
                row["teacher_relative_improvement_over_student"]
                for row in report["records"]),
        } for report in reports],
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
