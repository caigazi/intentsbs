"""Run a saved v2 Student checkpoint through complete obstacle-free scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from run_v2_teacher_smoke import minimum_center, scene
from sbs824.liveness import LocalSBSMonitor, LocalStep
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import apply_prepared_step, initialize_runtime, prepare_step
from sbs824.v2.student import IntentStudent, StudentArchitecture


CASES = ("n2_headon", "n5_partial", "n8_two_stream")


def load_checkpoint(checkpoint: Path, training_report: Path):
    report = json.loads(training_report.read_text(encoding="utf-8"))
    if report.get("formal_checkpoint", True):
        raise RuntimeError("full tiny rollout expects a development checkpoint")
    if report.get("protocol") != SYNC_EVENT_V2.manifest():
        raise RuntimeError("checkpoint protocol differs from runtime protocol")
    architecture = StudentArchitecture(**report["architecture"])
    model = IntentStudent(architecture)
    dummy_self = jnp.zeros((1, 9), dtype=jnp.float32)
    dummy_edge = jnp.zeros((1, 1, 9), dtype=jnp.float32)
    dummy_mask = jnp.zeros((1, 1), dtype=bool)
    template = model.init(
        jax.random.PRNGKey(0), dummy_self, dummy_edge, dummy_mask)["params"]
    params = serialization.from_bytes(template, checkpoint.read_bytes())
    count = int(sum(value.size for value in jax.tree_util.tree_leaves(params)))
    if count != report["parameter_count"]:
        raise RuntimeError("checkpoint parameter count differs from manifest")
    return model, params, report


def run_case(case: str, model, params, output: Path) -> dict:
    initial, goals, topology, scene_seed = scene(case)
    cfg = make_v2_config(
        n_agents=len(initial), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(initial, goals, gain, cfg, SYNC_EVENT_V2)
    monitors = [LocalSBSMonitor(
        dt=cfg.dt, window_steps=cfg.stall_window_steps,
        nominal_epsilon=cfg.sbs_nominal_progress_epsilon,
        ratio_threshold=cfg.sbs_progress_ratio_threshold,
        active_fraction_threshold=cfg.sbs_active_fraction)
        for _ in range(cfg.n_agents)]
    minimum = minimum_center(runtime.physical)
    sbs_agent_seconds = 0.0
    sbs_agents = np.zeros(cfg.n_agents, dtype=bool)
    inference_seconds = []
    physical_states = [runtime.physical.copy()]
    world_intents = [runtime.world_intent.copy()]
    clear_counts = [runtime.clear_counts.copy()]
    braking_latches = [runtime.braking_latch.copy()]
    previous_safe_forces = [runtime.previous_safe_force.copy()]
    modes = [runtime.mode.copy()]
    parameters = [runtime.parameters.copy()]
    evidence_masks = []
    applied_masks = []
    step_minimum_distances = []
    step_brake_agents = []
    step_outside_samples = []
    dwell = 0
    termination = "step_limit"
    total_brake = 0
    total_outside = 0
    refreshed_intents = 0
    active_agent_steps = 0
    release_agent_steps = 0
    started = perf_counter()

    for _ in range(cfg.steps):
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        before_inference = perf_counter()
        prediction = model.apply(
            {"params": params}, jnp.asarray(prepared.features.self_features),
            jnp.asarray(prepared.features.edge_features),
            jnp.asarray(prepared.features.edge_mask))
        prediction = np.asarray(jax.block_until_ready(prediction), dtype=float)
        inference_seconds.append(perf_counter() - before_inference)
        previous = runtime.physical.copy()
        runtime, trace = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2, prediction)
        minimum = min(minimum, minimum_center(runtime.physical))
        total_brake += trace.certified_brake_agents
        total_outside += trace.out_of_certificate_pair_samples
        refreshed_intents += int(trace.applied_mask.sum())
        active_agent_steps += int(np.sum(prepared.mode == IntentMode.ACTIVE))
        release_agent_steps += int(np.sum(prepared.mode == IntentMode.RELEASE))

        goal_delta = goals - previous[:, :2]
        direction = goal_delta / np.maximum(
            np.linalg.norm(goal_delta, axis=1, keepdims=True), 1e-9)
        nominal_progress = np.maximum(
            0.0, np.sum(direction * prepared.base_velocity, axis=1))
        actual_progress = np.maximum(
            0.0, np.sum(direction * runtime.physical[:, 2:], axis=1))
        inside = np.linalg.norm(
            runtime.physical[:, :2] - goals, axis=1) < cfg.reach_radius
        arrived = inside & (np.linalg.norm(
            runtime.physical[:, 2:], axis=1)
            < cfg.adaptive_goal_speed_epsilon)
        sbs_now = np.zeros(cfg.n_agents, dtype=bool)
        for agent, monitor in enumerate(monitors):
            value = monitor.update(LocalStep(
                nominal_goal_speed=float(nominal_progress[agent]),
                actual_goal_speed=float(actual_progress[agent]),
                active_agent_interaction=bool(
                    prepared.mode[agent] == IntentMode.ACTIVE),
                safe=bool(minimum >= SYNC_EVENT_V2.hard_center_distance),
                reached=bool(arrived[agent])))
            sbs_now[agent] = value.online_sbs
        sbs_agents |= sbs_now
        sbs_agent_seconds += float(sbs_now.sum()) * cfg.dt
        dwell = dwell + 1 if arrived.all() else 0
        physical_states.append(runtime.physical.copy())
        world_intents.append(runtime.world_intent.copy())
        clear_counts.append(runtime.clear_counts.copy())
        braking_latches.append(runtime.braking_latch.copy())
        previous_safe_forces.append(runtime.previous_safe_force.copy())
        modes.append(runtime.mode.copy())
        parameters.append(runtime.parameters.copy())
        evidence_masks.append(trace.evidence_mask.copy())
        applied_masks.append(trace.applied_mask.copy())
        step_minimum_distances.append(trace.minimum_center_distance)
        step_brake_agents.append(trace.certified_brake_agents)
        step_outside_samples.append(trace.out_of_certificate_pair_samples)

        if (trace.out_of_certificate_pair_samples > 0
                or trace.minimum_center_distance
                < SYNC_EVENT_V2.hard_center_distance):
            termination = "safety_invalid"
            break
        if dwell >= cfg.adaptive_goal_dwell_steps:
            termination = "all_goal_dwell"
            break

    wall_seconds = perf_counter() - started
    steady = np.asarray(inference_seconds[1:], dtype=float)
    np.savez_compressed(
        output / f"trajectory_{case}.npz",
        physical=np.asarray(physical_states), goals=goals,
        world_intent=np.asarray(world_intents),
        clear_counts=np.asarray(clear_counts),
        braking_latch=np.asarray(braking_latches),
        previous_safe_force=np.asarray(previous_safe_forces),
        modes=np.asarray(modes), parameters=np.asarray(parameters),
        evidence_mask=np.asarray(evidence_masks),
        applied_mask=np.asarray(applied_masks),
        step_minimum_center_distance=np.asarray(step_minimum_distances),
        step_certified_brake_agents=np.asarray(step_brake_agents),
        step_out_of_certificate_samples=np.asarray(step_outside_samples))
    return {
        "case": case,
        "topology": topology,
        "n_agents": cfg.n_agents,
        "termination": termination,
        "success": int(termination == "all_goal_dwell"),
        "runtime_steps": runtime.step,
        "simulation_seconds": runtime.step * cfg.dt,
        "wall_seconds": wall_seconds,
        "minimum_center_distance": minimum,
        "sbs_agents": int(sbs_agents.sum()),
        "sbs_agent_seconds": sbs_agent_seconds,
        "certified_brake_agent_steps": total_brake,
        "out_of_certificate_pair_samples": total_outside,
        "active_agent_steps": active_agent_steps,
        "release_agent_steps": release_agent_steps,
        "refreshed_intents": refreshed_intents,
        "communicated_float_count": 2 * refreshed_intents,
        "first_inference_seconds_including_compile": inference_seconds[0],
        "mean_steady_inference_seconds": (
            float(steady.mean()) if len(steady) else None),
        "p95_steady_inference_seconds": (
            float(np.percentile(steady, 95)) if len(steady) else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, params, training = load_checkpoint(
        args.checkpoint, args.training_report)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [run_case(case, model, params, args.output) for case in args.cases]
    report = {
        "audit_kind": "tiny_student_complete_training_scene_rollout",
        "formal_dataset": False,
        "formal_checkpoint": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "checkpoint": str(args.checkpoint),
        "training_report": str(args.training_report),
        "training_samples": training["training_samples"],
        "architecture": training["architecture"],
        "parameter_count": training["parameter_count"],
        "cases": rows,
        "all_cases_success": bool(all(row["success"] for row in rows)),
        "total_certificate_violation_samples": int(sum(
            row["out_of_certificate_pair_samples"] for row in rows)),
        "interpretation": (
            "Development Gate-2 training-scene rollout only. These are the "
            "same scene definitions that supplied short authoritative windows; "
            "this is not unseen-scene generalization or a paper result."),
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
