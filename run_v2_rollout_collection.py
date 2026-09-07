"""Collect systematic Student rollouts and select failure-driven label windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from run_v2_student_full_rollout import load_checkpoint
from run_v2_teacher_smoke import minimum_center
from sbs824.liveness import LocalSBSMonitor, LocalStep
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import IntentMode, SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import apply_prepared_step, initialize_runtime, prepare_step
from sbs824.v2.scenes import mixed_crossing_scene


DEFAULT_VALIDATION_SPECS = ((2, 8240), (4, 8241), (6, 8242), (8, 8243))
GCBF_STYLE_ROLLOUT_HORIZON = 32


def parse_scene_spec(value: str) -> tuple[int, int]:
    left, separator, right = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("scene spec must be N:SEED")
    n_agents, seed = int(left), int(right)
    if not 2 <= n_agents <= 8:
        raise argparse.ArgumentTypeError("N must be in [2,8]")
    return n_agents, seed


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def select_label_steps(
        applied_mask: np.ndarray, online_sbs_mask: np.ndarray,
        brake_agents: np.ndarray, outside_samples: np.ndarray,
        minimum_distances: np.ndarray, termination: str,
        hard_distance: float, horizon: int = GCBF_STYLE_ROLLOUT_HORIZON,
        maximum: int = 4) -> tuple[list[int], str, int | None]:
    """Choose a fixed-horizon window around the first causal failure signal."""
    applied = np.any(np.asarray(applied_mask, dtype=bool), axis=1)
    unsafe = ((np.asarray(outside_samples) > 0)
              | (np.asarray(minimum_distances) < hard_distance))
    persistent_sbs = np.any(np.asarray(online_sbs_mask, dtype=bool), axis=1)
    braking = np.asarray(brake_agents) > 0
    active_steps = np.flatnonzero(applied)
    if not len(active_steps):
        return [], "no_active_intent", None

    if termination == "all_goal_dwell":
        count = min(maximum, len(active_steps))
        positions = np.linspace(0, len(active_steps) - 1, count)
        selected = sorted({int(active_steps[int(round(pos))])
                           for pos in positions})
        return selected, "successful_active_coverage", None

    if termination == "step_limit":
        event_step = first_true(persistent_sbs)
        reason = "first_persistent_sbs"
        if event_step is None:
            event_step = int(active_steps[-1])
            reason = "step_limit_active_tail"
    else:
        candidates = (
            (unsafe, "safety_invalid", 0),
            (persistent_sbs, "first_persistent_sbs", 1),
            (braking, "first_certified_brake", 2),
        )
        events = [(step, priority, name)
                  for mask, name, priority in candidates
                  if (step := first_true(mask)) is not None]
        if events:
            event_step, _, reason = min(events)
        else:
            event_step = int(active_steps[-1])
            reason = "failed_active_tail"

    start = max(0, event_step - horizon + 1)
    window = active_steps[(active_steps >= start) & (active_steps <= event_step)]
    if not len(window):
        window = active_steps[active_steps <= event_step]
    count = min(maximum, len(window))
    positions = np.linspace(0, len(window) - 1, count)
    selected = sorted({int(window[int(round(pos))]) for pos in positions})
    return selected, reason, event_step


def make_label_manifest(rows: list[dict]) -> dict:
    return {
        "manifest_kind": "failure_driven_rollout_label_manifest_v1",
        "formal_dataset": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "rollout_horizon": GCBF_STYLE_ROLLOUT_HORIZON,
        "selection_policy": (
            "Successful rollouts receive uniform ACTIVE coverage. Failed "
            "rollouts receive at most four uniformly spaced ACTIVE states "
            "from the 32-step window ending at the first causal signal; "
            "step-limit rollouts use persistent SBS or the ACTIVE tail."),
        "entries": [
            {
                "scene_id": row["scene_id"],
                "n_agents": row["n_agents"],
                "seed": row["seed"],
                "topology": row["topology"],
                "trace": row["trace"],
                "snapshot_step": step,
                "selection_reason": row["selection_reason"],
                "failure_event_step": row["failure_event_step"],
            }
            for row in rows for step in row["selected_label_steps"]
        ],
    }


def reselect_existing_rows(rows: list[dict]) -> list[dict]:
    result = []
    for original in rows:
        row = dict(original)
        trace = np.load(row["trace"])
        selected, reason, event_step = select_label_steps(
            trace["applied_mask"], trace["online_sbs_mask"],
            trace["step_certified_brake_agents"],
            trace["step_out_of_certificate_samples"],
            trace["step_minimum_center_distance"], row["termination"],
            SYNC_EVENT_V2.hard_center_distance)
        row["selection_reason"] = reason
        row["failure_event_step"] = event_step
        row["selected_label_steps"] = selected
        result.append(row)
    return result


def run_scene(n_agents: int, seed: int, model, params, output: Path) -> dict:
    initial, goals, topology = mixed_crossing_scene(seed, n_agents)
    cfg = make_v2_config(
        n_agents=n_agents, n_obstacles=0, seed=seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(initial, goals, gain, cfg, SYNC_EVENT_V2)
    monitors = [LocalSBSMonitor(
        dt=cfg.dt, window_steps=cfg.stall_window_steps,
        nominal_epsilon=cfg.sbs_nominal_progress_epsilon,
        ratio_threshold=cfg.sbs_progress_ratio_threshold,
        active_fraction_threshold=cfg.sbs_active_fraction)
        for _ in range(n_agents)]
    scene_id = f"mixed_n{n_agents}_seed{seed}"
    physical = [runtime.physical.copy()]
    world_intent = [runtime.world_intent.copy()]
    clear_counts = [runtime.clear_counts.copy()]
    braking_latch = [runtime.braking_latch.copy()]
    previous_safe_force = [runtime.previous_safe_force.copy()]
    modes = [runtime.mode.copy()]
    parameters = [runtime.parameters.copy()]
    proposals, applied_masks, evidence_masks = [], [], []
    step_minimum, step_brakes, step_outside, online_sbs = [], [], [], []
    dwell = 0
    termination = "step_limit"
    started = perf_counter()

    for _ in range(cfg.steps):
        prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
        prediction = model.apply(
            {"params": params}, jnp.asarray(prepared.features.self_features),
            jnp.asarray(prepared.features.edge_features),
            jnp.asarray(prepared.features.edge_mask))
        prediction = np.asarray(jax.block_until_ready(prediction), dtype=float)
        previous = runtime.physical.copy()
        runtime, trace = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2, prediction)

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
        sbs_now = np.zeros(n_agents, dtype=bool)
        for agent, monitor in enumerate(monitors):
            result = monitor.update(LocalStep(
                nominal_goal_speed=float(nominal_progress[agent]),
                actual_goal_speed=float(actual_progress[agent]),
                active_agent_interaction=bool(
                    prepared.mode[agent] == IntentMode.ACTIVE),
                safe=bool(trace.minimum_center_distance
                          >= SYNC_EVENT_V2.hard_center_distance),
                reached=bool(arrived[agent])))
            sbs_now[agent] = result.online_sbs

        physical.append(runtime.physical.copy())
        world_intent.append(runtime.world_intent.copy())
        clear_counts.append(runtime.clear_counts.copy())
        braking_latch.append(runtime.braking_latch.copy())
        previous_safe_force.append(runtime.previous_safe_force.copy())
        modes.append(runtime.mode.copy())
        parameters.append(runtime.parameters.copy())
        proposals.append(prediction.copy())
        applied_masks.append(trace.applied_mask.copy())
        evidence_masks.append(trace.evidence_mask.copy())
        step_minimum.append(trace.minimum_center_distance)
        step_brakes.append(trace.certified_brake_agents)
        step_outside.append(trace.out_of_certificate_pair_samples)
        online_sbs.append(sbs_now)
        dwell = dwell + 1 if arrived.all() else 0
        if (trace.out_of_certificate_pair_samples > 0
                or trace.minimum_center_distance
                < SYNC_EVENT_V2.hard_center_distance):
            termination = "safety_invalid"
            break
        if dwell >= cfg.adaptive_goal_dwell_steps:
            termination = "all_goal_dwell"
            break

    trace_path = output / f"trajectory_{scene_id}.npz"
    np.savez_compressed(
        trace_path, physical=np.asarray(physical), goals=goals,
        world_intent=np.asarray(world_intent),
        clear_counts=np.asarray(clear_counts),
        braking_latch=np.asarray(braking_latch),
        previous_safe_force=np.asarray(previous_safe_force),
        modes=np.asarray(modes), parameters=np.asarray(parameters),
        student_proposals=np.asarray(proposals),
        applied_mask=np.asarray(applied_masks),
        evidence_mask=np.asarray(evidence_masks),
        online_sbs_mask=np.asarray(online_sbs),
        step_minimum_center_distance=np.asarray(step_minimum),
        step_certified_brake_agents=np.asarray(step_brakes),
        step_out_of_certificate_samples=np.asarray(step_outside))
    selected, reason, event_step = select_label_steps(
        np.asarray(applied_masks), np.asarray(online_sbs),
        np.asarray(step_brakes), np.asarray(step_outside),
        np.asarray(step_minimum), termination,
        SYNC_EVENT_V2.hard_center_distance)
    final_goal = np.linalg.norm(runtime.physical[:, :2] - goals, axis=1)
    return {
        "scene_id": scene_id,
        "n_agents": n_agents,
        "seed": seed,
        "topology": topology,
        "termination": termination,
        "success": int(termination == "all_goal_dwell"),
        "runtime_steps": runtime.step,
        "minimum_center_distance": float(min(step_minimum)),
        "final_mean_goal_distance": float(np.mean(final_goal)),
        "final_maximum_goal_distance": float(np.max(final_goal)),
        "trace": str(trace_path),
        "selection_reason": reason,
        "failure_event_step": event_step,
        "selected_label_steps": selected,
        "wall_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-spec", action="append", type=parse_scene_spec)
    parser.add_argument(
        "--reselect-report", type=Path,
        help="Rebuild only the automatic manifest from saved rollout traces")
    args = parser.parse_args()
    if args.reselect_report is not None:
        source = json.loads(args.reselect_report.read_text(encoding="utf-8"))
        if source.get("protocol") != SYNC_EVENT_V2.manifest():
            raise RuntimeError("source rollout protocol differs from runtime")
        args.output.mkdir(parents=True, exist_ok=False)
        rows = reselect_existing_rows(source["cases"])
        manifest_path = args.output / "LABEL_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(make_label_manifest(rows), indent=2), encoding="utf-8")
        report = dict(source)
        report["audit_kind"] = "systematic_rollout_manifest_reselection"
        report["source_report"] = str(args.reselect_report)
        report["cases"] = rows
        report["label_manifest"] = str(manifest_path)
        (args.output / "REPORT.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return
    if args.checkpoint is None or args.training_report is None:
        parser.error(
            "--checkpoint and --training-report are required for collection")
    specs = args.scene_spec or list(DEFAULT_VALIDATION_SPECS)
    model, params, training = load_checkpoint(
        args.checkpoint, args.training_report)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [run_scene(n_agents, seed, model, params, args.output)
            for n_agents, seed in specs]
    label_manifest = make_label_manifest(rows)
    manifest_path = args.output / "LABEL_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(label_manifest, indent=2), encoding="utf-8")
    report = {
        "audit_kind": "systematic_failure_driven_rollout_collection",
        "formal_dataset": False,
        "formal_checkpoint": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "checkpoint": str(args.checkpoint),
        "training_report": str(args.training_report),
        "training_samples": training["training_samples"],
        "validation_agent_counts": sorted({row["n_agents"] for row in rows}),
        "main_training_anchor": "N=8",
        "rollout_horizon": GCBF_STYLE_ROLLOUT_HORIZON,
        "cases": rows,
        "label_manifest": str(manifest_path),
        "all_cases_success": bool(all(row["success"] for row in rows)),
        "interpretation": (
            "Development validation of the collection pipeline. Multi-N here "
            "tests plumbing; the future main training distribution remains "
            "anchored at N=8."),
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
