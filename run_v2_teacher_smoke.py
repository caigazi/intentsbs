"""Closed-loop smoke for the clean v2 runtime and component CEM Teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from sbs824.liveness import LocalSBSMonitor, LocalStep
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import (apply_prepared_step, initialize_runtime,
                               prepare_step)
from sbs824.v2.scenes import headon_scene, mixed_crossing_scene
from sbs824.v2.teacher import (AuthoritativeComponentTeacher,
                               ComponentCEMTeacher, TeacherBudget,
                               TeacherFeedback)


ROOT = Path(__file__).resolve().parent


def scene(name: str):
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


def minimum_center(state: np.ndarray) -> float:
    delta = state[:, None, :2] - state[None, :, :2]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    return float(distance.min())


def run(name: str, output: Path, budget: TeacherBudget | None,
        *, debug_quick: bool) -> dict:
    state, goals, topology, seed = scene(name)
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(
        state, goals, gain, cfg, SYNC_EVENT_V2)
    teacher = (ComponentCEMTeacher(
        SYNC_EVENT_V2, budget=budget, seed=seed + 3000)
        if debug_quick else AuthoritativeComponentTeacher(
            SYNC_EVENT_V2, seed=seed + 3000))
    monitors = [LocalSBSMonitor(
        dt=cfg.dt, window_steps=cfg.stall_window_steps,
        nominal_epsilon=cfg.sbs_nominal_progress_epsilon,
        ratio_threshold=cfg.sbs_progress_ratio_threshold,
        active_fraction_threshold=cfg.sbs_active_fraction)
        for _ in range(cfg.n_agents)]
    feedback = TeacherFeedback()
    dwell = 0
    minimum = minimum_center(runtime.physical)
    sbs_agent_seconds = 0.0
    sbs_agents = np.zeros(cfg.n_agents, dtype=bool)
    positions = [runtime.physical[:, :2].copy()]
    parameters = [runtime.parameters.copy()]
    modes = [runtime.mode.copy()]
    sources = []
    started = perf_counter()
    termination = "step_limit"
    total_brake = 0
    total_outside = 0
    for _ in range(cfg.steps):
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)
        if prepared.refresh_mask.any():
            decision = teacher.decide(
                runtime, prepared, goals, gain, cfg, feedback)
            proposal = decision.parameters
            sources.append(decision.source.copy())
        else:
            proposal = runtime.parameters.copy()
            sources.append(np.full(cfg.n_agents, "bypass", dtype="U8"))
        previous = runtime.physical.copy()
        runtime, trace = apply_prepared_step(
            runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2,
            proposal)
        minimum = min(minimum, minimum_center(runtime.physical))
        total_brake += trace.certified_brake_agents
        total_outside += trace.out_of_certificate_pair_samples
        goal_delta = goals - previous[:, :2]
        direction = goal_delta / np.maximum(
            np.linalg.norm(goal_delta, axis=1, keepdims=True), 1e-9)
        nominal_progress = np.maximum(
            0.0, np.sum(direction * prepared.base_velocity, axis=1))
        actual_progress = np.maximum(
            0.0, np.sum(direction * runtime.physical[:, 2:], axis=1))
        ratio = float(actual_progress.sum() /
                      (nominal_progress.sum() + 1e-9))
        feedback = TeacherFeedback(
            certified_brake_fraction=(
                trace.certified_brake_agents / cfg.n_agents),
            out_of_certificate_pair_samples=(
                trace.out_of_certificate_pair_samples),
            progress_ratio=ratio)
        if (trace.out_of_certificate_pair_samples > 0
                or trace.minimum_center_distance
                < SYNC_EVENT_V2.hard_center_distance):
            termination = "safety_invalid"
            positions.append(runtime.physical[:, :2].copy())
            parameters.append(runtime.parameters.copy())
            modes.append(runtime.mode.copy())
            break
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
                    prepared.mode[agent] == 1),
                safe=bool(minimum >= SYNC_EVENT_V2.hard_center_distance),
                reached=bool(arrived[agent])))
            sbs_now[agent] = value.online_sbs
        sbs_agents |= sbs_now
        sbs_agent_seconds += float(sbs_now.sum()) * cfg.dt
        dwell = dwell + 1 if arrived.all() else 0
        positions.append(runtime.physical[:, :2].copy())
        parameters.append(runtime.parameters.copy())
        modes.append(runtime.mode.copy())
        if dwell >= cfg.adaptive_goal_dwell_steps:
            termination = "all_goal_dwell"
            break
    wall = perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / f"trajectory_{name}.npz",
        positions=np.asarray(positions), goals=goals,
        parameters=np.asarray(parameters), modes=np.asarray(modes),
        teacher_source=np.asarray(sources))
    row = {
        "case": name,
        "topology": topology,
        "n_agents": cfg.n_agents,
        "termination": termination,
        "success": int(termination == "all_goal_dwell"),
        "seconds": (len(positions) - 1) * cfg.dt,
        "wall_seconds": wall,
        "minimum_center_distance": minimum,
        "sbs_agents": int(sbs_agents.sum()),
        "sbs_agent_seconds": sbs_agent_seconds,
        "certified_brake_agent_steps": total_brake,
        "out_of_certificate_pair_samples": total_outside,
        "protocol": SYNC_EVENT_V2.manifest(),
        "teacher_tier": (
            "debug_quick_non_authoritative" if debug_quick
            else AuthoritativeComponentTeacher.tier),
        "teacher": teacher.statistics(),
    }
    (output / "REPORT.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(
        "n2_headon", "n5_partial", "n8_two_stream"),
        default="n2_headon")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "sync_event_v2_teacher_smoke")
    parser.add_argument(
        "--debug-quick", action="store_true",
        help="explicit non-authoritative execution/debug path")
    args = parser.parse_args()
    budget = (TeacherBudget(
        strong_samples=8, strong_iterations=2,
        light_samples=4, light_iterations=1,
        horizon_steps=24, max_reuse_ticks=3)
        if args.debug_quick else None)
    result = run(
        args.case, args.output, budget, debug_quick=args.debug_quick)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
