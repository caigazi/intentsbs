"""Replay a saved authoritative audit without rerunning the Teacher and render MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run_v2_teacher_smoke import scene
from run_v2_teacher_temporal_audit import local_observation_vector
from sbs824.simulation import _dare_gain
from sbs824.v2.intent import decode_world_velocity
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import apply_prepared_step, initialize_runtime, prepare_step


WIDTH, HEIGHT = 960, 720
PLOT = (70, 85, 890, 625)


def text(draw, xy, value, *, fill="black"):
    draw.text(xy, value, fill=fill, font=ImageFont.load_default())


def arrow(draw, start, vector, color, scale, width=3):
    end = (start[0] + scale * vector[0], start[1] - scale * vector[1])
    draw.line((start, end), fill=color, width=width)
    delta = np.asarray(end) - np.asarray(start)
    norm = np.linalg.norm(delta)
    if norm > 1e-6:
        unit = delta / norm
        side = np.array([-unit[1], unit[0]])
        tip = np.asarray(end)
        left = tip - 10 * unit + 5 * side
        right = tip - 10 * unit - 5 * side
        draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=color)


def frame(case, step, state, goals, mode, base_velocity, parameters,
          bounds, flip_agents):
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = PLOT
    draw.rectangle(PLOT, outline="#444444", width=2)
    low, high = bounds
    scale = min((right - left) / (high[0] - low[0]),
                (bottom - top) / (high[1] - low[1]))

    def point(world):
        return (left + (world[0] - low[0]) * scale,
                bottom - (world[1] - low[1]) * scale)

    for gx in np.linspace(low[0], high[0], 7):
        x, _ = point((gx, low[1]))
        draw.line((x, top, x, bottom), fill="#eeeeee")
    for gy in np.linspace(low[1], high[1], 7):
        _, y = point((low[0], gy))
        draw.line((left, y, right, y), fill="#eeeeee")

    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if np.linalg.norm(state[i, :2] - state[j, :2]) <= 0.50:
                draw.line((point(state[i, :2]), point(state[j, :2])),
                          fill="#c8c8c8", width=2)

    for agent, goal in enumerate(goals):
        x, y = point(goal)
        draw.line((x - 7, y - 7, x + 7, y + 7), fill="#26734d", width=2)
        draw.line((x - 7, y + 7, x + 7, y - 7), fill="#26734d", width=2)
        text(draw, (x + 8, y - 8), f"G{agent}", fill="#26734d")

    intent_world = decode_world_velocity(
        state, goals, base_velocity, parameters, SYNC_EVENT_V2)
    colors = {0: "#4c78a8", 1: "#e45756", 2: "#8e6bbd"}
    radius = max(7, int(0.10 * scale))
    for agent in range(len(state)):
        x, y = point(state[agent, :2])
        color = colors.get(int(mode[agent]), "#4c78a8")
        outline = "#ffbf00" if agent in flip_agents else "#111111"
        outline_width = 5 if agent in flip_agents else 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=color, outline=outline, width=outline_width)
        text(draw, (x - 3, y - 7), str(agent), fill="white")
        arrow(draw, (x, y), state[agent, 2:], "#555555", 36, 2)
        arrow(draw, (x, y), intent_world[agent], "#d62728", 48, 3)

    text(draw, (70, 28),
         f"{case} | runtime step {step} | t={step * 0.03:.2f}s")
    if flip_agents:
        text(draw, (70, 48),
             "TEACHER MODE SWITCH: agents " + ", ".join(map(str, flip_agents)),
             fill="#c00000")
    text(draw, (70, 650),
         "red arrow: selected Intent velocity | gray arrow: physical velocity | "
         "gray edge: sensed pair (<=0.50 m)")
    text(draw, (70, 670),
         "blue: non-ACTIVE | red: ACTIVE | purple: RELEASE | gold ring: beta sign flip")
    return np.asarray(image)


def replay(report_path: Path, output: Path, fps: int) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = {int(row["runtime_step"]): row for row in report["records"]}
    state, goals, _, scene_seed = scene(report["case"])
    cfg = make_v2_config(
        n_agents=len(state), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    runtime = initialize_runtime(state, goals, gain, cfg, SYNC_EVENT_V2)

    all_points = np.vstack((state[:, :2], goals))
    low = all_points.min(axis=0) - 0.35
    high = all_points.max(axis=0) + 0.35
    center = (low + high) / 2
    span = max(high - low)
    bounds = (center - span / 2, center + span / 2)

    flip_by_step = {}
    prior = {}
    for step in sorted(records):
        row = records[step]
        parameters = np.asarray(row["parameters"], dtype=float)
        flips = []
        for agent in row["active_ids"]:
            beta = parameters[agent, 1]
            if agent in prior and abs(beta) > 0.03 and abs(prior[agent]) > 0.03 \
                    and beta * prior[agent] < 0:
                flips.append(int(agent))
            prior[agent] = beta
        if flips:
            flip_by_step[step] = flips

    output.parent.mkdir(parents=True, exist_ok=True)
    max_error = 0.0
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        first_frame = None
        while runtime.step <= max(records):
            prepared = prepare_step(runtime, goals, gain, cfg, SYNC_EVENT_V2)
            row = records.get(runtime.step)
            if row is not None:
                actual_ids = np.flatnonzero(prepared.refresh_mask).astype(int).tolist()
                if actual_ids != row["active_ids"]:
                    raise RuntimeError(
                        f"active ids differ at step {runtime.step}: "
                        f"{actual_ids} != {row['active_ids']}")
                for agent in actual_ids:
                    observed = local_observation_vector(
                        prepared.features, agent, cfg.n_agents)
                    expected = np.asarray(
                        row["local_observations"][str(agent)], dtype=float)
                    max_error = max(max_error, float(np.max(np.abs(observed - expected))))
                proposal = np.asarray(row["parameters"], dtype=float)
            else:
                proposal = runtime.parameters.copy()
            image = frame(
                report["case"], runtime.step, runtime.physical, goals,
                prepared.mode, prepared.base_velocity, proposal, bounds,
                flip_by_step.get(runtime.step, []))
            if first_frame is None:
                first_frame = image
                for _ in range(fps):
                    writer.append_data(image)
            writer.append_data(image)
            if runtime.step in flip_by_step:
                for _ in range(2 * fps):
                    writer.append_data(image)
            runtime, _ = apply_prepared_step(
                runtime, prepared, goals, gain, cfg, SYNC_EVENT_V2, proposal)
        for _ in range(fps):
            writer.append_data(image)
    if max_error > 2e-6:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"saved trajectory replay diverged: max feature error {max_error}")
    print(json.dumps({
        "output": str(output),
        "max_saved_feature_error": max_error,
        "flip_steps": flip_by_step,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()
    replay(args.report, args.output, args.fps)


if __name__ == "__main__":
    main()
