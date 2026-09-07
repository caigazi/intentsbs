"""Offline local-observation/Teacher-label conflict audit for v2.

Only the future Student input is compared: the nine ego features and the
unordered set of nine-dimensional sensed-neighbor features. Scene, step and
agent identifiers are retained solely to identify diagnostic rows and never
enter the observation distance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Sample:
    case: str
    step: int
    agent: int
    ego: np.ndarray
    edges: np.ndarray
    action: np.ndarray

    @property
    def beta_mode(self) -> int:
        beta = float(self.action[1])
        return 1 if beta > 0.03 else (-1 if beta < -0.03 else 0)


def parse_observation(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if len(values) < 9 or (len(values) - 9) % 10:
        raise ValueError("unexpected flattened local observation shape")
    ego = values[:9].copy()
    rows = values[9:].reshape(-1, 10)
    edges = rows[rows[:, 9] > 0.5, :9].copy()
    return ego, edges


def load_samples(report_paths: list[Path]) -> tuple[list[Sample], list[dict]]:
    samples: list[Sample] = []
    sources = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        case = str(report["case"])
        sources.append({
            "case": case,
            "report": str(path),
            "active_ticks": int(report["active_ticks"]),
        })
        for record in report["records"]:
            parameters = np.asarray(record["parameters"], dtype=float)
            for agent in record["active_ids"]:
                ego, edges = parse_observation(
                    record["local_observations"][str(agent)])
                samples.append(Sample(
                    case=case, step=int(record["runtime_step"]),
                    agent=int(agent), ego=ego, edges=edges,
                    action=parameters[int(agent)].copy()))
    if len(samples) < 2:
        raise RuntimeError("local-observability audit requires multiple samples")
    return samples, sources


def observation_distance(left: Sample, right: Sample) -> float:
    """Permutation-invariant RMS distance with explicit unmatched-edge cost."""
    ego_distance = float(np.linalg.norm(left.ego - right.ego) / np.sqrt(9.0))
    count = max(len(left.edges), len(right.edges))
    if count == 0:
        edge_distance = 0.0
    else:
        costs = np.full((count, count), 1.0, dtype=float)
        if len(left.edges) and len(right.edges):
            delta = left.edges[:, None] - right.edges[None, :]
            costs[:len(left.edges), :len(right.edges)] = (
                np.linalg.norm(delta, axis=-1) / np.sqrt(9.0))
        rows, columns = linear_sum_assignment(costs)
        edge_distance = float(costs[rows, columns].mean())
    return float(np.hypot(ego_distance, edge_distance) / np.sqrt(2.0))


def pair_matrices(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray]:
    count = len(samples)
    observation = np.zeros((count, count), dtype=float)
    action = np.zeros((count, count), dtype=float)
    for left in range(count):
        for right in range(left + 1, count):
            observation[left, right] = observation[right, left] = (
                observation_distance(samples[left], samples[right]))
            action[left, right] = action[right, left] = float(np.linalg.norm(
                samples[left].action - samples[right].action))
    return observation, action


def opposite_chirality(left: Sample, right: Sample) -> bool:
    return left.beta_mode * right.beta_mode < 0


def pair_row(samples: list[Sample], left: int, right: int,
             observation: np.ndarray, action: np.ndarray) -> dict:
    return {
        "left": {
            "case": samples[left].case, "step": samples[left].step,
            "agent": samples[left].agent,
            "action": samples[left].action.tolist(),
            "beta_mode": samples[left].beta_mode,
        },
        "right": {
            "case": samples[right].case, "step": samples[right].step,
            "agent": samples[right].agent,
            "action": samples[right].action.tolist(),
            "beta_mode": samples[right].beta_mode,
        },
        "observation_distance": float(observation[left, right]),
        "action_l2": float(action[left, right]),
        "opposite_chirality": opposite_chirality(
            samples[left], samples[right]),
    }


def neighborhood_statistics(samples: list[Sample], observation: np.ndarray,
                            action: np.ndarray) -> dict:
    count = len(samples)
    masked = observation.copy()
    np.fill_diagonal(masked, np.inf)
    nearest_distances = masked.min(axis=1)
    summaries = {}
    for k in (1, 3, 5):
        effective = min(k, count - 1)
        pairs = [(left, int(right)) for left in range(count)
                 for right in np.argsort(masked[left])[:effective]]
        summaries[str(k)] = {
            "directed_neighbor_links": len(pairs),
            "opposite_chirality_rate": float(np.mean([
                opposite_chirality(samples[left], samples[right])
                for left, right in pairs])),
            "large_action_gap_rate": float(np.mean([
                action[left, right] > 0.5 for left, right in pairs])),
            "opposite_and_large_rate": float(np.mean([
                opposite_chirality(samples[left], samples[right])
                and action[left, right] > 0.5
                for left, right in pairs])),
        }
    threshold_rows = {}
    upper = np.triu_indices(count, 1)
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        selected = [
            (left, right) for left, right in zip(*upper)
            if observation[left, right] <= threshold]
        threshold_rows[f"{threshold:.2f}"] = {
            "undirected_pairs": len(selected),
            "opposite_chirality_rate": (float(np.mean([
                opposite_chirality(samples[left], samples[right])
                for left, right in selected])) if selected else None),
            "opposite_and_large_count": int(sum(
                opposite_chirality(samples[left], samples[right])
                and action[left, right] > 0.5
                for left, right in selected)),
        }
    return {
        "nearest_observation_distance_percentiles": {
            str(percentile): float(np.percentile(
                nearest_distances, percentile))
            for percentile in (10, 25, 50, 75, 90, 95)
        },
        "knn": summaries,
        "distance_thresholds": threshold_rows,
    }


def temporal_flips(samples: list[Sample], observation: np.ndarray,
                   action: np.ndarray) -> list[dict]:
    by_track: dict[tuple[str, int], list[int]] = {}
    for index, sample in enumerate(samples):
        by_track.setdefault((sample.case, sample.agent), []).append(index)
    flips = []
    for indices in by_track.values():
        indices.sort(key=lambda index: samples[index].step)
        for left, right in zip(indices, indices[1:]):
            if opposite_chirality(samples[left], samples[right]):
                row = pair_row(samples, left, right, observation, action)
                neighbors = np.argsort(np.where(
                    np.arange(len(samples)) == right, np.inf,
                    observation[right]))[:5]
                row["target_five_nearest"] = [
                    pair_row(samples, right, int(index), observation, action)
                    for index in neighbors]
                flips.append(row)
    return flips


def run(report_paths: list[Path], output: Path) -> dict:
    samples, sources = load_samples(report_paths)
    observation, action = pair_matrices(samples)
    dangerous = []
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if (observation[left, right] <= 0.20
                    and opposite_chirality(samples[left], samples[right])
                    and action[left, right] > 0.5):
                dangerous.append(pair_row(
                    samples, left, right, observation, action))
    dangerous.sort(key=lambda row: row["observation_distance"])
    result = {
        "audit_kind": "student_local_observability_label_conflict_audit",
        "formal_dataset": False,
        "input_contract": {
            "used_for_distance": (
                "9-D ego features plus unordered sensed-neighbor 9-D edges"),
            "identifier_usage": (
                "case/step/agent retained only for report row identification"),
            "excluded": [
                "Teacher cost", "component id", "global active state",
                "global map", "robot id as a feature"],
            "edge_matching": (
                "minimum-cost permutation matching with unit unmatched cost"),
        },
        "sources": sources,
        "samples": len(samples),
        "positive_mode_samples": int(sum(
            sample.beta_mode > 0 for sample in samples)),
        "negative_mode_samples": int(sum(
            sample.beta_mode < 0 for sample in samples)),
        "near_zero_mode_samples": int(sum(
            sample.beta_mode == 0 for sample in samples)),
        "neighborhood_statistics": neighborhood_statistics(
            samples, observation, action),
        "temporal_chirality_flips": temporal_flips(
            samples, observation, action),
        "dangerous_pairs_within_0_20": dangerous[:50],
        "dangerous_pair_count_within_0_20": len(dangerous),
        "interpretation": (
            "Low observation distance with opposite beta chirality and a large "
            "action gap is a local-inferability warning. Thresholds are a "
            "sensitivity report, not a learned-policy pass criterion."),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "REPORT.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.reports, args.output)
    stats = result["neighborhood_statistics"]
    print(json.dumps({
        "samples": result["samples"],
        "temporal_flips": len(result["temporal_chirality_flips"]),
        "dangerous_pairs_within_0_20": (
            result["dangerous_pair_count_within_0_20"]),
        "nearest_distance_percentiles": (
            stats["nearest_observation_distance_percentiles"]),
        "knn": stats["knn"],
        "report": str(args.output / "REPORT.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
