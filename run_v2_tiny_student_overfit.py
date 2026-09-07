"""Development-only tiny Student action overfit and short closed-loop replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from flax import serialization
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from run_v2_teacher_smoke import minimum_center, scene
from run_v2_teacher_temporal_audit import local_observation_vector
from sbs824.simulation import _dare_gain
from sbs824.v2.protocol import SYNC_EVENT_V2, make_v2_config
from sbs824.v2.runtime import apply_prepared_step, initialize_runtime, prepare_step
from sbs824.v2.student import IntentStudent, StudentArchitecture


def parse_observation(values, slots):
    values = np.asarray(values, dtype=np.float32)
    rows = values[9:].reshape(-1, 10)
    edge = np.zeros((slots, 9), dtype=np.float32)
    mask = np.zeros(slots, dtype=bool)
    count = min(slots, len(rows))
    edge[:count] = rows[:count, :9]
    mask[:count] = rows[:count, 9] > 0.5
    return values[:9], edge, mask


def load_training_data(report_paths):
    reports = [json.loads(path.read_text(encoding="utf-8"))
               for path in report_paths]
    slots = max(len(row["parameters"]) - 1
                for report in reports for row in report["records"])
    self_features, edge_features, edge_mask, labels, provenance = [], [], [], [], []
    for report, path in zip(reports, report_paths):
        if report.get("teacher_tier") != "authoritative_v2":
            raise RuntimeError(f"non-authoritative report rejected: {path}")
        if report.get("formal_dataset", True):
            raise RuntimeError("tiny audit expects development reports, not formal data")
        if report["protocol"] != SYNC_EVENT_V2.manifest():
            raise RuntimeError(f"protocol manifest differs: {path}")
        for row in report["records"]:
            parameters = np.asarray(row["parameters"], dtype=np.float32)
            for agent in row["active_ids"]:
                own, edges, mask = parse_observation(
                    row["local_observations"][str(agent)], slots)
                self_features.append(own)
                edge_features.append(edges)
                edge_mask.append(mask)
                labels.append(parameters[agent])
                provenance.append({
                    "case": report["case"], "step": row["runtime_step"],
                    "agent": agent,
                })
    return ({
        "self": np.asarray(self_features, dtype=np.float32),
        "edges": np.asarray(edge_features, dtype=np.float32),
        "mask": np.asarray(edge_mask, dtype=bool),
        "labels": np.asarray(labels, dtype=np.float32),
    }, reports, provenance)


def train(data, seed, steps, learning_rate, architecture):
    model = IntentStudent(architecture)
    key = jax.random.PRNGKey(seed)
    variables = model.init(
        key, jnp.asarray(data["self"][:1]), jnp.asarray(data["edges"][:1]),
        jnp.asarray(data["mask"][:1]))
    schedule = optax.cosine_decay_schedule(learning_rate, steps, alpha=0.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=optimizer)
    batch = {key: jnp.asarray(value) for key, value in data.items()}
    target_logits = jnp.arctanh(jnp.clip(
        batch["labels"], -0.999, 0.999))

    @jax.jit
    def step_fn(current):
        def loss_fn(params):
            logits = model.apply(
                {"params": params}, batch["self"], batch["edges"], batch["mask"],
                return_logits=True)
            error = logits - target_logits
            return jnp.mean(error * error)
        loss, gradients = jax.value_and_grad(loss_fn)(current.params)
        return current.apply_gradients(grads=gradients), loss

    history = []
    started = perf_counter()
    for index in range(steps):
        state, loss = step_fn(state)
        if index == 0 or (index + 1) % 500 == 0:
            logits = np.asarray(model.apply(
                {"params": state.params}, batch["self"], batch["edges"],
                batch["mask"], return_logits=True))
            prediction = np.asarray(model.apply(
                {"params": state.params}, batch["self"], batch["edges"],
                batch["mask"]))
            error = prediction - data["labels"]
            history.append({
                "step": index + 1,
                "logit_mse": float(np.mean(
                    (logits - np.asarray(target_logits)) ** 2)),
                "mse": float(np.mean(error ** 2)),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "maximum_action_l2": float(np.linalg.norm(error, axis=1).max()),
                "beta_sign_accuracy": float(np.mean(
                    np.sign(np.where(np.abs(prediction[:, 1]) > 0.03,
                                     prediction[:, 1], 0.0))
                    == np.sign(np.where(np.abs(data["labels"][:, 1]) > 0.03,
                                        data["labels"][:, 1], 0.0)))),
            })
    jax.block_until_ready(state.params)
    return model, state.params, history, perf_counter() - started


def feature_arrays(features):
    return (jnp.asarray(features.self_features),
            jnp.asarray(features.edge_features),
            jnp.asarray(features.edge_mask))


def replay_case(model, params, report):
    rows = {int(row["runtime_step"]): row for row in report["records"]}
    initial, goals, _, scene_seed = scene(report["case"])
    cfg = make_v2_config(
        n_agents=len(initial), n_obstacles=0, seed=scene_seed, steps=600,
        adaptive_goal_dwell_steps=20)
    gain = _dare_gain(cfg.dt, cfg.mass)
    student_runtime = initialize_runtime(initial, goals, gain, cfg, SYNC_EVENT_V2)
    teacher_runtime = initialize_runtime(initial, goals, gain, cfg, SYNC_EVENT_V2)
    student_minimum = minimum_center(student_runtime.physical)
    teacher_minimum = minimum_center(teacher_runtime.physical)
    action_errors, active_schedule_mismatches = [], 0
    position_errors, velocity_errors = [], []
    student_certificate_violations = 0
    teacher_replay_feature_error = 0.0
    while student_runtime.step <= max(rows):
        student_prepared = prepare_step(
            student_runtime, goals, gain, cfg, SYNC_EVENT_V2)
        teacher_prepared = prepare_step(
            teacher_runtime, goals, gain, cfg, SYNC_EVENT_V2)
        own, edges, mask = feature_arrays(student_prepared.features)
        proposal = np.asarray(model.apply(
            {"params": params}, own, edges, mask), dtype=float)
        teacher_row = rows.get(student_runtime.step)
        if teacher_row is not None:
            teacher_proposal = np.asarray(
                teacher_row["parameters"], dtype=float)
            saved_teacher_ids = teacher_row["active_ids"]
            replay_teacher_ids = np.flatnonzero(
                teacher_prepared.refresh_mask).astype(int).tolist()
            if replay_teacher_ids != saved_teacher_ids:
                raise RuntimeError(
                    f"saved Teacher replay differs at step {student_runtime.step}")
            for agent in replay_teacher_ids:
                actual = local_observation_vector(
                    teacher_prepared.features, agent, cfg.n_agents)
                expected = np.asarray(
                    teacher_row["local_observations"][str(agent)], dtype=float)
                teacher_replay_feature_error = max(
                    teacher_replay_feature_error,
                    float(np.max(np.abs(actual - expected))))
            student_ids = np.flatnonzero(
                student_prepared.refresh_mask).astype(int).tolist()
            teacher_ids = teacher_row["active_ids"]
            if student_ids != teacher_ids:
                active_schedule_mismatches += 1
            common = sorted(set(student_ids).intersection(teacher_ids))
            if common:
                action_errors.extend(np.linalg.norm(
                    proposal[common] - teacher_proposal[common], axis=1).tolist())
        else:
            teacher_proposal = teacher_runtime.parameters.copy()
            student_ids = np.flatnonzero(
                student_prepared.refresh_mask).astype(int).tolist()
            teacher_ids = np.flatnonzero(
                teacher_prepared.refresh_mask).astype(int).tolist()
            if student_ids != teacher_ids:
                active_schedule_mismatches += 1
        student_runtime, student_trace = apply_prepared_step(
            student_runtime, student_prepared, goals, gain, cfg,
            SYNC_EVENT_V2, proposal)
        teacher_runtime, teacher_trace = apply_prepared_step(
            teacher_runtime, teacher_prepared, goals, gain, cfg,
            SYNC_EVENT_V2, teacher_proposal)
        per_agent_position = np.linalg.norm(
            student_runtime.physical[:, :2]
            - teacher_runtime.physical[:, :2], axis=1)
        per_agent_velocity = np.linalg.norm(
            student_runtime.physical[:, 2:]
            - teacher_runtime.physical[:, 2:], axis=1)
        position_errors.extend(per_agent_position.tolist())
        velocity_errors.extend(per_agent_velocity.tolist())
        student_minimum = min(
            student_minimum, minimum_center(student_runtime.physical))
        teacher_minimum = min(
            teacher_minimum, minimum_center(teacher_runtime.physical))
        student_certificate_violations += (
            student_trace.out_of_certificate_pair_samples)
    return {
        "case": report["case"],
        "evaluated_through_runtime_step": max(rows),
        "student_minimum_center_distance": student_minimum,
        "teacher_minimum_center_distance": teacher_minimum,
        "student_certificate_violation_samples": (
            student_certificate_violations),
        "active_schedule_mismatch_steps": active_schedule_mismatches,
        "maximum_teacher_replay_feature_error": teacher_replay_feature_error,
        "maximum_agent_position_l2_from_teacher": max(
            position_errors, default=0.0),
        "mean_agent_position_l2_from_teacher": float(np.mean(position_errors)),
        "maximum_agent_velocity_l2_from_teacher": max(
            velocity_errors, default=0.0),
        "mean_agent_velocity_l2_from_teacher": float(np.mean(velocity_errors)),
        "same_step_cross_state_action_samples": len(action_errors),
        "mean_same_step_cross_state_action_l2": (
            float(np.mean(action_errors)) if action_errors else None),
        "maximum_same_step_cross_state_action_l2": max(
            action_errors, default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=824)
    args = parser.parse_args()
    data, reports, provenance = load_training_data(args.reports)
    architecture = StudentArchitecture()
    model, params, history, wall_seconds = train(
        data, args.seed, args.steps, args.learning_rate, architecture)
    fitted = np.asarray(model.apply(
        {"params": params}, jnp.asarray(data["self"]),
        jnp.asarray(data["edges"]), jnp.asarray(data["mask"])))
    fitted_l2 = np.linalg.norm(fitted - data["labels"], axis=1)
    boundary_fit = []
    for index, source in enumerate(provenance):
        if ((source["case"] == "n5_partial" and source["step"] in (19, 20))
                or (source["case"] == "n8_two_stream"
                    and source["step"] in (13, 14))):
            boundary_fit.append({
                **source,
                "target": data["labels"][index].tolist(),
                "prediction": fitted[index].tolist(),
                "action_l2": float(fitted_l2[index]),
            })
    worst_indices = np.argsort(fitted_l2)[-10:][::-1]
    worst_fit = [{
        **provenance[int(index)],
        "target": data["labels"][index].tolist(),
        "prediction": fitted[index].tolist(),
        "action_l2": float(fitted_l2[index]),
    } for index in worst_indices]
    replay_reports = [
        report for report in reports
        if report.get("audit_kind")
        != "tiny_student_failure_window_authoritative_labels"
    ]
    closed_loop = [
        replay_case(model, params, report) for report in replay_reports]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "checkpoint.msgpack").write_bytes(
        serialization.to_bytes(params))
    report = {
        "audit_kind": "tiny_student_action_overfit_and_short_closed_loop",
        "formal_dataset": False,
        "formal_checkpoint": False,
        "protocol": SYNC_EVENT_V2.manifest(),
        "architecture": architecture.manifest(),
        "parameter_count": int(sum(value.size for value in jax.tree_util.tree_leaves(params))),
        "training_sources": [str(path) for path in args.reports],
        "short_replay_sources": [
            report["case"] for report in replay_reports],
        "training_samples": len(data["labels"]),
        "training_steps": args.steps,
        "learning_rate": args.learning_rate,
        "training_loss": (
            "MSE on unclipped Student logits against atanh-clipped Teacher "
            "Intent; clip magnitude=0.999"),
        "seed": args.seed,
        "history": history,
        "final": history[-1],
        "boundary_fit": boundary_fit,
        "worst_fit": worst_fit,
        "short_closed_loop": closed_loop,
        "wall_seconds": wall_seconds,
        "interpretation": (
            "Development Gate-2 precursor only. It tests exact action memorization "
            "and the saved short windows; it is not a full-trajectory or "
            "generalization result."),
    }
    (args.output / "REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_samples": report["training_samples"],
        "parameter_count": report["parameter_count"],
        "final": report["final"],
        "short_closed_loop": closed_loop,
        "wall_seconds": wall_seconds,
        "report": str(args.output / "REPORT.json"),
    }), flush=True)


if __name__ == "__main__":
    main()
