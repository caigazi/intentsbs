import ast
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import jax.numpy as jnp

from sbs824.simulation import _dare_gain
from sbs824.v2.dataset import DecisionDataset
from sbs824.v2.features import build_local_features
from sbs824.v2.protocol import (IntentMode, SYNC_EVENT_V2,
                                make_v2_config)
from sbs824.v2.runtime import initialize_runtime, step_runtime
from sbs824.v2.runtime import prepare_step
from sbs824.v2.jax_safety import batched_wang_qp
from sbs824.v2.jax_rollout import batched_candidate_costs
from sbs824.v2.sampled_safety import sampled_wang_step
from sbs824.v2.teacher import (AuthoritativeComponentTeacher,
                               AuthoritativeTeacherBudget,
                               ComponentCEMTeacher, TeacherBudget,
                               TeacherDecision)
from sbs824.v2.trigger import (TriggerResult, local_shadow_ratios,
                               shadow_progress_ratio, ttc_candidates)
from sbs824.wang_safety import solve_wang_braking_qp, wang_pair_barrier


class SyncEventV2Test(unittest.TestCase):
    def setUp(self):
        self.cfg = make_v2_config(n_agents=3, n_obstacles=0, steps=1)
        self.gain = _dare_gain(self.cfg.dt, self.cfg.mass)

    def test_v2_core_has_no_legacy_experiment_imports(self):
        root = Path(__file__).parents[1] / "sbs824" / "v2"
        banned = ("train_", "phase1_", "phase2_", "rolling_cem_", "round1", "round2", "round3")
        for path in root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(
                any(token in name for token in banned for name in imports),
                f"legacy dependency leaked into {path.name}: {imports}")

    def test_out_of_range_robot_cannot_change_ego_trigger_or_features(self):
        state = np.array([
            [1.00, 1.00, 0.0, 0.0],
            [1.40, 1.00, 0.0, 0.0],
            [3.50, 3.50, 0.0, 0.0],
        ])
        goals = np.array([[2.0, 1.0], [0.0, 1.0], [3.8, 3.8]])
        base = np.array([[0.4, 0.0], [-0.4, 0.0], [0.1, 0.0]])
        messages = base.copy()
        mode = np.array([IntentMode.ACTIVE, IntentMode.ACTIVE, IntentMode.BYPASS])
        evidence = np.array([True, True, False])

        trigger_a = ttc_candidates(
            state, base, messages, self.cfg, SYNC_EVENT_V2)
        feature_a = build_local_features(
            state, goals, base, messages, mode, evidence,
            self.cfg, SYNC_EVENT_V2)
        changed_state = state.copy()
        changed_goals = goals.copy()
        changed_base = base.copy()
        changed_messages = messages.copy()
        changed_state[2] = [3.10, 3.75, -0.5, 0.5]
        changed_goals[2] = [0.1, 3.9]
        changed_base[2] = [-0.5, 0.5]
        changed_messages[2] = [0.5, -0.5]
        trigger_b = ttc_candidates(
            changed_state, changed_base, changed_messages,
            self.cfg, SYNC_EVENT_V2)
        feature_b = build_local_features(
            changed_state, changed_goals, changed_base, changed_messages,
            mode, np.array([True, True, True]), self.cfg,
            SYNC_EVENT_V2)
        self.assertEqual(trigger_a.evidence[0], trigger_b.evidence[0])
        np.testing.assert_allclose(
            feature_a.self_features[0], feature_b.self_features[0])
        np.testing.assert_allclose(
            feature_a.edge_features[0], feature_b.edge_features[0])
        np.testing.assert_array_equal(
            feature_a.edge_mask[0], feature_b.edge_mask[0])

    def test_neighbor_active_state_is_not_an_edge_feature(self):
        state = np.array([
            [1.00, 1.00, 0.0, 0.0],
            [1.40, 1.00, 0.0, 0.0],
            [1.80, 1.00, 0.0, 0.0],
        ])
        goals = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
        base = np.zeros((3, 2))
        messages = np.array([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0]])
        evidence = np.zeros(3, dtype=bool)
        first = build_local_features(
            state, goals, base, messages,
            np.array([IntentMode.BYPASS] * 3), evidence,
            self.cfg, SYNC_EVENT_V2)
        second = build_local_features(
            state, goals, base, messages,
            np.array([IntentMode.BYPASS, IntentMode.ACTIVE, IntentMode.RELEASE]),
            evidence, self.cfg, SYNC_EVENT_V2)
        # Ego 0 may know its own mode only; neighbor 1's mode cannot alter edge 0->1.
        np.testing.assert_allclose(first.edge_features[0], second.edge_features[0])

    def test_runtime_has_one_mask_source(self):
        state = np.array([
            [1.00, 1.00, 0.0, 0.0],
            [1.40, 1.00, 0.0, 0.0],
            [3.50, 3.50, 0.0, 0.0],
        ])
        goals = np.array([[2.0, 1.0], [0.0, 1.0], [3.8, 3.8]])
        runtime = initialize_runtime(
            state, goals, self.gain, self.cfg, SYNC_EVENT_V2)
        proposal = np.array([[0.3, 0.8], [0.2, 0.7], [-1.0, -1.0]])
        _, trace = step_runtime(
            runtime, goals, self.gain, self.cfg, SYNC_EVENT_V2,
            proposal)
        np.testing.assert_array_equal(trace.optimized_mask, trace.label_mask)
        np.testing.assert_array_equal(trace.label_mask, trace.applied_mask)
        self.assertTrue(trace.applied_mask[0])
        self.assertTrue(trace.applied_mask[1])
        self.assertFalse(trace.applied_mask[2])

    def test_frozen_manifest_is_written_and_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "formal"
            DecisionDataset().save(
                output, SYNC_EVENT_V2,
                teacher_tier="authoritative_v2")
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], SYNC_EVENT_V2.manifest())
            SYNC_EVENT_V2.assert_manifest_matches(manifest["protocol"])
            altered = dict(manifest["protocol"])
            altered["control_dt"] = 0.21
            with self.assertRaisesRegex(RuntimeError, "control_dt"):
                SYNC_EVENT_V2.assert_manifest_matches(altered)

    def test_development_protocol_refuses_formal_dataset(self):
        development = replace(
            SYNC_EVENT_V2, version="sync_event_v2_dev", development=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "development protocol"):
                DecisionDataset().save(
                    Path(directory) / "formal", development,
                    teacher_tier="authoritative_v2")

    def test_formal_dataset_rejects_debug_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "authoritative_v2"):
                DecisionDataset().save(
                    Path(directory) / "formal", SYNC_EVENT_V2,
                    teacher_tier="debug_quick_non_authoritative")

    def test_shadow_candidate_uses_all_sensed_neighbors(self):
        cfg = make_v2_config(n_agents=3, n_obstacles=0)
        state = np.array([
            [1.00, 1.00, 0.0, 0.0],
            [1.25, 1.00, 0.0, 0.0],
            [1.45, 1.00, 0.0, 0.0],
        ])
        reference = np.array([
            [0.2, 0.0], [-0.2, 0.0], [0.0, 0.1],
        ])
        candidates = TriggerResult(
            evidence=np.array([True, True, False]),
            edges=((0, 1),), components=((0, 1),))
        with patch(
                "sbs824.v2.trigger.shadow_progress_ratio",
                return_value=0.5) as ratio:
            result = local_shadow_ratios(
                state, reference, candidates, cfg, horizon_steps=2)
        self.assertEqual(ratio.call_count, 2)
        self.assertTrue(all(call.args[1].shape == (3, 4)
                            for call in ratio.call_args_list))
        np.testing.assert_allclose(result, [0.5, 0.5, 1.0])

    def test_internal_sampled_margin_preserves_public_hard_line(self):
        self.assertAlmostEqual(
            self.cfg.wang_pair_safe_radius_factor * self.cfg.car_radius,
            0.2025)
        self.assertAlmostEqual(SYNC_EVENT_V2.hard_center_distance, 0.20)
        self.assertEqual(self.cfg.wang_safety_substeps, 32)
        self.assertEqual(SYNC_EVENT_V2.integration_substeps, 32)
        self.assertEqual(TeacherBudget().planning_integration_substeps, 32)

    def test_runtime_solves_safety_qp_once_per_control_tick(self):
        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        gain = _dare_gain(cfg.dt, cfg.mass)
        state = np.array([[1.0, 1.0, 0.2, 0.0],
                          [1.4, 1.0, -0.2, 0.0]])
        goals = np.array([[2.0, 1.0], [0.0, 1.0]])
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        _, trace = step_runtime(
            runtime, goals, gain, cfg, SYNC_EVENT_V2,
            np.tile([1.0, 0.0], (2, 1)))
        self.assertEqual(trace.safety_qp_solves, 1)

        with patch(
                "sbs824.v2.trigger.sampled_wang_step",
                wraps=sampled_wang_step) as shared_safety:
            shadow_progress_ratio(
                0, state, np.array([[0.2, 0.0], [-0.2, 0.0]]), cfg,
                horizon_steps=2, safety_substeps=8)
        self.assertEqual(shared_safety.call_count, 2)
        self.assertTrue(all(
            call.kwargs["integration_substeps"] == 8
            for call in shared_safety.call_args_list))

    def test_low_speed_eq17_stops_before_tick_end_without_leaving_certificate(self):
        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        state = np.array([
            [2.20146151, 1.83823950, -0.05620068, 0.08980254],
            [2.23408278, 2.03960476, -0.09133391, -0.07095307],
        ])
        initial_barrier = wang_pair_barrier(state[0], state[1], cfg)
        self.assertGreater(initial_barrier, 0.0)

        result = sampled_wang_step(
            state, np.zeros((2, 2)), [], cfg, np.ones(2, dtype=bool),
            integration_substeps=SYNC_EVENT_V2.integration_substeps)

        self.assertEqual(result.certified_brake_agents, (0, 1))
        self.assertEqual(result.out_of_certificate_pair_samples, 0)
        self.assertGreaterEqual(
            result.minimum_center_distance,
            SYNC_EVENT_V2.hard_center_distance)
        np.testing.assert_allclose(
            result.next_state[:, 2:], np.zeros((2, 2)), atol=1e-12)
        self.assertGreaterEqual(
            wang_pair_barrier(
                result.next_state[0], result.next_state[1], cfg), 0.0)

    def test_jax_batched_qp_matches_numpy_reference_actions(self):
        cfg = make_v2_config(n_agents=5, n_obstacles=0)
        rng = np.random.default_rng(824)
        # Separated but interacting states exercise multiple local rows while
        # remaining inside the Wang certificate.
        angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
        position = np.column_stack((np.cos(angles), np.sin(angles))) * 0.22 + 2.0
        batches = []
        references = []
        for _ in range(3):
            velocity = rng.uniform(-0.18, 0.18, size=(5, 2))
            batches.append(np.column_stack((position, velocity)))
            references.append(rng.uniform(-0.8, 0.8, size=(5, 2)))
        states = np.asarray(batches)
        refs = np.asarray(references)
        latch = np.zeros((3, 5), dtype=bool)
        actual, _, _, _ = batched_wang_qp(
            jnp.asarray(states), jnp.asarray(refs), jnp.asarray(latch),
            n_agents=5, sense_radius=cfg.sense_radius, mass=cfg.mass,
            max_force=cfg.max_force, dt=cfg.dt,
            safe_distance=(cfg.wang_pair_safe_radius_factor * cfg.car_radius),
            wang_gamma=cfg.wang_gamma)
        expected = np.asarray([
            solve_wang_braking_qp(states[index], refs[index], [], cfg,
                                  braking_latch=latch[index].copy()).action
            for index in range(3)
        ])
        np.testing.assert_allclose(np.asarray(actual), expected,
                                   atol=2e-4, rtol=2e-4)

    def test_jax_candidate_cost_matches_numpy_reference_without_violation(self):
        from sbs824.v2.teacher import TeacherBudget

        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        gain = _dare_gain(cfg.dt, cfg.mass)
        state = np.array([[1.0, 1.0, 0.0, 0.0],
                          [1.4, 1.0, 0.0, 0.0]])
        goals = np.array([[2.0, 1.0], [0.0, 1.0]])
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)
        budget = TeacherBudget(horizon_steps=4,
                               planning_integration_substeps=4)
        teacher = ComponentCEMTeacher(
            SYNC_EVENT_V2, budget=budget, seed=1)
        candidates = np.asarray([
            [[0.3, 0.7], [0.3, 0.7]],
            [[-0.1, -0.8], [-0.1, -0.8]],
        ])
        active = prepared.mode == IntentMode.ACTIVE
        focus = active.copy()
        actual = batched_candidate_costs(
            jnp.asarray(runtime.physical), jnp.asarray(runtime.world_intent),
            jnp.asarray(runtime.braking_latch),
            jnp.asarray(runtime.previous_safe_force), jnp.asarray(candidates),
            jnp.asarray(runtime.parameters), jnp.asarray(goals),
            jnp.asarray(gain), jnp.asarray(active),
            jnp.zeros_like(jnp.asarray(active)), jnp.asarray(focus),
            n_agents=2, horizon_steps=4, integration_substeps=4,
            runtime_step=runtime.step, dt=cfg.dt, mass=cfg.mass,
            sense_radius=cfg.sense_radius, max_force=cfg.max_force,
            max_speed=cfg.max_speed,
            safe_distance=cfg.wang_pair_safe_radius_factor * cfg.car_radius,
            hard_distance=SYNC_EVENT_V2.hard_center_distance,
            wang_gamma=cfg.wang_gamma,
            lateral_speed=SYNC_EVENT_V2.lateral_speed,
            max_intent_accel=SYNC_EVENT_V2.max_intent_accel,
            intent_lookahead=SYNC_EVENT_V2.intent_lookahead,
            action_smooth_weight=SYNC_EVENT_V2.action_smooth_weight)
        ids = np.flatnonzero(active)
        expected = np.asarray([
            teacher._rollout_cost(
                runtime, goals, gain, cfg, candidate, ids,
                np.zeros(0, dtype=int), ids, integration_substeps=4)
            for candidate in candidates
        ])
        np.testing.assert_allclose(np.asarray(actual), expected,
                                   atol=2e-2, rtol=2e-4)

    def test_teacher_exact_jax_evaluator_matches_numpy_at_32_substeps(self):
        from sbs824.v2.teacher import TeacherBudget

        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        gain = _dare_gain(cfg.dt, cfg.mass)
        state = np.array([[1.0, 1.0, 0.0, 0.0],
                          [1.4, 1.0, 0.0, 0.0]])
        goals = np.array([[2.0, 1.0], [0.0, 1.0]])
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)
        teacher = ComponentCEMTeacher(
            SYNC_EVENT_V2,
            budget=TeacherBudget(horizon_steps=4), seed=1)
        candidates = np.asarray([
            [[0.3, 0.7], [0.3, 0.7]],
            [[-0.1, -0.8], [-0.1, -0.8]],
        ])
        active_ids = np.flatnonzero(prepared.mode == IntentMode.ACTIVE)
        release_ids = np.flatnonzero(prepared.mode == IntentMode.RELEASE)
        actual = teacher.evaluate_candidates_exact_jax(
            runtime, goals, gain, cfg, candidates, active_ids, release_ids,
            active_ids)
        expected = np.asarray([
            teacher._rollout_cost(
                runtime, goals, gain, cfg, candidate, active_ids,
                release_ids, active_ids,
                integration_substeps=SYNC_EVENT_V2.integration_substeps)
            for candidate in candidates
        ])
        np.testing.assert_allclose(actual, expected, atol=2e-2, rtol=2e-4)

    def test_teacher_reuse_runs_no_hidden_objective_rollout(self):
        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        state = np.array([[1.0, 1.0, 0.0, 0.0],
                          [1.4, 1.0, 0.0, 0.0]])
        goals = np.array([[2.0, 1.0], [0.0, 1.0]])
        gain = _dare_gain(cfg.dt, cfg.mass)
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)

        class CheapTeacher(ComponentCEMTeacher):
            def _rollout_cost(self, runtime, goals, gain, cfg, parameters,
                              active_ids, release_ids, focus_ids, **_kwargs):
                self.objective_rollouts += 1
                target = np.array([0.25, 0.75])
                return float(np.sum((parameters[focus_ids] - target) ** 2))

        teacher = CheapTeacher(
            SYNC_EVENT_V2,
            budget=TeacherBudget(candidate_backend="numpy"), seed=10)
        first = teacher.decide(runtime, prepared, goals, gain, cfg)
        after_search = teacher.objective_rollouts
        second = teacher.decide(runtime, prepared, goals, gain, cfg)
        self.assertGreater(after_search, 0)
        self.assertEqual(teacher.objective_rollouts, after_search)
        self.assertTrue(np.all(second.source[prepared.refresh_mask] == "reuse"))
        np.testing.assert_allclose(
            second.parameters[prepared.refresh_mask],
            first.parameters[prepared.refresh_mask])

    def test_teacher_branch_tie_uses_recovered_improvement_scale(self):
        self.assertTrue(ComponentCEMTeacher._branch_costs_near_tie(
            1000.0, 10.0, 15.0, 0.01))
        self.assertFalse(ComponentCEMTeacher._branch_costs_near_tie(
            1000.0, 10.0, 30.0, 0.01))
        self.assertFalse(ComponentCEMTeacher._branch_costs_near_tie(
            10.0, 10.0, 10.02, 0.01))

    def test_teacher_budget_uses_nontrivial_exact_shortlist(self):
        budget = TeacherBudget()
        self.assertGreaterEqual(budget.exact_shortlist_size, 16)
        self.assertEqual(budget.jax_evaluation_batch_size, 256)
        self.assertEqual(
            budget.planning_integration_substeps,
            SYNC_EVENT_V2.integration_substeps)

    def test_authoritative_teacher_budget_rejects_saving_shortcuts(self):
        with self.assertRaises(ValueError):
            AuthoritativeTeacherBudget(samples=4095)
        with self.assertRaises(ValueError):
            AuthoritativeTeacherBudget(iterations=2)
        with self.assertRaises(ValueError):
            AuthoritativeTeacherBudget(restarts=1)
        with self.assertRaises(ValueError):
            AuthoritativeTeacherBudget(horizon_steps=39)
        budget = AuthoritativeTeacherBudget()
        self.assertEqual(budget.restarts, 4)
        internal = budget.search_budget(SYNC_EVENT_V2)
        self.assertEqual(internal.strong_samples, 4096)
        self.assertEqual(internal.light_samples, internal.strong_samples)
        self.assertEqual(internal.light_iterations,
                         internal.strong_iterations)
        self.assertEqual(internal.planning_integration_substeps,
                         SYNC_EVENT_V2.integration_substeps)
        self.assertEqual(internal.candidate_backend, "jax_x64")
        self.assertEqual(AuthoritativeComponentTeacher.tier,
                         "authoritative_v2")

    def test_authoritative_teacher_uses_four_fresh_restarts_every_tick(self):
        cfg = make_v2_config(n_agents=2, n_obstacles=0)
        state = np.array([[1.0, 1.0, 0.0, 0.0],
                          [1.4, 1.0, 0.0, 0.0]])
        goals = np.array([[2.0, 1.0], [0.0, 1.0]])
        gain = _dare_gain(cfg.dt, cfg.mass)
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)
        self.assertTrue(prepared.refresh_mask.all())

        calls = {"value": 0}

        def fake_decide(_core, runtime, prepared, *_args, **_kwargs):
            parameters = runtime.parameters.copy()
            parameters[prepared.refresh_mask] = (
                [0.8, 0.5] if calls["value"] % 2 == 0
                else [0.9, 0.1])
            calls["value"] += 1
            source = np.full(cfg.n_agents, "strong", dtype="U8")
            finite = np.full(cfg.n_agents, 10.0)
            return TeacherDecision(
                parameters=parameters, source=source,
                teacher_cost=finite.copy(), incumbent_cost=finite.copy(),
                validated=prepared.refresh_mask.copy(),
                component_id=np.zeros(cfg.n_agents, dtype=np.int32),
                positive_branch_cost=finite.copy(),
                negative_branch_cost=finite.copy(),
                branch_near_tie=np.zeros(cfg.n_agents, dtype=bool))

        with (patch.object(
                ComponentCEMTeacher, "decide", autospec=True,
                side_effect=fake_decide) as decide_mock,
              patch.object(
                ComponentCEMTeacher, "evaluate_candidates_exact_jax",
                autospec=True,
                return_value=np.array([10.0, 20.0])) as exact_mock):
            teacher = AuthoritativeComponentTeacher(
                SYNC_EVENT_V2, seed=7)
            first = teacher.decide(runtime, prepared, goals, gain, cfg)
            self.assertEqual(decide_mock.call_count, 4)
            self.assertEqual(exact_mock.call_count, 4)
            self.assertTrue(np.all(
                first.source[prepared.refresh_mask] == "auth"))
            np.testing.assert_allclose(
                first.parameters[prepared.refresh_mask],
                np.tile([0.9, 0.1], (2, 1)))
            self.assertEqual(teacher.statistics()[
                "admissible_restart_counts"], [4])
            teacher.decide(runtime, prepared, goals, gain, cfg)
            self.assertEqual(decide_mock.call_count, 8)
            self.assertEqual(exact_mock.call_count, 8)

    def test_teacher_splits_disconnected_active_components(self):
        cfg = make_v2_config(n_agents=4, n_obstacles=0)
        state = np.array([
            [0.8, 1.0, 0.0, 0.0], [1.2, 1.0, 0.0, 0.0],
            [2.8, 3.0, 0.0, 0.0], [3.2, 3.0, 0.0, 0.0],
        ])
        goals = np.array([
            [1.5, 1.0], [0.5, 1.0], [3.5, 3.0], [2.5, 3.0],
        ])
        gain = _dare_gain(cfg.dt, cfg.mass)
        runtime = initialize_runtime(
            state, goals, gain, cfg, SYNC_EVENT_V2)
        prepared = prepare_step(
            runtime, goals, gain, cfg, SYNC_EVENT_V2)

        class CheapTeacher(ComponentCEMTeacher):
            def _rollout_cost(self, runtime, goals, gain, cfg, parameters,
                              active_ids, release_ids, focus_ids, **_kwargs):
                self.objective_rollouts += 1
                return float(np.sum(parameters[focus_ids] ** 2))

        teacher = CheapTeacher(
            SYNC_EVENT_V2,
            budget=TeacherBudget(candidate_backend="numpy"), seed=11)
        decision = teacher.decide(runtime, prepared, goals, gain, cfg)
        self.assertEqual(len(teacher.caches), 2)
        self.assertEqual({cache.members for cache in teacher.caches},
                         {(0, 1), (2, 3)})
        self.assertNotEqual(decision.component_id[0], decision.component_id[2])
        stats = teacher.statistics()
        # Cheap test override counts one incumbent, two exact branch checks,
        # plus 2 branches * 12 * 3 approximate candidates per component.
        self.assertEqual(stats["objective_rollouts"], 2 * (3 + 2 * 12 * 3))
        self.assertEqual(stats["incumbent_and_branch_validation_rollouts"], 6)


if __name__ == "__main__":
    unittest.main()
