import unittest

import numpy as np

from run_v2_label_rollout_manifest import select_validation_entries
from run_v2_rollout_collection import select_label_steps


class RolloutCollectionTests(unittest.TestCase):
    def test_first_signal_drives_failure_window(self):
        applied = np.ones((50, 3), dtype=bool)
        sbs = np.zeros((50, 3), dtype=bool)
        sbs[20:, 0] = True
        brake = np.zeros(50, dtype=int)
        brake[12] = 1
        outside = np.zeros(50, dtype=int)
        outside[30] = 1
        distance = np.full(50, 0.3)
        steps, reason, event = select_label_steps(
            applied, sbs, brake, outside, distance, "safety_invalid", 0.2)
        self.assertEqual(event, 12)
        self.assertEqual(reason, "first_certified_brake")
        self.assertEqual(steps, [0, 4, 8, 12])

    def test_success_samples_active_coverage(self):
        applied = np.zeros((20, 2), dtype=bool)
        applied[[2, 5, 9, 15], 0] = True
        steps, reason, event = select_label_steps(
            applied, np.zeros_like(applied), np.zeros(20, dtype=int),
            np.zeros(20, dtype=int), np.full(20, 0.3),
            "all_goal_dwell", 0.2, maximum=3)
        self.assertEqual(steps, [2, 9, 15])
        self.assertEqual(reason, "successful_active_coverage")
        self.assertIsNone(event)

    def test_step_limit_uses_active_tail_not_early_brake(self):
        applied = np.ones((50, 2), dtype=bool)
        brake = np.zeros(50, dtype=int)
        brake[5] = 1
        steps, reason, event = select_label_steps(
            applied, np.zeros_like(applied), brake,
            np.zeros(50, dtype=int), np.full(50, 0.3),
            "step_limit", 0.2)
        self.assertEqual(event, 49)
        self.assertEqual(reason, "step_limit_active_tail")
        self.assertEqual(steps, [18, 28, 39, 49])

    def test_validation_cap_is_per_scene_and_automatic(self):
        entries = [
            {"scene_id": "b", "snapshot_step": 5},
            {"scene_id": "a", "snapshot_step": 3},
            {"scene_id": "a", "snapshot_step": 7},
            {"scene_id": "b", "snapshot_step": 9},
        ]
        selected = select_validation_entries(entries, 1)
        self.assertEqual(
            [(row["scene_id"], row["snapshot_step"]) for row in selected],
            [("a", 7), ("b", 9)])


if __name__ == "__main__":
    unittest.main()
