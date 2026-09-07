import unittest

import numpy as np

from run_v2_label_rollout_manifest import teacher_label_is_accepted
from run_v2_small_dagger_train import teacher_record_is_accepted


class DaggerLabelAcceptanceTests(unittest.TestCase):
    def test_teacher_winner_is_accepted(self):
        accepted, source, baseline = teacher_label_is_accepted(
            np.asarray([10.0, 12.0, 15.0]))
        self.assertTrue(accepted)
        self.assertEqual(source, "student")
        self.assertEqual(baseline, 12.0)

    def test_optimizer_miss_abstains(self):
        row = {
            "teacher_exact_cost": 3816.8,
            "student_exact_cost": 648.8,
            "incumbent_exact_cost": 10221.2,
        }
        self.assertFalse(teacher_record_is_accepted(row))

    def test_legacy_authoritative_seed_record_is_accepted(self):
        self.assertTrue(teacher_record_is_accepted({"parameters": [[1, 0]]}))


if __name__ == "__main__":
    unittest.main()
