import unittest

import numpy as np

from sbs824.liveness import (CoordinationGate, GateMode, Intent, LocalSBSMonitor,
                             LocalStep, apply_velocity_intent)


class LivenessTests(unittest.TestCase):
    def test_monitor_detects_safe_interaction_stall(self):
        monitor = LocalSBSMonitor(window_steps=4, dt=0.25)
        evidence = None
        for _ in range(4):
            evidence = monitor.update(LocalStep(0.4, 0.01, True, True, False))
        self.assertTrue(evidence.ready)
        self.assertTrue(evidence.online_sbs)

    def test_monitor_excludes_obstacle_only_stop(self):
        monitor = LocalSBSMonitor(window_steps=4, dt=0.25)
        for _ in range(4):
            evidence = monitor.update(LocalStep(0.4, 0.0, False, True, False))
        self.assertFalse(evidence.online_sbs)

    def test_gate_requires_confirmation(self):
        gate = CoordinationGate(entry_confirm_steps=3)
        evidence = type("E", (), {"online_sbs": True, "reached": False, "ready": True,
                                   "progress_ratio": 0.0, "active_fraction": 1.0})()
        self.assertEqual(gate.update(evidence), GateMode.BYPASS)
        self.assertEqual(gate.update(evidence), GateMode.BYPASS)
        self.assertEqual(gate.update(evidence), GateMode.COORDINATE)

    def test_identity_intent_is_exact_bypass(self):
        preferred = np.array([0.3, -0.2])
        np.testing.assert_allclose(apply_velocity_intent(preferred, Intent()), preferred)

    def test_gate_releases_after_progress_recovers(self):
        gate = CoordinationGate(entry_confirm_steps=1, release_confirm_steps=2)
        stalled = type("E", (), {"online_sbs": True, "reached": False, "ready": True,
                                  "progress_ratio": 0.0, "active_fraction": 1.0})()
        moving = type("E", (), {"online_sbs": False, "reached": False, "ready": True,
                                 "progress_ratio": 0.8, "active_fraction": 1.0})()
        self.assertEqual(gate.update(stalled), GateMode.COORDINATE)
        self.assertEqual(gate.update(moving), GateMode.COORDINATE)
        self.assertEqual(gate.update(moving), GateMode.BYPASS)


if __name__ == "__main__":
    unittest.main()
