import unittest

import numpy as np

from sbs824.simulation import Config, _sbs_metrics, generate_scene


class AttributionTests(unittest.TestCase):
    def _base(self, steps: int = 8):
        cfg = Config(n_agents=1, stall_window_steps=4, dt=0.25,
                     sbs_nominal_progress_epsilon=0.05,
                     sbs_causal_progress_epsilon=0.02)
        q_nom = np.full((steps, 1), 0.4)
        q_safe = np.zeros((steps, 1))
        reached = np.zeros((steps + 1, 1), dtype=bool)
        collision = np.zeros((steps + 1, 1), dtype=bool)
        return cfg, q_nom, q_safe, reached, collision

    def test_persistent_agent_cause_is_a_sbs(self):
        cfg, q_nom, q_safe, reached, collision = self._base()
        yes = np.ones_like(q_nom, dtype=bool)
        no = np.zeros_like(yes)
        result = _sbs_metrics(q_nom, q_safe, yes, no, q_nom, q_safe,
                              reached, collision, cfg)
        self.assertEqual(result["final_class"][0], 2)
        self.assertTrue(result["a_persistent"][0])

    def test_persistent_obstacle_cause_is_o_sbs(self):
        cfg, q_nom, q_safe, reached, collision = self._base()
        yes = np.ones_like(q_nom, dtype=bool)
        no = np.zeros_like(yes)
        result = _sbs_metrics(q_nom, q_safe, no, yes, q_safe, q_nom,
                              reached, collision, cfg)
        self.assertEqual(result["final_class"][0], 1)
        self.assertTrue(result["o_persistent"][0])

    def test_obstacle_first_then_agent_is_m_sbs(self):
        cfg, q_nom, q_safe, reached, collision = self._base(steps=10)
        agent = np.zeros_like(q_nom, dtype=bool)
        obstacle = np.zeros_like(agent)
        agent[6:] = True
        obstacle[:4] = True
        q_no_agent = np.zeros_like(q_nom)
        q_no_obstacle = np.zeros_like(q_nom)
        q_no_agent[6:] = q_nom[6:]
        q_no_obstacle[:4] = q_nom[:4]
        result = _sbs_metrics(q_nom, q_safe, agent, obstacle,
                              q_no_agent, q_no_obstacle, reached, collision, cfg)
        self.assertEqual(result["final_class"][0], 3)
        self.assertTrue(result["m_persistent"][0])

    def test_generic_stall_before_agent_without_obstacle_is_not_m_sbs(self):
        cfg, q_nom, q_safe, reached, collision = self._base(steps=10)
        agent = np.zeros_like(q_nom, dtype=bool)
        obstacle = np.zeros_like(agent)
        agent[6:] = True
        q_no_agent = np.zeros_like(q_nom)
        q_no_agent[6:] = q_nom[6:]
        result = _sbs_metrics(q_nom, q_safe, agent, obstacle,
                              q_no_agent, q_safe, reached, collision, cfg)
        self.assertEqual(result["final_class"][0], 2)
        self.assertFalse(result["m_persistent"][0])

    def test_simultaneous_agent_obstacle_cause_is_ambiguous(self):
        cfg, q_nom, q_safe, reached, collision = self._base()
        pressure = np.ones_like(q_nom, dtype=bool)
        result = _sbs_metrics(q_nom, q_safe, pressure, pressure,
                              q_nom, q_nom, reached, collision, cfg)
        self.assertEqual(result["final_class"][0], 4)
        self.assertTrue(result["x_persistent"][0])

    def test_random_rectangles_do_not_form_overlapping_compounds(self):
        cfg = Config(n_agents=4, n_obstacles=8, seed=3)
        _, _, obstacles = generate_scene(cfg)
        for i, left in enumerate(obstacles):
            for right in obstacles[i + 1:]:
                conservative_gap = (np.linalg.norm(left.center - right.center)
                                    - 0.5 * np.linalg.norm(left.size)
                                    - 0.5 * np.linalg.norm(right.size))
                self.assertGreater(conservative_gap, cfg.obstacle_clearance)


if __name__ == "__main__":
    unittest.main()
