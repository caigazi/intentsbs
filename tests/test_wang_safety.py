import unittest

import numpy as np

from sbs824.geometry import Rectangle
from sbs824.simulation import Config
from sbs824.wang_safety import (
    certified_braking_force,
    obstacle_braking_barrier,
    obstacle_force_row,
    solve_wang_braking_qp,
    wang_pair_acceleration_halfspace,
    wang_pair_barrier,
    wang_pair_force_row,
)


class WangSafetyTests(unittest.TestCase):
    def test_eq16_a_hat_is_negative_velocity_gradient(self):
        cfg = Config(n_agents=2, n_obstacles=0)
        left = np.array([0.6, 0.2, -0.23, 0.17])
        right = np.array([-0.4, -0.1, 0.31, -0.08])
        a_hat, _, _ = wang_pair_acceleration_halfspace(left, right, cfg)
        epsilon = 1e-6
        gradient = np.zeros(2)
        for axis in range(2):
            plus, minus = left.copy(), left.copy()
            plus[2 + axis] += epsilon
            minus[2 + axis] -= epsilon
            gradient[axis] = (
                wang_pair_barrier(plus, right, cfg)
                - wang_pair_barrier(minus, right, cfg)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(a_hat, -gradient, rtol=2e-5, atol=2e-7)

    def test_two_decentralized_rhs_sum_to_joint_zcbf_rhs(self):
        cfg = Config(n_agents=2, n_obstacles=0)
        left = np.array([0.7, -0.1, -0.2, 0.13])
        right = np.array([-0.5, 0.2, 0.29, -0.07])
        _, b_ij, h = wang_pair_acceleration_halfspace(left, right, cfg)
        _, b_ji, h_reverse = wang_pair_acceleration_halfspace(right, left, cfg)
        alpha = cfg.max_force / cfg.mass
        vi, vj = left[2:], right[2:]
        delta_p, delta_v = left[:2] - right[:2], vi - vj
        q_delta = (delta_p
                   + np.linalg.norm(vi) * vi / (4.0 * alpha)
                   - np.linalg.norm(vj) * vj / (4.0 * alpha))
        d_safe = cfg.wang_pair_safe_radius_factor * cfg.car_radius
        gamma = cfg.wang_gamma / d_safe ** 4
        expected = 2.0 * float(q_delta @ delta_v) + gamma * h ** 3
        self.assertAlmostEqual(h, h_reverse, places=12)
        self.assertAlmostEqual(b_ij + b_ji, expected, places=10)

    def test_force_conversion_occurs_once(self):
        cfg = Config(n_agents=2, n_obstacles=0, mass=0.2, max_force=1.0)
        left = np.array([0.4, 0.0, -0.2, 0.1])
        right = np.array([-0.4, 0.0, 0.2, 0.0])
        a_hat, b_hat, _ = wang_pair_acceleration_halfspace(left, right, cfg)
        row, rhs, _ = wang_pair_force_row(left, right, cfg)
        np.testing.assert_allclose(row, -a_hat / cfg.mass)
        self.assertAlmostEqual(rhs, -b_hat)

    def test_eq17_commands_maximum_braking(self):
        cfg = Config(n_agents=1, dt=0.03, mass=0.1, max_force=1.0)
        velocity = np.array([0.02, -0.01])
        force = certified_braking_force(velocity, cfg)
        self.assertAlmostEqual(np.linalg.norm(force), cfg.max_force)
        self.assertLess(float(force @ velocity), 0.0)

    def test_static_obstacle_row_matches_barrier_velocity_gradient(self):
        cfg = Config(n_agents=1, n_obstacles=1)
        obstacle = Rectangle(np.array([0.0, 0.0]), np.array([0.4, 0.6]), 0.0)
        state = np.array([0.8, 0.12, -0.31, 0.09])
        force_row, _, _ = obstacle_force_row(state, obstacle, cfg)
        a_hat = -force_row * cfg.mass
        epsilon = 1e-6
        gradient = np.zeros(2)
        for axis in range(2):
            plus, minus = state.copy(), state.copy()
            plus[2 + axis] += epsilon
            minus[2 + axis] -= epsilon
            gradient[axis] = (
                obstacle_braking_barrier(plus, obstacle, cfg)
                - obstacle_braking_barrier(minus, obstacle, cfg)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(a_hat, -gradient, rtol=2e-5, atol=2e-7)

    def test_eq17_latch_is_certified_brake_not_fallback(self):
        cfg = Config(n_agents=3, n_obstacles=0, sense_radius=2.0)
        state = np.array([
            [0.0, 0.0, 0.4, 0.0],
            [0.24, 0.0, -0.4, 0.0],
            [-0.24, 0.0, 0.4, 0.0],
        ])
        reference = np.tile(np.array([1.0, 0.0]), (3, 1))
        latch = np.array([True, False, False])
        result = solve_wang_braking_qp(
            state, reference, [], cfg, braking_latch=latch)
        self.assertEqual(result.infeasible_components, [])
        self.assertTrue(result.certified_brake_agents)
        # Lemma VI.1 needs both endpoints to brake; the emergency bit therefore
        # propagates across this connected three-agent component.
        self.assertEqual(result.certified_brake_agents, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
