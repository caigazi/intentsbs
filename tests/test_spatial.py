import unittest

import numpy as np

from sbs824.spatial import radius_neighbor_lists, radius_neighbor_pairs


class SpatialNeighborTests(unittest.TestCase):
    def test_grid_matches_brute_force_radius_graph(self):
        rng = np.random.default_rng(824)
        for n_agents in (2, 8, 32, 128):
            positions = rng.uniform(-1.0, 5.0, size=(n_agents, 2))
            radius = 0.5
            expected = []
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    if np.linalg.norm(positions[i] - positions[j]) <= radius:
                        expected.append((i, j))
            self.assertEqual(radius_neighbor_pairs(positions, radius), expected)

    def test_neighbor_lists_are_symmetric_and_local(self):
        positions = np.array([[0.0, 0.0], [0.49, 0.0], [0.51, 0.0],
                              [2.0, 2.0]])
        neighbors = radius_neighbor_lists(positions, 0.5)
        self.assertEqual(neighbors[0].tolist(), [1])
        self.assertEqual(neighbors[1].tolist(), [0, 2])
        self.assertEqual(neighbors[2].tolist(), [1])
        self.assertEqual(neighbors[3].tolist(), [])


if __name__ == "__main__":
    unittest.main()
