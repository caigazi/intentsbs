import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sbs824.v2.student import IntentStudent


class IntentStudentTests(unittest.TestCase):
    def setUp(self):
        self.model = IntentStudent()
        self.self_features = jnp.asarray(np.linspace(-0.4, 0.4, 18).reshape(2, 9))
        self.edges = jnp.asarray(np.linspace(-0.8, 0.8, 54).reshape(2, 3, 9))
        self.mask = jnp.asarray([[True, True, False], [True, False, False]])
        self.variables = self.model.init(
            jax.random.PRNGKey(824), self.self_features, self.edges, self.mask)

    def test_output_shape_and_bounds(self):
        output = np.asarray(self.model.apply(
            self.variables, self.self_features, self.edges, self.mask))
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(np.all(np.abs(output) <= 1.0))

    def test_deployed_output_is_tanh_of_training_logits(self):
        logits = self.model.apply(
            self.variables, self.self_features, self.edges, self.mask,
            return_logits=True)
        output = self.model.apply(
            self.variables, self.self_features, self.edges, self.mask)
        np.testing.assert_allclose(output, np.tanh(np.asarray(logits)), atol=1e-6)

    def test_neighbor_permutation_invariance(self):
        order = jnp.asarray([1, 0, 2])
        original = self.model.apply(
            self.variables, self.self_features, self.edges, self.mask)
        permuted = self.model.apply(
            self.variables, self.self_features,
            self.edges[:, order], self.mask[:, order])
        np.testing.assert_allclose(original, permuted, atol=1e-6)

    def test_masked_neighbor_is_ignored(self):
        changed = np.asarray(self.edges).copy()
        changed[0, 2] = 1000.0
        changed[1, 1:] = -1000.0
        original = self.model.apply(
            self.variables, self.self_features, self.edges, self.mask)
        modified = self.model.apply(
            self.variables, self.self_features, jnp.asarray(changed), self.mask)
        np.testing.assert_allclose(original, modified, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
